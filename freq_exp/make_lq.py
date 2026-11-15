"""Создание LQ из HQ слепой деградацией Real-ESRGAN (две ступени) без сдвига относительно HQ.

    python freq_exp/make_lq.py --hq_dir data/train/hq --out_dir data/train/lq
    python freq_exp/make_lq.py --hq_dir data/val/hq   --out_dir data/val/lq --seed 1

Что сохранено от Real-ESRGAN: случайные ядра размытия (изотропные, анизотропные, обобщённые, «плато», sinc),
случайное изменение размера, гауссов и пуассоновский шум, JPEG, вторая ступень и финальный sinc-фильтр.
Параметры по умолчанию взяты из конфига DiffBIR, которым делались прежние LQ.

Чем отличается от оригинала (ради совмещения):
  * размер при каждом изменении задаётся явно (size=...), а не через scale_factor — именно scale_factor
    с округлением вниз давал случайный сдвиг;
  * все ядра нечётного размера и симметричны относительно центра;
  * JPEG настоящий (Pillow), а не дифференцируемое приближение.
Не делается повышение резкости HQ (use_sharpener из DiffBIR): эталон остаётся исходным снимком.

Пресеты (--preset):
  diffbir (по умолчанию) — конфиг DiffBIR/Real-ESRGAN, рассчитан на фото из интернета: две ступени, шум σ до 10,
                           JPEG 30–95, случайный масштаб 0,15–1,5, sinc-фильтры.
  mild                   — мягкий вариант под снимок другого сенсора с меньшим разрешением: PSF-размытие,
                           точное понижение в 4 раза (интегрирование по площади), небольшой шум сенсора на уровне LQ,
                           редкий мягкий JPEG. Без случайного масштаба и sinc-ореолов.

Деградация каждого файла определяется его именем и --seed, поэтому повторный запуск даёт те же LQ.
Скрипт не зависит от basicsr; нужны torch, numpy, scipy, Pillow.
"""
import os, sys, io, argparse, zlib, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import special

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
MODES = ["area", "bilinear", "bicubic"]

# Параметры деградации (как в конфиге DiffBIR)
OPT = dict(
    kernel_sizes=list(range(7, 22, 2)),            # 7, 9, ..., 21
    kernel_list=["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"],
    kernel_prob=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    betag_range=[0.5, 4], betap_range=[1, 2],
    # первая ступень
    sinc_prob=0.1, blur_sigma=[0.2, 1.5],
    resize_prob=[0.2, 0.7, 0.1], resize_range=[0.15, 1.5],          # вероятности: увеличить, уменьшить, оставить
    gaussian_noise_prob=0.3, noise_range=[1, 10], poisson_scale_range=[0.05, 1.0], gray_noise_prob=0.2,
    jpeg_range=[30, 95],
    # вторая ступень
    second_blur_prob=0.8, sinc_prob2=0.1, blur_sigma2=[0.2, 1.0],
    resize_prob2=[0.3, 0.4, 0.3], resize_range2=[0.3, 1.2],
    gaussian_noise_prob2=0.3, noise_range2=[1, 8], poisson_scale_range2=[0.05, 0.8], gray_noise_prob2=0.2,
    jpeg_range2=[30, 95],
    final_sinc_prob=0.8,
)

# Мягкий пресет (вариант B). Размытие задано в пикселях HQ, шум и JPEG применяются уже на масштабе LQ.
OPT_MILD = dict(
    kernel_sizes=[9, 11, 13, 15, 17],
    kernel_list=["iso", "aniso", "generalized_iso", "plateau_iso"],
    kernel_prob=[0.55, 0.30, 0.10, 0.05],
    betag_range=[0.5, 4], betap_range=[1, 2],
    blur_sigma=[0.8, 2.5],                       # ≈ 0.2–0.6 пикселя LQ при масштабе 4: PSF оптики и сенсора
    resize_modes=["area", "bicubic", "bilinear"], resize_prob=[0.6, 0.3, 0.1],   # bicubic/bilinear — с антиалиасингом
    gaussian_noise_prob=0.5, noise_range=[0.5, 3], poisson_scale_range=[0.05, 0.3], gray_noise_prob=0.1,
    jpeg_prob=0.7, jpeg_range=[70, 95],
)
PRESETS = {"diffbir": OPT, "mild": OPT_MILD}


# ----------------------------- ядра размытия -----------------------------
def _grid(k):
    ax = np.arange(k) - (k - 1) / 2.0
    return np.meshgrid(ax, ax)                       # xx, yy; центр ядра точно в середине (k нечётное)


def _quad_form(k, sx, sy, theta):
    """x^T Σ^-1 x для повёрнутой гауссовой формы."""
    xx, yy = _grid(k)
    c, s = np.cos(theta), np.sin(theta)
    u, v = c * xx + s * yy, -s * xx + c * yy
    return (u / sx) ** 2 + (v / sy) ** 2


def mixed_kernel(rng, k, sigma_range, o):
    kind = rng.choice(o["kernel_list"], p=o["kernel_prob"])
    sx = rng.uniform(*sigma_range)
    if kind.endswith("aniso"):
        sy, theta = rng.uniform(*sigma_range), rng.uniform(-np.pi, np.pi)
    else:
        sy, theta = sx, 0.0
    q = _quad_form(k, sx, sy, theta)
    if kind.startswith("generalized"):
        b = rng.uniform(o["betag_range"][0], 1) if rng.uniform() < 0.5 else rng.uniform(1, o["betag_range"][1])
        ker = np.exp(-0.5 * np.power(q, b))
    elif kind.startswith("plateau"):
        b = rng.uniform(o["betap_range"][0], 1) if rng.uniform() < 0.5 else rng.uniform(1, o["betap_range"][1])
        ker = 1.0 / (np.power(q, b) + 1.0)
    else:
        ker = np.exp(-0.5 * q)
    return ker / ker.sum()


def sinc_kernel(k, omega_c):
    """Идеальный круговой фильтр нижних частот (вызывает ореолы, как в Real-ESRGAN)."""
    xx, yy = _grid(k)
    r = np.sqrt(xx ** 2 + yy ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ker = omega_c * special.j1(omega_c * r) / (2 * np.pi * r)
    ker[(k - 1) // 2, (k - 1) // 2] = omega_c ** 2 / (4 * np.pi)
    return ker / ker.sum()


def random_kernel(rng, sinc_prob, sigma_range, o):
    k = int(rng.choice(o["kernel_sizes"]))
    if rng.uniform() < sinc_prob:
        omega = rng.uniform(np.pi / 3, np.pi) if k < 13 else rng.uniform(np.pi / 5, np.pi)
        return sinc_kernel(k, omega)
    return mixed_kernel(rng, k, sigma_range, o)


def filter2d(x, ker):
    """Свёртка с отражением на краях; размер не меняется, сдвига нет (ядро нечётное)."""
    k = ker.shape[0]
    w = torch.from_numpy(ker).float().to(x.device)[None, None].repeat(x.shape[1], 1, 1, 1)
    p = k // 2
    while p > 0:                                     # отражение допускает отступ меньше размера картинки
        step = min(p, min(x.shape[-2:]) - 1)
        x = F.pad(x, (step,) * 4, mode="reflect")
        p -= step
    return F.conv2d(x, w, groups=x.shape[1])


# ----------------------------- шум и JPEG -----------------------------
def add_noise(rng, x, gauss_prob, sigma_range, poisson_range, gray_prob):
    gray = rng.uniform() < gray_prob
    c, h, w = x.shape[1:]
    if rng.uniform() < gauss_prob:
        sigma = rng.uniform(*sigma_range) / 255.0
        n = rng.standard_normal((1, 1 if gray else c, h, w)) * sigma
    else:
        scale = rng.uniform(*poisson_range)
        img = np.clip((x.cpu().numpy() * 255.0).round(), 0, 255) / 255.0
        if gray:
            img = (0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3])
            img = np.clip((img * 255.0).round(), 0, 255) / 255.0
        vals = 2 ** np.ceil(np.log2(max(len(np.unique(img)), 2)))
        n = (rng.poisson(img * vals) / vals - img) * scale
    n = torch.from_numpy(np.broadcast_to(n, (1, c, h, w)).copy()).float().to(x.device)
    return (x + n).clamp(0, 1)


def jpeg(rng, x, q_range):
    q = int(round(rng.uniform(*q_range)))
    arr = (x[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=q)
    buf.seek(0)
    out = np.array(Image.open(buf).convert("RGB"))
    return torch.from_numpy(out).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(x.device)


def resize(x, size, mode, antialias=False):
    """Только явный размер: координаты считаются по отношению размеров, центры совпадают."""
    size = (max(int(size[0]), 8), max(int(size[1]), 8))
    if tuple(x.shape[-2:]) == size:
        return x
    kw = {} if mode == "area" else {"align_corners": False}
    if antialias and mode != "area":
        kw["antialias"] = True
    return F.interpolate(x, size=size, mode=mode, **kw).clamp(0, 1)


def pick_scale(rng, prob, rng_range):
    t = rng.choice(["up", "down", "keep"], p=prob)
    if t == "up":
        return rng.uniform(1, rng_range[1])
    if t == "down":
        return rng.uniform(rng_range[0], 1)
    return 1.0


# ----------------------------- деградация -----------------------------
@torch.no_grad()
def degrade(hq, rng, scale=4, o=OPT):
    """hq: (1,3,H,W) в [0,1]. Возвращает LQ (1,3,H/scale,W/scale) в [0,1]."""
    H, W = hq.shape[-2:]
    lq_size = (H // scale, W // scale)

    # --- первая ступень ---
    out = filter2d(hq, random_kernel(rng, o["sinc_prob"], o["blur_sigma"], o))
    s = pick_scale(rng, o["resize_prob"], o["resize_range"])
    out = resize(out, (round(H * s), round(W * s)), str(rng.choice(MODES)))
    out = add_noise(rng, out, o["gaussian_noise_prob"], o["noise_range"], o["poisson_scale_range"], o["gray_noise_prob"])
    out = jpeg(rng, out, o["jpeg_range"])

    # --- вторая ступень ---
    if rng.uniform() < o["second_blur_prob"]:
        out = filter2d(out, random_kernel(rng, o["sinc_prob2"], o["blur_sigma2"], o))
    s = pick_scale(rng, o["resize_prob2"], o["resize_range2"])
    out = resize(out, (round(lq_size[0] * s), round(lq_size[1] * s)), str(rng.choice(MODES)))
    out = add_noise(rng, out, o["gaussian_noise_prob2"], o["noise_range2"], o["poisson_scale_range2"], o["gray_noise_prob2"])

    # --- приведение к размеру LQ + финальный sinc-фильтр и JPEG в случайном порядке ---
    if rng.uniform() < o["final_sinc_prob"]:
        fk = sinc_kernel(int(rng.choice(o["kernel_sizes"])), rng.uniform(np.pi / 3, np.pi))
    else:
        fk = None
    mode = str(rng.choice(MODES))
    if rng.uniform() < 0.5:
        out = resize(out, lq_size, mode)
        if fk is not None:
            out = filter2d(out, fk)
        out = jpeg(rng, out.clamp(0, 1), o["jpeg_range2"])
    else:
        out = jpeg(rng, out, o["jpeg_range2"])
        out = resize(out, lq_size, mode)
        if fk is not None:
            out = filter2d(out, fk)
    return out.clamp(0, 1)


@torch.no_grad()
def degrade_mild(hq, rng, scale=4, o=OPT_MILD):
    """Мягкий пресет: размытие PSF -> точное понижение в scale раз -> шум сенсора -> (иногда) мягкий JPEG."""
    H, W = hq.shape[-2:]
    out = filter2d(hq, random_kernel(rng, 0.0, o["blur_sigma"], o))
    mode = str(rng.choice(o["resize_modes"], p=o["resize_prob"]))
    out = resize(out, (H // scale, W // scale), mode, antialias=True)
    out = add_noise(rng, out, o["gaussian_noise_prob"], o["noise_range"], o["poisson_scale_range"], o["gray_noise_prob"])
    if rng.uniform() < o["jpeg_prob"]:
        out = jpeg(rng, out, o["jpeg_range"])
    return out.clamp(0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hq_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--preset", choices=sorted(PRESETS), default="diffbir",
                   help="diffbir — конфиг Real-ESRGAN/DiffBIR; mild — мягкая деградация (см. описание в начале файла)")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="для валидации возьмите другой сид, чем для обучения")
    p.add_argument("--n", type=int, default=0, help="обработать только первые n файлов (0 = все)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true", help="пересоздать уже существующие файлы")
    p.add_argument("--check", type=int, default=200, help="на скольких парах проверить совмещение (0 = не проверять)")
    a = p.parse_args()

    names = sorted(f for f in os.listdir(a.hq_dir) if f.lower().endswith(IMG_EXT))
    assert names, f"в {a.hq_dir} нет изображений"
    if a.n > 0:
        names = names[:a.n]
    os.makedirs(a.out_dir, exist_ok=True)
    meta = os.path.join(a.out_dir, "_degradation.json")
    if os.path.exists(meta) and not a.overwrite:
        with open(meta) as f:
            prev = json.load(f).get("preset", "diffbir")      # файлы без поля preset созданы конфигом DiffBIR
        if prev != a.preset:
            raise SystemExit(f"В {a.out_dir} уже лежат LQ с пресетом '{prev}', а запрошен '{a.preset}'. "
                             f"Укажите другую папку (--out_dir) или добавьте --overwrite.")

    done = skipped = 0
    for i, nm in enumerate(names):
        dst = os.path.join(a.out_dir, os.path.splitext(nm)[0] + ".png")
        if os.path.exists(dst) and not a.overwrite:
            skipped += 1
            continue
        im = Image.open(os.path.join(a.hq_dir, nm)).convert("RGB")
        w, h = im.size
        assert h % a.scale == 0 and w % a.scale == 0, \
            f"{nm}: размер {w}x{h} не делится на {a.scale}; обрежьте HQ, иначе пары не совместятся"
        hq = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(a.device)
        rng = np.random.default_rng([a.seed, zlib.crc32(nm.encode())])   # зависит только от имени файла и сида
        lq = (degrade_mild if a.preset == "mild" else degrade)(hq, rng, a.scale)
        arr = (lq[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(arr).save(dst)
        done += 1
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(names)}", flush=True)
    print(f"создано: {done}, пропущено (уже были): {skipped}, папка: {a.out_dir}")
    with open(os.path.join(a.out_dir, "_degradation.json"), "w") as f:
        json.dump({"preset": a.preset, "seed": a.seed, "scale": a.scale, "opt": PRESETS[a.preset]}, f, ensure_ascii=False, indent=1)

    if a.check > 0:
        from diag_alignment import estimate_shift
        g = lambda x: 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
        sh = []
        for nm in names[:a.check]:
            hq_im = Image.open(os.path.join(a.hq_dir, nm)).convert("RGB")
            lq_im = Image.open(os.path.join(a.out_dir, os.path.splitext(nm)[0] + ".png")).convert("RGB")
            lq_im = lq_im.resize(hq_im.size, Image.BICUBIC)
            t = lambda im: torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255.0)
            sh.append(estimate_shift(g(t(hq_im)), g(t(lq_im))))
        sh = np.array(sh)
        print(f"\nпроверка совмещения на {len(sh)} парах (пикс. HQ):")
        print(f"  средний сдвиг: dy = {sh[:, 0].mean():+.3f}, dx = {sh[:, 1].mean():+.3f}")
        print(f"  медиана |сдвига|: {np.median(np.abs(sh)):.3f};  ст. откл.: dy = {sh[:, 0].std():.3f}, dx = {sh[:, 1].std():.3f}")
        print("  Среднее должно быть около 0. Разброс здесь — в основном погрешность оценки на шумных LQ.")


if __name__ == "__main__":
    main()
