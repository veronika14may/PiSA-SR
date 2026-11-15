"""К6: дообучение декодера VAE на аэроснимках при замороженном энкодере.

    python freq_exp/train_vae_decoder.py --pretrained_model_path /path/sd-2-1-base \
        --hq_dir data/train/hq --val_hq_dir data/val/hq --output_dir experiments/freq/E17_K6_vae_ffl --w_ffl 1.0

Схема: эталон -> энкодер VAE (заморожен) -> латент -> декодер (обучается) -> реконструкция.
Потери между реконструкцией и эталоном: L1 + LPIPS и, по выбору, те же частотные члены, что в train_freq.py
(Focal Frequency Loss, Fourier Space Loss, FDL). UNet и LQ здесь не участвуют.

Энкодер не меняется, поэтому латентное пространство остаётся прежним: готовый чекпоинт PiSA-SR, LoRA и
учитель CSD работают с новым декодером без переобучения. Декодер подключается флагом --vae_decoder_ckpt
в train_freq.py, infer_eval.py и diag_vae_ceiling.py.

Валидация — тот же расчёт, что в diag_vae_ceiling.py (центральный кроп 512, среднее латента), поэтому
строка шага 0 совпадает с исходным потолком VAE. Запускать из папки PiSA-SR-main.
"""
import os, sys, csv, json, time, random, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import losses as FL
import metrics as FM
from io_utils import IMG_EXT, load01, psnr
from vae_utils import load_vae, load_decoder, save_decoder, decoder_params

TERMS = ("l1", "lpips", "ffl", "fsl", "fdl")
BASE_TERMS = ("l1", "lpips")
FREQ_TERMS = ("ffl", "fsl", "fdl")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_path", required=True)
    p.add_argument("--hq_dir", required=True)
    p.add_argument("--val_hq_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--init_decoder", default=None, help="продолжить с сохранённого чекпоинта декодера")
    # обучение
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--crop", type=int, default=256, help="сторона обучающего кропа (кратна 8)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--latent", choices=["sample", "mode"], default="sample",
                   help="sample: случайный латент, как при обучении PiSA-SR; mode: среднее")
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    # потери (вес 0 = выключено)
    p.add_argument("--w_l1", type=float, default=1.0)
    p.add_argument("--w_lpips", type=float, default=1.0)
    p.add_argument("--w_ffl", type=float, default=0.0)
    p.add_argument("--w_fsl", type=float, default=0.0)
    p.add_argument("--w_fdl", type=float, default=0.0)
    p.add_argument("--ffl_alpha", type=float, default=1.0)
    p.add_argument("--ffl_patch_factor", type=int, default=1)
    p.add_argument("--fdl_phase_weight", type=float, default=1.0)
    p.add_argument("--fdl_num_proj", type=int, default=256)
    p.add_argument("--aux_device", default="auto", help="куда поставить VGG для FDL: auto = cuda:1, если карт две")
    p.add_argument("--auto_weight_ratio", type=float, default=0.0,
                   help=">0: перед обучением подобрать веса частотных членов так, чтобы норма градиента каждого "
                        "составляла эту долю от нормы градиента базовой части (L1 + LPIPS с их весами)")
    p.add_argument("--auto_weight_batches", type=int, default=16)
    # валидация и логи
    p.add_argument("--n_val", type=int, default=100)
    p.add_argument("--val_crop", type=int, default=512)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--log_grad_every", type=int, default=100)
    p.add_argument("--gsd_cm", type=float, default=8.0)
    a = p.parse_args()
    assert a.crop % 8 == 0, "--crop должен быть кратен 8"
    a.device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.aux_device == "auto":
        a.aux_device = ("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0") if a.device == "cuda" else "cpu"
    return a


class HQCrops(Dataset):
    """Случайные кропы эталонов с отражениями и поворотами на 90° (для съёмки в надир все они допустимы)."""

    def __init__(self, hq_dir, crop):
        self.paths = sorted(os.path.join(hq_dir, f) for f in os.listdir(hq_dir) if f.lower().endswith(IMG_EXT))
        assert self.paths, f"в {hq_dir} нет изображений"
        self.crop = crop
        print(f"[train] {len(self.paths)} эталонов из {hq_dir}, кроп {crop}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        c = self.crop
        if min(im.size) < c:
            s = c / min(im.size)
            im = im.resize((max(c, round(im.width * s)), max(c, round(im.height * s))), Image.BICUBIC)
        x = random.randint(0, im.width - c); y = random.randint(0, im.height - c)
        t = torch.from_numpy(np.array(im.crop((x, y, x + c, y + c)))).permute(2, 0, 1).float() / 127.5 - 1
        if random.random() < 0.5: t = t.flip(-1)
        if random.random() < 0.5: t = t.flip(-2)
        return torch.rot90(t, random.randint(0, 3), dims=(-2, -1))


def load_val(hq_dir, n, crop):
    """Те же снимки и кропы, что в diag_vae_ceiling.py."""
    names = sorted(f for f in os.listdir(hq_dir) if f.lower().endswith(IMG_EXT))
    names = names[:n] if n > 0 else names
    out = []
    for nm in names:
        hq = load01(os.path.join(hq_dir, nm))
        h, w = hq.shape[-2:]
        c = min(crop, h - h % 8, w - w % 8)
        i, j = (h - c) // 2, (w - c) // 2
        out.append(hq[..., i:i + c, j:j + c])
    print(f"[val] {len(out)} снимков из {hq_dir}, кроп {crop}")
    return out


class Losses:
    """Члены потерь между реконструкцией и эталоном (оба в [-1, 1]). unit() возвращает невзвешенные значения."""

    def __init__(self, a):
        import lpips
        self.w = {k: getattr(a, f"w_{k}") for k in TERMS}
        self.lpips = lpips.LPIPS(net="vgg").to(a.device).requires_grad_(False) if self.w["lpips"] else None
        self.ffl = FL.build_ffl(a.ffl_alpha, a.ffl_patch_factor) if self.w["ffl"] else None
        self.fdl = (FL.build_fdl(a.fdl_phase_weight, a.fdl_num_proj, device=a.aux_device) if self.w["fdl"] else None)

    def unit(self, rec, hq):
        p, t = rec.float(), hq.float()
        out = {}
        if self.w["l1"]:
            out["l1"] = F.l1_loss(p, t)
        if self.w["lpips"]:
            out["lpips"] = self.lpips(p, t).mean()
        if self.w["ffl"]:
            out["ffl"] = self.ffl(p, t)
        if self.w["fsl"]:
            out["fsl"] = FL.fourier_space_loss(p, t)
        if self.w["fdl"]:
            out["fdl"] = FL.fdl_call(self.fdl, p, t, "full").to(p.device)
        return out

    def total(self, units):
        return sum(self.w[k] * v for k, v in units.items())


def grad_norms(units, leaf):
    out = {}
    for k, v in units.items():
        g, = torch.autograd.grad(v, leaf, retain_graph=True, allow_unused=True)
        out[k] = float(g.float().norm()) if g is not None else 0.0
    return out


@torch.no_grad()
def evaluate(vae, val, lp, a):
    acc = np.zeros((FM.N_BANDS, 4)); ps, lps, vols, vols_gt = [], [], [], []
    for hq in val:
        hq = hq.to(a.device)
        z = vae.encode(hq * 2 - 1).latent_dist.mode()
        rec = (vae.decode(z).sample.float().clamp(-1, 1) * 0.5 + 0.5)
        acc += FM.band_stats(rec.cpu(), hq.cpu()); ps.append(psnr(rec.cpu(), hq.cpu()))
        vols.append(FM.vol_of(transforms.ToPILImage()(rec[0].cpu())))
        vols_gt.append(FM.vol_of(transforms.ToPILImage()(hq[0].cpu())))
        if lp is not None:
            lps.append(float(lp(rec * 2 - 1, hq * 2 - 1)))
    s, bands = FM.summary(acc, gsd_cm=a.gsd_cm)
    r = {"psnr": float(np.mean(ps)), "lpips": float(np.mean(lps)) if lps else float("nan"),
         "vol_ratio": float(np.mean(vols) / np.mean(vols_gt)), **s}
    return r, bands


def main():
    a = get_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    ck_dir = os.path.join(a.output_dir, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    with open(os.path.join(a.output_dir, "args_vae.json"), "w") as jf:
        json.dump(vars(a), jf, indent=2, ensure_ascii=False)
    dev = a.device

    vae = load_vae(a.pretrained_model_path).to(dev)
    vae.requires_grad_(False); vae.eval()
    if a.init_decoder:
        load_decoder(vae, a.init_decoder)
    params = decoder_params(vae)
    for p_ in params:
        p_.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)   # без затухания: обучаются сами веса, не LoRA
    bank = Losses(a)
    lp_val = bank.lpips
    if lp_val is None:
        import lpips
        lp_val = lpips.LPIPS(net="vgg").to(dev).requires_grad_(False)

    train_dl = DataLoader(HQCrops(a.hq_dir, a.crop), batch_size=a.batch_size, shuffle=True,
                          num_workers=a.num_workers, drop_last=True)
    val = load_val(a.val_hq_dir, a.n_val, a.val_crop)

    use_amp = a.mixed_precision != "no" and dev == "cuda"
    amp_dtype = torch.bfloat16 if a.mixed_precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(dev, enabled=(use_amp and a.mixed_precision == "fp16"))

    f = open(os.path.join(a.output_dir, "curve_vae.csv"), "w", newline=""); w = csv.writer(f)
    w.writerow(["global_step", "psnr", "lpips_val", "vol_ratio", "hf_frc", "hf_err", "hf_energy", "f_eff",
                "eff_gsd_cm", "mem0_gb", "it_per_s"])
    fb = open(os.path.join(a.output_dir, "bands_vae.csv"), "w", newline=""); wb = csv.writer(fb)
    wb.writerow(["global_step", "metric"] + [f"b{i}" for i in range(FM.N_BANDS)])
    ft = open(os.path.join(a.output_dir, "train_log_vae.csv"), "w", newline=""); wt = csv.writer(ft)
    wt.writerow(["global_step"] + [f"val_{k}" for k in TERMS] + [f"w_{k}" for k in TERMS] + [f"gnorm_{k}" for k in TERMS])
    state = {"speed": float("nan"), "bands": None}

    def log_eval(step):
        r, bands = evaluate(vae, val, lp_val, a)
        state["bands"] = bands
        mem = torch.cuda.max_memory_allocated(0) / 2 ** 30 if dev == "cuda" else float("nan")
        w.writerow([step, f"{r['psnr']:.3f}", f"{r['lpips']:.5f}", f"{r['vol_ratio']:.4f}", f"{r['hf_frc']:.4f}",
                    f"{r['hf_err']:.4f}", f"{r['hf_energy']:.4f}", f"{r['f_eff']:.4f}", f"{r['eff_gsd_cm']:.2f}",
                    f"{mem:.2f}", f"{state['speed']:.3f}"]); f.flush()
        for name in ("frc", "err", "energy"):
            wb.writerow([step, name] + [f"{v:.4f}" for v in bands[name]])
        fb.flush()
        print(f"[eval] шаг {step:5d}  PSNR {r['psnr']:.2f} дБ  LPIPS {r['lpips']:.4f}  VoL/GT {r['vol_ratio']:.3f}  "
              f"ВЧ: FRC {r['hf_frc']:.3f} ошибка {r['hf_err']:.3f} энергия {r['hf_energy']:.3f}  "
              f"эфф. разрешение {r['eff_gsd_cm']:.1f} см")
        return r

    def forward(hq):
        with torch.no_grad():
            dist = vae.encode(hq).latent_dist
            z = dist.sample() if a.latent == "sample" else dist.mode()
        with torch.autocast(dev, dtype=amp_dtype, enabled=use_amp):
            return vae.decode(z).sample

    def calibrate():
        sums, n = {}, 0
        it = iter(train_dl)
        for _ in range(a.auto_weight_batches):
            try:
                hq = next(it).to(dev)
            except StopIteration:
                break
            with torch.no_grad():
                rec = forward(hq)
            leaf = rec.detach().float().requires_grad_(True)
            units = bank.unit(leaf, hq)
            units["base"] = sum(bank.w[k] * units[k] for k in BASE_TERMS if k in units)
            for k, v in grad_norms(units, leaf).items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        ref = sums["base"] / n
        for k in FREQ_TERMS:
            if bank.w[k] and sums.get(k, 0) > 0:
                new = a.auto_weight_ratio * ref / (sums[k] / n)
                print(f"[калибровка] {k}: вес {bank.w[k]:.3g} -> {new:.3g} (норма градиента = {a.auto_weight_ratio:.2f} от L1 + LPIPS)")
                bank.w[k] = new

    if a.auto_weight_ratio > 0 and any(bank.w[k] for k in BASE_TERMS) and any(bank.w[k] for k in FREQ_TERMS):
        calibrate()
    print(f"[К6] обучаемых параметров декодера: {sum(p_.numel() for p_ in params) / 1e6:.1f}M; "
          f"веса потерь: { {k: v for k, v in bank.w.items() if v} }; устройство {dev}, VGG для FDL на {a.aux_device}")

    first = log_eval(0)
    step, t0 = 0, time.time()
    while step < a.steps:
        for hq in train_dl:
            hq = hq.to(dev)
            rec = forward(hq)
            with torch.autocast(dev, dtype=amp_dtype, enabled=use_amp):
                units = bank.unit(rec, hq)
                loss = bank.total(units)
            if not torch.isfinite(loss):
                raise RuntimeError(f"потеря стала {float(loss)} на шаге {step + 1}: уменьшите --lr или веса частотных членов")
            if a.log_grad_every and (step + 1) % a.log_grad_every == 0:
                gn = grad_norms(units, rec)
                vals = {k: float(v.detach()) for k, v in units.items()}
                wt.writerow([step + 1] + [f"{vals.get(k, float('nan')):.6g}" for k in TERMS]
                            + [f"{bank.w[k]:.4g}" for k in TERMS]
                            + [f"{gn.get(k, float('nan')):.4g}" for k in TERMS]); ft.flush()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
            step += 1
            if step % a.log_every == 0:
                state["speed"] = step / (time.time() - t0)
                print(f"[К6] {step}/{a.steps} " + " ".join(f"{k} {float(v.detach()) * bank.w[k]:.4f}" for k, v in units.items())
                      + f" ({state['speed']:.2f} шаг/с)")
            if step % a.eval_every == 0 or step == a.steps:
                last = log_eval(step)
                save_decoder(vae, os.path.join(ck_dir, "vae_decoder_last.pt"), {"step": step, **last})
            if step >= a.steps:
                break
    final = os.path.join(ck_dir, "vae_decoder_final.pt")
    save_decoder(vae, final, {"step": step, "weights": dict(bank.w), **last})
    f.close(); fb.close(); ft.close()

    print("\nПотолок VAE после дообучения:\n" + FM.format_bands(state["bands"]))
    print(f"\nБыло -> стало: PSNR {first['psnr']:.2f} -> {last['psnr']:.2f} дБ; ошибка в ВЧ {first['hf_err']:.3f} -> "
          f"{last['hf_err']:.3f}; энергия ВЧ {first['hf_energy']:.3f} -> {last['hf_energy']:.3f}; эффективное "
          f"разрешение {first['eff_gsd_cm']:.1f} -> {last['eff_gsd_cm']:.1f} см")
    try:
        import pandas as pd, matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        d = pd.read_csv(os.path.join(a.output_dir, "curve_vae.csv"))
        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        axes[0].plot(d.global_step, d.psnr, "-o", ms=2.5); axes[0].set_ylabel("PSNR реконструкции, дБ ↑")
        axes[1].plot(d.global_step, d.hf_err, "-o", ms=2.5, label="ошибка")
        axes[1].plot(d.global_step, d.hf_energy, "-o", ms=2.5, label="энергия")
        axes[1].axhline(1.0, ls=":", c="k"); axes[1].set_ylabel("ВЧ-полосы"); axes[1].legend()
        axes[2].plot(d.global_step, d.eff_gsd_cm, "-o", ms=2.5); axes[2].set_ylabel("эффективное разрешение, см ↓")
        axes[2].set_xlabel("шаг дообучения декодера")
        for ax in axes: ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(a.output_dir, "curve_vae.png"), dpi=200)
    except Exception as e:
        print("график не построен:", e)
    print("готово:", final)


if __name__ == "__main__":
    main()
