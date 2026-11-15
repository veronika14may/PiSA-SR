"""Частотные потери и вспомогательные карты для экспериментов с PiSA-SR.

Все функции потерь взяты из опубликованных работ, собственных формул здесь нет:

  * Focal Frequency Loss      — Jiang et al., ICCV 2021 (официальный пакет focal-frequency-loss)
  * Fourier Space Loss        — Fuoli, Van Gool, Timofte, ICCV 2021, формулы (7)-(9)
  * Frequency Distribution Loss (FDL) — Ni et al., CVPR 2024 (официальный пакет fdl-pytorch)
  * Карта деталей (DAW)       — FiDeSR, формулы (3)-(7)
  * Карта границ Собеля       — RS-AdaDiff, формула (13)
  * Фильтр Баттерворта НЧ/ВЧ  — FiDeSR (LFIM), деление на полосы — TFDSR

Соглашение о диапазонах: изображения приходят в [-1, 1] (как в PiSA-SR);
функции, которым нужен [0, 1], переводят сами.
"""
import math

import torch
import torch.nn.functional as F


def to01(x):
    return (x.float() * 0.5 + 0.5).clamp(0, 1)


# --------------------------------------------------------------------------- #
# Fourier Space Loss (Fuoli et al., ICCV 2021)
# --------------------------------------------------------------------------- #
def _hann2d(h, w, device):
    return (torch.hann_window(h, device=device)[:, None]
            * torch.hann_window(w, device=device)[None, :])


def fourier_space_loss(pred, target, return_parts=False):
    """L_F = 1/2 * L_amp + 1/2 * L_phase  (формула 9).

    L_amp   — L1 между амплитудами (формула 7),
    L_phase — L1 по свёрнутой разности фаз (формула 8),
    перед БПФ — окно Ханна, суммирование по полуплоскости (rfft2) из-за эрмитовой симметрии.
    """
    p, t = pred.float(), target.float()
    win = _hann2d(p.shape[-2], p.shape[-1], p.device)
    P = torch.fft.rfft2(p * win)
    T = torch.fft.rfft2(t * win)
    l_amp = (P.abs() - T.abs()).abs().mean()
    d = torch.angle(P) - torch.angle(T)
    d = torch.atan2(torch.sin(d), torch.cos(d))          # свёртка разности в (-pi, pi]
    l_phase = d.abs().mean()
    loss = 0.5 * l_amp + 0.5 * l_phase
    if return_parts:
        return loss, l_amp, l_phase
    return loss


# --------------------------------------------------------------------------- #
# Focal Frequency Loss (официальная реализация)
# --------------------------------------------------------------------------- #
def build_ffl(alpha=1.0, patch_factor=1):
    try:
        from focal_frequency_loss import FocalFrequencyLoss
    except ImportError as e:
        raise ImportError("Нужен пакет: pip install focal-frequency-loss") from e
    return FocalFrequencyLoss(loss_weight=1.0, alpha=alpha, patch_factor=patch_factor)


# --------------------------------------------------------------------------- #
# Карты: детали (FiDeSR), границы (RS-AdaDiff)
# --------------------------------------------------------------------------- #
def _gray(x01):
    return 0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]


def sobel_mag(x01):
    """Карта границ h(I) = sqrt((Kx*I)^2 + (Ky*I)^2), считается поканально."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x01.device)
    ky = kx.t()
    c = x01.shape[1]
    wx = kx[None, None].repeat(c, 1, 1, 1)
    wy = ky[None, None].repeat(c, 1, 1, 1)
    xp = F.pad(x01, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(xp, wx, groups=c)
    gy = F.conv2d(xp, wy, groups=c)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)


def detail_map(hq, var_ks=7):
    """D = (Sobel + Laplacian + Variance) / 3 по эталону (FiDeSR, формула 3).

    В статье не сказано, как нормируются три отклика; здесь каждый делится на свой
    максимум по изображению, чтобы они были сопоставимы. Возвращает (N,1,H,W) в [0,1].
    """
    g = _gray(to01(hq))
    sob = sobel_mag(g)
    lap_k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=g.device)[None, None]
    lap = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), lap_k).abs()
    pad = var_ks // 2
    gp = F.pad(g, (pad, pad, pad, pad), mode="replicate")
    mu = F.avg_pool2d(gp, var_ks, stride=1)
    var = (F.avg_pool2d(gp ** 2, var_ks, stride=1) - mu ** 2).clamp(min=0)

    def norm(m):
        return m / m.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)

    return ((norm(sob) + norm(lap) + norm(var)) / 3.0).detach()


def daw_weight(hq, alpha=1.0, pred=None, lpips_map=None, p=0.5):
    """Пространственный вес потерь.

    Режим 'detail' (pred=None): w = 1 + alpha * D — вес растёт в областях с деталями
    (так же устроен TADL в TexADiff: 1 + alpha * M).
    Режим 'detail_error' (задан pred): w = 1 + alpha * D ⊙ E, где
    E = (1 - p) * |pred - hq| + p * LPIPS-карта (FiDeSR, формулы 4-7).
    Вес нормируется к среднему 1, чтобы не менять масштаб потери.
    """
    d = detail_map(hq)
    if pred is not None:
        e = (pred.detach().float() - hq.float()).abs().mean(1, keepdim=True)
        if lpips_map is not None:
            e = (1 - p) * e + p * lpips_map.detach().float()
        e = e / e.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        d = d * e
        d = d / d.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    w = 1.0 + alpha * d
    return (w / w.mean(dim=(-2, -1), keepdim=True)).detach()


def best_detail_crop(hq, size):
    """Координаты окна size x size с максимальной средней детальностью (по карте D)."""
    h, w = hq.shape[-2:]
    if size >= min(h, w):
        return 0, 0
    d = detail_map(hq).mean(0, keepdim=True)
    s = F.avg_pool2d(d, size, stride=max(1, size // 4))
    idx = int(torch.argmax(s.flatten()))
    ny, nx = s.shape[-2:]
    step = max(1, size // 4)
    return (idx // nx) * step, (idx % nx) * step


# --------------------------------------------------------------------------- #
# Деление на частотные полосы фильтром Баттерворта
# --------------------------------------------------------------------------- #
def butterworth_split(x, cutoff=0.25, order=2):
    """Возвращает (НЧ, ВЧ) составляющие. cutoff — доля частоты Найквиста."""
    x = x.float()
    h, w = x.shape[-2:]
    fy = torch.fft.fftfreq(h, device=x.device)[:, None]
    fx = torch.fft.rfftfreq(w, device=x.device)[None, :]
    r = torch.sqrt(fx ** 2 + fy ** 2) / 0.5
    lp = 1.0 / (1.0 + (r / cutoff) ** (2 * order))
    X = torch.fft.rfft2(x)
    low = torch.fft.irfft2(X * lp, s=(h, w))
    return low, x - low


# --------------------------------------------------------------------------- #
# FDL: официальный класс + расчёт по областям карты деталей
# --------------------------------------------------------------------------- #
def build_fdl(phase_weight=1.0, num_proj=256, patch_size=5, device="cuda"):
    # В пакете fdl-pytorch 1.0 в теле класса остались строки с .cuda(): на машине без CUDA
    # импорт падает. На время импорта подменяем .cuda() пустышкой (на GPU ничего не меняется).
    _orig_cuda = torch.Tensor.cuda
    if not torch.cuda.is_available():
        torch.Tensor.cuda = lambda self, *a, **k: self
    try:
        from FDL_pytorch import FDL_loss
    except ImportError as e:
        raise ImportError("Нужен пакет: pip install fdl-pytorch") from e
    finally:
        torch.Tensor.cuda = _orig_cuda

    class FDLRegion(FDL_loss):
        """Тот же расчёт, что в FDL_loss.forward, но признаки VGG можно домножить на
        пространственную карту (N,1,H,W) перед преобразованием Фурье: статистика
        амплитуд и фаз тогда собирается преимущественно по областям с деталями."""

        def forward(self, x, y, region=None, amp_weight=1.0, return_parts=False):
            fx = self.model(x)
            fy = self.model(y)
            amp_total, ph_total = 0.0, 0.0
            for i in range(len(fx)):
                a, b = fx[i], fy[i]
                if region is not None:
                    m = F.interpolate(region, size=a.shape[-2:], mode="bilinear", align_corners=False)
                    a, b = a * m, b * m
                fa = torch.fft.fftn(a, dim=(-2, -1))
                fb = torch.fft.fftn(b, dim=(-2, -1))
                amp_total = amp_total + self.forward_once(torch.abs(fa), torch.abs(fb), i)
                ph_total = ph_total + self.forward_once(torch.angle(fa), torch.angle(fb), i)
            amp, ph = amp_total.mean(), ph_total.mean()
            loss = amp_weight * amp + self.phase_weight * ph
            if return_parts:
                return loss, amp, ph
            return loss

    m = FDLRegion(patch_size=patch_size, num_proj=num_proj, model="VGG", phase_weight=phase_weight)
    m.requires_grad_(False)
    return m.eval().to(device)


def fdl_call(fdl, pred, hq, mode="full", region=None, crop=0, crop_ij=None,
             bw_cutoff=0.25, bw_order=2, hf_weight=1.0, lf_weight=0.25):
    """Единая точка вызова FDL. pred/hq в [-1,1], любой device — переносятся на device FDL.

    mode:
      'full'  — FDL как в статье (вход — изображения);
      'edge'  — FDL между картами Собеля (приём RS-AdaDiff, вместо DISTS — FDL);
      'bands' — FDL отдельно по ВЧ и НЧ составляющим (фильтр Баттерворта), ВЧ с большим весом.
    region — карта (N,1,H,W) для расчёта по областям; crop — сторона окна (0 = без кропа).
    """
    dev = next(fdl.buffers()).device
    p = pred.float().to(dev)
    t = hq.float().to(dev)
    if region is not None:
        region = region.float().to(dev)
    if crop and crop < min(p.shape[-2:]):
        i, j = crop_ij if crop_ij is not None else (0, 0)
        p, t = p[..., i:i + crop, j:j + crop], t[..., i:i + crop, j:j + crop]
        if region is not None:
            region = region[..., i:i + crop, j:j + crop]
    with torch.autocast(device_type=dev.type, enabled=False):
        if mode == "full":
            return fdl(to01(p), to01(t), region=region)
        if mode == "edge":
            ep, et = sobel_mag(to01(p)), sobel_mag(to01(t))
            return fdl(ep, et, region=region)
        if mode == "bands":
            lp, hp = butterworth_split(to01(p), bw_cutoff, bw_order)
            lt, ht = butterworth_split(to01(t), bw_cutoff, bw_order)
            return (hf_weight * fdl(hp + 0.5, ht + 0.5, region=region)
                    + lf_weight * fdl(lp, lt, region=region))
    raise ValueError(mode)
