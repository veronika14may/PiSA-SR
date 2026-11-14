"""Метрики сохранения высоких частот: по кольцам частот и интегральные.

Для каждого кольца (доля частоты Найквиста) считаются:
  frc    — кольцевая корреляция Фурье: согласие фаз с весом амплитуд (1 — совпадают, 0 — не связаны);
  err    — относительная ошибка sum|F_sr - F_gt|^2 / sum|F_gt|^2
           (0 — идеально, 1 — ничего не восстановлено, >1 — детали выдуманы);
  energy — отношение энергий sum|F_sr|^2 / sum|F_gt|^2 (обобщение VoL/GT).
Эффективное разрешение — частота, на которой FRC падает ниже порога.
"""
import numpy as np
import torch

N_BANDS = 8


def _to_y(x01):
    return 0.299 * x01[:, 0] + 0.587 * x01[:, 1] + 0.114 * x01[:, 2]


@torch.no_grad()
def band_stats(sr01, gt01, n_bands=N_BANDS):
    """sr01, gt01: (N,3,H,W) в [0,1]. Возвращает суммы по кольцам (для накопления по набору)."""
    sr, gt = _to_y(sr01.float()), _to_y(gt01.float())
    h, w = gt.shape[-2:]
    win = torch.hann_window(h, device=gt.device)[:, None] * torch.hann_window(w, device=gt.device)[None, :]
    S = torch.fft.fft2(sr * win, norm="ortho")
    G = torch.fft.fft2(gt * win, norm="ortho")
    r = torch.sqrt(torch.fft.fftfreq(h, device=gt.device)[:, None] ** 2
                   + torch.fft.fftfreq(w, device=gt.device)[None, :] ** 2) / 0.5
    out = np.zeros((n_bands, 4), dtype=np.float64)       # cross, |S|^2, |G|^2, |S-G|^2
    for i in range(n_bands):
        m = (r >= i / n_bands) & (r < (i + 1) / n_bands)
        out[i, 0] = (S * G.conj()).real[:, m].sum().item()
        out[i, 1] = (S.abs() ** 2)[:, m].sum().item()
        out[i, 2] = (G.abs() ** 2)[:, m].sum().item()
        out[i, 3] = ((S - G).abs() ** 2)[:, m].sum().item()
    return out


def finalize(acc):
    """acc: сумма band_stats по набору -> dict с массивами frc, err, energy."""
    cross, ps, pg, pe = acc[:, 0], acc[:, 1], acc[:, 2], acc[:, 3]
    frc = cross / np.sqrt(np.maximum(ps * pg, 1e-30))
    return {"frc": frc, "err": pe / np.maximum(pg, 1e-30), "energy": ps / np.maximum(pg, 1e-30)}


def effective_resolution(frc, threshold=0.5):
    """Доля частоты Найквиста, на которой FRC впервые опускается ниже порога (линейная интерполяция
    между центрами колец). 1.0 — корреляция выше порога на всех частотах."""
    n = len(frc)
    centers = (np.arange(n) + 0.5) / n
    for i in range(n):
        if frc[i] < threshold:
            if i == 0:
                return float(centers[0])
            f0, f1 = frc[i - 1], frc[i]
            t = (f0 - threshold) / max(f0 - f1, 1e-12)
            return float(centers[i - 1] + t * (centers[i] - centers[i - 1]))
    return 1.0


def summary(acc, gsd_cm=None, hf_from=0.5):
    """Сводка: средние по ВЧ-кольцам (>= hf_from Найквиста) и эффективное разрешение."""
    m = finalize(acc)
    n = len(m["frc"])
    hf = np.arange(n) >= int(round(hf_from * n))
    cross, ps, pg, pe = acc[hf, 0].sum(), acc[hf, 1].sum(), acc[hf, 2].sum(), acc[hf, 3].sum()
    f_eff = effective_resolution(m["frc"])
    s = {
        "hf_frc": float(cross / np.sqrt(max(ps * pg, 1e-30))),
        "hf_err": float(pe / max(pg, 1e-30)),
        "hf_energy": float(ps / max(pg, 1e-30)),
        "f_eff": f_eff,
    }
    if gsd_cm is not None:
        s["eff_gsd_cm"] = gsd_cm / max(f_eff, 1e-6)
    return s, m


def vol_of(pil_img):
    """Дисперсия лапласиана (как в исходном скрипте)."""
    g = np.array(pil_img.convert("L"))
    try:
        import cv2
        return float(cv2.Laplacian(g, cv2.CV_64F).var())
    except ImportError:
        g = g.astype(np.float64)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def format_bands(m, n_bands=N_BANDS):
    lines = [f"{'полоса (доля Найквиста)':>26} {'FRC':>7} {'ошибка':>8} {'энергия':>8}"]
    for i in range(n_bands):
        lines.append(f"{i / n_bands:>12.3f}-{(i + 1) / n_bands:<13.3f} {m['frc'][i]:7.3f} "
                     f"{m['err'][i]:8.3f} {m['energy'][i]:8.3f}")
    return "\n".join(lines)
