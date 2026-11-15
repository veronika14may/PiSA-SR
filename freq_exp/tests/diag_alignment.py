"""Проверка субпиксельного совмещения пар LQ–HQ.

    python freq_exp/diag_alignment.py --lq_dir data/train/lq --hq_dir data/train/hq --n 200

LQ приводится к размеру HQ бикубической интерполяцией (как в paired_dataset.py), затем сдвиг
оценивается фазовой корреляцией по низким частотам, которые есть в обоих снимках.
Сдвиг больше ~0,25 пикселя HQ заметно портит потери, чувствительные к положению (FFL, Fourier Space Loss).
"""
import os, sys, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from io_utils import list_pairs, load01


def estimate_shift(a, b, max_freq=0.2):
    """Сдвиг b относительно a в пикселях (dy, dx); a, b: (H,W). Положительный dx — b смещён вправо."""
    h, w = a.shape
    win = torch.hann_window(h)[:, None] * torch.hann_window(w)[None, :]
    A, B = torch.fft.fft2((a - a.mean()) * win), torch.fft.fft2((b - b.mean()) * win)
    fy, fx = torch.fft.fftfreq(h)[:, None], torch.fft.fftfreq(w)[None, :]
    m = (torch.sqrt(fy ** 2 + fx ** 2) / 0.5 < max_freq)
    m[0, 0] = False
    X = A * B.conj()
    ph = torch.angle(X)[m]                      # ≈ 2*pi*(fy*dy + fx*dx)
    wgt = X.abs()[m]
    M = torch.stack([fy.expand(h, w)[m], fx.expand(h, w)[m]], 1) * 2 * np.pi
    Mw = M * wgt[:, None]
    sol = torch.linalg.lstsq(Mw.T @ M, Mw.T @ ph[:, None]).solution.flatten()
    return float(sol[0]), float(sol[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lq_dir", required=True)
    p.add_argument("--hq_dir", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max_freq", type=float, default=0.2,
                   help="верхняя частота для оценки, доля Найквиста HQ (для x4 берите не больше 0.2)")
    a = p.parse_args()
    names = list_pairs(a.lq_dir, a.hq_dir, a.n)
    sh = []
    for nm in names:
        hq = load01(os.path.join(a.hq_dir, nm))
        lq = load01(os.path.join(a.lq_dir, nm), size=(hq.shape[-1], hq.shape[-2]))
        g = lambda x: (0.299 * x[0, 0] + 0.587 * x[0, 1] + 0.114 * x[0, 2])
        sh.append(estimate_shift(g(hq), g(lq), a.max_freq))
    sh = np.array(sh)
    print(f"пар: {len(names)}")
    print(f"средний сдвиг LQ относительно HQ: dy = {sh[:, 0].mean():+.3f}, dx = {sh[:, 1].mean():+.3f} пикс. HQ")
    print(f"разброс (ст. откл.):               dy = {sh[:, 0].std():.3f},  dx = {sh[:, 1].std():.3f}")
    worst = np.abs(sh).max(1)
    print(f"доля пар со сдвигом > 0.25 пикс.: {(worst > 0.25).mean():.1%}")
    if np.abs(sh.mean(0)).max() > 0.25:
        print("ВНИМАНИЕ: систематический сдвиг. Проверьте, как создавались LQ (выравнивание сетки при уменьшении).")
    else:
        print("Систематического сдвига нет.")


if __name__ == "__main__":
    main()
