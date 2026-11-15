"""Тест подмены амплитуды и фазы (по TFDSR, формула 7): чего модели не хватает больше.

    python freq_exp/diag_swap_test.py --sr_dir out/E01 --hq_dir data/val/hq

Для каждого снимка строятся два гибрида и сравниваются с эталоном:
  A_модели + P_эталона — показывает, насколько плоха АМПЛИТУДА модели (фаза идеальна);
  A_эталона + P_модели — показывает, насколько плоха ФАЗА модели (амплитуда идеальна).
Чем ниже PSNR гибрида, тем больше ошибка соответствующей составляющей.
Считается по всему спектру и отдельно по высоким частотам (выше --hf_from Найквиста).
"""
import os, sys, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from io_utils import list_pairs, load01


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sr_dir", required=True)
    p.add_argument("--hq_dir", required=True)
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--hf_from", type=float, default=0.25)
    a = p.parse_args()

    names = list_pairs(a.sr_dir, a.hq_dir, a.n)
    keys = ["выход модели", "A модели + P эталона", "A эталона + P модели"]
    mse = {k: [] for k in keys}; mse_hf = {k: [] for k in keys}
    for nm in names:
        hq = load01(os.path.join(a.hq_dir, nm))
        sr = load01(os.path.join(a.sr_dir, nm), size=(hq.shape[-1], hq.shape[-2]))
        S, G = torch.fft.fft2(sr), torch.fft.fft2(hq)
        h, w = hq.shape[-2:]
        r = torch.sqrt(torch.fft.fftfreq(h)[:, None] ** 2 + torch.fft.fftfreq(w)[None, :] ** 2) / 0.5
        hf = (r >= a.hf_from).float()
        hyb = {"выход модели": S,
               "A модели + P эталона": torch.polar(S.abs(), torch.angle(G)),
               "A эталона + P модели": torch.polar(G.abs(), torch.angle(S))}
        for k, F_ in hyb.items():
            d = F_ - G
            # по равенству Парсеваля MSE в пикселях = средний квадрат разности спектров / (H*W)
            mse[k].append(float((d.abs() ** 2).mean()) / (h * w))
            mse_hf[k].append(float(((d.abs() ** 2) * hf).mean()) / (h * w))

    def ps(v):
        return 10 * np.log10(1.0 / max(np.mean(v), 1e-12))

    print(f"снимков: {len(names)}; ВЧ = выше {a.hf_from} частоты Найквиста")
    print(f"{'':28s}{'PSNR, весь спектр':>20}{'PSNR по ВЧ-ошибке':>22}")
    for k in keys:
        print(f"{k:28s}{ps(mse[k]):20.2f}{ps(mse_hf[k]):22.2f}")
    da = np.mean(mse_hf["A модели + P эталона"]); dp = np.mean(mse_hf["A эталона + P модели"])
    print(f"\nДоля ВЧ-ошибки: от амплитуды {da / (da + dp):.0%}, от фазы {dp / (da + dp):.0%} "
          f"(оценка по двум гибридам).")


if __name__ == "__main__":
    main()
