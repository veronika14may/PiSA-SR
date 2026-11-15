"""Оценка папки с результатами сверхразрешения по частотным полосам.

    python freq_exp/eval_bands.py --sr_dir out/E05 --hq_dir data/val/hq
    python freq_exp/eval_bands.py --sr_dir out/E05 --sr_dir_b out/E01 --hq_dir data/val/hq   # парное сравнение

Печатает таблицу по кольцам частот (FRC, ошибка, энергия), сводку по ВЧ, эффективное разрешение,
PSNR, VoL/GT и (если установлен пакет lpips) LPIPS. С --sr_dir_b считает разности A − B
с 95 % доверительным интервалом парным бутстрепом по снимкам.
"""
import os, sys, csv, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from PIL import Image

import metrics as FM
from io_utils import list_pairs, load01, psnr


def collect(sr_dir, hq_dir, names, lp):
    accs, rows = [], []
    for nm in names:
        hq = load01(os.path.join(hq_dir, nm))
        sr = load01(os.path.join(sr_dir, nm), size=(hq.shape[-1], hq.shape[-2]))
        acc = FM.band_stats(sr, hq)
        s, _ = FM.summary(acc)
        r = {"name": nm, "psnr": psnr(sr, hq), "hf_err": s["hf_err"], "hf_frc": s["hf_frc"],
             "hf_energy": s["hf_energy"],
             "vol": FM.vol_of(Image.open(os.path.join(sr_dir, nm))),
             "vol_gt": FM.vol_of(Image.open(os.path.join(hq_dir, nm)))}
        if lp is not None:
            with torch.no_grad():
                r["lpips"] = float(lp(sr * 2 - 1, hq * 2 - 1))
        accs.append(acc); rows.append(r)
    return np.stack(accs), rows


def aggregate(accs, rows, idx, gsd):
    s, bands = FM.summary(accs[idx].sum(0), gsd_cm=gsd)
    s["psnr"] = float(np.mean([rows[i]["psnr"] for i in idx]))
    s["vol_ratio"] = float(np.mean([rows[i]["vol"] for i in idx]) / np.mean([rows[i]["vol_gt"] for i in idx]))
    if "lpips" in rows[0]:
        s["lpips"] = float(np.mean([rows[i]["lpips"] for i in idx]))
    return s, bands


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sr_dir", required=True)
    p.add_argument("--hq_dir", required=True)
    p.add_argument("--sr_dir_b", default=None, help="вторая папка для парного сравнения (A − B)")
    p.add_argument("--gsd_cm", type=float, default=8.0)
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--out_csv", default=None, help="куда записать метрики по каждому снимку")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--no_lpips", action="store_true")
    a = p.parse_args()

    lp = None
    if not a.no_lpips:
        try:
            import lpips
            lp = lpips.LPIPS(net="vgg").eval()
        except Exception as e:
            print("LPIPS пропущен:", e)

    names = list_pairs(a.sr_dir, a.hq_dir, a.n)
    if a.sr_dir_b:
        names = [n for n in names if os.path.exists(os.path.join(a.sr_dir_b, n))]
    print(f"снимков: {len(names)}")
    accs, rows = collect(a.sr_dir, a.hq_dir, names, lp)
    all_idx = np.arange(len(names))
    s, bands = aggregate(accs, rows, all_idx, a.gsd_cm)
    print(f"\n== {a.sr_dir}\n" + FM.format_bands(bands))
    print("сводка:", {k: round(v, 4) for k, v in s.items()})
    if a.out_csv:
        with open(a.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("по снимкам:", a.out_csv)

    if a.sr_dir_b:
        accs_b, rows_b = collect(a.sr_dir_b, a.hq_dir, names, lp)
        sb, bands_b = aggregate(accs_b, rows_b, all_idx, a.gsd_cm)
        print(f"\n== {a.sr_dir_b}\n" + FM.format_bands(bands_b))
        print("сводка:", {k: round(v, 4) for k, v in sb.items()})
        rng = np.random.RandomState(0)
        keys = [k for k in s if k in sb]
        diffs = {k: [] for k in keys}
        for _ in range(a.n_boot):
            idx = rng.randint(0, len(names), len(names))
            sa_, _ = aggregate(accs, rows, idx, a.gsd_cm)
            sb_, _ = aggregate(accs_b, rows_b, idx, a.gsd_cm)
            for k in keys:
                diffs[k].append(sa_[k] - sb_[k])
        print("\nразность A − B, 95 % интервал (парный бутстреп по снимкам):")
        for k in keys:
            lo, hi = np.percentile(diffs[k], [2.5, 97.5])
            mark = "значимо" if (lo > 0 or hi < 0) else "в пределах шума"
            print(f"  {k:12s} {s[k] - sb[k]:+9.4f}  [{lo:+.4f}; {hi:+.4f}]  {mark}")


if __name__ == "__main__":
    main()
