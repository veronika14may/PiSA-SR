"""Инференс чекпоинта PiSA-SR и оценка по частотным полосам; перебор lambda_sem.

    # один режим (lambda_pix = lambda_sem = 1), результаты сохраняются в out/E05/lsem_1.0
    python freq_exp/infer_eval.py --pretrained_model_path /path/sd-2-1-base \
        --ckpt experiments/E05/checkpoints/pisasr_final.pkl --lq_dir data/val/lq --hq_dir data/val/hq --out_dir out/E05

    # перебор lambda_sem: показывает, что VoL/GT можно довести до 1 без роста точности
    python freq_exp/infer_eval.py ... --lambda_sem 0 0.5 1.0 1.2 1.5 2.0

Инференс тот же, что в sweep_pisasr.py (PiSASR_eval, пустой промпт, цветокоррекция AdaIN).
Запускать из папки PiSA-SR-main, нужен GPU.
"""
import os, sys, csv, glob, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
import torch
from PIL import Image

import metrics as FM
from io_utils import IMG_EXT, load01, psnr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_path", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--lq_dir", required=True)
    p.add_argument("--hq_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--lambda_pix", type=float, default=1.0)
    p.add_argument("--lambda_sem", type=float, nargs="+", default=[1.0])
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--process_size", type=int, default=512)
    p.add_argument("--align_method", choices=["adain", "wavelet", "nofix"], default="adain")
    p.add_argument("--n", type=int, default=0, help="сколько снимков брать (0 = все)")
    p.add_argument("--gsd_cm", type=float, default=8.0)
    p.add_argument("--mixed_precision", default="fp16")
    a = p.parse_args()

    from pisasr import PiSASR_eval
    from sweep_pisasr import run_inference

    class Args:
        pretrained_model_path = a.pretrained_model_path
        pretrained_path = a.ckpt
        seed = 42
        process_size = a.process_size
        upscale = a.upscale
        align_method = a.align_method
        lambda_pix = a.lambda_pix
        lambda_sem = 1.0
        vae_decoder_tiled_size = 224
        vae_encoder_tiled_size = 1024
        latent_tiled_size = 96
        latent_tiled_overlap = 32
        mixed_precision = a.mixed_precision
        default = False

    names = sorted(f for f in os.listdir(a.lq_dir) if f.lower().endswith(IMG_EXT)
                   and os.path.exists(os.path.join(a.hq_dir, f)))
    if a.n > 0:
        names = names[:a.n]
    paths = [os.path.join(a.lq_dir, n) for n in names]
    print(f"снимков: {len(names)}")

    args = Args()
    model = PiSASR_eval(args); model.set_eval()
    try:
        import lpips
        lp = lpips.LPIPS(net="vgg").eval()
    except Exception:
        lp = None

    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for ls in a.lambda_sem:
        model.lambda_sem = torch.tensor([ls], device=model.device)
        model.lambda_pix = torch.tensor([a.lambda_pix], device=model.device)
        out = os.path.join(a.out_dir, f"lsem_{ls}")
        run_inference(model, args, paths, out)
        acc = np.zeros((FM.N_BANDS, 4)); ps, lps, v, vg = [], [], [], []
        for nm in names:
            hq = load01(os.path.join(a.hq_dir, nm))
            sr = load01(os.path.join(out, nm), size=(hq.shape[-1], hq.shape[-2]))
            acc += FM.band_stats(sr, hq); ps.append(psnr(sr, hq))
            v.append(FM.vol_of(Image.open(os.path.join(out, nm)))); vg.append(FM.vol_of(Image.open(os.path.join(a.hq_dir, nm))))
            if lp is not None:
                with torch.no_grad():
                    lps.append(float(lp(sr * 2 - 1, hq * 2 - 1)))
        s, bands = FM.summary(acc, gsd_cm=a.gsd_cm)
        r = {"lambda_sem": ls, "psnr": np.mean(ps), "lpips": np.mean(lps) if lps else float("nan"),
             "vol_ratio": np.mean(v) / np.mean(vg), **s}
        rows.append(r)
        print(f"\n== lambda_sem = {ls}\n" + FM.format_bands(bands))

    keys = list(rows[0].keys())
    with open(os.path.join(a.out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("\n" + " ".join(f"{k:>11}" for k in keys))
    for r in rows:
        print(" ".join(f"{r[k]:11.4f}" for k in keys))
    print(f"\nтаблица: {os.path.join(a.out_dir, 'summary.csv')}")
    if len(rows) > 1:
        print("Как читать: если с ростом lambda_sem vol_ratio приближается к 1, а hf_err растёт выше 1, "
              "резкость добавляется выдуманными деталями.")


if __name__ == "__main__":
    main()
