import os, glob
import cv2
import numpy as np
import torch
import pyiqa
from PIL import Image
import torchvision.transforms.functional as TF

# ---- настрой ----
GT_DIR = "/kaggle/input/datasets/vende14/hq-lq-netherlands-val/hq_images"
SWEEP_GLOB = "/kaggle/working/sweep/lsem_*"   # папки из bash-цикла
device = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------

# FR — нужен GT; NR — без GT. Те же метрики, что в твоей таблице.
m_psnr = pyiqa.create_metric("psnr",   device=device)
m_ssim = pyiqa.create_metric("ssim",   device=device)
m_lpips= pyiqa.create_metric("lpips",  device=device)
m_dists= pyiqa.create_metric("dists",  device=device)
m_musiq= pyiqa.create_metric("musiq",  device=device)
m_clip = pyiqa.create_metric("clipiqa",device=device)

def load(path):
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img).unsqueeze(0).to(device)   # [1,3,H,W] в [0,1]

def vol(path):  # variance of Laplacian (резкость)
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(g, cv2.CV_64F).var()

folders = sorted(glob.glob(SWEEP_GLOB))
rows = []
for fd in folders:
    label = os.path.basename(fd)
    names = sorted(set(os.listdir(fd)) & set(os.listdir(GT_DIR)))
    if not names:
        print(f"[{label}] нет совпадающих имён с GT — пропуск"); continue

    acc = {k: [] for k in ["psnr","ssim","lpips","dists","musiq","clipiqa","vol"]}
    for n in names:
        p, g = os.path.join(fd, n), os.path.join(GT_DIR, n)
        tp, tg = load(p), load(g)
        with torch.no_grad():
            acc["psnr"].append(m_psnr(tp, tg).item())
            acc["ssim"].append(m_ssim(tp, tg).item())
            acc["lpips"].append(m_lpips(tp, tg).item())
            acc["dists"].append(m_dists(tp, tg).item())
            acc["musiq"].append(m_musiq(tp).item())
            acc["clipiqa"].append(m_clip(tp).item())
        acc["vol"].append(vol(p))
    rows.append((label, {k: np.mean(v) for k, v in acc.items()}, len(names)))

# GT-резкость как референс
gt_names = sorted(os.listdir(GT_DIR))[:len(names)]
gt_vol = np.mean([vol(os.path.join(GT_DIR, n)) for n in gt_names])

hdr = f"{'lambda_sem':12s}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>9}{'DISTS':>9}{'MUSIQ':>8}{'CLIPIQA':>9}{'VoL':>9}{'VoL/GT':>8}"
print(hdr); print("-" * len(hdr))
for label, m, k in rows:
    print(f"{label:12s}{m['psnr']:8.2f}{m['ssim']:8.4f}{m['lpips']:9.4f}"
          f"{m['dists']:9.4f}{m['musiq']:8.2f}{m['clipiqa']:9.4f}"
          f"{m['vol']:9.1f}{m['vol']/gt_vol:8.2f}")
print("-" * len(hdr))
print(f"{'GT':12s}{'':8}{'':8}{'':9}{'':9}{'':8}{'':9}{gt_vol:9.1f}{1.0:8.2f}")
print("\nцель режима A: VoL/GT ближе к 1.0 при минимальных LPIPS/DISTS")
