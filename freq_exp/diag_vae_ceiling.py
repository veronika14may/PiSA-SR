"""Потолок VAE: сколько высоких частот теряется при кодировании и декодировании эталона без UNet.

    python freq_exp/diag_vae_ceiling.py --pretrained_model_path /path/sd-2-1-base --hq_dir data/val/hq --n 100

Эталон -> энкодер VAE -> декодер VAE; результат сравнивается с эталоном теми же метриками, что и выход
модели. Это верхняя граница для любого варианта, где VAE заморожен: лучше этих чисел дообучение LoRA
дать не может. Запускать из папки PiSA-SR-main, нужен GPU.
"""
import os, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
import torch
from torchvision import transforms

import metrics as FM
from io_utils import list_pairs, load01, psnr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_path", required=True)
    p.add_argument("--hq_dir", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--gsd_cm", type=float, default=8.0)
    p.add_argument("--save_dir", default=None, help="куда сохранить реконструкции (необязательно)")
    p.add_argument("--sample", action="store_true", help="брать случайный латент (как при обучении), а не среднее")
    p.add_argument("--vae_decoder_ckpt", default=None, help="К6: дообученный декодер VAE (по умолчанию исходный)")
    a = p.parse_args()

    from vae_utils import load_vae, load_decoder
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vae = load_vae(a.pretrained_model_path).to(dev).eval()
    if a.vae_decoder_ckpt:
        load_decoder(vae, a.vae_decoder_ckpt)
    names = list_pairs(a.hq_dir, a.hq_dir, a.n)
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
    acc = np.zeros((FM.N_BANDS, 4)); ps, vols, vols_gt = [], [], []
    torch.manual_seed(0)
    for nm in names:
        hq = load01(os.path.join(a.hq_dir, nm))
        h, w = hq.shape[-2:]
        c = min(a.crop, h - h % 8, w - w % 8)
        i, j = (h - c) // 2, (w - c) // 2
        hq = hq[..., i:i + c, j:j + c].to(dev)
        with torch.no_grad():
            dist = vae.encode(hq * 2 - 1).latent_dist
            z = dist.sample() if a.sample else dist.mode()
            rec = (vae.decode(z).sample.clamp(-1, 1) * 0.5 + 0.5)
        rec_pil = transforms.ToPILImage()(rec[0].cpu()); hq_pil = transforms.ToPILImage()(hq[0].cpu())
        if a.save_dir:
            rec_pil.save(os.path.join(a.save_dir, nm))
        acc += FM.band_stats(rec.cpu(), hq.cpu())
        ps.append(psnr(rec.cpu(), hq.cpu())); vols.append(FM.vol_of(rec_pil)); vols_gt.append(FM.vol_of(hq_pil))
    s, bands = FM.summary(acc, gsd_cm=a.gsd_cm)
    print(f"снимков: {len(names)}, кроп {a.crop}, латент: {'случайный' if a.sample else 'среднее'}")
    print(FM.format_bands(bands))
    print(f"PSNR {np.mean(ps):.2f} дБ; VoL/GT {np.mean(vols) / np.mean(vols_gt):.3f}; ВЧ: FRC {s['hf_frc']:.3f}, "
          f"ошибка {s['hf_err']:.3f}, энергия {s['hf_energy']:.3f}; эффективное разрешение {s['eff_gsd_cm']:.1f} см")
    print("Как читать: если ошибка в ВЧ-полосах здесь близка к 1, а энергия заметно ниже 1, узкое место — VAE, "
          "и потери на UNet дадут ограниченный эффект.")


if __name__ == "__main__":
    main()
