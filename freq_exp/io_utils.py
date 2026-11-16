"""Общие функции чтения изображений для скриптов оценки и диагностики."""
import os

import numpy as np
import torch
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def pick_even(names, n):
    """n имён, равномерно распределённых по отсортированному списку (n <= 0 или n >= len: все).
    Первые n по алфавиту дали бы один класс и пару тайлов: имена начинаются с класса и тайла."""
    if n <= 0 or n >= len(names):
        return list(names)
    return [names[int(i * len(names) / n)] for i in range(n)]


def list_pairs(a_dir, b_dir, n=0):
    """Имена файлов, которые есть в обеих папках (n равномерно выбранных, если n > 0)."""
    names = sorted(f for f in os.listdir(b_dir) if f.lower().endswith(IMG_EXT)
                   and os.path.exists(os.path.join(a_dir, f)))
    assert names, f"нет совпадающих имён в {a_dir} и {b_dir}"
    return pick_even(names, n)


def load01(path, size=None):
    """(1,3,H,W) в [0,1]. size=(W,H) — привести бикубической интерполяцией."""
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.BICUBIC)
    return torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(1.0 / mse))
