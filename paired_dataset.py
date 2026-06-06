import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as TF

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

class PairedLQHQDataset(Dataset):
    def __init__(self, lq_dir, hq_dir, resolution=512,
                 prompt="", neg_prompt="", augment=True):
        super().__init__()
        self.lq_dir = lq_dir
        self.hq_dir = hq_dir
        self.resolution = resolution
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.augment = augment

        names = sorted(f for f in os.listdir(hq_dir) if f.lower().endswith(IMG_EXT))
        self.names = [n for n in names if os.path.exists(os.path.join(lq_dir, n))]
        assert len(self.names) > 0, f"Не найдено ни одной пары LQ-HQ в {lq_dir} / {hq_dir}"
        print(f"[PairedLQHQDataset] найдено пар: {len(self.names)}")

    def __len__(self):
        return len(self.names)

    def _load_pair(self, name):
        lq = Image.open(os.path.join(self.lq_dir, name)).convert("RGB")
        hq = Image.open(os.path.join(self.hq_dir, name)).convert("RGB")
        if lq.size != hq.size: # привести LQ к размеру HQ
            lq = lq.resize(hq.size, Image.BICUBIC)
        return lq, hq

    def __getitem__(self, idx):
        name = self.names[idx]
        lq, hq = self._load_pair(name)
        r = self.resolution

        if min(hq.size) < r:
            lq, hq = lq.resize((r, r), Image.BICUBIC), hq.resize((r, r), Image.BICUBIC)
            i = j = 0
        else:
            i, j, h, w = RandomCrop.get_params(hq, (r, r))
            lq, hq = TF.crop(lq, i, j, h, w), TF.crop(hq, i, j, h, w)

        if self.augment and torch.rand(1).item() < 0.5:  # одинаковый флип для пары
            lq, hq = TF.hflip(lq), TF.hflip(hq)

        # в тензоры и в диапазон [-1, 1]
        lq = TF.normalize(TF.to_tensor(lq), [0.5] * 3, [0.5] * 3)
        hq = TF.normalize(TF.to_tensor(hq), [0.5] * 3, [0.5] * 3)

        return {
            "conditioning_pixel_values": lq, # LQ
            "output_pixel_values": hq, # HQ
            "prompt": self.prompt,
            "neg_prompt": self.neg_prompt,
            "null_prompt": "",
            "base_name": name,
        }
