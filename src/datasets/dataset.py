import os
import random
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from pathlib import Path


class PairedSROnlineTxtDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()

        self.args = args
        self.split = split

        if split == 'train':
            self.resolution_tgt = args.resolution_tgt

            # GT paths
            with open(args.dataset_txt_paths, 'r') as f:
                self.gt_list = [line.strip() for line in f.readlines()]

            # LQ paths (parallel to GT)
            with open(args.dataset_txt_paths_lq, 'r') as f:
                self.lq_list = [line.strip() for line in f.readlines()]

            assert len(self.lq_list) == len(self.gt_list), \
                f"LQ ({len(self.lq_list)}) and GT ({len(self.gt_list)}) lists must have the same length"

        elif split == 'test':
            self.input_folder = args.lq_test_folder
            self.output_folder = args.gt_test_folder
            self.lr_list = []
            self.gt_list = []
            lr_names = sorted(os.listdir(self.input_folder))
            gt_names = sorted(os.listdir(self.output_folder))
            assert len(lr_names) == len(gt_names)
            for i in range(len(lr_names)):
                self.lr_list.append(os.path.join(self.input_folder, lr_names[i]))
                self.gt_list.append(os.path.join(self.output_folder, gt_names[i]))
            assert len(self.lr_list) == len(self.gt_list)

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'train':
            # read GT (512x512) and LQ (128x128) from disk
            gt_img = Image.open(self.gt_list[idx]).convert('RGB')
            lq_img = Image.open(self.lq_list[idx]).convert('RGB')

            # upsample LQ to GT size (bicubic) — required because VAE expects same spatial size
            lq_img = F.resize(
                lq_img,
                (self.resolution_tgt, self.resolution_tgt),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )

            # synchronous random horizontal flip
            if random.random() < 0.5:
                gt_img = F.hflip(gt_img)
                lq_img = F.hflip(lq_img)

            # to tensor [0, 1]
            output_t = F.to_tensor(gt_img)
            img_t = F.to_tensor(lq_img)

            # normalize to [-1, 1]
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_csd
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t

            return example

        elif self.split == 'test':
            input_img = Image.open(self.lr_list[idx]).convert('RGB')   # 128x128
            output_img = Image.open(self.gt_list[idx]).convert('RGB')  # 512x512
        
            # LQ → 512x512
            input_img = F.resize(
                input_img,
                (self.args.resolution_tgt, self.args.resolution_tgt),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        
            img_t = F.to_tensor(input_img)
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            output_t = F.to_tensor(output_img)
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])
        
            example = {}
            example["neg_prompt"] = self.args.neg_prompt_csd
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t
            example["base_name"] = os.path.basename(self.lr_list[idx])
            return example
