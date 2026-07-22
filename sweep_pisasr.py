import os
import glob
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

from pisasr import PiSASR_eval
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

# ---- config ----
PRETRAINED_MODEL_PATH = "/kaggle/working/sd_v2_1_full"
CKPT_DIR = "/kaggle/working/exp_full/checkpoints"
CKPT_STEPS = [500, 1000, 1500, 2000]          # какие чекпоинты сравнить
LSEM_VALUES = [0.0, 0.5, 0.7, 1.0, 1.2]        # сетка по lambda_sem
LAMBDA_PIX = 1.0
INPUT_DIR = "/kaggle/input/datasets/vende14/hq-lq-netherlands-val/lq_images"
OUT_ROOT = "/kaggle/working/sweep"
N_IMAGES = 15                                   # ограничение для быстрой сверки


class Args:
    pretrained_model_path = PRETRAINED_MODEL_PATH
    pretrained_path = None  # выставляется на каждой итерации по чекпоинту
    seed = 42
    process_size = 512
    upscale = 4
    align_method = "adain"
    lambda_pix = LAMBDA_PIX
    lambda_sem = 1.0
    vae_decoder_tiled_size = 224
    vae_encoder_tiled_size = 1024
    latent_tiled_size = 96
    latent_tiled_overlap = 32
    mixed_precision = "fp16"
    default = False


def run_inference(model, args, image_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for image_name in image_names:
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False

        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True

        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = os.path.basename(image_name)

        with torch.no_grad():
            c_t = F.to_tensor(input_image).unsqueeze(0).cuda() * 2 - 1
            _, output_image = model(args.default, c_t, prompt='')

        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

        if args.align_method == 'adain':
            output_pil = adain_color_fix(target=output_pil, source=input_image)
        elif args.align_method == 'wavelet':
            output_pil = wavelet_color_fix(target=output_pil, source=input_image)

        if resize_flag:
            output_pil = output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))
        output_pil.save(os.path.join(output_dir, bname))


if __name__ == "__main__":
    image_names = sorted(glob.glob(f"{INPUT_DIR}/*.png"))[:N_IMAGES]
    print(f"Using {len(image_names)} images for the sweep.")

    for step in CKPT_STEPS:
        args = Args()
        args.pretrained_path = f"{CKPT_DIR}/pisasr_{step}.pkl"

        print(f"\n=== Loading checkpoint step {step} ===")
        model = PiSASR_eval(args)
        model.set_eval()

        for lsem in LSEM_VALUES:
            print(f"  -- lambda_sem={lsem}")
            model.lambda_sem = torch.tensor([lsem], device=model.device)
            model.lambda_pix = torch.tensor([LAMBDA_PIX], device=model.device)
            out_dir = f"{OUT_ROOT}/ckpt_{step}/lsem_{lsem}"
            run_inference(model, args, image_names, out_dir)

        del model
        torch.cuda.empty_cache()
