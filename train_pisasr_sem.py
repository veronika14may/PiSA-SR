import os
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import lpips

from pisasr import PiSASR, CSDLoss
from paired_dataset import PairedLQHQDataset

ALL_ADAPTERS = ["default_encoder_pix", "default_decoder_pix", "default_others_pix",
                "default_encoder_sem", "default_decoder_sem", "default_others_sem"]

class _Acc:
    mixed_precision = "fp16"
    device = torch.device("cuda")

def get_args():
    p = argparse.ArgumentParser()
    # модель
    p.add_argument("--pretrained_model_path", default="preset/models/stable-diffusion-2-1-base")
    p.add_argument("--pretrained_model_path_csd", default=None,
                   help="по умолчанию = pretrained_model_path")
    p.add_argument("--init_ckpt", required=True,
                   help="ОБЯЗАТЕЛЬНО: .pkl с уже обученной pixel-веткой")
    p.add_argument("--resume_ckpt", default=None)
    p.add_argument("--lora_rank_unet_pix", type=int, default=4)
    p.add_argument("--lora_rank_unet_sem", type=int, default=4)
    p.add_argument("--timesteps1", type=int, default=1)
    p.add_argument("--null_text_ratio", type=float, default=0.5)
    p.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    # данные
    p.add_argument("--lq_dir", required=True)
    p.add_argument("--hq_dir", required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--prompt", default="", help="фолбэк-промпт, если RAM не задан")
    p.add_argument("--pos_prompt_csd", default="")
    p.add_argument("--neg_prompt", default=(
        "painting, oil painting, illustration, drawing, art, sketch, cartoon, "
        "CG Style, 3D render, blurring, dirty, messy, worst quality, low quality, "
        "frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth"))
    # RAM
    p.add_argument("--ram_path", default=None, help="ram_swin_large_14m.pth")
    # обучение
    p.add_argument("--output_dir", default="experiments/sem")
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lambda_l2", type=float, default=1.0)
    p.add_argument("--lambda_lpips", type=float, default=2.0)
    p.add_argument("--lambda_csd", type=float, default=1.0)
    p.add_argument("--cfg_csd", type=float, default=7.5)
    p.add_argument("--min_dm_step_ratio", type=float, default=0.02)
    p.add_argument("--max_dm_step_ratio", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--log_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    args = p.parse_args()
    if args.pretrained_model_path_csd is None:
        args.pretrained_model_path_csd = args.pretrained_model_path
    return args

def load_lora(net, ckpt):
    sd = torch.load(ckpt, map_location="cpu")["state_dict_unet"]
    msd = net.unet.state_dict()
    n = 0
    for k, v in sd.items():
        if k in msd and msd[k].shape == v.shape:
            msd[k].copy_(v); n += 1
    print(f"[init_ckpt] загружено LoRA-тензоров: {n}")

def build_ram(args):
    if not args.ram_path or not os.path.exists(args.ram_path):
        print("[RAM] не задан — используется фиксированный промпт")
        return None, None
    from ram.models.ram_lora import ram
    RAM = ram(pretrained=args.ram_path, pretrained_condition=None,
              image_size=384, vit="swin_l").eval().to("cuda", dtype=torch.float16)
    tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    print("[RAM] загружен")
    return RAM, tf

def get_prompts(args, RAM, ram_tf, hq, bs):
    if RAM is None:
        base = args.prompt
    else:
        from ram import inference_ram
        with torch.no_grad():
            tags = inference_ram(ram_tf(hq * 0.5 + 0.5).to(torch.float16), RAM)
        return [f"{t}, {args.pos_prompt_csd}" for t in tags]
    return [f"{base}, {args.pos_prompt_csd}" if args.pos_prompt_csd else base] * bs

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda"
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    net = PiSASR(args)
    load_lora(net, args.init_ckpt)

    net.unet.set_adapter(ALL_ADAPTERS)
    net.set_train_sem()
    net.unet.enable_gradient_checkpointing()
    net.unet.train()

    params = [p for n, p in net.unet.named_parameters()
              if "lora" in n and "sem" in n and p.requires_grad]
    print(f"обучаемых sem-LoRA параметров: {sum(p.numel() for p in params)/1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=args.lr)

    csd = CSDLoss(args=args, accelerator=_Acc())
    csd.requires_grad_(False)
    lpips_fn = lpips.LPIPS(net="vgg").to(device); lpips_fn.requires_grad_(False)
    RAM, ram_tf = build_ram(args)

    ds = PairedLQHQDataset(args.lq_dir, args.hq_dir, resolution=args.resolution,
                           neg_prompt=args.neg_prompt)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True)

    use_amp = args.mixed_precision != "no"
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))

    step = 0
    while step < args.max_steps:
        for batch in dl:
            lq = batch["conditioning_pixel_values"].to(device)
            hq = batch["output_pixel_values"].to(device)
            batch["prompt"] = get_prompts(args, RAM, ram_tf, hq, lq.shape[0])

            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                pred, latents_pred, prompt_embeds, neg_prompt_embeds = net(lq, hq, batch=batch, args=args)
                loss_l2 = F.mse_loss(pred.float(), hq.float()) * args.lambda_l2
                loss_lp = lpips_fn(pred.float(), hq.float()).mean() * args.lambda_lpips
                loss_csd = csd.cal_csd(latents_pred, prompt_embeds, neg_prompt_embeds, args) * args.lambda_csd
                loss = loss_l2 + loss_lp + loss_csd

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()

            step += 1
            if step % args.log_steps == 0:
                print(f"step {step}/{args.max_steps}  l2 {loss_l2.item():.4f}  "
                      f"lpips {loss_lp.item():.4f}  csd {loss_csd.item():.4f}")
            if step % args.save_steps == 0:
                outf = os.path.join(args.output_dir, "checkpoints", f"pisasr_sem_{step}.pkl")
                net.save_model(outf); print("saved", outf)
            if step >= args.max_steps:
                break

    outf = os.path.join(args.output_dir, "checkpoints", "pisasr_sem_final.pkl")
    net.save_model(outf); print("done ->", outf)

if __name__ == "__main__":
    main()