import os, csv, random, argparse
os.environ["MPLBACKEND"] = "Agg"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import lpips as lpips_pkg

from pisasr import PiSASR, CSDLoss
from paired_dataset import PairedLQHQDataset, IMG_EXT
from torchvision import transforms
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from train_pisasr_pix import PIX_ADAPTERS
from train_pisasr_sem import ALL_ADAPTERS, _Acc, load_lora, build_ram, get_prompts


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_path", required=True)
    p.add_argument("--init_ckpt", default=None, help="релизный pisa_sr.pkl (нужен для --stage pix/both)")
    p.add_argument("--stage", choices=["pix", "sem", "both"], default="both",
                   help="pix: только pix-этап; sem: только sem-этап поверх pisasr_pix_final.pkl; both: оба подряд")
    p.add_argument("--sem_init_ckpt", default=None,
                   help="для --stage sem: чекпоинт после pix-этапа (по умолчанию output_dir/checkpoints/pisasr_pix_final.pkl)")
    p.add_argument("--ram_path", default=None)
    p.add_argument("--lora_rank_unet_pix", type=int, default=4)
    p.add_argument("--lora_rank_unet_sem", type=int, default=4)
    p.add_argument("--timesteps1", type=int, default=1)
    p.add_argument("--null_text_ratio", type=float, default=0.5)
    p.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    p.add_argument("--resume_ckpt", default=None)
    # данные
    p.add_argument("--lq_dir", required=True, help="train LQ")
    p.add_argument("--hq_dir", required=True, help="train HQ")
    p.add_argument("--val_lq_dir", required=True, help="отложенный LQ (не из train!)")
    p.add_argument("--val_hq_dir", required=True)
    p.add_argument("--align_method", choices=["adain", "wavelet", "nofix"], default="adain",
                   help="цветокоррекция перед метрикой, как в test_pisasr.py (по умолчанию adain)")
    p.add_argument("--n_val", type=int, default=100, help="сколько val-кропов брать (фикс. набор)")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--prompt", default="")
    p.add_argument("--pos_prompt_csd", default="")
    p.add_argument("--neg_prompt", default=(
        "painting, oil painting, illustration, drawing, art, sketch, cartoon, "
        "CG Style, 3D render, blurring, dirty, messy, worst quality, low quality, "
        "frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth"))
    # обучение
    p.add_argument("--output_dir", default="experiments/curve")
    p.add_argument("--pix_steps", type=int, default=4000)
    p.add_argument("--sem_steps", type=int, default=4000)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--save_ckpt", action="store_true", help="сохранять LoRA каждые eval_every шагов")
    p.add_argument("--pix_with_lpips", action="store_true",
                   help="добавить LPIPS в loss pix-стадии (в оригинале csslc/PiSA-SR pix-стадия = только L2)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lambda_l2", type=float, default=1.0)
    p.add_argument("--lambda_lpips", type=float, default=2.0)
    p.add_argument("--lambda_csd", type=float, default=1.0)
    p.add_argument("--cfg_csd", type=float, default=7.5)
    p.add_argument("--min_dm_step_ratio", type=float, default=0.02)
    p.add_argument("--max_dm_step_ratio", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    a = p.parse_args()
    a.pretrained_model_path_csd = a.pretrained_model_path
    return a


class FixedValSet(Dataset):
    """Фиксированные кропы 512x512 из val: координаты задаются один раз по seed, без аугментаций."""
    def __init__(self, lq_dir, hq_dir, n, res, seed=0):
        names = sorted(f for f in os.listdir(hq_dir) if f.lower().endswith(IMG_EXT))
        names = [x for x in names if os.path.exists(os.path.join(lq_dir, x))]
        rng = np.random.RandomState(seed)
        if len(names) > n:
            names = [names[i] for i in sorted(rng.choice(len(names), n, replace=False))]
        self.items = []
        for nm in names:
            hq = Image.open(os.path.join(hq_dir, nm)).convert("RGB")
            w, h = hq.size
            if min(w, h) < res:
                self.items.append((nm, None))
            else:
                self.items.append((nm, (rng.randint(0, h - res + 1), rng.randint(0, w - res + 1))))
        self.lq_dir, self.hq_dir, self.res = lq_dir, hq_dir, res
        print(f"[val] {len(self.items)} кропов из {hq_dir}")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        nm, ij = self.items[i]
        lq = Image.open(os.path.join(self.lq_dir, nm)).convert("RGB")
        hq = Image.open(os.path.join(self.hq_dir, nm)).convert("RGB")
        if lq.size != hq.size:
            lq = lq.resize(hq.size, Image.BICUBIC)
        r = self.res
        if ij is None:
            lq, hq = lq.resize((r, r), Image.BICUBIC), hq.resize((r, r), Image.BICUBIC)
        else:
            lq, hq = TF.crop(lq, ij[0], ij[1], r, r), TF.crop(hq, ij[0], ij[1], r, r)
        n = lambda x: TF.normalize(TF.to_tensor(x), [0.5] * 3, [0.5] * 3)
        return n(lq), n(hq)


@torch.no_grad()
def evaluate(net, val_dl, metric, args, amp_dtype, use_amp):
    """LPIPS на фикс. val. Пустой промпт (как в test_pisasr.py), фикс. шум VAE-энкодера."""
    was_training = net.unet.training
    net.unet.eval()
    old_ratio = args.null_text_ratio
    args.null_text_ratio = 1.1          # random.random() < 1.1 всегда -> всегда null-промпт
    vals = []
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(1234)
        for lq, hq in val_dl:
            lq, hq = lq.cuda(), hq.cuda()
            b = {"prompt": [""] * lq.shape[0], "neg_prompt": [""] * lq.shape[0],
                 "null_prompt": [""] * lq.shape[0]}
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                pred, _, _, _ = net(lq, hq, batch=b, args=args)
            # как в test_pisasr.py: -> PIL (8 бит, как PNG на диске) -> цветокоррекция по LQ -> метрика
            for k in range(pred.shape[0]):
                out = transforms.ToPILImage()((pred[k].float().cpu() * 0.5 + 0.5).clamp(0, 1))
                src = transforms.ToPILImage()((lq[k].float().cpu() * 0.5 + 0.5).clamp(0, 1))
                if args.align_method == "adain":
                    out = adain_color_fix(target=out, source=src)
                elif args.align_method == "wavelet":
                    out = wavelet_color_fix(target=out, source=src)
                tp = TF.to_tensor(out).unsqueeze(0).cuda()
                tg = ((hq[k:k + 1].float() * 0.5 + 0.5).clamp(0, 1))
                v = metric(tp, tg)          # pyiqa lpips, [0,1] — как в eval_sweep.py
                vals.extend(v.flatten().tolist())
    args.null_text_ratio = old_ratio
    net.unet.train(was_training)
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals)))


def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    ck_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)

    net = PiSASR(args)
    if args.stage == "sem":
        start = args.sem_init_ckpt or os.path.join(ck_dir, "pisasr_pix_final.pkl")
    else:
        start = args.init_ckpt
    assert start and os.path.exists(start), f"нет стартового чекпоинта: {start}"
    load_lora(net, start)
    net.unet.enable_gradient_checkpointing()

    import pyiqa
    metric = pyiqa.create_metric("lpips", device="cuda")      # как в eval_sweep.py
    lpips_train = lpips_pkg.LPIPS(net="vgg").cuda().requires_grad_(False)   # как в train-скриптах (loss)
    RAM, ram_tf = build_ram(args)
    csd = CSDLoss(args=args, accelerator=_Acc()); csd.requires_grad_(False)

    train_ds = PairedLQHQDataset(args.lq_dir, args.hq_dir, resolution=args.resolution,
                                 prompt=args.prompt, neg_prompt=args.neg_prompt)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True)
    val_dl = DataLoader(FixedValSet(args.val_lq_dir, args.val_hq_dir, args.n_val, args.resolution),
                        batch_size=1, shuffle=False, num_workers=1)

    use_amp = args.mixed_precision != "no"
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))

    log_path = os.path.join(args.output_dir, f"curve_{args.stage}.csv")
    f = open(log_path, "w", newline=""); w = csv.writer(f)
    w.writerow(["global_step", "stage", "lpips_val", "lpips_val_sem", "train_l2", "train_lpips"])

    def log_eval(gstep, stage, l2=float("nan"), lp=float("nan")):
        m, se = evaluate(net, val_dl, metric, args, amp_dtype, use_amp)
        w.writerow([gstep, stage, f"{m:.5f}", f"{se:.5f}", f"{l2:.5f}", f"{lp:.5f}"]); f.flush()
        print(f"[eval] step {gstep:5d} ({stage})  LPIPS val = {m:.4f} ± {se:.4f}")
        if args.save_ckpt and gstep > 0:
            net.save_model(os.path.join(ck_dir, f"pisasr_{gstep}.pkl"))

    def run_stage(stage, n_steps, offset):
        if stage == "pix":
            net.unet.set_adapter(PIX_ADAPTERS); net.set_train_pix()
            params = [p for n, p in net.unet.named_parameters() if "lora" in n and p.requires_grad]
        else:
            net.unet.set_adapter(ALL_ADAPTERS); net.set_train_sem()
            params = [p for n, p in net.unet.named_parameters()
                      if "lora" in n and "sem" in n and p.requires_grad]
        net.unet.train()
        opt = torch.optim.AdamW(params, lr=args.lr)
        print(f"[{stage}] обучаемых параметров: {sum(p.numel() for p in params)/1e6:.2f}M")
        step = 0
        while step < n_steps:
            for batch in train_dl:
                lq = batch["conditioning_pixel_values"].cuda()
                hq = batch["output_pixel_values"].cuda()
                batch["prompt"] = get_prompts(args, RAM, ram_tf, hq, lq.shape[0])
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    pred, lat, pe, npe = net(lq, hq, batch=batch, args=args)
                    l2 = F.mse_loss(pred.float(), hq.float()) * args.lambda_l2
                    lp_w = args.lambda_lpips if (stage == "sem" or args.pix_with_lpips) else 0.0
                    lp = lpips_train(pred.float(), hq.float()).mean() * lp_w
                    loss = l2 + lp
                    if stage == "sem":
                        loss = loss + csd.cal_csd(lat, pe, npe, args) * args.lambda_csd
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update()
                step += 1
                if step % 20 == 0:
                    print(f"[{stage}] {step}/{n_steps} l2 {l2.item():.4f} lpips {lp.item():.4f}")
                if step % args.eval_every == 0:
                    log_eval(offset + step, stage, l2.item(), lp.item())
                if step >= n_steps:
                    break

    if args.stage in ("pix", "both"):
        net.unet.set_adapter(PIX_ADAPTERS)                # на pix-этапе активны только pix-LoRA
        log_eval(0, "init")                               # точка отсчёта: исходный pisa_sr.pkl
        run_stage("pix", args.pix_steps, 0)
        net.save_model(os.path.join(ck_dir, "pisasr_pix_final.pkl"))
    if args.stage in ("sem", "both"):
        net.unet.set_adapter(ALL_ADAPTERS)                # на sem-этапе включаются обе ветки
        log_eval(args.pix_steps, "sem_start")             # скачок при включении sem-LoRA, ещё без обучения sem
        run_stage("sem", args.sem_steps, args.pix_steps)
        net.save_model(os.path.join(ck_dir, "pisasr_final.pkl"))
    f.close()

    # --- график ---
    import pandas as pd, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import glob
    d = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(args.output_dir, "curve_*.csv")))])
    d = d.drop_duplicates(["global_step", "stage"], keep="last").sort_values("global_step")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, col, sel in [("pix-этап", "tab:blue", d.stage.isin(["init", "pix"])),
                           ("sem-этап", "tab:orange", d.stage.isin(["sem_start", "sem"]))]:
        s = d[sel]
        ax.plot(s.global_step, s.lpips_val, "-o", ms=2.5, color=col, label=name)
        ax.fill_between(s.global_step, s.lpips_val - s.lpips_val_sem, s.lpips_val + s.lpips_val_sem,
                        color=col, alpha=0.15)
    ax.axvline(args.pix_steps, ls="--", c="gray")
    ax.set_xlabel("шаг дообучения"); ax.set_ylabel("LPIPS (val), ↓"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.output_dir, "curve.png"), dpi=200)
    print("saved", log_path, "и curve.png")


if __name__ == "__main__":
    main()
