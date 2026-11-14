"""Дообучение PiSA-SR с частотными потерями из статей (копия train_pisasr_curve_exp.py с расширениями).

Запуск из папки PiSA-SR-main:
    python freq_exp/train_freq.py --pretrained_model_path ... --lq_dir ... --hq_dir ... \
        --val_lq_dir ... --val_hq_dir ... --init_ckpt pisa_sr.pkl --stage both --pix_ffl 1.0 --sem_fdl 0.003

Что добавлено к исходному скрипту:
  * набор потерь задаётся отдельно для pix- и sem-этапа (--pix_* / --sem_*):
      ffl       — Focal Frequency Loss (ICCV 2021)
      fsl       — Fourier Space Loss (ICCV 2021): L1 по амплитуде + L1 по фазе
      fdl       — FDL (CVPR 2024), как в статье
      fdl_edge  — FDL между картами Собеля (приём RS-AdaDiff)
      fdl_bands — FDL по ВЧ/НЧ составляющим (фильтр Баттерворта из FiDeSR)
      daw       — вес L2/LPIPS по карте деталей (FiDeSR / TexADiff), опционально и для FDL, и для CSD
  * замороженные модели (учитель CSD, RAM, VGG для FDL) можно вынести на вторую видеокарту (--aux_device);
  * накопление градиента (--grad_accum), EMA весов LoRA (--ema_decay);
  * лог значений и норм градиента каждого члена потерь (train_log_*.csv);
  * одноразовая калибровка весов частотных членов по норме градиента (--auto_weight_ratio);
  * в валидации, кроме LPIPS и VoL, считаются метрики по частотным полосам (freq_exp/metrics.py).
Исходный train_pisasr_curve_exp.py не изменяется.
"""
import os, sys, csv, json, time, random, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.environ["MPLBACKEND"] = "Agg"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import lpips as lpips_pkg

from pisasr import PiSASR, CSDLoss
from paired_dataset import PairedLQHQDataset, IMG_EXT
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from train_pisasr_pix import PIX_ADAPTERS
from train_pisasr_sem import ALL_ADAPTERS, load_lora

import losses as FL
import metrics as FM

STAGES = ("pix", "sem")
FREQ_TERMS = ("ffl", "fsl", "fdl", "fdl_edge", "fdl_bands")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_path", required=True)
    p.add_argument("--init_ckpt", default=None, help="релизный pisa_sr.pkl (нужен для --stage pix/both)")
    p.add_argument("--stage", choices=["pix", "sem", "both"], default="both")
    p.add_argument("--sem_init_ckpt", default=None,
                   help="для --stage sem: стартовый чекпоинт (по умолчанию output_dir/checkpoints/pisasr_pix_final.pkl)")
    p.add_argument("--ram_path", default=None)
    p.add_argument("--lora_rank_unet_pix", type=int, default=4)
    p.add_argument("--lora_rank_unet_sem", type=int, default=4)
    p.add_argument("--timesteps1", type=int, default=1)
    p.add_argument("--null_text_ratio", type=float, default=0.5)
    p.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    p.add_argument("--resume_ckpt", default=None)
    # данные
    p.add_argument("--lq_dir", required=True)
    p.add_argument("--hq_dir", required=True)
    p.add_argument("--val_lq_dir", required=True)
    p.add_argument("--val_hq_dir", required=True)
    p.add_argument("--align_method", choices=["adain", "wavelet", "nofix"], default="adain")
    p.add_argument("--n_val", type=int, default=100)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--gsd_cm", type=float, default=8.0, help="размер пикселя эталона, см (для эффективного разрешения)")
    p.add_argument("--prompt", default="")
    p.add_argument("--pos_prompt_csd", default="")
    p.add_argument("--neg_prompt", default=(
        "painting, oil painting, illustration, drawing, art, sketch, cartoon, "
        "CG Style, 3D render, blurring, dirty, messy, worst quality, low quality, "
        "frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth"))
    # обучение
    p.add_argument("--output_dir", default="experiments/freq")
    p.add_argument("--pix_steps", type=int, default=4000)
    p.add_argument("--sem_steps", type=int, default=4000)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--save_ckpt", action="store_true")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=1, help="шагов накопления градиента на одно обновление")
    p.add_argument("--ema_decay", type=float, default=0.0, help="EMA весов LoRA (0 = выключено, обычно 0.999)")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    p.add_argument("--aux_device", default="auto",
                   help="куда вынести учителя CSD, RAM и VGG для FDL: auto = cuda:1, если карт две, иначе cuda:0")
    # базовые потери
    p.add_argument("--lambda_l2", type=float, default=1.0)
    p.add_argument("--pix_lpips", type=float, default=0.0, help="в оригинале pix-этап = только L2")
    p.add_argument("--sem_lpips", type=float, default=2.0)
    p.add_argument("--pix_csd", type=float, default=0.0)
    p.add_argument("--sem_csd", type=float, default=1.0)
    p.add_argument("--cfg_csd", type=float, default=7.5)
    p.add_argument("--min_dm_step_ratio", type=float, default=0.02)
    p.add_argument("--max_dm_step_ratio", type=float, default=0.5)
    # частотные потери по этапам (вес 0 = выключено)
    for s in STAGES:
        for t in FREQ_TERMS:
            p.add_argument(f"--{s}_{t}", type=float, default=0.0)
        p.add_argument(f"--{s}_daw", type=float, default=0.0,
                       help="alpha карты деталей для L2/LPIPS: вес = 1 + alpha*D (0 = выключено)")
    p.add_argument("--daw_mode", choices=["detail", "detail_error"], default="detail",
                   help="detail: только карта деталей эталона; detail_error: D ⊙ E, как в FiDeSR")
    p.add_argument("--daw_fdl", action="store_true", help="считать FDL по областям карты деталей")
    p.add_argument("--daw_csd", action="store_true", help="взвешивать CSD картой деталей (как в FiDeSR)")
    p.add_argument("--ffl_alpha", type=float, default=1.0)
    p.add_argument("--ffl_patch_factor", type=int, default=1)
    p.add_argument("--fdl_phase_weight", type=float, default=1.0, help="lambda между амплитудой и фазой в FDL")
    p.add_argument("--fdl_num_proj", type=int, default=256)
    p.add_argument("--fdl_crop", type=int, default=0, help="сторона окна для FDL (0 = весь кроп; 256 экономит память)")
    p.add_argument("--fdl_crop_mode", choices=["random", "detail"], default="random")
    p.add_argument("--bw_cutoff", type=float, default=0.25, help="срез фильтра Баттерворта, доля Найквиста")
    p.add_argument("--bw_order", type=int, default=2)
    p.add_argument("--bands_hf_weight", type=float, default=1.0)
    p.add_argument("--bands_lf_weight", type=float, default=0.25)
    # логирование и калибровка
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--log_grad_every", type=int, default=100, help="как часто писать нормы градиентов членов (0 = никогда)")
    p.add_argument("--auto_weight_ratio", type=float, default=0.0,
                   help=">0: перед этапом один раз подобрать веса частотных членов так, чтобы норма градиента "
                        "каждого составляла эту долю от нормы главного члена этапа (L2 для pix, LPIPS для sem)")
    p.add_argument("--auto_weight_batches", type=int, default=16)
    a = p.parse_args()
    a.pretrained_model_path_csd = a.pretrained_model_path
    if a.aux_device == "auto":
        a.aux_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    return a


# --------------------------------------------------------------------------- #
# данные для валидации (как в исходном скрипте)
# --------------------------------------------------------------------------- #
class FixedValSet(Dataset):
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


# --------------------------------------------------------------------------- #
# замороженные модели на вспомогательной карте
# --------------------------------------------------------------------------- #
class _Acc:
    mixed_precision = "fp16"

    def __init__(self, device):
        self.device = torch.device(device)


class CSDLossAux(CSDLoss):
    """CSD из PiSA-SR; учитель может стоять на другой видеокарте.
    weight — необязательная карта (N,1,h,w) в разрешении латента (взвешивание, как в FiDeSR)."""

    def cal_csd(self, latents, prompt_embeds, negative_prompt_embeds, args, weight=None):
        dev = next(self.unet_fix.parameters()).device
        bsz = latents.shape[0]
        n_t = self.sched.config.num_train_timesteps
        timestep = torch.randint(int(n_t * args.min_dm_step_ratio), int(n_t * args.max_dm_step_ratio),
                                 (bsz,), device=dev).long()
        lat = latents.detach().float().to(dev)
        noise = torch.randn_like(lat)
        noisy = self.sched.add_noise(lat, noise, timestep)
        with torch.no_grad():
            emb = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(dev, dtype=torch.float16)
            noise_pred = self.forward_latent(self.unet_fix, latents=torch.cat([noisy] * 2).to(dtype=torch.float16),
                                             timestep=torch.cat([timestep] * 2), prompt_embeds=emb)
            uncond, text = noise_pred.float().chunk(2)
            guided = uncond + args.cfg_csd * (text - uncond)
            pred_real = self.eps_to_mu(self.sched, guided, noisy, timestep)
            pred_fake = self.eps_to_mu(self.sched, uncond, noisy, timestep)
            wf = torch.abs(lat - pred_real).mean(dim=[1, 2, 3], keepdim=True)
            grad = ((pred_fake - pred_real) / wf).to(latents.device, latents.dtype)
        target = (latents - grad).detach()
        if weight is None:
            return F.mse_loss(latents, target)
        return (weight.to(latents.dtype) * (latents - target) ** 2).mean()


def build_ram(args, dev):
    if not args.ram_path or not os.path.exists(args.ram_path):
        print("[RAM] не задан — используется фиксированный промпт")
        return None, None
    from ram.models.ram_lora import ram
    RAM = ram(pretrained=args.ram_path, pretrained_condition=None,
              image_size=384, vit="swin_l").eval().to(dev, dtype=torch.float16)
    tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    print(f"[RAM] загружен на {dev}")
    return RAM, tf


def get_prompts(args, RAM, ram_tf, hq, bs, dev):
    if RAM is None:
        base = args.prompt
        return [f"{base}, {args.pos_prompt_csd}" if args.pos_prompt_csd else base] * bs
    from ram import inference_ram
    with torch.no_grad():
        tags = inference_ram(ram_tf(hq * 0.5 + 0.5).to(dev, dtype=torch.float16), RAM)
    return [f"{t}, {args.pos_prompt_csd}" for t in tags]


# --------------------------------------------------------------------------- #
# потери
# --------------------------------------------------------------------------- #
class LossBank:
    """Все члены потерь в пространстве изображений. unit() возвращает НЕвзвешенные значения,
    веса хранятся отдельно в self.w[stage] — это нужно для лога норм градиентов и калибровки."""

    def __init__(self, args):
        self.args = args
        self.w = {s: {"l2": args.lambda_l2, "lpips": getattr(args, f"{s}_lpips")} for s in STAGES}
        for s in STAGES:
            for t in FREQ_TERMS:
                self.w[s][t] = getattr(args, f"{s}_{t}")
        self.lpips_plain = lpips_pkg.LPIPS(net="vgg").cuda().requires_grad_(False)
        need_spatial = any(getattr(args, f"{s}_daw") > 0 for s in STAGES)
        self.lpips_spatial = (lpips_pkg.LPIPS(net="vgg", spatial=True).cuda().requires_grad_(False)
                              if need_spatial else None)
        self.ffl = (FL.build_ffl(args.ffl_alpha, args.ffl_patch_factor)
                    if any(self.w[s]["ffl"] for s in STAGES) else None)
        need_fdl = any(self.w[s][t] for s in STAGES for t in ("fdl", "fdl_edge", "fdl_bands"))
        self.fdl = (FL.build_fdl(args.fdl_phase_weight, args.fdl_num_proj, device=args.aux_device)
                    if need_fdl else None)
        if self.fdl is not None:
            print(f"[FDL] VGG19 на {args.aux_device}, проекций {args.fdl_num_proj}, окно {args.fdl_crop or 'весь кроп'}")

    def unit(self, stage, pred, hq):
        """dict: имя -> невзвешенное значение (только члены с ненулевым весом). Плюс карта деталей."""
        a, w = self.args, self.w[stage]
        p, t = pred.float(), hq.float()
        alpha = getattr(a, f"{stage}_daw")
        out, dmap = {}, None
        lp_map = None
        if w["lpips"]:
            if alpha > 0:
                lp_map = self.lpips_spatial(p, t)
            else:
                out["lpips"] = self.lpips_plain(p, t).mean()
        if alpha > 0:
            dmap = FL.daw_weight(t, alpha, pred=p if a.daw_mode == "detail_error" else None, lpips_map=lp_map)
            if lp_map is not None:
                out["lpips"] = (dmap * lp_map).mean()
            out["l2"] = (dmap * (p - t) ** 2).mean()
        else:
            out["l2"] = F.mse_loss(p, t)
        if w["ffl"]:
            out["ffl"] = self.ffl(p, t)
        if w["fsl"]:
            out["fsl"] = FL.fourier_space_loss(p, t)
        if self.fdl is not None and any(w[k] for k in ("fdl", "fdl_edge", "fdl_bands")):
            crop_ij = None
            if a.fdl_crop and a.fdl_crop < min(p.shape[-2:]):
                if a.fdl_crop_mode == "detail":
                    crop_ij = FL.best_detail_crop(t, a.fdl_crop)
                else:
                    crop_ij = (random.randint(0, p.shape[-2] - a.fdl_crop), random.randint(0, p.shape[-1] - a.fdl_crop))
            region = dmap if (a.daw_fdl and dmap is not None) else None
            kw = dict(region=region, crop=a.fdl_crop, crop_ij=crop_ij, bw_cutoff=a.bw_cutoff, bw_order=a.bw_order,
                      hf_weight=a.bands_hf_weight, lf_weight=a.bands_lf_weight)
            for name, mode in (("fdl", "full"), ("fdl_edge", "edge"), ("fdl_bands", "bands")):
                if w[name]:
                    out[name] = FL.fdl_call(self.fdl, p, t, mode, **kw).to(p.device)
        return out, dmap

    def total(self, stage, units):
        return sum(self.w[stage][k] * v for k, v in units.items())


def grad_norms(units, pred):
    out = {}
    for k, v in units.items():
        g, = torch.autograd.grad(v, pred, retain_graph=True, allow_unused=True)
        out[k] = float(g.float().norm()) if g is not None else 0.0
    return out


class EMA:
    def __init__(self, params, decay):
        self.params, self.decay = params, decay
        self.shadow = [p.detach().clone() for p in params]
        self.backup = None

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def swap_in(self):
        self.backup = [p.detach().clone() for p in self.params]
        for s, p in zip(self.shadow, self.params):
            p.copy_(s)

    @torch.no_grad()
    def swap_out(self):
        for b, p in zip(self.backup, self.params):
            p.copy_(b)
        self.backup = None


# --------------------------------------------------------------------------- #
# валидация
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(net, val_dl, metric, args, amp_dtype, use_amp):
    was_training = net.unet.training
    net.unet.eval()
    old_ratio = args.null_text_ratio
    args.null_text_ratio = 1.1                       # всегда пустой промпт
    vals, vols, vols_gt = [], [], []
    acc = np.zeros((FM.N_BANDS, 4))
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(1234)
        for lq, hq in val_dl:
            lq, hq = lq.cuda(), hq.cuda()
            b = {"prompt": [""] * lq.shape[0], "neg_prompt": [""] * lq.shape[0], "null_prompt": [""] * lq.shape[0]}
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                pred, _, _, _ = net(lq, hq, batch=b, args=args)
            for k in range(pred.shape[0]):
                out = transforms.ToPILImage()((pred[k].float().cpu() * 0.5 + 0.5).clamp(0, 1))
                src = transforms.ToPILImage()((lq[k].float().cpu() * 0.5 + 0.5).clamp(0, 1))
                if args.align_method == "adain":
                    out = adain_color_fix(target=out, source=src)
                elif args.align_method == "wavelet":
                    out = wavelet_color_fix(target=out, source=src)
                gt_pil = transforms.ToPILImage()((hq[k].float().cpu() * 0.5 + 0.5).clamp(0, 1))
                vols.append(FM.vol_of(out)); vols_gt.append(FM.vol_of(gt_pil))
                tp = TF.to_tensor(out).unsqueeze(0).cuda()
                tg = (hq[k:k + 1].float() * 0.5 + 0.5).clamp(0, 1)      # как в исходном скрипте
                vals.extend(metric(tp, tg).flatten().tolist())
                acc += FM.band_stats(tp, tg)
    args.null_text_ratio = old_ratio
    net.unet.train(was_training)
    s, bands = FM.summary(acc, gsd_cm=args.gsd_cm)
    res = {"lpips": float(np.mean(vals)), "lpips_se": float(np.std(vals) / np.sqrt(len(vals))),
           "vol": float(np.mean(vols)), "vol_gt": float(np.mean(vols_gt))}
    res.update(s)
    return res, bands


def mem_gb():
    m = [torch.cuda.max_memory_allocated(i) / 2 ** 30 for i in range(torch.cuda.device_count())]
    return (m + [float("nan"), float("nan")])[:2]


# --------------------------------------------------------------------------- #
def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    ck_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"args_{args.stage}.json"), "w") as jf:
        json.dump(vars(args), jf, indent=2, ensure_ascii=False)

    net = PiSASR(args)
    if args.stage == "sem":
        start = args.sem_init_ckpt or os.path.join(ck_dir, "pisasr_pix_final.pkl")
    else:
        start = args.init_ckpt
    assert start and os.path.exists(start), f"нет стартового чекпоинта: {start}"
    load_lora(net, start)
    net.unet.enable_gradient_checkpointing()

    import pyiqa
    metric = pyiqa.create_metric("lpips", device="cuda")
    bank = LossBank(args)
    aux = args.aux_device
    RAM, ram_tf = build_ram(args, aux)
    need_csd = (args.stage in ("sem", "both") and args.sem_csd > 0) or (args.stage in ("pix", "both") and args.pix_csd > 0)
    csd = None
    if need_csd:
        csd = CSDLossAux(args=args, accelerator=_Acc(aux)); csd.requires_grad_(False)
        print(f"[CSD] учитель на {aux}")

    train_ds = PairedLQHQDataset(args.lq_dir, args.hq_dir, resolution=args.resolution,
                                 prompt=args.prompt, neg_prompt=args.neg_prompt)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True)
    val_dl = DataLoader(FixedValSet(args.val_lq_dir, args.val_hq_dir, args.n_val, args.resolution),
                        batch_size=1, shuffle=False, num_workers=1)

    use_amp = args.mixed_precision != "no"
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))

    # --- логи ---
    log_path = os.path.join(args.output_dir, f"curve_{args.stage}.csv")
    f = open(log_path, "w", newline=""); w = csv.writer(f)
    w.writerow(["global_step", "stage", "lpips_val", "lpips_val_sem", "vol", "vol_gt", "vol_ratio",
                "train_l2", "train_lpips", "hf_frc", "hf_err", "hf_energy", "f_eff", "eff_gsd_cm",
                "mem0_gb", "mem1_gb", "it_per_s"])
    fb = open(os.path.join(args.output_dir, f"bands_{args.stage}.csv"), "w", newline=""); wb = csv.writer(fb)
    wb.writerow(["global_step", "stage", "metric"] + [f"b{i}" for i in range(FM.N_BANDS)])
    term_names = ["l2", "lpips", "csd"] + list(FREQ_TERMS)
    ft = open(os.path.join(args.output_dir, f"train_log_{args.stage}.csv"), "w", newline=""); wt = csv.writer(ft)
    wt.writerow(["global_step", "stage"] + [f"val_{k}" for k in term_names] + [f"w_{k}" for k in term_names]
                + [f"gnorm_{k}" for k in term_names])

    state = {"ema": None, "speed": float("nan")}

    def log_eval(gstep, stage, l2=float("nan"), lp=float("nan")):
        if state["ema"] is not None:
            state["ema"].swap_in()
        r, bands = evaluate(net, val_dl, metric, args, amp_dtype, use_amp)
        if args.save_ckpt and gstep > 0:
            net.save_model(os.path.join(ck_dir, f"pisasr_{gstep}.pkl"))
        if state["ema"] is not None:
            state["ema"].swap_out()
        m = mem_gb()
        w.writerow([gstep, stage, f"{r['lpips']:.5f}", f"{r['lpips_se']:.5f}", f"{r['vol']:.2f}", f"{r['vol_gt']:.2f}",
                    f"{r['vol'] / r['vol_gt']:.4f}", f"{l2:.5f}", f"{lp:.5f}", f"{r['hf_frc']:.4f}",
                    f"{r['hf_err']:.4f}", f"{r['hf_energy']:.4f}", f"{r['f_eff']:.4f}", f"{r['eff_gsd_cm']:.2f}",
                    f"{m[0]:.2f}", f"{m[1]:.2f}", f"{state['speed']:.3f}"]); f.flush()
        for name in ("frc", "err", "energy"):
            wb.writerow([gstep, stage, name] + [f"{v:.4f}" for v in bands[name]])
        fb.flush()
        print(f"[eval] step {gstep:5d} ({stage})  LPIPS {r['lpips']:.4f} ± {r['lpips_se']:.4f}  VoL/GT "
              f"{r['vol'] / r['vol_gt']:.3f}  ВЧ: FRC {r['hf_frc']:.3f} ошибка {r['hf_err']:.3f} энергия "
              f"{r['hf_energy']:.3f}  эфф. разрешение {r['eff_gsd_cm']:.1f} см  память {m[0]:.1f}/{m[1]:.1f} ГБ")

    def forward_batch(batch, stage):
        lq = batch["conditioning_pixel_values"].cuda()
        hq = batch["output_pixel_values"].cuda()
        batch["prompt"] = get_prompts(args, RAM, ram_tf, hq, lq.shape[0], aux)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            pred, lat, pe, npe = net(lq, hq, batch=batch, args=args)
        return pred, hq, lat, pe, npe

    def calibrate(stage):
        """Одноразовый подбор весов частотных членов по норме градиента относительно главного члена этапа."""
        ref = "lpips" if bank.w[stage]["lpips"] else "l2"
        sums, n = {}, 0
        it = iter(train_dl)
        for _ in range(args.auto_weight_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            with torch.no_grad():
                pred, hq, _, _, _ = forward_batch(batch, stage)
            leaf = pred.detach().float().requires_grad_(True)
            units, _ = bank.unit(stage, leaf, hq)
            for k, v in grad_norms(units, leaf).items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        ref_norm = bank.w[stage][ref] * sums[ref] / n
        for k in FREQ_TERMS:
            if bank.w[stage][k] and sums.get(k, 0) > 0:
                new = args.auto_weight_ratio * ref_norm / (sums[k] / n)
                print(f"[калибровка {stage}] {k}: вес {bank.w[stage][k]:.3g} -> {new:.3g} "
                      f"(норма градиента = {args.auto_weight_ratio:.2f} от {ref})")
                bank.w[stage][k] = new

    def run_stage(stage, n_steps, offset):
        if stage == "pix":
            net.unet.set_adapter(PIX_ADAPTERS); net.set_train_pix()
            params = [p for n, p in net.unet.named_parameters() if "lora" in n and p.requires_grad]
        else:
            net.unet.set_adapter(ALL_ADAPTERS); net.set_train_sem()
            params = [p for n, p in net.unet.named_parameters() if "lora" in n and "sem" in n and p.requires_grad]
        net.unet.train()
        opt = torch.optim.AdamW(params, lr=args.lr)
        state["ema"] = EMA(params, args.ema_decay) if args.ema_decay > 0 else None
        lam_csd = getattr(args, f"{stage}_csd")
        if args.auto_weight_ratio > 0 and any(bank.w[stage][k] for k in FREQ_TERMS):
            calibrate(stage)
        active = {k: v for k, v in bank.w[stage].items() if v}
        print(f"[{stage}] обучаемых параметров: {sum(p.numel() for p in params) / 1e6:.2f}M; "
              f"веса потерь: {active}, csd: {lam_csd}, карта деталей: {getattr(args, stage + '_daw')}")
        step, t0 = 0, time.time()
        opt.zero_grad(set_to_none=True)
        while step < n_steps:
            for batch in train_dl:
                pred, hq, lat, pe, npe = forward_batch(batch, stage)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    units, dmap = bank.unit(stage, pred, hq)
                    loss = bank.total(stage, units)
                    csd_val = None
                    if lam_csd > 0:
                        wl = F.avg_pool2d(dmap, 8) if (args.daw_csd and dmap is not None) else None
                        csd_val = csd.cal_csd(lat, pe, npe, args, weight=wl)
                        loss = loss + lam_csd * csd_val
                gstep = offset + step + 1
                if args.log_grad_every and (step + 1) % args.log_grad_every == 0:
                    gn = grad_norms(units, pred)
                    vals = {k: float(v) for k, v in units.items()}
                    wts = dict(bank.w[stage])
                    if csd_val is not None:
                        g, = torch.autograd.grad(csd_val, lat, retain_graph=True)
                        gn["csd"], vals["csd"], wts["csd"] = float(g.float().norm()), float(csd_val), lam_csd
                    wt.writerow([gstep, stage] + [f"{vals.get(k, float('nan')):.6g}" for k in term_names]
                                + [f"{wts.get(k, 0):.4g}" for k in term_names]
                                + [f"{gn.get(k, float('nan')):.4g}" for k in term_names]); ft.flush()
                scaler.scale(loss / args.grad_accum).backward()
                step += 1
                if step % args.grad_accum == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)
                    if state["ema"] is not None:
                        state["ema"].update()
                l2v = float(units["l2"]) * bank.w[stage]["l2"]
                lpv = float(units["lpips"]) * bank.w[stage]["lpips"] if "lpips" in units else 0.0
                if step % args.log_every == 0:
                    state["speed"] = step / (time.time() - t0)
                    extra = " ".join(f"{k} {float(v) * bank.w[stage][k]:.4f}" for k, v in units.items()
                                     if k not in ("l2", "lpips"))
                    print(f"[{stage}] {step}/{n_steps} l2 {l2v:.4f} lpips {lpv:.4f} {extra} "
                          f"{'csd %.4f' % float(csd_val) if csd_val is not None else ''} ({state['speed']:.2f} шаг/с)")
                if step % args.eval_every == 0:
                    log_eval(offset + step, stage, l2v, lpv)
                if step >= n_steps:
                    break

    def save_final(name):
        net.save_model(os.path.join(ck_dir, name + ".pkl"))
        if state["ema"] is not None:
            state["ema"].swap_in()
            net.save_model(os.path.join(ck_dir, name + "_ema.pkl"))
            state["ema"].swap_out()

    if args.stage in ("pix", "both"):
        net.unet.set_adapter(PIX_ADAPTERS)
        log_eval(0, "init")
        run_stage("pix", args.pix_steps, 0)
        save_final("pisasr_pix_final")
        if state["ema"] is not None:                    # sem-этап продолжает с EMA-весов pix-этапа
            state["ema"].swap_in(); state["ema"].backup = None; state["ema"] = None
    if args.stage in ("sem", "both"):
        net.unet.set_adapter(ALL_ADAPTERS)
        log_eval(args.pix_steps, "sem_start")
        run_stage("sem", args.sem_steps, args.pix_steps)
        save_final("pisasr_final")
    f.close(); fb.close(); ft.close()

    # --- график ---
    try:
        import glob, pandas as pd, matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        d = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(args.output_dir, "curve_*.csv")))])
        d = d.drop_duplicates(["global_step", "stage"], keep="last").sort_values("global_step")
        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        for name, col, sel in [("pix-этап", "tab:blue", d.stage.isin(["init", "pix"])),
                               ("sem-этап", "tab:orange", d.stage.isin(["sem_start", "sem"]))]:
            s = d[sel]
            axes[0].plot(s.global_step, s.lpips_val, "-o", ms=2.5, color=col, label=name)
            axes[0].fill_between(s.global_step, s.lpips_val - s.lpips_val_sem, s.lpips_val + s.lpips_val_sem,
                                 color=col, alpha=0.15)
            axes[1].plot(s.global_step, s.vol_ratio, "-o", ms=2.5, color=col)
            axes[2].plot(s.global_step, s.hf_err, "-o", ms=2.5, color=col)
        for a_ in axes:
            a_.axvline(args.pix_steps, ls="--", c="gray"); a_.grid(alpha=0.3)
        axes[1].axhline(1.0, ls=":", c="k"); axes[2].axhline(1.0, ls=":", c="k")
        axes[0].set_ylabel("LPIPS (val), ↓"); axes[0].legend()
        axes[1].set_ylabel("VoL / VoL(GT), цель ≈ 1")
        axes[2].set_ylabel("ошибка в ВЧ-полосах, ↓ (1 = не восстановлено)")
        axes[2].set_xlabel("шаг дообучения")
        fig.tight_layout(); fig.savefig(os.path.join(args.output_dir, "curve.png"), dpi=200)
    except Exception as e:
        print("график не построен:", e)
    print("готово:", log_path)


if __name__ == "__main__":
    main()
