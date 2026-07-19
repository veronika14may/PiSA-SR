import os
import gc
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from pisasr import CSDLoss, PiSASR
from src.my_utils.training_utils import parse_args
from src.datasets.dataset import PairedSROnlineTxtDataset

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
import random


def _activate_semantic_branch(net_pisasr, is_module):
    """Switch UNet to combined pix+sem adapters and mark sem params trainable."""
    if is_module:
        net_pisasr.module.unet.set_adapter([
            'default_encoder_pix', 'default_decoder_pix', 'default_others_pix',
            'default_encoder_sem', 'default_decoder_sem', 'default_others_sem',
        ])
        net_pisasr.module.set_train_sem()
    else:
        net_pisasr.unet.set_adapter([
            'default_encoder_pix', 'default_decoder_pix', 'default_others_pix',
            'default_encoder_sem', 'default_decoder_sem', 'default_others_sem',
        ])
        net_pisasr.set_train_sem()


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    net_pisasr = PiSASR(args)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pisasr.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pisasr.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # init CSDLoss model
    net_csd = CSDLoss(args=args, accelerator=accelerator)
    net_csd.requires_grad_(False)

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # set gen adapter (pix-only by default; may be switched to pix+sem below on resume)
    net_pisasr.unet.set_adapter(['default_encoder_pix', 'default_decoder_pix', 'default_others_pix'])
    net_pisasr.set_train_pix()  # first to remove degradation

    # make the optimizer
    layers_to_opt = []
    for n, _p in net_pisasr.unet.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(
        layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,
    )

    # initialize the dataset
    dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
    dataset_val = PairedSROnlineTxtDataset(split="test", args=args)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # init RAM for text prompt extractor
    from ram.models.ram_lora import ram
    from ram import inference_ram as inference
    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    RAM = ram(pretrained=args.ram_path,
              pretrained_condition=None,
              image_size=384,
              vit='swin_l')
    RAM.eval()
    RAM.to("cuda", dtype=torch.float16)

    # Prepare everything with our `accelerator`.
    net_pisasr, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_pisasr, optimizer, dl_train, lr_scheduler
    )
    net_lpips = accelerator.prepare(net_lpips)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    global_step = 0
    lambda_l2 = args.lambda_l2
    lambda_lpips = 0
    lambda_csd = 0

    if args.resume_ckpt is not None:
        print(f"====> resuming training state from {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("  loaded optimizer state (Adam moments)")
        else:
            print("  WARNING: no optimizer_state_dict in ckpt - Adam moments will restart from zero")

        if "lr_scheduler_state_dict" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
            print("  loaded lr_scheduler state (warmup will not replay)")
        else:
            print("  WARNING: no lr_scheduler_state_dict in ckpt - warmup will replay from zero")

        if "global_step" in ckpt:
            global_step = ckpt["global_step"]
            print(f"  resumed at global_step={global_step}")
        else:
            print("  WARNING: no global_step in ckpt - starting count from 0")

        del ckpt
        gc.collect()
        torch.cuda.empty_cache()

        # If we resumed past pix_steps, switch to semantic branch immediately.
        if global_step >= args.pix_steps:
            _activate_semantic_branch(net_pisasr, args.is_module)
            lambda_lpips = args.lambda_lpips
            lambda_csd = args.lambda_csd
            print(f"====> resumed past pix_steps ({global_step} >= {args.pix_steps}), semantic branch active")

    progress_bar = tqdm(
        range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # ------------------------------------------------------------------
    # Skip already-processed micro-batches so that we do not train on
    # them again (which would keep advancing the LR scheduler and
    # optimizer past the intended point).
    #
    # NOTE: this counts micro-batches, not the exact original order of
    # samples (DataLoader shuffle uses fresh RNG each session). For LoRA
    # fine-tuning this is a fine approximation.
    # ------------------------------------------------------------------
    resume_step = global_step
    steps_to_skip = resume_step * args.gradient_accumulation_steps
    steps_skipped = 0
    if resume_step > 0:
        print(f"  will skip ~{steps_to_skip} micro-batches on the first epoch to catch up to step {resume_step}")

    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):

            # Skip micro-batches until we catch up to the resumed step.
            if steps_skipped < steps_to_skip:
                steps_skipped += 1
                continue

            with accelerator.accumulate(net_pisasr):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]

                # get text prompts from GT
                x_tgt_ram = ram_transforms(x_tgt * 0.5 + 0.5)
                caption = inference(x_tgt_ram.to(dtype=torch.float16), RAM)
                batch["prompt"] = [f'{each_caption}, {args.pos_prompt_csd}' for each_caption in caption]

                if global_step == args.pix_steps:
                    # begin the semantic optimization
                    _activate_semantic_branch(net_pisasr, args.is_module)
                    lambda_l2 = args.lambda_l2
                    lambda_lpips = args.lambda_lpips
                    lambda_csd = args.lambda_csd

                x_tgt_pred, latents_pred, prompt_embeds, neg_prompt_embeds = net_pisasr(x_src, x_tgt, batch=batch, args=args)
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * lambda_lpips
                loss = loss_l2 + loss_lpips
                # reg loss
                loss_csd = net_csd.cal_csd(latents_pred, prompt_embeds, neg_prompt_embeds, args) * lambda_csd
                loss = loss + loss_csd
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    logs["loss_csd"] = loss_csd.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    progress_bar.set_postfix(**logs)

                    # checkpoint the model + full training state for clean resume
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pisasr).save_model(
                            outf,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            global_step=global_step,
                        )

                    # eval on the held-out test set
                    if global_step % args.eval_freq == 1:
                        os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        for _step, batch_val in enumerate(dl_val):
                            x_src = batch_val["conditioning_pixel_values"].cuda()
                            x_tgt = batch_val["output_pixel_values"].cuda()
                            x_basename = batch_val["base_name"][0]
                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # get text prompts from LR
                                x_src_ram = ram_transforms(x_src * 0.5 + 0.5)
                                caption = inference(x_src_ram.to(dtype=torch.float16), RAM)
                                batch_val["prompt"] = caption
                                # forward pass
                                x_tgt_pred, latents_pred, _, _ = accelerator.unwrap_model(net_pisasr)(
                                    x_src, x_tgt, batch=batch_val, args=args
                                )
                                # save the output
                                output_pil = transforms.ToPILImage()(x_tgt_pred[0].cpu() * 0.5 + 0.5)
                                input_image = transforms.ToPILImage()(x_src[0].cpu() * 0.5 + 0.5)
                                if args.align_method == 'adain':
                                    output_pil = adain_color_fix(target=output_pil, source=input_image)
                                elif args.align_method == 'wavelet':
                                    output_pil = wavelet_color_fix(target=output_pil, source=input_image)
                                else:
                                    pass
                                outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"{x_basename}")
                                output_pil.save(outf)
                        gc.collect()
                        torch.cuda.empty_cache()
                        accelerator.log(logs, step=global_step)

                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

        # Reset skip counter so subsequent epochs don't skip anything.
        steps_to_skip = 0
        if global_step >= args.max_train_steps:
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
