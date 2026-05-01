#!/usr/bin/env python
# coding=utf-8
"""
Stage-2 training: fine-tune remaining DualFlowControlNet_traj parameters on
RARP multimodal dataset.

Stage-1 trains:  flow_encoder, controlnet_cond_embedding, SegCondEncoder
Stage-2 trains:  all other ControlNet params (down/mid blocks, zero convs)
Stage-2 freezes: flow_encoder, controlnet_cond_embedding, SegCondEncoder, UNet, VAE, CLIP

Data: identical to Stage-1 – pre-computed RGB/flow/depth/mask, letterbox to 320×192.
"""
import argparse
import logging
import math
import os
import shutil
import random
from pathlib import Path

import accelerate
import numpy as np
from PIL import Image
import torch
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

import diffusers
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version

from train_utils.rarp_dataset import RARPDataset
from train_utils.rarp_losses import SegCondEncoder, compute_multitask_loss
from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
from pipeline.pipeline import DualFlowControlNetPipeline
from models.svdxt_featureflow_forward_controlnet_s2d_fixcmp_norefine import DualFlowControlNet_traj

check_min_version("0.24.0.dev0")
logger = get_logger(__name__, log_level="INFO")

# ---------------------------------------------------------------------------
# Sigma / noise schedule helpers (identical to stage1)
# ---------------------------------------------------------------------------
import math as _math

def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high,
                              sigma_data=1., min_value=1e-3, max_value=1e3,
                              device='cpu', dtype=torch.float32):
    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = _math.atan(_math.exp(-0.5 * logsnr_max))
        t_max = _math.atan(_math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * _math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_cosine_interp(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        low  = logsnr_cosine_shifted(t, image_d, noise_d_low,  logsnr_min, logsnr_max)
        high = logsnr_cosine_shifted(t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(low, high, t)

    logsnr_min = -2 * _math.log(min_value / sigma_data)
    logsnr_max = -2 * _math.log(max_value / sigma_data)
    u = stratified_uniform(shape, group=0, groups=1, dtype=dtype, device=device)
    logsnr = logsnr_cosine_interp(u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data


min_value   = 0.002
max_value   = 700
image_d     = 64
noise_d_low = 32
noise_d_high= 64
sigma_data  = 0.5


# ---------------------------------------------------------------------------
# Antialiased resize (for CLIP encoder input)
# ---------------------------------------------------------------------------

def _compute_padding(kernel_size):
    computed = [k - 1 for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]
    for i in range(len(kernel_size)):
        c = computed[-(i + 1)]
        out_padding[2 * i]     = c // 2
        out_padding[2 * i + 1] = c - c // 2
    return out_padding


def _filter2d(input, kernel):
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]
    input = torch.nn.functional.pad(input, _compute_padding([height, width]), mode="reflect")
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    return output.view(b, c, h, w)


def _gaussian(window_size, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])
    bs = sigma.shape[0]
    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(bs, -1)
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))
    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)
    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    return _filter2d(_filter2d(input, kernel_x[..., None, :]), kernel_y[..., None])


def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])
    sigmas = (max((factors[0] - 1.0) / 2.0, 0.001), max((factors[1] - 1.0) / 2.0, 0.001))
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))
    if ks[0] % 2 == 0: ks = ks[0] + 1, ks[1]
    if ks[1] % 2 == 0: ks = ks[0], ks[1] + 1
    input = _gaussian_blur2d(input, ks, sigmas)
    return torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)


def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]
    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    return latents * vae.config.scaling_factor


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Stage-2 RARP ControlNet training")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--stage1_controlnet_path", type=str, required=True,
                        help="Path to Stage-1 controlnet checkpoint directory.")
    parser.add_argument("--stage1_seg_encoder_path", type=str, required=True,
                        help="Path to Stage-1 seg_encoder.pth file.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="logs/train_stage2_rarp")
    parser.add_argument("--width",  type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--num_frames", type=int, default=21)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--max_flow", type=float, default=20.0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_gpu_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--conditioning_dropout_prob", type=float, default=0.1)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpointing_steps", type=int, default=3000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--validation_steps", type=int, default=3000)
    parser.add_argument("--num_validation_images", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--lambda_fg", type=float, default=2.0)
    parser.add_argument("--lambda_flow_tv", type=float, default=0.01)
    parser.add_argument("--lambda_depth_tv", type=float, default=0.05)
    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed or 42)

    # ------------------------------------------------------------------
    # Load base models (frozen)
    # ------------------------------------------------------------------
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", variant="fp16")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", variant="fp16")
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
        low_cpu_mem_usage=True, variant="fp16")

    # ------------------------------------------------------------------
    # Load Stage-1 controlnet checkpoint
    # ------------------------------------------------------------------
    logger.info(f"[RANK 0] Loading Stage-1 controlnet from {args.stage1_controlnet_path}")
    controlnet = DualFlowControlNet_traj.from_pretrained(
        args.stage1_controlnet_path, low_cpu_mem_usage=False)

    # Load SegCondEncoder from stage1
    seg_encoder = SegCondEncoder()
    ckpt_path = args.stage1_seg_encoder_path
    if os.path.exists(ckpt_path):
        seg_encoder.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        logger.info(f"[RANK 0] Loaded SegCondEncoder from {ckpt_path}")
    else:
        logger.warning(f"[RANK 0] seg_encoder.pth not found at {ckpt_path}, using random init")

    # ------------------------------------------------------------------
    # Freeze / unfreeze parameters
    # Stage-2 freezes: vae, image_encoder, unet, flow_encoder,
    #                  controlnet_cond_embedding, seg_encoder
    # Stage-2 trains:  all other controlnet params (down/mid/zero convs)
    # ------------------------------------------------------------------
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    seg_encoder.requires_grad_(False)

    trainable_params = []
    for name, param in controlnet.named_parameters():
        if "flow_encoder" in name or "controlnet_cond_embedding" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
            trainable_params.append(param)

    logger.info(f"[RANK 0] Stage-2 trainable params: "
                f"{sum(p.numel() for p in trainable_params) / 1e6:.1f}M")

    # ------------------------------------------------------------------
    # dtype
    # ------------------------------------------------------------------
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    seg_encoder.to(accelerator.device, dtype=weight_dtype)

    # Gradient checkpointing on controlnet
    controlnet.enable_gradient_checkpointing()

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ------------------------------------------------------------------
    # Save / load hooks (mirrors stage1 pattern)
    # ------------------------------------------------------------------
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            for model in models:
                if isinstance(model, DualFlowControlNet_traj):
                    model.save_pretrained(os.path.join(output_dir, "controlnet"))
                weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                model = models.pop()
                if isinstance(model, DualFlowControlNet_traj):
                    load_model = DualFlowControlNet_traj.from_pretrained(
                        input_dir, subfolder="controlnet", low_cpu_mem_usage=False)
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # ------------------------------------------------------------------
    # Dataset & dataloader
    # ------------------------------------------------------------------
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    train_dataset = RARPDataset(
        data_root=args.data_root,
        sample_n_frames=args.num_frames,
        sample_stride=args.sample_stride,
        sample_size=[args.height, args.width],
        max_flow=args.max_flow,
        val_split=args.val_split,
        split="train",
        seed=args.seed,
        augment=True,
    )
    val_dataset = RARPDataset(
        data_root=args.data_root,
        sample_n_frames=args.num_frames,
        sample_stride=args.sample_stride,
        sample_size=[args.height, args.width],
        max_flow=args.max_flow,
        val_split=args.val_split,
        split="val",
        seed=args.seed,
        augment=False,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # LR scheduler & accelerate prepare
    # ------------------------------------------------------------------
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    controlnet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        controlnet, optimizer, lr_scheduler, train_dataloader)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("stage2_rarp", config=vars(args))

    # ------------------------------------------------------------------
    # encode_image helper (CLIP)
    # ------------------------------------------------------------------
    def encode_image(pixel_values):
        pv = pixel_values * 2.0 - 1.0
        pv = _resize_with_antialiasing(pv, (224, 224))
        pv = (pv + 1.0) / 2.0
        pv = feature_extractor(
            images=pv, do_normalize=True, do_center_crop=False,
            do_resize=False, do_rescale=False, return_tensors="pt"
        ).pixel_values.to(device=accelerator.device, dtype=weight_dtype)
        return image_encoder(pv).image_embeds.unsqueeze(1)

    def _get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, dtype, batch_size):
        add_time_ids = torch.tensor(
            [[fps, motion_bucket_id, noise_aug_strength]] * batch_size, dtype=dtype)
        passed_dim = unet.config.addition_time_embed_dim * add_time_ids.shape[1]
        expected_dim = unet.add_embedding.linear_1.in_features
        if passed_dim != expected_dim:
            raise ValueError(f"add_time_ids dim mismatch: {passed_dim} vs {expected_dim}")
        return add_time_ids

    # ------------------------------------------------------------------
    # Training info
    # ------------------------------------------------------------------
    total_batch_size = (args.per_gpu_batch_size * accelerator.num_processes
                        * args.gradient_accumulation_steps)
    logger.info("***** Running RARP Stage-2 Training *****")
    logger.info(f"  Num clips (train)  = {len(train_dataset)}")
    logger.info(f"  Num clips (val)    = {len(val_dataset)}")
    logger.info(f"  Num Epochs         = {args.num_train_epochs}")
    logger.info(f"  Total optim steps  = {args.max_train_steps}")
    logger.info(f"  Mixed precision    = {args.mixed_precision}")

    global_step = 0
    first_epoch = 0

    # ------------------------------------------------------------------
    # Optional resume
    # ------------------------------------------------------------------
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        else:
            path = os.path.basename(args.resume_from_checkpoint)
        if path is None:
            accelerator.print("No checkpoint found. Starting fresh.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                accelerator.load_state(os.path.join(args.output_dir, path), strict=False)
            except FileNotFoundError as e:
                accelerator.print(f"[WARNING] Incomplete checkpoint: {e}")
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    train_noise_aug = 0.02

    for epoch in range(first_epoch, args.num_train_epochs):
        controlnet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(controlnet):
                pixel_values = batch["pixel_values"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True)   # [B,T,3,H,W]
                flows  = batch["flows"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True)   # [B,T-1,2,H,W]
                depths = batch["depths"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True)   # [B,T,1,H,W]
                masks  = batch["masks"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True)   # [B,T,1,H,W]

                B = pixel_values.shape[0]

                latents = tensor_to_vae_latent(pixel_values, vae).to(weight_dtype)

                noise = torch.randn_like(latents)
                sigmas = rand_cosine_interpolated(
                    shape=[B], image_d=image_d, noise_d_low=noise_d_low,
                    noise_d_high=noise_d_high, sigma_data=sigma_data,
                    min_value=min_value, max_value=max_value
                ).to(device=latents.device, dtype=weight_dtype)

                sigmas_r = sigmas.clone()
                while len(sigmas_r.shape) < len(latents.shape):
                    sigmas_r = sigmas_r.unsqueeze(-1)

                small_noise_latents = latents + noise * train_noise_aug
                conditional_latents = small_noise_latents[:, 0] / vae.config.scaling_factor

                noisy_latents = latents + noise * sigmas_r
                timesteps = torch.tensor(
                    [0.25 * sigma.log() for sigma in sigmas], device=latents.device).to(weight_dtype)

                inp_noisy_latents = noisy_latents / ((sigmas_r ** 2 + 1) ** 0.5)

                encoder_hidden_states = encode_image(pixel_values[:, 0].float())

                added_time_ids = _get_add_time_ids(
                    6, 127, train_noise_aug,
                    encoder_hidden_states.dtype, B
                ).to(latents.device)

                # Conditioning dropout
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(B, device=latents.device, generator=generator)
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    encoder_hidden_states = torch.where(
                        prompt_mask.reshape(B, 1, 1),
                        torch.zeros_like(encoder_hidden_states),
                        encoder_hidden_states)
                    img_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(conditional_latents.dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(conditional_latents.dtype)
                    )
                    conditional_latents = img_mask.reshape(B, 1, 1, 1) * conditional_latents

                conditional_latents = conditional_latents.unsqueeze(1).repeat(
                    1, noisy_latents.shape[1], 1, 1, 1)
                inp_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)

                # Build ControlNet conditioning
                controlnet_image = pixel_values[:, 0]                       # [B,3,H,W]
                controlnet_depth = depths[:, 0]                             # [B,1,H,W]
                first_mask = masks[:, 0].to(dtype=next(seg_encoder.parameters()).dtype)  # [B,1,H,W]
                controlnet_mask = seg_encoder(first_mask)                   # [B,256,H/4,W/4]
                controlnet_flow = flows                                     # [B,T-1,2,H,W]

                down_block_res_samples, mid_block_res_sample, _, _ = controlnet(
                    inp_noisy_latents, timesteps, encoder_hidden_states,
                    added_time_ids=added_time_ids,
                    controlnet_cond=controlnet_image,
                    controlnet_flow=controlnet_flow,
                    controlnet_mask=controlnet_mask,
                    controlnet_depth=controlnet_depth,
                    return_dict=False,
                )

                model_pred = unet(
                    inp_noisy_latents, timesteps, encoder_hidden_states,
                    added_time_ids=added_time_ids,
                    down_block_additional_residuals=[
                        s.to(dtype=weight_dtype) for s in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                c_out = -sigmas_r / ((sigmas_r ** 2 + 1) ** 0.5)
                c_skip = 1 / (sigmas_r ** 2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas_r ** 2) * (sigmas_r ** -2.0)

                losses = compute_multitask_loss(
                    denoised_latents, latents, weighing,
                    masks, flows, depths,
                    lambda_fg=args.lambda_fg,
                    lambda_flow_tv=args.lambda_flow_tv,
                    lambda_depth_tv=args.lambda_depth_tv,
                )
                loss = losses["total"]

                avg_loss = accelerator.gather(loss.repeat(B)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    # ── checkpoint save ──────────────────────────────────
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            ckpts = sorted(
                                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                                key=lambda x: int(x.split("-")[1]))
                            if len(ckpts) >= args.checkpoints_total_limit:
                                for rm in ckpts[:len(ckpts) - args.checkpoints_total_limit + 1]:
                                    shutil.rmtree(os.path.join(args.output_dir, rm))
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    # ── validation ───────────────────────────────────────
                    if global_step % args.validation_steps == 0 or global_step == 1:
                        controlnet.eval()
                        val_iter = iter(torch.utils.data.DataLoader(
                            val_dataset, batch_size=1, shuffle=True, num_workers=2))
                        pipeline = DualFlowControlNetPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            controlnet=accelerator.unwrap_model(controlnet),
                            image_encoder=accelerator.unwrap_model(image_encoder),
                            vae=accelerator.unwrap_model(vae),
                            torch_dtype=weight_dtype,
                        ).to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)
                        val_save_dir = os.path.join(
                            args.output_dir, "validation_images",
                            f"step_{global_step:08d}")
                        os.makedirs(val_save_dir, exist_ok=True)

                        with torch.no_grad():
                            for val_idx in range(args.num_validation_images):
                                try:
                                    vbatch = next(val_iter)
                                except StopIteration:
                                    break
                                vpv = vbatch["pixel_values"].to(weight_dtype).to(accelerator.device)
                                vflows  = vbatch["flows"].to(weight_dtype).to(accelerator.device)
                                vdepths = vbatch["depths"].to(weight_dtype).to(accelerator.device)
                                vmasks  = vbatch["masks"].to(weight_dtype).to(accelerator.device)

                                val_seg = seg_encoder(vmasks[:, 0].to(dtype=next(seg_encoder.parameters()).dtype))
                                pil_frames = [Image.fromarray(
                                    (vpv[0, i].permute(1, 2, 0).cpu().float().numpy() * 255
                                     ).astype(np.uint8))
                                    for i in range(vpv.shape[1])]

                                video_frames = pipeline(
                                    pil_frames[0], pil_frames[0], vflows,
                                    vdepths[:, 0], val_seg,
                                    height=args.height, width=args.width,
                                    num_frames=args.num_frames,
                                    decode_chunk_size=8,
                                    motion_bucket_id=127, fps=7,
                                    noise_aug_strength=train_noise_aug,
                                ).frames[0]

                                # Build side-by-side strip: GT | generated
                                gt_np  = (vpv[0].permute(0,2,3,1).cpu().float().numpy()*255).astype(np.uint8)
                                gen_np = np.stack([np.array(f) for f in video_frames])
                                strip  = np.concatenate([gt_np, gen_np], axis=2)
                                vname  = vbatch["video_name"][0].replace("/","_")
                                strip_frames = [Image.fromarray(strip[i]) for i in range(strip.shape[0])]
                                strip_grid   = np.concatenate([np.array(f) for f in strip_frames], axis=0)
                                Image.fromarray(strip_grid).save(
                                    os.path.join(val_save_dir, f"{val_idx:03d}-{vname}_grid.png"))

                        del pipeline
                        torch.cuda.empty_cache()
                        controlnet.train()

            logs = {"step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_controlnet = accelerator.unwrap_model(controlnet)
        pipeline = DualFlowControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            image_encoder=accelerator.unwrap_model(image_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            controlnet=final_controlnet,
        )
        pipeline.save_pretrained(args.output_dir)
        torch.save(seg_encoder.state_dict(),
                   os.path.join(args.output_dir, "seg_encoder.pth"))

    accelerator.end_training()


if __name__ == "__main__":
    main()
