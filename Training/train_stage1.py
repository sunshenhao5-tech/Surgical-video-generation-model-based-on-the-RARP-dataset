#!/usr/bin/env python
# coding=utf-8
"""
Stage-1 training of DualFlowControlNet on the RARP multimodal surgical dataset.

Key changes vs. original SurgSora code:
  - Dataset: RARPDataset loads pre-computed RGB / optical-flow / depth / mask
    from disk. UniMatch, DAv2, and SAM online inference are fully removed.
  - Seg conditioning: SegCondEncoder (lightweight CNN) replaces SAM image
    encoder, mapping binary masks → [B,256,H/4,W/4] features.
  - Loss: multi-task objective (foreground-weighted diffusion MSE +
    flow TV + depth TV).
  - Memory: BF16 forced, gradient checkpointing forced on controlnet.
  - Logging: Tensorboard/Wandb multimodal video strips (RGB | flow | depth |
    mask | generated).
"""

import argparse
import logging
import math
import os
import shutil
from pathlib import Path

import accelerate
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
from torch.utils.data import RandomSampler

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from einops import rearrange
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

import diffusers
from diffusers import (
    AutoencoderKLTemporalDecoder,
    EulerDiscreteScheduler,
    UNetSpatioTemporalConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available

from train_utils.rarp_dataset import RARPDataset
from train_utils.rarp_losses import SegCondEncoder, compute_multitask_loss

from models.unet_spatio_temporal_condition_controlnet import (
    UNetSpatioTemporalConditionControlNetModel,
)
from pipeline.pipeline import DualFlowControlNetPipeline
from models.svdxt_featureflow_forward_controlnet_s2d_fixcmp_norefine import (
    DualFlowControlNet_traj,
)

check_min_version("0.24.0.dev0")
logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Utility – flow visualisation (HSV colour wheel, no external dep)
# ---------------------------------------------------------------------------

def flow_to_image_np(flow_np: np.ndarray) -> np.ndarray:
    """
    flow_np: HxWx2 float32 (u, v)
    Returns HxWx3 uint8 RGB image.
    """
    import cv2
    u, v = flow_np[..., 0], flow_np[..., 1]
    mag = np.sqrt(u ** 2 + v ** 2)
    ang = np.arctan2(v, u)                          # [-π, π]
    h = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    max_mag = mag.max() if mag.max() > 0 else 1.0
    s = (mag / max_mag * 255).astype(np.uint8)
    val = np.full_like(s, 255)
    hsv = np.stack([h, s, val], axis=-1)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# VAE latent helpers
# ---------------------------------------------------------------------------

def tensor_to_vae_latent(t: torch.Tensor, vae) -> torch.Tensor:
    """t: [B, T, C, H, W] → latents [B, T, C', H', W']"""
    video_length = t.shape[1]
    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    return latents * vae.config.scaling_factor


# ---------------------------------------------------------------------------
# Cosine-interpolated sigma schedule (from SVD paper)
# ---------------------------------------------------------------------------

min_value   = 0.002
max_value   = 700
image_d     = 64
noise_d_low = 32
noise_d_high = 64
sigma_data  = 0.5


def _stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high,
                              sigma_data=1., min_value=1e-3, max_value=1e3,
                              device="cpu", dtype=torch.float32):
    def logsnr_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_interp(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        lo = logsnr_shifted(t, image_d, noise_d_low, logsnr_min, logsnr_max)
        hi = logsnr_shifted(t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(lo, hi, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = _stratified_uniform(shape, group=0, groups=1, dtype=dtype, device=device)
    logsnr = logsnr_interp(u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data


# ---------------------------------------------------------------------------
# Anti-aliased resize (used for CLIP conditioning)
# ---------------------------------------------------------------------------

def _compute_padding(kernel_size):
    computed = [k - 1 for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]
    for i in range(len(kernel_size)):
        pad_front = computed[-(i + 1)] // 2
        pad_rear  = computed[-(i + 1)] - pad_front
        out_padding[2 * i]     = pad_front
        out_padding[2 * i + 1] = pad_rear
    return out_padding


def _filter2d(input, kernel):
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]
    padding_shape = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    return output.view(b, c, h, w)


def _gaussian(window_size, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])
    bs = sigma.shape[0]
    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype)
         - window_size // 2).expand(bs, -1)
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
    out_x = _filter2d(input, kernel_x[..., None, :])
    return _filter2d(out_x, kernel_y[..., None])


def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))
    if ks[0] % 2 == 0:
        ks = ks[0] + 1, ks[1]
    if ks[1] % 2 == 0:
        ks = ks[0], ks[1] + 1
    input = _gaussian_blur2d(input, ks, sigmas)
    return torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)


# ---------------------------------------------------------------------------
# Validation iterator
# ---------------------------------------------------------------------------

def create_iterator(sample_size, sample_dataset):
    while True:
        loader = torch.utils.data.DataLoader(
            dataset=sample_dataset, batch_size=sample_size, drop_last=True)
        for item in loader:
            yield item


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage-1 training: DualFlowControlNet on RARP dataset.")

    # Paths
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
        help="Path to the SVD XT 1.1 model dir (HuggingFace layout).")
    parser.add_argument("--data_root", type=str,
        default="/opt/data/private/RarpSora/data",
        help="Root that contains knotting_videos/, needleGrasping_videos/, needlePuncture_videos/.")
    parser.add_argument("--output_dir", type=str, default="./outputs",
        help="Output directory for checkpoints and validation videos.")
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None,
        help="Resume controlnet from a previous checkpoint dir.")
    parser.add_argument("--pretrain_unet", type=str, default=None,
        help="Optional path to a fine-tuned UNet (defaults to SVD UNet).")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")

    # Dataset
    parser.add_argument("--num_frames", type=int, default=21,
        help="Number of frames T sampled per clip.")
    parser.add_argument("--width",  type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--max_flow", type=float, default=20.0,
        help="Max optical-flow magnitude (pixels) assumed during HSV decode.")
    parser.add_argument("--val_split", type=float, default=0.1)

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_gpu_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--conditioning_dropout_prob", type=float, default=0.1)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)

    # Loss weights
    parser.add_argument("--lambda_fg", type=float, default=2.0,
        help="Foreground (instrument) up-weighting multiplier in diffusion loss.")
    parser.add_argument("--lambda_flow_tv", type=float, default=0.01,
        help="Weight for optical-flow total-variation loss.")
    parser.add_argument("--lambda_depth_tv", type=float, default=0.05,
        help="Weight for depth gradient total-variation loss.")

    # Precision – BF16 is the default for 4-modality training
    parser.add_argument("--mixed_precision", type=str, default="bf16",
        choices=["no", "fp16", "bf16"])

    # Checkpointing / validation
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--validation_steps", type=int, default=1000)
    parser.add_argument("--num_validation_images", type=int, default=2)
    parser.add_argument("--report_to", type=str, default="tensorboard",
        choices=["tensorboard", "wandb", "all"])
    parser.add_argument("--local_rank", type=int, default=-1)

    # xFormers
    parser.add_argument("--enable_xformers_memory_efficient_attention",
        action="store_true")

    # Hub (kept for compatibility, usually unused)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--non_ema_revision", type=str, default=None)
    parser.add_argument("--rank", type=int, default=128)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None", "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. "
                "Please make sure to use `--variant=non_ema` instead."
            ),
        )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    generator = torch.Generator(device=accelerator.device).manual_seed(23123134)

    if args.report_to in ("wandb", "all"):
        if not is_wandb_available():
            raise ImportError("Install wandb: pip install wandb")
        import wandb

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

    # ------------------------------------------------------------------
    # Load models (SVD architecture)
    # ------------------------------------------------------------------
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor",
        revision=args.revision)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder",
        revision=args.revision, variant="fp16")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant="fp16")
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None
        else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = DualFlowControlNet_traj.from_pretrained(
            args.controlnet_model_name_or_path)
    else:
        logger.info("Initialising controlnet weights from UNet")
        controlnet = DualFlowControlNet_traj.from_unet(unet)

    # SegCondEncoder: replaces SAM online inference
    seg_encoder = SegCondEncoder()

    # Freeze everything except controlnet + seg_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)   # will be re-enabled after dtype cast

    # ------------------------------------------------------------------
    # Dtype setup – BF16 by default
    # ------------------------------------------------------------------
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    # controlnet and seg_encoder stay in fp32 for stability; AMP casts inputs

    if args.use_ema:
        ema_controlnet = EMAModel(
            controlnet.parameters(),
            model_cls=UNetSpatioTemporalConditionModel,
            model_config=unet.config,
        )

    # ------------------------------------------------------------------
    # xFormers
    # ------------------------------------------------------------------
    if args.enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 has known training issues on some GPUs. "
                    "Please update to ≥0.0.17.")
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xFormers not available.")

    # ------------------------------------------------------------------
    # Accelerate save/load hooks
    # ------------------------------------------------------------------
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_controlnet.save_pretrained(
                    os.path.join(output_dir, "controlnet_ema"))
            for model in models:
                if isinstance(model, DualFlowControlNet_traj):
                    model.save_pretrained(os.path.join(output_dir, "controlnet"))
                elif isinstance(model, SegCondEncoder):
                    torch.save(model.state_dict(),
                               os.path.join(output_dir, "seg_encoder.pth"))
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"),
                    UNetSpatioTemporalConditionModel)
                ema_controlnet.load_state_dict(load_model.state_dict())
                ema_controlnet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                model = models.pop()
                if isinstance(model, DualFlowControlNet_traj):
                    load_model = DualFlowControlNet_traj.from_pretrained(
                        input_dir, subfolder="controlnet", low_cpu_mem_usage=False)
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                elif isinstance(model, SegCondEncoder):
                    ckpt_path = os.path.join(input_dir, "seg_encoder.pth")
                    if os.path.exists(ckpt_path):
                        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # ------------------------------------------------------------------
    # Gradient checkpointing (always enabled to handle 4-modality VRAM)
    # ------------------------------------------------------------------
    controlnet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.per_gpu_batch_size
            * accelerator.num_processes
        )

    # ------------------------------------------------------------------
    # Optimiser – trains controlnet + seg_encoder jointly
    # ------------------------------------------------------------------
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Install bitsandbytes for 8-bit Adam.")
    else:
        optimizer_cls = torch.optim.AdamW

    controlnet.requires_grad_(True)
    seg_encoder.requires_grad_(True)

    trainable_params = list(controlnet.parameters()) + list(seg_encoder.parameters())
    optimizer = optimizer_cls(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Log parameter names
    if accelerator.is_main_process:
        with open("rec_para_frozen.txt", "w") as f1, \
             open("rec_para_train.txt",  "w") as f2:
            for name, param in list(controlnet.named_parameters()) + \
                               [("seg_encoder." + n, p)
                                for n, p in seg_encoder.named_parameters()]:
                (f2 if param.requires_grad else f1).write(name + "\n")

    # ------------------------------------------------------------------
    # Datasets & Dataloaders
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

    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = create_iterator(1, val_dataset)

    # ------------------------------------------------------------------
    # LR scheduler
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

    # ------------------------------------------------------------------
    # Accelerate prepare
    # ------------------------------------------------------------------
    (unet, seg_encoder, optimizer, lr_scheduler,
     train_dataloader, controlnet) = accelerator.prepare(
        unet, seg_encoder, optimizer, lr_scheduler, train_dataloader, controlnet)

    if args.use_ema:
        ema_controlnet.to(accelerator.device)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("RARPStage1", config=vars(args))

    # ------------------------------------------------------------------
    # Encoding helpers (CLIP image encoder for first-frame conditioning)
    # ------------------------------------------------------------------
    def encode_image(pixel_values):
        """pixel_values: [B, 3, H, W] float [0,1]"""
        pv = pixel_values * 2.0 - 1.0
        pv = _resize_with_antialiasing(pv, (224, 224))
        pv = (pv + 1.0) / 2.0
        pv = feature_extractor(
            images=pv,
            do_normalize=True, do_center_crop=False,
            do_resize=False, do_rescale=False,
            return_tensors="pt",
        ).pixel_values
        pv = pv.to(device=accelerator.device, dtype=weight_dtype)
        embeds = image_encoder(pv).image_embeds
        return embeds.unsqueeze(1)   # [B, 1, D]

    def _get_add_time_ids(fps, motion_bucket_ids, noise_aug_strength,
                          dtype, batch_size, unet):
        motion_ids = torch.tensor(
            [motion_bucket_ids], dtype=dtype).unsqueeze(0).repeat(batch_size, 1)
        base = torch.tensor([fps, noise_aug_strength],
                            dtype=dtype).repeat(batch_size, 1)
        add_time_ids = torch.cat([base, motion_ids.to(base)], dim=1)
        passed = unet.config.addition_time_embed_dim * add_time_ids.size(1)
        expected = unet.add_embedding.linear_1.in_features
        if expected != passed:
            raise ValueError(
                f"UNet expects add_time_ids dim {expected}, got {passed}.")
        return add_time_ids

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir)
                    if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None

        if path is None:
            accelerator.print(
                f"No checkpoint found. Starting fresh training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                accelerator.load_state(
                    os.path.join(args.output_dir, path), strict=False)
            except FileNotFoundError as e:
                accelerator.print(
                    f"[WARNING] Incomplete checkpoint (missing optimizer/scheduler state): {e}\n"
                    f"  Model weights were loaded via load_model_hook; continuing without optimizer state.")
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # ------------------------------------------------------------------
    # Training info
    # ------------------------------------------------------------------
    total_batch_size = (args.per_gpu_batch_size
                        * accelerator.num_processes
                        * args.gradient_accumulation_steps)
    logger.info("***** Running RARP Stage-1 Training *****")
    logger.info(f"  Num clips (train)  = {len(train_dataset)}")
    logger.info(f"  Num clips (val)    = {len(val_dataset)}")
    logger.info(f"  Num Epochs         = {args.num_train_epochs}")
    logger.info(f"  Batch/device       = {args.per_gpu_batch_size}")
    logger.info(f"  Total batch size   = {total_batch_size}")
    logger.info(f"  Grad accum steps   = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optim steps  = {args.max_train_steps}")
    logger.info(f"  Mixed precision    = {args.mixed_precision}")
    logger.info(f"  Loss weights: fg={args.lambda_fg}  "
                f"flow_tv={args.lambda_flow_tv}  depth_tv={args.lambda_depth_tv}")

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    train_noise_aug = 0.02

    # ==================================================================
    # Training loop
    # ==================================================================
    for epoch in range(first_epoch, args.num_train_epochs):
        controlnet.train()
        seg_encoder.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            if (args.resume_from_checkpoint and epoch == first_epoch
                    and step < resume_step):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(controlnet):
                # ------------------------------------------------------
                # Unpack batch
                # ------------------------------------------------------
                pixel_values = batch["pixel_values"].to(
                    weight_dtype, non_blocking=True).to(accelerator.device)
                flows  = batch["flows"].to(
                    weight_dtype, non_blocking=True).to(accelerator.device)
                depths = batch["depths"].to(
                    weight_dtype, non_blocking=True).to(accelerator.device)
                masks  = batch["masks"].to(
                    weight_dtype, non_blocking=True).to(accelerator.device)
                # pixel_values: [B, T, 3, H, W]
                # flows:        [B, T-1, 2, H, W]
                # depths:       [B, T, 1, H, W]
                # masks:        [B, T, 1, H, W]

                B, T = pixel_values.shape[:2]

                # ------------------------------------------------------
                # VAE encoding
                # ------------------------------------------------------
                latents = tensor_to_vae_latent(pixel_values, vae)
                # latents: [B, T, C, Hl, Wl]

                # ------------------------------------------------------
                # Noise / sigma schedule
                # ------------------------------------------------------
                noise = torch.randn_like(latents)
                sigmas = rand_cosine_interpolated(
                    shape=[B],
                    image_d=image_d, noise_d_low=noise_d_low,
                    noise_d_high=noise_d_high, sigma_data=sigma_data,
                    min_value=min_value, max_value=max_value,
                    device=latents.device, dtype=latents.dtype,
                )
                sigmas_reshaped = sigmas.clone()
                while len(sigmas_reshaped.shape) < len(latents.shape):
                    sigmas_reshaped = sigmas_reshaped.unsqueeze(-1)

                # First-frame conditional latent
                small_noise_latents = latents + noise * train_noise_aug
                conditional_latents = (
                    small_noise_latents[:, 0] / vae.config.scaling_factor)

                noisy_latents = latents + noise * sigmas_reshaped
                timesteps = torch.tensor(
                    [0.25 * sigma.log() for sigma in sigmas],
                    device=latents.device)

                inp_noisy_latents = noisy_latents / (
                    (sigmas_reshaped ** 2 + 1) ** 0.5)

                # ------------------------------------------------------
                # CLIP image conditioning
                # ------------------------------------------------------
                encoder_hidden_states = encode_image(
                    pixel_values[:, 0].float())

                added_time_ids = _get_add_time_ids(
                    6, 127, train_noise_aug,
                    encoder_hidden_states.dtype, B, unet)
                added_time_ids = added_time_ids.to(latents.device)

                # Conditioning dropout
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(B, device=latents.device,
                                         generator=generator)
                    prompt_mask = (random_p < 2 * args.conditioning_dropout_prob
                                   ).reshape(B, 1, 1)
                    null_cond = torch.zeros_like(encoder_hidden_states)
                    encoder_hidden_states = torch.where(
                        prompt_mask, null_cond, encoder_hidden_states)

                    img_mask_dtype = conditional_latents.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(img_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(img_mask_dtype)
                    )
                    image_mask = image_mask.reshape(B, 1, 1, 1)
                    conditional_latents = image_mask * conditional_latents

                conditional_latents = conditional_latents.unsqueeze(1).repeat(
                    1, noisy_latents.shape[1], 1, 1, 1)
                inp_noisy_latents = torch.cat(
                    [inp_noisy_latents, conditional_latents], dim=2)

                # ------------------------------------------------------
                # Multimodal conditioning inputs for ControlNet
                # ------------------------------------------------------
                # controlnet_cond : first RGB frame [B, 3, H, W]
                controlnet_image = pixel_values[:, 0]

                # controlnet_depth : first-frame depth [B, 1, H, W]
                controlnet_depth = depths[:, 0]          # [B, 1, H, W]

                # controlnet_mask : SegCondEncoder → [B, 256, H/4, W/4]
                # Use first frame mask; ControlNet's FuseFlowConditioningEmbeddingSVD
                # expects seg features of shape [B, 256, *, *].
                first_mask = masks[:, 0].float()         # [B, 1, H, W]
                controlnet_mask = seg_encoder(first_mask)  # [B, 256, H/4, W/4]

                # controlnet_flow : [B, T-1, 2, H, W]
                controlnet_flow = flows

                # ------------------------------------------------------
                # ControlNet forward
                # ------------------------------------------------------
                down_block_res_samples, mid_block_res_sample, _, _ = controlnet(
                    inp_noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_time_ids=added_time_ids,
                    controlnet_cond=controlnet_image,
                    controlnet_flow=controlnet_flow,
                    controlnet_mask=controlnet_mask,
                    controlnet_depth=controlnet_depth,
                    return_dict=False,
                )

                # UNet prediction
                model_pred = unet(
                    inp_noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_time_ids=added_time_ids,
                    down_block_additional_residuals=[
                        s.to(dtype=weight_dtype)
                        for s in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(
                        dtype=weight_dtype),
                ).sample

                # Denoising parameterisation (SVD v-prediction)
                c_out = -sigmas_reshaped / ((sigmas_reshaped ** 2 + 1) ** 0.5)
                c_skip = 1 / (sigmas_reshaped ** 2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas_reshaped ** 2) * (sigmas_reshaped ** -2.0)

                # ------------------------------------------------------
                # Multi-task loss
                # ------------------------------------------------------
                loss_dict = compute_multitask_loss(
                    denoised_latents=denoised_latents,
                    target_latents=latents,
                    weighing=weighing,
                    masks_pixel=masks.float(),
                    flows=flows.float(),
                    depths=depths.float(),
                    lambda_fg=args.lambda_fg,
                    lambda_flow_tv=args.lambda_flow_tv,
                    lambda_depth_tv=args.lambda_depth_tv,
                )
                loss = loss_dict["total"]

                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ----------------------------------------------------------
            # After optimiser step
            # ----------------------------------------------------------
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_controlnet.step(controlnet.parameters())
                progress_bar.update(1)
                global_step += 1

                accelerator.log(
                    {
                        "train/loss_total":    train_loss,
                        "train/loss_diffusion": loss_dict["diffusion"].item(),
                        "train/loss_flow_tv":   loss_dict["flow_tv"].item(),
                        "train/loss_depth_tv":  loss_dict["depth_tv"].item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )
                train_loss = 0.0

                if accelerator.is_main_process:
                    # --------------------------------------------------
                    # Checkpointing
                    # --------------------------------------------------
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = sorted(
                                [d for d in os.listdir(args.output_dir)
                                 if d.startswith("checkpoint")],
                                key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                n_remove = (len(checkpoints)
                                            - args.checkpoints_total_limit + 1)
                                for ckpt in checkpoints[:n_remove]:
                                    shutil.rmtree(
                                        os.path.join(args.output_dir, ckpt))
                                    logger.info(f"Removed checkpoint: {ckpt}")

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        # Also save seg_encoder separately
                        torch.save(
                            accelerator.unwrap_model(seg_encoder).state_dict(),
                            os.path.join(save_path, "seg_encoder.pth"))
                        logger.info(f"Saved state to {save_path}")

                    # --------------------------------------------------
                    # Validation + multimodal logging
                    # --------------------------------------------------
                    if (global_step % args.validation_steps == 0
                            or global_step == 1):
                        logger.info(
                            f"Running validation... "
                            f"({args.num_validation_images} samples)")
                        if args.use_ema:
                            ema_controlnet.store(controlnet.parameters())
                            ema_controlnet.copy_to(controlnet.parameters())

                        pipeline = DualFlowControlNetPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            controlnet=accelerator.unwrap_model(controlnet),
                            image_encoder=accelerator.unwrap_model(image_encoder),
                            vae=accelerator.unwrap_model(vae),
                            revision=args.revision,
                            torch_dtype=weight_dtype,
                        )
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)

                        val_save_dir = os.path.join(
                            args.output_dir, "validation_images")
                        os.makedirs(val_save_dir, exist_ok=True)

                        _unwrapped_seg = accelerator.unwrap_model(seg_encoder)
                        _unwrapped_seg.eval()

                        with torch.autocast(
                            str(accelerator.device).replace(":0", ""),
                            enabled=accelerator.mixed_precision != "no",
                            dtype=weight_dtype,
                        ):
                            for val_idx in range(args.num_validation_images):
                                val_batch = next(val_loader)
                                vpv    = val_batch["pixel_values"].to(
                                    weight_dtype).to(accelerator.device)
                                vflows = val_batch["flows"].to(
                                    weight_dtype).to(accelerator.device)
                                vdepths = val_batch["depths"].to(
                                    weight_dtype).to(accelerator.device)
                                vmasks  = val_batch["masks"].to(
                                    weight_dtype).to(accelerator.device)

                                # Seg features for pipeline
                                val_seg_feats = _unwrapped_seg(
                                    vmasks[:, 0].float())

                                pil_frames = [
                                    Image.fromarray(
                                        (vpv[0, i].permute(1, 2, 0)
                                         .cpu().float().numpy() * 255
                                         ).astype(np.uint8))
                                    for i in range(vpv.shape[1])
                                ]

                                with torch.no_grad():
                                    video_frames = pipeline(
                                        pil_frames[0],
                                        pil_frames[0],
                                        vflows,
                                        vdepths[:, 0],
                                        val_seg_feats,
                                        height=args.height,
                                        width=args.width,
                                        num_frames=args.num_frames,
                                        decode_chunk_size=8,
                                        motion_bucket_id=127,
                                        fps=7,
                                        noise_aug_strength=train_noise_aug,
                                    ).frames[0]

                                # Build multimodal video strip:
                                # GT | flow_viz | depth | mask | generated
                                gt_np = (vpv[0].permute(0, 2, 3, 1)
                                         .cpu().float().numpy() * 255
                                         ).astype(np.uint8)        # [T,H,W,3]

                                # Flow visualisation (T-1 frames; pad first)
                                flow_vis = []
                                for fi in range(vflows.shape[1]):
                                    fl_np = (vflows[0, fi].permute(1, 2, 0)
                                             .cpu().float().numpy())
                                    flow_vis.append(flow_to_image_np(fl_np))
                                blank = np.zeros_like(flow_vis[0])
                                flow_vis_np = np.stack(
                                    [blank] + flow_vis)                # [T,H,W,3]

                                # Depth (pseudo-colour: replicate channel)
                                depth_np = (vdepths[0].permute(0, 2, 3, 1)
                                            .cpu().float().numpy()
                                            * 255).astype(np.uint8)  # [T,H,W,1]
                                depth_np = np.repeat(depth_np, 3, axis=-1)  # [T,H,W,3]

                                # Mask
                                mask_np = (vmasks[0].permute(0, 2, 3, 1)
                                           .cpu().float().numpy()
                                           * 255).astype(np.uint8)   # [T,H,W,1]
                                mask_np = np.repeat(mask_np, 3, axis=-1)    # [T,H,W,3]

                                # Generated frames
                                gen_np = np.stack(
                                    [np.array(f) for f in video_frames])  # [T,H,W,3]

                                # Horizontal concat per frame
                                strip = np.concatenate(
                                    [gt_np, flow_vis_np, depth_np,
                                     mask_np, gen_np], axis=2)        # [T,H,5W,3]

                                video_name = val_batch["video_name"][0].replace("/", "_")
                                out_path = os.path.join(
                                    val_save_dir,
                                    f"step_{global_step:06d}",
                                    f"{val_idx:03d}-{video_name}.mp4")
                                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                                torchvision.io.write_video(
                                    out_path, strip, fps=8,
                                    video_codec="h264",
                                    options={"crf": "18"})
                                logger.info(f"Saved validation video: {out_path}")

                                # Save first-frame modality grid as PNG
                                log_imgs = {
                                    "gt":        gt_np[0],
                                    "flow":      flow_vis_np[1] if flow_vis_np.shape[0] > 1 else flow_vis_np[0],
                                    "depth":     depth_np[0],
                                    "mask":      mask_np[0],
                                    "generated": gen_np[0],
                                }
                                img_grid = np.concatenate(
                                    list(log_imgs.values()), axis=1)  # [H, 5W, 3]
                                grid_path = os.path.join(
                                    val_save_dir,
                                    f"step_{global_step:06d}",
                                    f"{val_idx:03d}-{video_name}_grid.png")
                                Image.fromarray(img_grid).save(grid_path)

                                # Wandb image logging (if available)
                                if args.report_to in ("wandb", "all"):
                                    try:
                                        import wandb
                                        accelerator.log(
                                            {f"val/{k}": wandb.Image(v)
                                             for k, v in log_imgs.items()},
                                            step=global_step)
                                    except Exception:
                                        pass

                        if args.use_ema:
                            ema_controlnet.restore(controlnet.parameters())

                        _unwrapped_seg.train()
                        del pipeline
                        torch.cuda.empty_cache()

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        seg_enc    = accelerator.unwrap_model(seg_encoder)

        if args.use_ema:
            ema_controlnet.copy_to(controlnet.parameters())

        pipeline = DualFlowControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            image_encoder=accelerator.unwrap_model(image_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            controlnet=controlnet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)
        torch.save(seg_enc.state_dict(),
                   os.path.join(args.output_dir, "seg_encoder.pth"))
        logger.info(f"Final model saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
