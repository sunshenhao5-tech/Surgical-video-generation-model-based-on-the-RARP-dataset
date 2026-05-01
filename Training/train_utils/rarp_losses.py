"""
rarp_losses.py – auxiliary modules and loss functions for RARP Stage-1 training.

Contents:
  SegCondEncoder  – lightweight CNN that maps binary mask [B,1,H,W] →
                    [B,256,H/4,W/4], matching the SAM feature tensor shape
                    that DualFlowControlNet_traj.controlnet_cond_embedding expects.
                    This replaces online SAM inference entirely.

  compute_multitask_loss – multi-task loss on top of the diffusion denoising MSE:
    1. Foreground-weighted diffusion loss  (instrument region up-weighted)
    2. Optical-flow total-variation loss   (temporal smoothness prior)
    3. Depth gradient total-variation loss (geometric smoothness prior)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SegCondEncoder
# ---------------------------------------------------------------------------

class SegCondEncoder(nn.Module):
    """
    Maps a binary segmentation mask of arbitrary spatial size to a feature
    tensor of shape [B, 256, H/4, W/4].

    The SAM ViT-H image encoder maps a 256×256 RGB image to [1, 256, 64, 64].
    Here we replicate that spatial compression ratio (factor 4) using a small
    convolutional encoder so that DualFlowControlNet_traj receives a tensor of
    the same shape without running SAM online.

    Input  : [B, 1, H, W]   float32  {0, 1}  binary mask
    Output : [B, 256, H/4, W/4]  float32
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # stride-2 downsample: H → H/2
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            # stride-2 downsample: H/2 → H/4
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            # refine without changing spatial size
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        mask: [B, 1, H, W]  float32
        Returns [B, 256, H/4, W/4]
        """
        return self.encoder(mask)


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------

def _tv_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Isotropic total-variation loss over the last two spatial dimensions.
    x: [..., H, W]
    """
    diff_h = (x[..., 1:, :] - x[..., :-1, :]).abs()
    diff_w = (x[..., :, 1:] - x[..., :, :-1]).abs()
    return diff_h.mean() + diff_w.mean()


def compute_multitask_loss(
    denoised_latents: torch.Tensor,   # [B, T, C, Hl, Wl]  latent-space prediction
    target_latents:   torch.Tensor,   # [B, T, C, Hl, Wl]  latent-space target
    weighing:         torch.Tensor,   # [B, 1, 1, 1, 1]    per-sample SNR weight
    masks_pixel:      torch.Tensor,   # [B, T, 1, H, W]    {0,1} binary mask (pixel space)
    flows:            torch.Tensor,   # [B, T-1, 2, H, W]  (u,v) flow
    depths:           torch.Tensor,   # [B, T, 1, H, W]    depth [0,1]
    lambda_fg:        float = 2.0,    # foreground up-weight multiplier
    lambda_flow_tv:   float = 0.01,
    lambda_depth_tv:  float = 0.05,
) -> dict:
    """
    Computes a multi-task training objective.

    Returns a dict with individual loss scalars and the combined `total` loss.

    Loss 1 – Foreground-weighted diffusion MSE
    ------------------------------------------
    Down-sample the binary mask to latent spatial resolution via average pool.
    At latent resolution (H/8, W/8) the mask value is in [0,1].
    Weight = 1 + (lambda_fg - 1) * mask_latent  → [1, lambda_fg].
    This keeps background pixels contributing to the loss while boosting the
    instrument region by lambda_fg.

    Loss 2 – Flow total-variation (temporal smoothness)
    ---------------------------------------------------
    Penalises spatial gradients of the optical-flow field, encouraging the
    predicted/given flow to be spatially smooth across the whole sequence.

    Loss 3 – Depth gradient total-variation (geometric smoothness)
    -------------------------------------------------------------
    Same TV penalty on the depth map sequence, discouraging sharp synthetic
    edges introduced by depth estimation artefacts.
    """
    B, T, C, Hl, Wl = denoised_latents.shape

    # ------------------------------------------------------------------
    # Loss 1: foreground-weighted diffusion MSE
    # ------------------------------------------------------------------
    # masks_pixel: [B, T, 1, H, W] → need [B, T, 1, Hl, Wl]
    masks_flat = masks_pixel.reshape(B * T, 1, masks_pixel.shape[-2], masks_pixel.shape[-1])
    mask_latent_flat = F.adaptive_avg_pool2d(masks_flat, (Hl, Wl))
    mask_latent = mask_latent_flat.reshape(B, T, 1, Hl, Wl)

    # Spatial weight: background=1, foreground=lambda_fg
    spatial_weight = 1.0 + (lambda_fg - 1.0) * mask_latent.float()  # [B,T,1,Hl,Wl]

    per_pixel_sq = (denoised_latents.float() - target_latents.float()) ** 2  # [B,T,C,Hl,Wl]
    weighted_sq = weighing.float() * spatial_weight * per_pixel_sq

    loss_diffusion = weighted_sq.reshape(B, -1).mean(dim=1).mean()

    # ------------------------------------------------------------------
    # Loss 2: flow total-variation
    # ------------------------------------------------------------------
    # flows: [B, T-1, 2, H, W]
    loss_flow_tv = _tv_loss(flows.float()) if lambda_flow_tv > 0 else flows.new_tensor(0.0)

    # ------------------------------------------------------------------
    # Loss 3: depth gradient total-variation
    # ------------------------------------------------------------------
    # depths: [B, T, 1, H, W]
    loss_depth_tv = _tv_loss(depths.float()) if lambda_depth_tv > 0 else depths.new_tensor(0.0)

    total = (
        loss_diffusion
        + lambda_flow_tv  * loss_flow_tv
        + lambda_depth_tv * loss_depth_tv
    )

    return {
        "total":        total,
        "diffusion":    loss_diffusion,
        "flow_tv":      loss_flow_tv,
        "depth_tv":     loss_depth_tv,
    }
