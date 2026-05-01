#!/usr/bin/env python
# Pure-PyTorch bilinear soft-splatting — replaces the CuPy/CUDA-kernel version.
# Identical public API: softsplat(tenIn, tenFlow, tenMetric, strMode)

import torch
import torch.nn.functional as F


def softsplat(tenIn: torch.Tensor, tenFlow: torch.Tensor,
              tenMetric: torch.Tensor, strMode: str) -> torch.Tensor:
    assert strMode.split('-')[0] in ['sum', 'avg', 'linear', 'soft']

    if strMode == 'sum':    assert tenMetric is None
    if strMode == 'avg':    assert tenMetric is None
    if strMode.split('-')[0] == 'linear': assert tenMetric is not None
    if strMode.split('-')[0] == 'soft':   assert tenMetric is not None

    if strMode == 'avg':
        tenIn = torch.cat([tenIn,
                           tenIn.new_ones([tenIn.shape[0], 1,
                                           tenIn.shape[2], tenIn.shape[3]])], 1)
    elif strMode.split('-')[0] == 'linear':
        tenIn = torch.cat([tenIn * tenMetric, tenMetric], 1)
    elif strMode.split('-')[0] == 'soft':
        tenIn = torch.cat([tenIn * tenMetric.exp(), tenMetric.exp()], 1)

    tenOut = softsplat_func.apply(tenIn, tenFlow)

    if strMode.split('-')[0] in ['avg', 'linear', 'soft']:
        tenNormalize = tenOut[:, -1:, :, :]
        if len(strMode.split('-')) == 1:
            tenNormalize = tenNormalize + 1e-7
        elif strMode.split('-')[1] == 'addeps':
            tenNormalize = tenNormalize + 1e-7
        elif strMode.split('-')[1] == 'zeroeps':
            tenNormalize[tenNormalize == 0.0] = 1.0
        elif strMode.split('-')[1] == 'clipeps':
            tenNormalize = tenNormalize.clip(1e-7, None)
        tenOut = tenOut[:, :-1, :, :] / tenNormalize

    return tenOut


class softsplat_func(torch.autograd.Function):
    """Bilinear forward-warp (scatter) implemented in pure PyTorch."""

    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(ctx, tenIn: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
        B, C, H, W = tenIn.shape
        tenOut = tenIn.new_zeros([B, C, H, W])

        # Source pixel coordinates
        gy, gx = torch.meshgrid(
            torch.arange(H, device=tenIn.device, dtype=tenIn.dtype),
            torch.arange(W, device=tenIn.device, dtype=tenIn.dtype),
            indexing='ij',
        )
        gx = gx.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        gy = gy.unsqueeze(0).expand(B, -1, -1)

        # Destination (splatting target) coordinates
        fltX = gx + tenFlow[:, 0]   # [B, H, W]
        fltY = gy + tenFlow[:, 1]

        nwX = torch.floor(fltX).long()
        nwY = torch.floor(fltY).long()

        # 4 bilinear neighbours: (dst_x, dst_y, bilinear_weight)
        neighbors = [
            (nwX,     nwY,     (nwX + 1 - fltX) * (nwY + 1 - fltY)),  # NW
            (nwX + 1, nwY,     (fltX - nwX)     * (nwY + 1 - fltY)),  # NE
            (nwX,     nwY + 1, (nwX + 1 - fltX) * (fltY - nwY)),      # SW
            (nwX + 1, nwY + 1, (fltX - nwX)     * (fltY - nwY)),      # SE
        ]

        out_flat = tenOut.view(B, C, H * W)
        for cx, cy, w in neighbors:
            valid = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)   # [B,H,W]
            w = w * valid.to(tenIn.dtype)                           # zero out-of-bounds
            cx_c = cx.clamp(0, W - 1)
            cy_c = cy.clamp(0, H - 1)
            idx = (cy_c * W + cx_c).view(B, 1, H * W).expand(B, C, H * W)
            weighted = (tenIn * w.unsqueeze(1)).view(B, C, H * W)
            out_flat.scatter_add_(2, idx, weighted)

        ctx.save_for_backward(tenIn, tenFlow)
        return tenOut

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, tenOutgrad: torch.Tensor):
        tenIn, tenFlow = ctx.saved_tensors
        B, C, H, W = tenIn.shape

        tenIngrad = tenFlowgrad = None

        gy, gx = torch.meshgrid(
            torch.arange(H, device=tenIn.device, dtype=tenIn.dtype),
            torch.arange(W, device=tenIn.device, dtype=tenIn.dtype),
            indexing='ij',
        )
        gx = gx.unsqueeze(0).expand(B, -1, -1)
        gy = gy.unsqueeze(0).expand(B, -1, -1)
        fltX = gx + tenFlow[:, 0]
        fltY = gy + tenFlow[:, 1]

        if ctx.needs_input_grad[0]:
            # grad(tenIn) at (y,x) = bilinear sample of tenOutgrad at (fltX, fltY)
            # i.e. backward-warp tenOutgrad by the same flow
            norm_x = 2.0 * fltX / max(W - 1, 1) - 1.0
            norm_y = 2.0 * fltY / max(H - 1, 1) - 1.0
            grid = torch.stack([norm_x, norm_y], dim=-1)  # [B, H, W, 2]
            tenIngrad = F.grid_sample(
                tenOutgrad.contiguous(), grid,
                mode='bilinear', padding_mode='zeros', align_corners=True,
            )

        if ctx.needs_input_grad[1]:
            nwX = torch.floor(fltX).long()
            nwY = torch.floor(fltY).long()
            nwXf = nwX.float()
            nwYf = nwY.float()

            # Derivative of bilinear weights w.r.t. fltX / fltY
            # neighbour: (cx, cy, dw/dfltX, dw/dfltY)
            neighbors_grad = [
                (nwX,     nwY,     -(nwYf + 1 - fltY), -(nwXf + 1 - fltX)),
                (nwX + 1, nwY,     +(nwYf + 1 - fltY), -(fltX - nwXf)),
                (nwX,     nwY + 1, -(fltY - nwYf),     +(nwXf + 1 - fltX)),
                (nwX + 1, nwY + 1, +(fltY - nwYf),     +(fltX - nwXf)),
            ]

            tenFlowgrad = tenFlow.new_zeros(tenFlow.shape)
            og_flat = tenOutgrad.contiguous().view(B, C, H * W)

            for cx, cy, dwdx, dwdy in neighbors_grad:
                valid = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
                cx_c = cx.clamp(0, W - 1)
                cy_c = cy.clamp(0, H - 1)
                idx = (cy_c * W + cx_c).view(B, 1, H * W).expand(B, C, H * W)

                # Gather tenOutgrad at the neighbour destination
                og_nb = og_flat.gather(2, idx).view(B, C, H, W)

                vm = valid.to(tenIn.dtype)
                tenFlowgrad[:, 0] += (tenIn * og_nb *
                                      (dwdx * vm).unsqueeze(1)).sum(dim=1)
                tenFlowgrad[:, 1] += (tenIn * og_nb *
                                      (dwdy * vm).unsqueeze(1)).sum(dim=1)

        return tenIngrad, tenFlowgrad
