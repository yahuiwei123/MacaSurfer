import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

align_corners = False

# ==========================
#   displacements -> warps
# ==========================
def displacements_to_warps(displacements: List[torch.Tensor]) -> List[torch.Tensor]:
    warps = []
    for disp in displacements:
        # disp: [B, H, W, D, 3] or [B, H, W, 2]
        shape = disp.shape[1:-1]
        dims = len(shape)
        grid = F.affine_grid(
            torch.eye(dims, dims + 1, device=disp.device).unsqueeze(0),
            [1, 1] + list(shape),
            align_corners=align_corners,
        )
        warps.append(grid + disp)
    return warps

def v2img_2d(velocity: torch.Tensor):
    ''' convert [B, H, W, chan] to [B, chan, H, W] '''
    return velocity.permute(0, 3, 1, 2)

def img2v_2d(image: torch.Tensor):
    ''' convert [B, chan, H, W] to [B, H, W, chan] '''
    return image.permute(0, 2, 3, 1)

def v2img_3d(velocity: torch.Tensor):
    ''' convert [B, D, H, W, chan] to [B, chan, D, H, W] '''
    return velocity.permute(0, 4, 1, 2, 3)

def img2v_3d(image: torch.Tensor):
    ''' convert [B, chan, D, H, W] to [B, D, H, W, chan] '''
    return image.permute(0, 2, 3, 4, 1)


import torch
import torch.nn.functional as F

align_corners = True


# ---------------------------
# 1. Build identity grid
# ---------------------------
def make_identity_grid(shape, device):
    """
    shape = [D,H,W]
    return [1,D,H,W,3] normalized grid
    """
    D, H, W = shape
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, D, device=device),
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    grid = torch.stack((xx, yy, zz), dim=-1)  # [D,H,W,3]
    return grid.unsqueeze(0)  # [1,D,H,W,3]


# ---------------------------
# 2. Euler integration (your code)
# ---------------------------
def integrate_velocity(v, base_grid, n_steps=5):
    """
    v: [1,Df,Hf,Wf,3]
    base_grid: identity grid at desired shape
    """
    dt = 1.0 / n_steps
    v_img = v.permute(0, 4, 1, 2, 3)  # [1,3,Df,Hf,Wf]

    phi = base_grid.clone()

    for _ in range(n_steps):
        v_phi = F.grid_sample(
            v_img,
            phi,
            mode="bilinear",
            padding_mode="border",
            align_corners=align_corners,
        ).permute(0, 2, 3, 4, 1)

        phi = phi + dt * v_phi

    disp = phi - base_grid
    return disp, phi


# ---------------------------
# 3. Fit velocity in fixed space
# ---------------------------
def fit_velocity(disp_fixed, n_steps=5, iters=300, lr=1e-3):
    """
    disp_fixed: [1,Df,Hf,Wf,3]
    Fit v such that exp(v) = disp.
    """
    device = disp_fixed.device
    B, Df, Hf, Wf, _ = disp_fixed.shape

    base_fixed = make_identity_grid((Df, Hf, Wf), device)

    # trainable velocity
    v = torch.zeros_like(disp_fixed, requires_grad=True)

    optimizer = torch.optim.Adam([v], lr=lr)
    target_warp = base_fixed + disp_fixed

    for i in range(iters):
        optimizer.zero_grad()

        pred_disp, pred_warp = integrate_velocity(v, base_fixed, n_steps)

        loss = ((pred_warp - target_warp) ** 2).mean()
        loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:
            print(f"[Fit Velocity] Iter {i+1}/{iters}, Loss={loss.item():.6f}")

    return v.detach()


# ---------------------------
# 4. Compute inverse displacement in moving space
# ---------------------------
def compute_inverse_displacement(disp_fixed, moving_shape, n_steps=5, iters=300, lr=1e-3):
    """
    disp_fixed: [1,Df,Hf,Wf,3], forward displacement (fixed → moving)
    moving_shape: (Dm, Hm, Wm)
    """

    device = disp_fixed.device

    # Step 1 — fit velocity in fixed space
    v = fit_velocity(disp_fixed, n_steps=n_steps, iters=iters, lr=lr)

    # Step 2 — build moving identity grid
    base_moving = make_identity_grid(moving_shape, device)

    # Step 3 — integrate negative velocity field
    inv_disp, inv_warp = integrate_velocity(-v, base_moving, n_steps)

    return inv_disp, inv_warp, v
