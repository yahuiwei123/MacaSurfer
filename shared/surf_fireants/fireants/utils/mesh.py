import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List, Union, Optional
import nibabel as nib
import numpy as np

from pytorch3d.loss import chamfer_distance


def affine_mesh(
    verts: torch.Tensor,
    affine: torch.Tensor,
    fix_shape: List[int],
    mov_shape: List[int],
) -> torch.Tensor:
    """
    Apply a 3D affine transform to mesh vertices.

    Args:
        verts (torch.Tensor):
            Vertex coordinates, shape [B, N, 3],
            ordered as (x, y, z). The coordinate system (voxel / world) must
            be consistent with the affine matrix.
        affine (torch.Tensor):
            4x4 affine matrix, shape [B, 4, 4], that maps input coordinates to
            output coordinates in the same space as `verts`.

    Returns:
        torch.Tensor:
            Transformed vertices, shape [B, N, 3], in the same coordinate system
            as defined by the affine.
    """
    if verts.ndim != 3 or verts.shape[2] != 3:
        raise ValueError(
            f"`verts` must be of shape [B, N, 3], but got {verts.shape}."
        )

    if affine.ndim != 3 or affine.shape[1:] != (4, 4):
        raise ValueError(
            f"`affine` must be of shape [B, 4, 4], but got {affine.shape}."
        )

    # Ensure float tensors and same device
    B, N, _ = verts.shape
    D, H, W = fix_shape
    verts = verts.to(dtype=torch.float32)
    affine = affine.to(dtype=torch.float32, device=verts.device)

    # transform verts to normalize space (-1, 1)
    x = verts[..., 2]
    y = verts[..., 1]
    z = verts[..., 0]
    nx = 2.0 / W * (x + 0.5) - 1.0
    ny = 2.0 / H * (y + 0.5) - 1.0
    nz = 2.0 / D * (z + 0.5) - 1.0

    normalize_verts = torch.stack([nx, ny, nz], dim=-1)

    # Convert to homogeneous coordinates: [x, y, z] -> [x, y, z, 1]
    ones = torch.ones(B, N, 1, dtype=verts.dtype, device=verts.device)
    verts_h = torch.cat([normalize_verts, ones], dim=2)  # [B, N, 4]

    # Apply affine: [B, 4, 4] @ [B, 4, N] -> [B, 4, N] -> [B, N, 4]
    verts_h = verts_h.transpose(1, 2)
    transformed = affine @ verts_h
    transformed = transformed.transpose(1, 2)

    # Drop homogeneous coordinate and return [B, N, 3]
    moved_verts = transformed[..., :3]

    recover_coord = torch.tensor(
        [mov_shape[2], mov_shape[1], mov_shape[0]],
        device=moved_verts.device,
        dtype=torch.float32,
    ) / 2.0

    moved_verts = moved_verts * recover_coord + recover_coord
    moved_verts = moved_verts.flip(dims=[2])

    return moved_verts


def deform_mesh(
    verts: torch.Tensor,
    deform_field: torch.Tensor,
    mov_shape: List[int],
) -> torch.Tensor:
    if deform_field.ndim != 5:
        raise ValueError(f"deform_field should be [B,3,D,H,W], got {deform_field.shape}")

    B, C, D, H, W = deform_field.shape
    if C != 3:
        raise ValueError(f"Channel of deform_field must be 3, got {C}")

    if verts.ndim != 3 or verts.shape[2] != 3:
        raise ValueError(f"verts must be [B,N,3], got {verts.shape}")

    verts = verts.to(device=deform_field.device, dtype=torch.float32)   # [B,N,3]

    N = verts.shape[1]

    x = verts[..., 2]
    y = verts[..., 1]
    z = verts[..., 0]

    nx = 2.0 / W * (x + 0.5) - 1.0
    ny = 2.0 / H * (y + 0.5) - 1.0
    nz = 2.0 / D * (z + 0.5) - 1.0

    grid = torch.stack([nx, ny, nz], dim=-1)          # [B,N,3]
    grid = grid.view(B, 1, 1, N, 3)

    sampled = F.grid_sample(
        deform_field,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )   # [B,3,1,1,N]

    sampled = sampled.permute(0, 2, 3, 4, 1).squeeze(0).squeeze(0)

    recover_coord = torch.tensor(
        [mov_shape[2], mov_shape[1], mov_shape[0]],
        device=deform_field.device,
        dtype=torch.float32,
    ) / 2.0

    moved = sampled * recover_coord + recover_coord
    moved = moved.flip(dims=[2])

    return moved


class MeshDeformLoss(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        cal_loss: bool = True,
        loss_mode: str = "mse",
    ):
        super(MeshDeformLoss, self).__init__()
        self.device = device
        self.cal_loss = cal_loss
        if loss_mode not in ("l1", "mse", "chamfer"):
            raise ValueError(f"Unsupported loss_mode: {loss_mode}")
        self.loss_mode = loss_mode

    def forward(
        self,
        fix_mesh: torch.Tensor,       # [B, Nf, 3]
        mov_mesh: torch.Tensor,       # [B, Nm, 3]
        fix_roi: torch.Tensor,        # [B, Nf]
        mov_roi: torch.Tensor,        # [B, Nm]
        deform_field: torch.Tensor,   # [B, 3, D, H, W]
        mov_shape: List[int],
        norm: bool = True
    ):
        if deform_field.ndim != 5:
            raise ValueError(f"deform_field should be [B,3,D,H,W], got {deform_field.shape}")

        B = deform_field.shape[0]

        if fix_mesh.ndim != 3 or fix_mesh.shape[2] != 3:
            raise ValueError(f"fix_mesh must be [B,Nf,3], got {fix_mesh.shape}")
        if mov_mesh.ndim != 3 or mov_mesh.shape[2] != 3:
            raise ValueError(f"mov_mesh must be [B,Nm,3], got {mov_mesh.shape}")

        device = deform_field.device

        fix_mesh = fix_mesh.to(device=device, dtype=torch.float32)
        mov_mesh = mov_mesh.to(device=device, dtype=torch.float32)

        # deform source mesh: [B,Nf,3]
        fix_moved_verts = deform_mesh(
            verts=fix_mesh,
            deform_field=deform_field,
            mov_shape=mov_shape,
        )   # [B,Nf,3]

        if not self.cal_loss:
            return fix_moved_verts, None

        # ---------------- Loss computation ---------------- #
        if norm:
            norm = torch.tensor(mov_shape, device=device, dtype=torch.float32)
        else:
            norm = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)
        norm = norm.view(1, 1, 3)
        
        weights = torch.ones((*fix_moved_verts.shape[:-1], 1), device=device, dtype=torch.float32)
        if fix_roi is not None and mov_roi is not None:
            roi = torch.logical_and(fix_roi > 0, mov_roi > 0)
            weights *= 2e-1
            weights[roi] = 1
            

        if self.loss_mode == "mse":
            if fix_moved_verts.shape != mov_mesh.shape:
                raise ValueError(
                    f"MSE requires same shape: fix_moved {fix_moved_verts.shape} != mov_mesh {mov_mesh.shape}"
                )
            loss = F.mse_loss(weights * fix_moved_verts / norm, weights * mov_mesh / norm)
        
        elif self.loss_mode == "l1":
            if fix_moved_verts.shape != mov_mesh.shape:
                raise ValueError(
                    f"L1 requires same shape: fix_moved {fix_moved_verts.shape} != mov_mesh {mov_mesh.shape}"
                )
            loss = F.l1_loss(weights * fix_moved_verts / norm, weights * mov_mesh / norm)

        elif self.loss_mode == "chamfer":
            # chamfer_distance expects:
            #   x: [B, Nf, 3], y: [B, Nm, 3]
            from pytorch3d.loss import chamfer_distance
            if fix_roi is not None and mov_roi is not None:
                loss, _ = chamfer_distance(fix_moved_verts[fix_roi > 0] / norm, mov_mesh[mov_roi > 0] / norm)
            else:
                loss, _ = chamfer_distance(fix_moved_verts / norm, mov_mesh / norm)

        else:
            raise RuntimeError(f"Unexpected loss_mode: {self.loss_mode}")

        return fix_moved_verts, loss


def vox2phy(
    verts: torch.Tensor,
    affine: torch.Tensor
) -> torch.Tensor:
    if verts.ndim != 3 or verts.shape[2] != 3:
        raise ValueError(
            f"`verts` must be of shape [B, N, 3], but got {verts.shape}."
        )

    if affine.ndim != 3 or affine.shape[1:] != (4, 4):
        raise ValueError(
            f"`affine` must be of shape [B, 4, 4], but got {affine.shape}."
        )

    # Ensure float tensors and same device
    B, N, _ = verts.shape
    verts = verts.to(dtype=torch.float32)
    affine = affine.to(dtype=torch.float32, device=verts.device)

    # Convert to homogeneous coordinates: [x, y, z] -> [x, y, z, 1]
    ones = torch.ones(B, N, 1, dtype=verts.dtype, device=verts.device)
    verts_h = torch.cat([verts, ones], dim=2)  # [B, N, 4]

    # Apply affine: [B, 4, 4] @ [B, 4, N] -> [B, 4, N] -> [B, N, 4]
    verts_h = verts_h.transpose(1, 2)
    transformed = affine @ verts_h
    transformed = transformed.transpose(1, 2)

    # Drop homogeneous coordinate and return [B, N, 3]
    moved_verts = transformed[..., :3]

    return moved_verts