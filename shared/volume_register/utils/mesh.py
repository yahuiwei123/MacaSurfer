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
    verts = verts.to(dtype=torch.float32)
    affine = affine.to(dtype=torch.float32, device=verts.device)

    # Convert to homogeneous coordinates: [x, y, z] -> [x, y, z, 1]
    num_verts = verts.shape[0]
    ones = torch.ones(B, N, 1, dtype=verts.dtype, device=verts.device)
    verts_h = torch.cat([verts, ones], dim=2)  # [B, N, 4]

    # Apply affine: [B, 4, 4] @ [B, 4, N] -> [B, 4, N] -> [B, N, 4]
    verts_h = verts_h.transpose(1, 2)
    transformed = affine @ verts_h
    transformed = transformed.transpose(1, 2)

    # Drop homogeneous coordinate and return [N, 3]
    moved_verts = transformed[:, :3]

    return moved_verts


def deform_mesh(
    verts: torch.Tensor,
    deform_field: torch.Tensor,
    mov_shape: List[int]
) -> torch.Tensor:
    """
    Apply a 3D deformation field to mesh vertices.

    Args:
        verts (torch.Tensor): Vertex coordinates in voxel space, shape [N, 3],
            ordered as (x, y, z) in image index coordinates.
        deform_field (torch.Tensor): Deformation / sampling grid on moving image,
            shape [B, 3, D, H, W]. The 3 channels correspond to (z, y, x)
            in normalized coordinates for grid_sample.
        mov_shape (List[int]): Spatial shape of moving image [D, H, W].

    Returns:
        torch.Tensor: Deformed vertices in voxel space, shape [N, 3].
    """
    if deform_field.ndim != 5:
        raise ValueError(
            f"deform_field should be 5D [B, 3, D, H, W], "
            f"but got shape {deform_field.shape}"
        )

    B, C, D, H, W = deform_field.shape
    if C != 3:
        raise ValueError(
            f"Channel dimension of deform_field must be 3, but got {C}."
        )

    # verts: [N, 3] in (x, y, z) voxel coordinates
    sample_verts = verts.to(device=deform_field.device, dtype=torch.float32)  # [N, 3]

    # Convert voxel coordinates to normalized coordinates in (z, y, x) order
    # for grid_sample; note the xyz -> zyx reordering here.
    normalized_verts = torch.zeros_like(sample_verts)  # [N, 3]
    normalized_verts[:, 2] = 2.0 / D * (sample_verts[:, 0] + 0.5) - 1.0  # z
    normalized_verts[:, 1] = 2.0 / H * (sample_verts[:, 1] + 0.5) - 1.0  # y
    normalized_verts[:, 0] = 2.0 / W * (sample_verts[:, 2] + 0.5) - 1.0  # x

    # grid_sample expects a grid of shape [N, D, H, W, 3]
    # here we pack points into a fake volume dimension: [1, 1, 1, N, 3]
    grid = normalized_verts.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, N, 3]

    # mov_shape is [D, H, W]; build scale for each axis in (z, y, x)
    recover_coord = (
        torch.tensor([mov_shape[2], mov_shape[1], mov_shape[0]],
                     device=deform_field.device, dtype=torch.float32) / 2.0
    )

    # Sample deformation field at the vertex locations
    # Output: [1, 3, 1, 1, N]
    sampled = F.grid_sample(
        deform_field,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    )

    # Rearrange to [N, 3] in (z, y, x) normalized space
    # 1,3,1,1,N -> 1,1,1,N,3 -> N,3
    moved_verts = sampled.permute(0, 2, 3, 4, 1).squeeze(0).squeeze(0).squeeze(0)

    # Convert from normalized coordinates back to voxel coordinates
    moved_verts = moved_verts * recover_coord + recover_coord  # still (z, y, x)

    # Swap (z, y, x) back to (x, y, z) by flipping last dimension
    moved_verts = torch.flip(moved_verts, dims=[1])

    return moved_verts


class MeshDeformLoss(nn.Module):
    """
    Compute the deformed mesh and a distance loss between the deformed mesh
    and a target mesh.

    Two loss modes are supported:
        - 'mse': per-vertex MSE, requires one-to-one correspondence and equal
                 number of vertices.
        - 'chamfer': use PyTorch3D chamfer_distance, does NOT require
                     correspondence or equal vertex counts.
    """

    def __init__(
        self,
        device: str = "cuda",
        cal_loss: bool = True,
        loss_mode: str = "mse",
    ):
        """
        Args:
            device:
                Default device string, e.g. 'cuda' or 'cpu'. This is mainly
                for consistency; tensors are actually moved to deform_field.device
                in forward().
            cal_loss:
                If False, only returns the deformed mesh and None as loss.
            loss_mode:
                'mse' or 'chamfer'.
                - 'mse': use F.mse_loss between vertex sets (requires same shape).
                - 'chamfer': use pytorch3d.loss.chamfer_distance between point sets.
        """
        super(MeshDeformLoss, self).__init__()
        self.device = device
        self.cal_loss = cal_loss
        if loss_mode not in ("mse", "chamfer"):
            raise ValueError(f"Unsupported loss_mode: {loss_mode}")
        self.loss_mode = loss_mode

    def forward(
        self,
        fix_mesh: torch.Tensor,
        mov_mesh: torch.Tensor,
        deform_field: torch.Tensor,
        mov_shape: List[int]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            fix_mesh (torch.Tensor):
                The mesh to be deformed, [N_f, 3].
            mov_mesh (torch.Tensor):
                The target mesh, [N_m, 3] (for chamfer, N_f and N_m can differ).
            deform_field (torch.Tensor):
                Deformation field / sampling grid applied to fix_mesh,
                shape [B, 3, D, H, W] (typically B=1).
            mov_shape (List[int]):
                The spatial shape of moving volume [D, H, W].

        Returns:
            fix_moved_mesh (torch.Tensor):
                Deformed fix_mesh, [N_f, 3].
            loss (torch.Tensor | None):
                - If loss_mode == 'mse': MSE distance between vertices (requires
                  N_f == N_m).
                - If loss_mode == 'chamfer': Chamfer distance between point sets.
                - None if `cal_loss=False`.
        """
        if deform_field.ndim != 5:
            raise ValueError(
                f"deform_field should be 5D [B, 3, D, H, W], "
                f"but got shape {deform_field.shape}"
            )

        if deform_field.shape[1] != 3:
            raise ValueError(
                f"Dimension of warp field {deform_field.shape[1]} "
                f"does not match 3D coordinates."
            )

        # Move meshes to the same device as deform_field
        device = deform_field.device
        fix_verts = fix_mesh.to(device=device, dtype=torch.float32)

        if self.cal_loss:
            mov_verts = mov_mesh.to(device=device, dtype=torch.float32)

        # Deform the source mesh vertices
        fix_moved_verts = deform_mesh(
            verts=fix_verts,
            deform_field=deform_field,
            mov_shape=mov_shape
        )

        if not self.cal_loss:
            return fix_moved_verts, None

        # ------------------ Loss computation ------------------ #
        if self.loss_mode == "mse":
            # Require one-to-one correspondence and same number of vertices
            if fix_moved_verts.shape != mov_verts.shape:
                raise ValueError(
                    f"Number of vertices in fix_mesh {fix_moved_verts.shape} "
                    f"does not match mov_mesh {mov_verts.shape} for MSE loss. "
                    f"Use loss_mode='chamfer' if the meshes are not aligned."
                )
            norm_shape = torch.tensor(mov_shape).to(fix_moved_verts.device)
            distance_loss = F.mse_loss(fix_moved_verts / norm_shape, mov_verts / norm_shape)

        elif self.loss_mode == "chamfer":
            # Chamfer distance expects batched point clouds of shape [B, N, 3]
            # fix_moved_verts: [N_f, 3], mov_verts: [N_m, 3]
            src = fix_moved_verts.unsqueeze(0)  # [1, N_f, 3]
            tgt = mov_verts.unsqueeze(0)        # [1, N_m, 3]
            distance_loss, _ = chamfer_distance(src, tgt)

        else:
            raise RuntimeError(f"Unexpected loss_mode: {self.loss_mode}")

        return fix_moved_verts, distance_loss
