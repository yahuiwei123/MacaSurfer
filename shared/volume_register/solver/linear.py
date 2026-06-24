import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math
from utils.filter import GaussianFilter3d
align_corners = False

class AffineWarpField(nn.Module):
    """
    无镜像的仿射场：
        phi(x) = A x + t
        A = R @ diag(exp(s)), det(A) > 0

    支持 2D / 3D:
      - 2D: rot: [B,1]  旋转角; log_scale: [B,2]; trans: [B,2]
      - 3D: rot: [B,3]  Euler 角 (rx,ry,rz); log_scale: [B,3]; trans: [B,3]
    forward 接口:
        moved, grid = affine_model(moving, out_shape)
    """
    def __init__(
        self,
        batch_size: int,
        n_dims: int,
        device: torch.device,
    ):
        super().__init__()
        assert n_dims in (2, 3)
        self.batch_size = batch_size
        self.n_dims = n_dims
        self.device = device

        if n_dims == 2:
            # [B,1] 旋转角
            self.rot = nn.Parameter(torch.zeros(batch_size, 1, device=device))
        else:
            # [B,3] Euler angles (rx,ry,rz)
            self.rot = nn.Parameter(torch.zeros(batch_size, 3, device=device))

        # log_scale -> scale = exp(log_scale) > 0
        self.log_scale = nn.Parameter(torch.zeros(batch_size, n_dims, device=device))
        # 平移
        self.trans = nn.Parameter(torch.zeros(batch_size, n_dims, device=device))

    def _build_rotation(self) -> torch.Tensor:
        """
        2D: [B,2,2]
        3D: [B,3,3] (Z-Y-X Euler)
        """
        if self.n_dims == 2:
            theta = self.rot[:, 0]          # [B]
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            # [[ cos, -sin],
            #  [ sin,  cos]]
            R = torch.stack([
                torch.stack([cos_t, -sin_t], dim=-1),
                torch.stack([sin_t,  cos_t], dim=-1),
            ], dim=-2)                      # [B,2,2]
            return R

        # 3D: Z-Y-X Euler 角
        rx, ry, rz = self.rot[:, 0], self.rot[:, 1], self.rot[:, 2]  # [B]
        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)

        # Rz
        Rz = torch.stack([
            torch.stack([cz, -sz, torch.zeros_like(cz)], dim=-1),
            torch.stack([sz,  cz, torch.zeros_like(cz)], dim=-1),
            torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)], dim=-1),
        ], dim=-2)  # [B,3,3]

        # Ry
        Ry = torch.stack([
            torch.stack([ cy, torch.zeros_like(cy), sy], dim=-1),
            torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], dim=-1),
            torch.stack([-sy, torch.zeros_like(cy), cy], dim=-1),
        ], dim=-2)  # [B,3,3]

        # Rx
        Rx = torch.stack([
            torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], dim=-1),
            torch.stack([torch.zeros_like(cx),  cx, -sx], dim=-1),
            torch.stack([torch.zeros_like(cx),  sx,  cx], dim=-1),
        ], dim=-2)  # [B,3,3]

        # Z-Y-X
        R = Rz @ Ry @ Rx                # [B,3,3]
        return R

    def _build_theta(self) -> torch.Tensor:
        """
        构造 θ: [B, n_dims, n_dims+1]
        """
        R = self._build_rotation()                          # [B,n,n]
        S = torch.diag_embed(torch.exp(self.log_scale))     # [B,n,n], diag>0
        A = torch.bmm(R, S)                                 # [B,n,n], det>0

        t = self.trans.unsqueeze(-1)                        # [B,n,1]
        theta = torch.cat([A, t], dim=-1)                   # [B,n,n+1]
        return theta

    def forward(
        self,
        moving: torch.Tensor,
        out_shape: Tuple[int, ...],
    ):
        """
        moving: 2D -> [B, C, Hm, Wm]
                3D -> [B, C, Dm, Hm, Wm]
        out_shape: 2D -> (Hf, Wf)
                   3D -> (Df, Hf, Wf)

        返回:
            moved: [B, C, *out_shape]
            grid : [B, *out_shape, n_dims]
        """
        theta = self._build_theta()  # [B,n,n+1]

        grid = F.affine_grid(theta, size=out_shape, align_corners=align_corners)
        moved = F.grid_sample(
            moving,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=align_corners,
        )
        return moved, grid

    @property
    def matrix(self) -> torch.Tensor:
        """当前的 θ: [B, n_dims, n_dims+1]"""
        return self._build_theta()


# class AffineWarpField(nn.Module):
#     """
#     Learnable global affine transform.

#     使用 nn.Parameter 形式的 θ:
#         - 2D: θ 形状 [B, 2, 3]
#         - 3D: θ 形状 [B, 3, 4]

#     forward:
#         - 调用 F.affine_grid 生成 sampling_grid
#         - 调用 F.grid_sample 进行重采样
#     """
#     def __init__(
#         self,
#         batch_size: int,
#         n_dims: int,
#         device: torch.device,
#         init_theta: Optional[torch.Tensor] = None,
#     ):
#         super().__init__()
#         assert n_dims in (2, 3), "AffineWarpField only supports 2D or 3D."

#         self.batch_size = batch_size
#         self.n_dims = n_dims
#         self.device = device

#         if init_theta is None:
#             # 生成 identity 仿射矩阵
#             # 2D: [2,3], 3D: [3,4]
#             eye = torch.eye(n_dims + 1, device=device)[:n_dims, :]  # [n_dims, n_dims+1]
#             theta = eye.unsqueeze(0).repeat(batch_size, 1, 1)       # [B, n_dims, n_dims+1]
#         else:
#             theta = init_theta.to(device)
#             assert theta.shape == (batch_size, n_dims, n_dims + 1), \
#                 f"init_theta shape must be [B,{n_dims},{n_dims+1}]"

#         self.theta = nn.Parameter(theta)  # [B, n_dims, n_dims+1]

#     def forward(self, moving: torch.Tensor, fixed_shape: List) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         moving:
#             2D: [B, C, H, W]
#             3D: [B, C, D, H, W]

#         Returns
#         -------
#         moved : warped moving image
#         grid  : sampling grid in normalized coords [-1, 1]
#                 2D: [B, H, W, 2]
#                 3D: [B, D, H, W, 3]
#         """
#         mode = 'bilinear'

#         grid = F.affine_grid(
#             self.theta,          # [B, n_dims, n_dims+1]
#             fixed_shape,        # output size
#             align_corners=align_corners,
#         )  # [B, H, W, 2] or [B, D, H, W, 3]

#         moved = F.grid_sample(
#             moving,
#             grid,
#             mode=mode,
#             padding_mode='border',
#             align_corners=align_corners,
#         )

#         return moved, grid

#     @property
#     def matrix(self) -> torch.Tensor:
#         """返回当前的 affine 参数 θ: [B, n_dims, n_dims+1]."""
#         return self.theta