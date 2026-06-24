# Copyright (c) 2025 Rohit Jena. All rights reserved.
#
# NOTE: This version is modified to work directly with torch.Tensor images
# instead of FireANTs BatchedImages. All BatchedImages-related wrappers
# are removed; only core optimization algorithm is preserved.

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import List, Optional, Union, Callable
from tqdm import tqdm

from fireants.utils.globals import MIN_IMG_SIZE
from fireants.registration.deformation.svf import StationaryVelocity
from fireants.registration.deformation.compositive import CompositiveWarp
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.utils.imageutils import downsample
from fireants.interpolator import fireants_interpolator
from fireants.utils.mesh import deform_mesh, affine_mesh
from fireants.utils.weights import MGDASolver

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------
# 简单的收敛监控器（替代 AbstractRegistration 内的）
# ------------------------
class SimpleConvergenceMonitor:
    def __init__(self, tolerance: float = 1e-6, max_iters: int = 20, max_increase_iters: int = 20):
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.max_increase_iters = max_increase_iters
        self.buffer: List[float] = []
        self.increase_count = 0

    def reset(self):
        self.buffer = []
        self.increase_count = 0

    def converged(self, loss_value: float) -> bool:
        self.buffer.append(loss_value)
        
        if len(self.buffer) < self.max_iters:
            return False

        recent = self.buffer[-self.max_iters:]
        for i in range(1, len(recent)):
            if recent[i] > recent[i-1]:
                self.increase_count += 1
            else:
                self.increase_count = 0
        if self.increase_count >= self.max_increase_iters:
            return True
        diffs = [abs(recent[i] - recent[i-1]) for i in range(1, len(recent))]
        return max(diffs) < self.tolerance



class GreedyRegistration(nn.Module):
    """
    Greedy deformable registration class for non-linear image alignment,
    modified to directly use torch.Tensor images instead of BatchedImages.

    Args:
        scales: multi-resolution downsample factors, e.g. [4,2,1]
        iterations: num iterations per scale, same length as scales
        fixed_images: torch.Tensor, shape [B, C, D, H, W]
        moving_images: torch.Tensor, shape [B, C, D, H, W]
        loss_fn: similarity loss between moved and fixed, e.g. NCC, MI, MSE
        deformation_type: 'compositive' or 'geodesic'
        optimizer: 'Adam' or 'SGD' (used internally by warp object)
        optimizer_lr: learning rate for warp.optimizer
        warp_reg: regularization on warped coordinates (optional)
        displacement_reg: regularization on displacement field (optional)
        blur: whether to blur at coarse scales
        freeform: pass-through to CompositiveWarp
        progress_bar: whether to show tqdm
        tolerance / max_tolerance_iters: for early stopping
        fixed_surfs / moving_surfs / surf_loss_func: optional surface loss
    """

    def __init__(
        self,
        scales: List[int],
        iterations: List[int],
        fixed_images: torch.Tensor,
        moving_images: torch.Tensor,
        img_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        fixed_surfs: Optional[torch.Tensor] = None,
        moving_surfs: Optional[torch.Tensor] = None,
        fixed_roi: Optional[torch.Tensor] = None,
        moving_roi: Optional[torch.Tensor] = None,
        surf_loss_func: Optional[Callable] = None,
        surf_weight: float = 1.0,
        deformation_type: str = "compositive",
        optimizer: str = "Adam",
        optimizer_params: dict = {},
        optimizer_lr: float = 0.5,
        use_mgda: bool = True,
        integrator_n: Union[str, int] = 7,
        smooth_warp_sigma: float = 0.00,
        smooth_grad_sigma: float = 0.00,
        reduction: str = "mean",
        tolerance: float = 1e-16,
        max_tolerance_iters: int = 20,
        init_affine: Optional[torch.Tensor] = None,
        warp_reg: Optional[Union[Callable, nn.Module]] = None,
        displacement_reg: Optional[Union[Callable, nn.Module]] = None,
        blur: bool = True,
        freeform: bool = False,
        progress_bar: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        # 基本属性
        self.scales = scales
        self.iterations = iterations
        self.fixed_images = fixed_images      # [B, C, D, H, W]
        self.moving_images = moving_images
        self.img_loss_fn = img_loss_fn
        self.dims = fixed_images.ndim - 2     # 3 for [B,C,D,H,W]
        self.blur = blur
        self.reduction = reduction
        self.device = fixed_images.device
        self.dtype = fixed_images.dtype
        self.progress_bar = progress_bar
        self.use_mgda = use_mgda
        self.mgda_solver = MGDASolver(['volume', 'surface', 'displacement', 'consistency'])

        # convergence monitor
        self.convergence_monitor = SimpleConvergenceMonitor(
            tolerance=tolerance,
            max_iters=max_tolerance_iters,
        )

        # regularization
        self.fwd_warp_reg = warp_reg
        self.displacement_reg = displacement_reg
        self.deformation_type = deformation_type

        # surface 相关
        self.fixed_surfs = fixed_surfs       # [Nv, 3] in voxel / normalized coords
        self.moving_surfs = moving_surfs
        
        self.fixed_roi = fixed_roi
        self.moving_roi = moving_roi
        
        self.surf_loss_func = surf_loss_func
        self.surf_weight = surf_weight

        # 选择 warp 类型
        smooth_warp_sigma = 0
        self.fwd_warp = CompositiveWarp(
            fixed_images, moving_images,
            optimizer=optimizer,
            optimizer_lr=optimizer_lr,
            optimizer_params=optimizer_params,
            dtype=self.dtype,
            smoothing_grad_sigma=smooth_grad_sigma,
            smoothing_warp_sigma=smooth_warp_sigma,
            init_scale=scales[0],
            freeform=freeform,
        )

        self.rev_warp = CompositiveWarp(
            moving_images, fixed_images,
            optimizer=optimizer,
            optimizer_lr=optimizer_lr,
            optimizer_params=optimizer_params,
            dtype=self.dtype,
            smoothing_grad_sigma=smooth_grad_sigma,
            smoothing_warp_sigma=smooth_warp_sigma,
            init_scale=scales[0],
            freeform=freeform,
        )

        self.smooth_warp_sigma = smooth_warp_sigma  # in voxels

        # 初始化 affine
        B = fixed_images.shape[0]
        if init_affine is None:
            # [B, D+1, D+1]
            init_affine = torch.eye(self.dims + 1, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(B, 1, 1)
        assert init_affine.shape[0] == B, "init_affine batch size mismatch"
        D1, D2 = init_affine.shape[1], init_affine.shape[2]
        if D1 == self.dims + 1 and D2 == self.dims + 1:
            self.affine = init_affine.detach().to(self.dtype)
        elif D1 == self.dims and D2 == self.dims + 1:
            row = torch.zeros(B, 1, self.dims + 1, device=self.device, dtype=self.dtype)
            row[:, 0, -1] = 1.0
            self.affine = torch.cat([init_affine.detach(), row], dim=1).to(self.dtype)
        else:
            raise ValueError(f"Invalid initial affine shape: {init_affine.shape}")
        self.affine = self.affine.contiguous()
        self.inv_affine = torch.linalg.inv(self.affine)

    # ------------------------------
    # 获取 warp（非必须修改部分，但改成 tensor 版本）
    # ------------------------------
    def get_warp_parameters(self, shape=None, displacement: bool = False):
        """
        Get transformed coordinates for warping the moving image.

        Returns:
            dict with:
                'affine': [B, D, D+1] linear+translation map in normalized space
                'grid':   [B, *shape[2:], D] displacement or full grid
        """
        fixed_arrays = self.fixed_images
        if shape is None:
            shape = list(fixed_arrays.shape)
        else:
            shape = [fixed_arrays.shape[0], fixed_arrays.shape[1]] + list(shape)

        affine_map_init = (self.affine[:, :-1]).contiguous()  # [B,D,D+1]
        warp_field = self.fwd_warp.get_warp()                     # [B, H, W, D, dims] in normalized displacement

        mode = "bilinear" if self.dims == 2 else "trilinear"
        if tuple(warp_field.shape[1:-1]) != tuple(shape[2:]):
            warp_field = F.interpolate(
                warp_field.permute(*self.fwd_warp.permute_vtoimg),
                size=shape[2:],
                mode=mode,
                align_corners=False,
            ).permute(*self.fwd_warp.permute_imgtov)

        if self.smooth_warp_sigma > 0:
            warp_gaussian = [
                gaussian_1d(s, truncated=2)
                for s in (torch.zeros(self.dims, device=self.device, dtype=self.dtype) + self.smooth_warp_sigma)
            ]
            warp_field = separable_filtering(
                warp_field.permute(*self.fwd_warp.permute_vtoimg),
                warp_gaussian,
            ).permute(*self.fwd_warp.permute_imgtov)

        if displacement:
            grid = warp_field
        else:
            # 生成 identity grid 并加上位移
            B = warp_field.shape[0]
            eye = torch.eye(self.dims + 1, device=self.device, dtype=warp_field.dtype)[: self.dims, :]
            eye = eye.unsqueeze(0).repeat(B, 1, 1)
            identity_grid = F.affine_grid(eye, shape, align_corners=False)
            grid = identity_grid + warp_field

        return {"affine": affine_map_init, "grid": grid}

    # ------------------------------
    # 主优化流程（算法几乎未改，只把 BatchedImages 全换成 tensor）
    # ------------------------------
    def optimize(self):
        fixed_arrays = self.fixed_images         # [B,C,D,H,W]
        moving_arrays = self.moving_images
        fixed_size = fixed_arrays.shape[2:]
        moving_size = moving_arrays.shape[2:]

        affine_map_init = (self.affine[:, :-1]).contiguous().to(self.dtype)
        inv_affine_map_init = (self.inv_affine[:, :-1]).contiguous().to(self.dtype)

        # 用于平滑位移场的高斯核
        warp_gaussian = [
            gaussian_1d(s, truncated=2)
            for s in (torch.zeros(self.dims, device=self.device, dtype=self.dtype) + self.smooth_warp_sigma)
        ]

        for scale, iters in zip(self.scales, self.iterations):
            self.convergence_monitor.reset()

            fixed_size_down = [max(int(s / scale), MIN_IMG_SIZE) for s in fixed_size]
            moving_size_down = [max(int(s / scale), MIN_IMG_SIZE) for s in moving_size]

            # 下采样 fixed / moving（这里直接用 F.interpolate 或 downsample）
            if self.blur and scale > 1:
                sigmas = 0.5 * torch.tensor(
                    [sz / szdown for sz, szdown in zip(fixed_size, fixed_size_down)],
                    device=self.device,
                    dtype=fixed_arrays.dtype,
                )
                gaussians = [gaussian_1d(s, truncated=2) for s in sigmas]
                fixed_image_down = downsample(
                    fixed_arrays,
                    size=fixed_size_down,
                    mode="trilinear",
                    gaussians=gaussians,
                )
                moving_image_blur = downsample(
                    moving_arrays,
                    size=moving_size_down,
                    mode="trilinear",
                    gaussians=gaussians,
                )
            else:
                if scale > 1:
                    fixed_image_down = F.interpolate(
                        fixed_arrays,
                        size=fixed_size_down,
                        mode="trilinear",
                        align_corners=False,
                    )
                    moving_image_blur = F.interpolate(
                        moving_arrays,
                        size=moving_size_down,
                        mode="trilinear",
                        align_corners=False,
                    )
                else:
                    fixed_image_down = fixed_arrays
                    moving_image_blur = moving_arrays

            if (
                self.surf_loss_func is not None
                and self.fixed_surfs is not None
                and self.moving_surfs is not None
                ):
                fixed_surf = self.fixed_surfs * torch.tensor([p / q for p, q in zip(fixed_size_down, fixed_size)]).to(self.device)
                moving_surf = self.moving_surfs * torch.tensor([p / q for p, q in zip(moving_size_down, moving_size)]).to(self.device)

            # 设置 warp 的当前尺度 size
            self.fwd_warp.set_size(fixed_size_down)
            self.rev_warp.set_size(moving_size_down)

            if self.reduction == "mean":
                scale_factor = 1.0
            else:
                scale_factor = float(np.prod(fixed_image_down.shape))

            pbar = tqdm(range(iters)) if self.progress_bar else range(iters)

            for i in pbar:
                self.fwd_warp.set_zero_grad()
                self.rev_warp.set_zero_grad()

                fwd_disp = self.fwd_warp.get_warp()  # [N, HWD, 3]
                rev_disp = self.rev_warp.get_warp()


                # smooth if required
                if self.smooth_warp_sigma > 0:
                    fwd_warp_field = separable_filtering(fwd_warp_field.permute(*self.fwd_warp.permute_vtoimg), warp_gaussian).permute(*self.fwd_warp.permute_imgtov)
                    rev_warp_field = separable_filtering(rev_warp_field.permute(*self.rev_warp.permute_vtoimg), warp_gaussian).permute(*self.rev_warp.permute_imgtov)
                
                # disp(affine(p))
                fwd_warp_grid = F.affine_grid(affine_map_init, fixed_image_down.size(), align_corners=False) + fwd_disp
                
                # inv_affine(inv_disp(p))
                rev_warp_grid = F.affine_grid(torch.eye(4)[None, :-1, :].to(self.device), moving_image_blur.size(), align_corners=False) + rev_disp
                rev_warp_grid = torch.bmm(
                    self.inv_affine,
                    torch.cat([rev_warp_grid, torch.ones(*rev_warp_grid.shape[:-1], 1, device=self.device, dtype=self.dtype)], dim=-1).view(rev_warp_grid.shape[0], -1, 4).transpose(1, 2)
                ).transpose(1, 2)[..., :-1].view(*rev_warp_grid.shape[:-1], 3)

                # symmetic loss
                fwd_consist = F.grid_sample(
                    fwd_warp_grid.permute(*self.fwd_warp.permute_vtoimg),
                    rev_warp_grid.detach(),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                ).permute(*self.fwd_warp.permute_imgtov) - F.affine_grid(torch.eye(4)[None, :-1, :].to(self.device), moving_image_blur.size(), align_corners=False)
                fwd_consist_loss = (fwd_consist ** 2).mean()

                rev_consist = F.grid_sample(
                    rev_warp_grid.permute(*self.rev_warp.permute_vtoimg),
                    fwd_warp_grid.detach(),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                ).permute(*self.rev_warp.permute_imgtov) - F.affine_grid(torch.eye(4)[None, :-1, :].to(self.device), fixed_image_down.size(), align_corners=False)
                rev_consist_loss = (rev_consist ** 2).mean()
                consist_loss = 1 * (fwd_consist_loss + rev_consist_loss)
                
                # image loss
                moving_image_warped = F.grid_sample(
                    moving_image_blur,
                    fwd_warp_grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
                fixed_image_warped = F.grid_sample(
                    fixed_image_down,
                    rev_warp_grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
                fwd_img_loss = self.img_loss_fn(moving_image_warped, fixed_image_down)
                rev_img_loss = self.img_loss_fn(fixed_image_warped, moving_image_blur)
                img_loss = 1 * (fwd_img_loss + rev_img_loss)

                # from fireants.neuio.nifti import save_vol_nii
                # save_vol_nii(moving_image_warped.detach().cpu().numpy()[0, 0, ...], np.eye(4), '/home/weiyahui/projects/Monkey_Surface/test/test_tissue_guide_reg/vis/moving_image_warped.nii.gz')
                # save_vol_nii(fixed_image_warped.detach().cpu().numpy()[0, 0, ...], np.eye(4), '/home/weiyahui/projects/Monkey_Surface/test/test_tissue_guide_reg/vis/fixed_image_warped.nii.gz')
                # save_vol_nii(fixed_image_down.detach().cpu().numpy()[0, 0, ...], np.eye(4), '/home/weiyahui/projects/Monkey_Surface/test/test_tissue_guide_reg/vis/fixed_image_down.nii.gz')
                # save_vol_nii(moving_image_blur.detach().cpu().numpy()[0, 0, ...], np.eye(4), '/home/weiyahui/projects/Monkey_Surface/test/test_tissue_guide_reg/vis/moving_image_blur.nii.gz')
                # return
                

                # surface loss（如果提供）
                if (
                    self.surf_loss_func is not None
                    and self.fixed_surfs is not None
                    and self.moving_surfs is not None
                ):
                    _, fwd_surf_loss = self.surf_loss_func(
                        fixed_surf,
                        moving_surf,
                        self.fixed_roi,
                        self.moving_roi,
                        fwd_warp_grid.permute(0, 4, 1, 2, 3),
                        moving_size_down,
                    )

                    _, rev_surf_loss = self.surf_loss_func(
                        moving_surf,
                        fixed_surf,
                        self.fixed_roi,
                        self.moving_roi,
                        rev_warp_grid.permute(0, 4, 1, 2, 3),
                        fixed_size_down,
                    )
                    surf_loss = 1 * (fwd_surf_loss + rev_surf_loss)

                # 位移场正则
                if self.displacement_reg is not None:
                    fwd_reg_loss = self.displacement_reg(fwd_disp.permute(0, 4, 1, 2, 3).contiguous())
                    rev_reg_loss = self.displacement_reg(rev_disp.permute(0, 4, 1, 2, 3).contiguous())
                    reg_loss = 1 * (fwd_reg_loss + rev_reg_loss)

                # 如果你有 warp_reg（作用在坐标上），可以在此调用
                if self.fwd_warp_reg is not None:
                    # 这里简单调用 get_warp_parameters, 然后把 grid 传给 warp_reg
                    coords = self.get_warp_parameters(shape=fixed_size_down, displacement=False)
                    warp_loss = self.fwd_warp_reg(coords["grid"])

                total_loss = 5e-5 * img_loss + 1e0 * surf_loss + 5e-2 * reg_loss + 1e0 * consist_loss

                # Weighted loss
                # fwd_losses = [fwd_img_loss, fwd_surf_loss, fwd_reg_loss, fwd_consist_loss]
                # rev_losses = [rev_img_loss, rev_surf_loss, rev_reg_loss, rev_consist_loss]
                # fwd_task_grads = []
                # rev_task_grads = []
                # for i, (fwd_loss, rev_loss) in enumerate(zip(fwd_losses, rev_losses)):
                #     fwd_grad = torch.autograd.grad(fwd_loss, self.fwd_warp.warp, retain_graph=True)[0].detach()
                #     fwd_task_grads.append(fwd_grad)
                #     rev_grad = torch.autograd.grad(rev_loss, self.rev_warp.warp, retain_graph=True)[0].detach()
                #     rev_task_grads.append(rev_grad)
                
                # if self.use_mgda:
                #     fwd_weights = self.mgda_solver.solve_mgda_weights([grad.flatten() for grad in fwd_task_grads])
                #     rev_weights = self.mgda_solver.solve_mgda_weights([grad.flatten() for grad in rev_task_grads])
                # else:
                #     fwd_weights = rev_weights = [1.0, 1.0, 1.0, 1.0]

                # total_fwd_loss = 0.0
                # total_rev_loss = 0.0
                # # total_fwd_grad = 0.0
                # # total_rev_grad = 0.0

                # for i, (fwd_weight, rev_weight, fwd_grad, rev_grad, fwd_loss, rev_loss) in enumerate(zip(fwd_weights, rev_weights, fwd_task_grads, rev_task_grads, fwd_losses, rev_losses)):
                #     if i == 0:
                #         fwd_weight *= 2.0
                #         rev_weight *= 2.0
                #     if i == 1:
                #         fwd_weight *= 1.0
                #         rev_weight *= 1.0
                #     if i == 2:
                #         fwd_weight *= 1e-3
                #         rev_weight *= 1e-3
                #     if i == 3:
                #         fwd_weight *= 1.0
                #         rev_weight *= 1.0

                    # total_fwd_grad += fwd_weight * fwd_grad
                    # total_rev_grad += rev_weight * rev_grad
                    # total_fwd_loss += fwd_weight * fwd_loss
                    # total_rev_loss += rev_weight * rev_loss

                # self.fwd_warp.warp.grad = total_fwd_grad.contiguous()
                # self.rev_warp.warp.grad = total_rev_grad.contiguous()
                # total_fwd_loss.backward(retain_graph=True)
                # total_rev_loss.backward()

                total_loss.backward()

                if self.progress_bar:
                    pbar.set_postfix(
                        total=f"{total_loss.item():.3e}",
                        vol=f"{fwd_img_loss.item():.3e}",
                        surf=f"{rev_surf_loss.item():.3e}",
                        reg=f"{fwd_reg_loss.item():.3e}",
                        consist=f"{consist_loss.item():.3e}",
                    )

                self.fwd_warp.step()
                self.rev_warp.step()

                # if self.convergence_monitor.converged(total_loss.item()):
                #     break
