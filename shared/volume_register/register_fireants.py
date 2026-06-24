import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import gaussian_1d, v2img_2d, v2img_3d, img2v_2d, img2v_3d, separable_filtering, save_surf_gii, save_nii_gz
from losses import LocalNormalizedCrossCorrelationLoss
from monai.losses import DiceLoss
from monai.losses.hausdorff_loss import HausdorffDTLoss
import numpy as np
import nibabel as nib
import argparse
from typing import List, Union, Callable, Optional
from tqdm import tqdm
from mesh_loss import MeshDeformLoss
from pytorch3d.structures import Meshes
from LibMTL.weighting.MGDA import MGDA

align_corners = False

def displacements_to_warps(displacements):
    ''' given a list of displacements, add warps to them '''
    warps = []
    for disp in displacements:
        # disp is of shape [batch, H, W, D, 3] or [batch, H, W, 2]
        shape = disp.shape[1:-1]
        dims = len(shape)
        grid = F.affine_grid(torch.eye(dims, dims+1, device=disp.device).unsqueeze(0), [1, 1] + list(shape), align_corners=align_corners)
        warps.append(grid + disp)
    return warps

def multi_scale_diffeomorphic_solver(
        fixed_images: List[torch.Tensor],
        moving_images: List[torch.Tensor],
        fixed_labels: List[torch.Tensor], 
        moving_labels: List[torch.Tensor], 
        fixed_surfs: List[torch.Tensor], 
        moving_surfs: List[torch.Tensor], 
        iterations: List[int],
        image_loss_func: Union[nn.Module, Callable],
        label_loss_func: Union[nn.Module, Callable],
        surf_loss_func: Union[nn.Module, Callable],
        disp_loss_func: Optional[Callable] = None,
        gaussian_warp: Optional[Union[torch.Tensor]] = None,
        gaussian_grad: Optional[Union[torch.Tensor]] = None,
        learning_rate: List[float] = [1e1],
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
        convergence_tol: int = 4,
        convergence_eps: float = 1e-3,
        use_mgda: bool = True,  # 新增：是否使用MGDA
        mgda_gn: str = 'l2'     # 新增：MGDA归一化方法
):
    '''
    Implements multi-scale diffeomorphic riemannian adam for feature images and labels.
    '''
    if fixed_labels is None:
        fixed_labels = len(fixed_images) * [None]
        moving_labels = len(moving_images) * [None]
        
    if fixed_surfs is None:
        fixed_surfs = len(fixed_images) * [None]
        moving_surfs = len(moving_images) * [None]

    # 初始化MGDA
    if use_mgda:
        mgda = MGDA()
        mgda.device = fixed_images[0].device
        mgda.rep_grad = False

    # collect statistics
    batch_size, shape = fixed_images[0].shape[0], fixed_images[0].shape[2:]
    n_dims = len(shape)
    # initialize flow
    warp = torch.zeros((batch_size, *shape, n_dims), dtype=torch.float32, device=fixed_images[0].device)
    exp_avg = torch.zeros_like(warp)
    exp_sq_avg = torch.zeros_like(warp)
    all_warps = []
    global_step = 1

    # set functions for v2img and img2v
    v2img = v2img_2d if n_dims == 2 else v2img_3d
    img2v = img2v_2d if n_dims == 2 else img2v_3d

    losses = []

    # iterate over scales
    for level, (iter_scale, (fixed_image, moving_image, fixed_label, moving_label, fixed_surf, moving_surf)) in enumerate(
        zip(iterations, zip(fixed_images, moving_images, fixed_labels, moving_labels, fixed_surfs, moving_surfs))):

        half_res = 1.0 / (max(fixed_image.shape[2:]) - 1)
        grid = F.affine_grid(
            torch.eye(n_dims, n_dims + 1, device=fixed_image.device).unsqueeze(0).expand(batch_size, -1, -1),
            fixed_image.shape, align_corners=align_corners
        )

        warp.requires_grad_(True)
        exp_avg = exp_avg.detach()
        exp_sq_avg = exp_sq_avg.detach()
        last_loss = float('inf')
        iters_since_divergent = 0

        # tqdm 进度条
        from tqdm import tqdm
        pbar = tqdm(range(1, iter_scale + 1), total=iter_scale, desc=f"Level {level}")

        for step in pbar:
            # —— 前向：计算各个损失 ——
            moved_image = F.grid_sample(
                moving_image.detach(), grid + warp, mode='bilinear',
                padding_mode='zeros', align_corners=align_corners
            )
            img_loss = image_loss_func(moved_image, fixed_image.detach())

            if moving_label is not None:
                moved_label = F.grid_sample(
                    moving_label.float(), grid + warp, mode='bilinear',
                    padding_mode='zeros', align_corners=align_corners
                )
                lbl_loss = label_loss_func(moved_label, fixed_label)
            else:
                lbl_loss = torch.tensor(0)
            
            if moving_surf is not None:
                fixed_deformed_surf, surf_loss = surf_loss_func(fixed_surf, moving_surf, (grid + warp).permute(0, 4, 1, 2, 3), moving_image.shape[2:])
            else:
                surf_loss = torch.tensor(0)

            reg_loss = 0.0
            if disp_loss_func is not None:
                reg_loss = disp_loss_func(warp)

            # —— 使用MGDA自动平衡损失 ——
            if use_mgda:
                # 收集所有损失
                all_losses = [img_loss, lbl_loss, surf_loss]
                if disp_loss_func is not None:
                    all_losses.append(reg_loss)
                
                # 过滤掉值为0的损失（对应没有启用的任务）
                valid_losses = []
                valid_indices = []
                for i, loss in enumerate(all_losses):
                    if isinstance(loss, torch.Tensor) and loss.item() != 0:
                        valid_losses.append(loss)
                        valid_indices.append(i)
                
                if len(valid_losses) > 1:
                    # 使用MGDA计算最优权重
                    weights = mgda.backward(valid_losses, mgda_gn=mgda_gn)
                    
                    # 应用MGDA权重
                    weighted_total_loss = 0.0
                    for i, (loss_idx, weight) in enumerate(zip(valid_indices, weights)):
                        if loss_idx == 0:  # img_loss
                            weighted_total_loss += weight * img_loss
                        elif loss_idx == 1:  # lbl_loss
                            weighted_total_loss += weight * lbl_loss
                        elif loss_idx == 2:  # surf_loss
                            weighted_total_loss += weight * surf_loss
                        elif loss_idx == 3:  # reg_loss
                            weighted_total_loss += weight * reg_loss
                    
                    total_loss = weighted_total_loss
                    
                    # 记录权重信息用于显示
                    mgda_weights_info = f"MGDA_w: [{', '.join([f'{w:.3f}' for w in weights])}]"
                else:
                    # 只有一个有效损失，直接使用
                    total_loss = valid_losses[0] if valid_losses else torch.tensor(0.0)
                    mgda_weights_info = "Single_w"
            else:
                # 不使用MGDA，使用原来的手动权重
                if level < 2:
                    total_loss = 1.0 * img_loss + 0.30 * lbl_loss + 5e-2 * surf_loss + 0.10 * reg_loss
                else:
                    total_loss = 1.0 * img_loss + 0.30 * lbl_loss + 2e-2 * surf_loss + 0.10 * reg_loss
                mgda_weights_info = "Manual_w"

            # 反向得到对 warp 的梯度
            warp_grad = torch.autograd.grad(total_loss, warp, retain_graph=False, create_graph=False)[0].detach()

            # 收敛/发散检测
            lossitem = float(total_loss.item())
            rel_loss = lossitem / (1e-8 + last_loss) - 1.0
            if rel_loss < -convergence_eps[level] if isinstance(convergence_eps, (list, tuple)) else -convergence_eps:
                iters_since_divergent = 0
            else:
                iters_since_divergent += 1
                if iters_since_divergent >= convergence_tol:
                    pbar.write(f"Early stop at step {step} (rel_loss={rel_loss:.3e})")
                    break
            last_loss = lossitem

            # 可选：梯度平滑
            if gaussian_grad is not None:
                warp_grad = img2v(separable_filtering(v2img(warp_grad), gaussian_grad))

            # 归一化+步长
            gradmax = eps + warp_grad.norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
            gradmax = gradmax.reshape(-1, *([1]) * (n_dims + 1)).clamp(min=1)
            warp_grad = warp_grad / gradmax * half_res
            warp_grad.mul_(-learning_rate[level])

            # 黏滞/对偶更新
            warp_update = warp_grad + img2v(
                F.grid_sample(v2img(warp), grid + warp_grad, mode='bilinear', align_corners=align_corners)
            )
            if gaussian_warp is not None:
                warp_update = img2v(separable_filtering(v2img(warp_update), gaussian_warp))

            # 原地更新
            warp.data.copy_(warp_update)

            # tqdm 显示 - 添加MGDA权重信息
            pbar.set_postfix(
                total=f"{lossitem:.3e}",
                img=f"{float(img_loss.item()):.3e}",
                lbl=f"{float(lbl_loss.item()):.3e}",
                surf=f"{float(surf_loss.item()):.3e}",
                reg=f"{float(reg_loss if isinstance(reg_loss, float) else reg_loss.item()):.3e}",
                weights=mgda_weights_info if use_mgda else "Manual"
            )

        all_warps.append(warp)
        if level != len(iterations) - 1:
            new_shape = fixed_images[level+1].shape[2:]
            warp = img2v(F.interpolate(v2img(warp.detach()), size=new_shape, mode='bilinear' if n_dims == 2 else 'trilinear', align_corners=align_corners))
            exp_avg = img2v(F.interpolate(v2img(exp_avg), size=new_shape, mode='bilinear' if n_dims == 2 else 'trilinear', align_corners=align_corners))
            exp_sq_avg = img2v(F.interpolate(v2img(exp_sq_avg), size=new_shape, mode='bilinear' if n_dims == 2 else 'trilinear', align_corners=align_corners))

    return all_warps

def load_surface(surface_path, volume_affine = None):
    """
    Load surface file (GIFTI format)
    """
    gii = nib.load(str(surface_path))

    verts = gii.agg_data('pointset')
    faces = gii.agg_data('triangle')
    faces = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("Invalid vertices array shape, expected (N, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Invalid faces array shape, expected (M, 3).")
    
    if volume_affine is not None:
        verts = nib.affines.apply_affine(np.linalg.inv(volume_affine), verts)

    return Meshes(verts=[torch.from_numpy(verts).float()], faces=[torch.from_numpy(faces).long()])

def main(args):
    ###################
    # Read volume data
    ###################
    src_vol = nib.load(args.src_vol)
    trg_vol = nib.load(args.trg_vol)

    # Normalize
    src_vol_data = 1 * (src_vol.get_fdata() - src_vol.get_fdata().min()) / (src_vol.get_fdata().max() - src_vol.get_fdata().min())
    trg_vol_data = 1 * (trg_vol.get_fdata() - trg_vol.get_fdata().min()) / (trg_vol.get_fdata().max() - trg_vol.get_fdata().min())

    moving_image = torch.from_numpy(src_vol_data).float()[None][None]  # [B, 1, D, H, W]
    fixed_image = torch.from_numpy(trg_vol_data).float()[None][None]  # [B, 1, D, H, W]
    fixed_image, moving_image = fixed_image.to('cuda'), moving_image.to('cuda')
    fixed_images = [
        F.interpolate(fixed_image, scale_factor=[0.25, 0.25, 0.25], mode='trilinear', align_corners=align_corners),
        F.interpolate(fixed_image, scale_factor=[0.5, 0.5, 0.5], mode='trilinear', align_corners=align_corners),
        fixed_image]
    moving_images = [
        F.interpolate(moving_image, scale_factor=[0.25, 0.25, 0.25], mode='trilinear', align_corners=align_corners),
        F.interpolate(moving_image, scale_factor=[0.5, 0.5, 0.5], mode='trilinear', align_corners=align_corners), 
        moving_image]


    ###################
    # Read label data
    ###################
    if args.src_lbl:
        src_lbl = nib.load(args.src_lbl)
        trg_lbl = nib.load(args.trg_lbl)
        moving_label = torch.from_numpy(src_lbl.get_fdata()).long()[None][None]  # [B, 1, D, H, W]
        fixed_label = torch.from_numpy(trg_lbl.get_fdata()).long()[None][None]  # [B, 1, D, H, W]
        fixed_label, moving_label = F.one_hot(fixed_label.to('cuda').squeeze(1), num_classes=4).permute(0, 4, 1, 2, 3).float(), F.one_hot(moving_label.to('cuda').squeeze(1), num_classes=4).permute(0, 4, 1, 2, 3).float()
        fixed_labels = [
            F.interpolate(fixed_label, scale_factor=[0.25, 0.25, 0.25], mode='trilinear'),
            F.interpolate(fixed_label, scale_factor=[0.5, 0.5, 0.5], mode='trilinear'),
            fixed_label]
        moving_labels = [
            F.interpolate(moving_label, scale_factor=[0.25, 0.25, 0.25], mode='trilinear'),
            F.interpolate(moving_label, scale_factor=[0.5, 0.5, 0.5], mode='trilinear'),
            moving_label]
    else:
        fixed_labels, moving_labels = None, None
    # moving_label = torch.where(moving_label == 3, 1, 0).long()
    # fixed_label = torch.where(fixed_label == 3, 1, 0).long()
    
    
    ####################
    # Read surface data
    ####################
    if args.src_surf:
        src_surf = load_surface(args.src_surf, src_vol.affine)
        trg_surf = load_surface(args.trg_surf, trg_vol.affine)
        fixed_surf, moving_surf = trg_surf.verts_packed().to('cuda'), src_surf.verts_packed().to('cuda')
        fixed_surfs = [
            fixed_surf * 0.25,
            fixed_surf * 0.50,
            fixed_surf
        ]
        
        moving_surfs = [
            moving_surf * 0.25,
            moving_surf * 0.50,
            moving_surf
        ]
    else:
        fixed_surfs, moving_surfs = None, None

    # for i, fix in enumerate(fixed_images):
    #     nib.save(nib.Nifti1Image(fix[0, 0, ...].cpu().numpy(), np.eye(4)), f'vis/fixed_vol_{i}.nii.gz')
    
    # for i, mov in enumerate(moving_images):
    #     nib.save(nib.Nifti1Image(mov[0, 0, ...].cpu().numpy(), np.eye(4)), f'vis/moving_vol_{i}.nii.gz')

    # for i, fix in enumerate(fixed_surfs):
    #     save_surf_gii(fix.cpu().numpy(), src_surf.faces_packed(), f'vis/fixed_surfs_{i}.surf.gii')

    # for i, mov in enumerate(moving_surfs):
    #     save_surf_gii(mov.cpu().numpy(), src_surf.faces_packed(), f'vis/moving_surfs_{i}.surf.gii')

    ###################
    # Register
    ###################
    ncc_img_loss_fn = LocalNormalizedCrossCorrelationLoss(
        spatial_dims=3, kernel_size=7, reduction='mean'
    )
    
    dice_loss_fn = DiceLoss(to_onehot_y=False, softmax=False, weight=[0.05, 0.25, 0.35, 0.35])
    # dice_loss_fn = HausdorffDTLoss(alpha=2.0, include_background=False, to_onehot_y=False, sigmoid=False, softmax=False, other_act=None, reduction='mean', batch=False)
    
    mesh_loss_fn = MeshDeformLoss()

    
    # Setup Gaussian filters
    gaussian_grad = gaussian_1d(
        torch.tensor(0.8), truncated=2
    ).cuda()

    gaussian_warp = gaussian_1d(
        torch.tensor(0.8), truncated=2
    ).cuda()

    displacements = multi_scale_diffeomorphic_solver(
        fixed_images, moving_images,
        fixed_labels, moving_labels,
        fixed_surfs, moving_surfs,
        iterations=args.iterations,
        image_loss_func=ncc_img_loss_fn,
        label_loss_func=dice_loss_fn,
        surf_loss_func=mesh_loss_fn, 
        disp_loss_func=None,
        convergence_eps=args.convergence_eps,
        learning_rate=args.learning_rate,
        gaussian_grad=gaussian_grad,
        gaussian_warp=gaussian_warp
    )
    
    warps = displacements_to_warps(displacements)
    
    batch_size, shape = fixed_images[-1].shape[0], fixed_images[-1].shape[2:]
    n_dims = len(shape)
    moved_image = F.grid_sample(moving_image.detach(), warps[-1], align_corners=align_corners)[0, 0, ...].detach().cpu().numpy()
    nib.save(nib.Nifti1Image(moved_image, np.eye(4)), './moved_vol.nii.gz')
    nib.save(nib.Nifti1Image(fixed_image[0, 0, ...].detach().cpu().numpy(), np.eye(4)), './fixed_vol.nii.gz')
    nib.save(nib.Nifti1Image(moving_image[0, 0, ...].detach().cpu().numpy(), np.eye(4)), './moving_vol.nii.gz')
    
    if args.src_lbl:
        moved_label = F.grid_sample(
            moving_label.float(), warps[-1], mode='bilinear',
            padding_mode='zeros', align_corners=align_corners
        ).squeeze(0).detach().cpu().numpy()
        moved_label = np.argmax(moved_label, axis=0).astype(np.uint8)
        nib.save(nib.Nifti1Image(moved_label, np.eye(4)), './moved_lbl.nii.gz')
    
    if args.src_surf:
        deformed_surf, _ = mesh_loss_fn(fixed_surf, moving_surf, warps[-1].permute(0, 4, 1, 2, 3), moving_image.shape[2:])
        save_surf_gii(deformed_surf.detach().cpu().numpy(), src_surf.faces_packed(), './moved_surf.surf.gii')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Registration with Label-based Alignment")
    
    # Volume data arguments
    parser.add_argument('--src_vol', type=str, required=True, help="Source volume file name")
    parser.add_argument('--trg_vol', type=str, required=True, help="Target volume file name")
    
    # Label data arguments
    parser.add_argument('--src_lbl', type=str, default=None, help="Source label file name")
    parser.add_argument('--trg_lbl', type=str, default=None, help="Target label file name")
    
    # Surface data arguments
    parser.add_argument('--src_surf', type=str, default=None, help="Source surface file name")
    parser.add_argument('--trg_surf', type=str, default=None, help="Target surface file name")
    
    # Configuration parameters
    parser.add_argument('--iterations', type=int, nargs='+', default=[400, 300, 250], help="Number of iterations per scale")
    parser.add_argument('--learning_rate', type=float, nargs='+', default=[1e3, 1e4, 1e5], help="Learning rate")
    parser.add_argument('--convergence_eps', type=float, nargs='+', default=[-1e3, -1e3, -1e3], help="Convergence epsilon")
    
    args = parser.parse_args()
    
    main(args)