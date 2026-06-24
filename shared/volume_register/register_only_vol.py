import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import nibabel as nib

from typing import List, Union, Tuple, Callable, Optional

import argparse
from tqdm import tqdm
from monai.losses.ssim_loss import SSIMLoss
from monai.losses.hausdorff_loss import HausdorffDTLoss
from monai.losses import DiceLoss, BendingEnergyLoss, LocalNormalizedCrossCorrelationLoss, GlobalMutualInformationLoss, DiffusionLoss

from neuio.gifti import load_surf_gii, save_surf_gii
from neuio.nifti import load_vol_nii, save_vol_nii
from utils.mesh import MeshDeformLoss, deform_mesh, affine_mesh
from utils.grid import displacements_to_warps, v2img_3d, img2v_3d
from utils.schedule import build_scheduler
from utils.loss import MultiLossWrapper, JacobianDeterminantLoss
from solver.nonlinear import VelocityDiffeoWarpField, CompositeDiffeoWarpField, FreeFormWarpField
from solver.linear import AffineWarpField

align_corners = False

def multi_scale_diffeomorphic_solver(
        fixed_images: List[torch.Tensor],
        moving_images: List[torch.Tensor],
        fixed_labels: Optional[List[torch.Tensor]],
        moving_labels: Optional[List[torch.Tensor]],
        fixed_surfs: Optional[List[torch.Tensor]],
        moving_surfs: Optional[List[torch.Tensor]],
        iterations: List[int],
        image_loss_func: Union[nn.Module, Callable],
        label_loss_func: Union[nn.Module, Callable],
        surf_loss_func: Union[nn.Module, Callable],
        disp_loss_func: Optional[Callable] = None,
        gaussian_sigma: List[float] = [0.2, 0.3, 0.5],
        learning_rate: List[float] = [1e-1, 1e-2, 1e-2],
        eps: float = 1e-8,
        convergence_tol: int = 10,
        convergence_eps: Union[float, List[float]] = 1e-2,
        use_mgda: bool = False,
        mgda_gn: str = "loss",
        lr_schedule: str = "cosine"
):
    """
    Multi-scale diffeomorphic registration solver.

    This function performs:
        1) Global affine alignment at the coarsest scale.
        2) Multi-scale diffeomorphic refinement using VelocityDiffeoWarpField.

    Args:
        fixed_images: List of fixed images at different scales.
        moving_images: List of moving images at different scales.
        fixed_labels, moving_labels: Optional segmentation labels.
        fixed_surfs, moving_surfs: Optional surface data.
        iterations: Number of iterations at each scale.
        image_loss_func: Image similarity loss.
        label_loss_func: Label alignment loss.
        surf_loss_func: Surface alignment loss.
        disp_loss_func: Optional displacement regularizer.
        gaussian_sigma: Per-scale Gaussian smoothing coefficients.
        learning_rate: Per-scale learning rate.
        convergence_tol: Early-stop window size.
        convergence_eps: Early-stop threshold.

    Returns:
        displacements_per_scale, inverse_displacements_per_scale
    """

    device = fixed_images[0].device
    batch_size = fixed_images[0].shape[0]
    init_shape = fixed_images[0].shape[2:]
    n_dims = len(init_shape)

    # Ensure lists for labels/surfaces
    if fixed_labels is None:
        fixed_labels = [None] * len(fixed_images)
        moving_labels = [None] * len(moving_images)
    if fixed_surfs is None:
        fixed_surfs = [None] * len(fixed_images)
        moving_surfs = [None] * len(moving_images)

    # Function selection for reshaping v <-> image
    v2img = v2img_2d if n_dims == 2 else v2img_3d
    img2v = img2v_2d if n_dims == 2 else img2v_3d

    # Results
    all_displacements = []
    all_inv_displacements = []

    prev_disp = None  # cross-scale initial flow

    # ----------------------------------------------------------------------
    # 1) Affine alignment at coarsest scale
    # ----------------------------------------------------------------------
    fixed_lvl0 = fixed_images[0].to(device)
    moving_lvl0 = moving_images[0].to(device)

    affine_model = AffineWarpField(
        batch_size=batch_size,
        n_dims=n_dims,
        device=device,
    )

    affine_iters = max(100, iterations[0] // 2)
    affine_optimizer = torch.optim.AdamW(
        affine_model.parameters(),
        lr=1e-4,
        weight_decay=0.9,
    )

    for _ in range(affine_iters):
        moved_lvl0, affine_grid = affine_model(
            moving_lvl0.detach(),
            fixed_lvl0.shape
        )

        loss_img = image_loss_func(moved_lvl0, fixed_lvl0.detach())

        surf_loss_val = torch.tensor(0.0, device=device)
        if moving_surfs[0] is not None:
            mv_surf = moving_surfs[0].to(device)
            fx_surf = fixed_surfs[0].to(device)
            _, surf_loss_val = surf_loss_func(
                fx_surf,
                mv_surf,
                affine_grid.permute(0, 4, 1, 2, 3),
                moving_lvl0.shape[2:]
            )

        loss_affine = 0.5 * loss_img + 0.5 * surf_loss_val
        affine_optimizer.zero_grad()
        loss_affine.backward()
        affine_optimizer.step()

    # Convert final affine grid into displacement
    with torch.no_grad():
        _, final_affine_grid = affine_model(moving_lvl0.detach(), fixed_lvl0.shape)
        eye = torch.eye(n_dims + 1, device=device, dtype=final_affine_grid.dtype)[:n_dims, :]
        eye = eye.unsqueeze(0).repeat(batch_size, 1, 1)
        identity_grid = F.affine_grid(
            eye, fixed_lvl0.shape, align_corners=align_corners
        )
        prev_disp = (final_affine_grid - identity_grid).detach()

    # ----------------------------------------------------------------------
    # 2) Multi-scale diffeomorphic refinement
    # ----------------------------------------------------------------------
    loss_buffer = []

    for lvl, (n_iters, data_pack) in enumerate(
            zip(
                iterations,
                zip(fixed_images, moving_images,
                    fixed_labels, moving_labels,
                    fixed_surfs, moving_surfs)
            )
    ):
        fixed_img, moving_img, fixed_lbl, moving_lbl, fixed_surf, moving_surf = data_pack
        fixed_img = fixed_img.to(device)
        moving_img = moving_img.to(device)
        spatial_shape = fixed_img.shape[2:]

        # Upsample previous displacement as initialization
        if prev_disp is not None:
            prev_img = v2img(prev_disp)
            mode = "bilinear" if n_dims == 2 else "trilinear"
            prev_resized = F.interpolate(
                prev_img,
                size=spatial_shape,
                mode=mode,
                align_corners=align_corners,
            )
            init_warp = img2v(prev_resized)
        else:
            init_warp = None

        warp_field = VelocityDiffeoWarpField(
            batch_size=batch_size,
            shape=list(spatial_shape),
            n_dims=n_dims,
            device=device,
            init_warp=init_warp,
            n_steps=4
        )

        # warp_field = FreeFormWarpField(
        #     batch_size=batch_size,
        #     shape=list(spatial_shape),
        #     n_dims=n_dims,
        #     device=device,
        #     init_warp=init_warp,
        #     downsample_factor=1
        # )


        optimizer = torch.optim.Adam(
            warp_field.parameters(),
            lr=learning_rate[lvl],
            betas=(0.9, 0.999)
        )

        # optimizer = torch.optim.SGD(
        #     warp_field.parameters(),
        #     lr=learning_rate[lvl],
        #     momentum=0.9
        # )

        # Build LR scheduler for this level (if any)
        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_type=lr_schedule,
            base_lr=learning_rate[lvl],
            n_iters=n_iters
        )

        pbar = tqdm(range(1, n_iters + 1), desc=f"Level {lvl}")

        for step in pbar:
            disp, warp_grid = warp_field()
            inv_disp, inv_warp = warp_field.get_inv(out_shape=moving_img.shape[2:])

            moved = F.grid_sample(
                moving_img.detach(),
                warp_grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=align_corners,
            )
            moved_inv = F.grid_sample(
                fixed_img,
                inv_warp,
                mode='bilinear',
                padding_mode='border',
                align_corners=align_corners,
            )

            img_loss = (
                image_loss_func(moved, fixed_img.detach())
                + image_loss_func(moved_inv, moving_img.detach())
            )

            # Label loss
            if moving_lbl is not None:
                moving_lbl = moving_lbl.to(device)
                fixed_lbl = fixed_lbl.to(device)
                moved_lbl = F.grid_sample(
                    moving_lbl.float(),
                    inv_warp,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=align_corners
                )
                lbl_loss = label_loss_func(moved_lbl, fixed_lbl)
            else:
                lbl_loss = torch.tensor(0.0, device=device)

            # Surface loss
            if moving_surf is not None:
                fx_surf = fixed_surf.to(device)
                mv_surf = moving_surf.to(device)

                _, surf_loss_fw = surf_loss_func(
                    fx_surf,
                    mv_surf,
                    warp_grid.permute(0, 4, 1, 2, 3),
                    moving_img.shape[2:]
                )
                _, surf_loss_bw = surf_loss_func(
                    mv_surf,
                    fx_surf,
                    inv_warp.permute(0, 4, 1, 2, 3),
                    fixed_img.shape[2:]
                )
                surf_loss = surf_loss_fw + surf_loss_bw
            else:
                surf_loss = torch.tensor(0.0, device=device)

            # Displacement regularization
            if disp_loss_func is not None:
                reg_loss = disp_loss_func(disp.permute(0, 4, 1, 2, 3))
            else:
                reg_loss = torch.tensor(0.0, device=device)

            # Weighted loss
            total_loss = (
                1.00 * img_loss
                + 0.00 * lbl_loss
                + 0.00 * surf_loss
                # + 0.0 * surf_loss
                + 3e-3 * reg_loss
            )

            loss_buffer.append(total_loss)
            # Early stopping
            if len(loss_buffer) >= convergence_tol:
                recent = loss_buffer[-convergence_tol:]
                diffs = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
                if all(d < convergence_eps[lvl] for d in diffs):
                    print(f"[Early Stop] Level {lvl}, step {step}")
                    break

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(warp_field.parameters(), max_norm=1.0)
            optimizer.step()

            # Step LR scheduler (if enabled)
            if scheduler is not None:
                scheduler.step()

            pbar.set_postfix(
                total=f"{total_loss.item():.3e}",
                vol=f"{img_loss.item():.3e}",
                surf=f"{surf_loss.item():.3e}",
                reg=f"{reg_loss.item():.3e},",
                lr=f"{optimizer.param_groups[0]['lr']:.3e}"
            )

        # Save final displacement
        with torch.no_grad():
            inv_disp, _ = warp_field.get_inv(moving_img.shape[2:])
        all_displacements.append(disp.detach())
        all_inv_displacements.append(inv_disp.detach())
        prev_disp = disp.detach()

    return all_displacements, all_inv_displacements

# ==========================
#   main
# ==========================
def main(args):
    ###################
    # Read volume data
    ###################
    src_vol = nib.load(args.src_vol)
    trg_vol = nib.load(args.trg_vol)

    # Normalize
    src_vol_data = 255 * (src_vol.get_fdata() - src_vol.get_fdata().min()) / (
            src_vol.get_fdata().max() - src_vol.get_fdata().min()
    )
    trg_vol_data = 255 * (trg_vol.get_fdata() - trg_vol.get_fdata().min()) / (
            trg_vol.get_fdata().max() - trg_vol.get_fdata().min()
    )

    moving_image = torch.from_numpy(src_vol_data).float()[None][None]  # [B, 1, D, H, W]
    fixed_image = torch.from_numpy(trg_vol_data).float()[None][None]   # [B, 1, D, H, W]
    fixed_image, moving_image = fixed_image.to('cuda'), moving_image.to('cuda')

    fixed_images = [
        F.interpolate(fixed_image, scale_factor=[0.25, 0.25, 0.25], mode='trilinear', align_corners=align_corners),
        F.interpolate(fixed_image, scale_factor=[0.5, 0.5, 0.5], mode='trilinear', align_corners=align_corners),
        fixed_image,
    ]
    moving_images = [
        F.interpolate(moving_image, scale_factor=[0.25, 0.25, 0.25], mode='trilinear', align_corners=align_corners),
        F.interpolate(moving_image, scale_factor=[0.5, 0.5, 0.5], mode='trilinear', align_corners=align_corners),
        moving_image,
    ]

    ###################
    # Read label data
    ###################
    if args.src_lbl:
        src_lbl = nib.load(args.src_lbl)
        trg_lbl = nib.load(args.trg_lbl)
        moving_label = torch.from_numpy(src_lbl.get_fdata()).long()[None][None].to('cuda')
        fixed_label = torch.from_numpy(trg_lbl.get_fdata()).long()[None][None].to('cuda')
        fixed_labels, moving_labels = None, None
    else:
        fixed_labels, moving_labels = None, None

    ####################
    # Read surface data
    ####################
    if args.src_surf:
        src_surf = load_surf_gii(args.src_surf, src_vol.affine)
        trg_surf = load_surf_gii(args.trg_surf, trg_vol.affine)
        fixed_surf = trg_surf.verts_packed().to('cuda')
        moving_surf = src_surf.verts_packed().to('cuda')

        fixed_surfs = [
            fixed_surf * 0.25,
            fixed_surf * 0.50,
            fixed_surf,
        ]
        moving_surfs = [
            moving_surf * 0.25,
            moving_surf * 0.50,
            moving_surf,
        ]
    else:
        fixed_surfs, moving_surfs = None, None

    ###################
    # Register
    ###################
    ncc_img_loss_fn = LocalNormalizedCrossCorrelationLoss(kernel_size=3)
    # ncc_img_loss_fn = SSIMLoss(spatial_dims=3, win_size=3)

    dice_loss_fn = DiceLoss(
        to_onehot_y=False,
        softmax=False,
        weight=[0.05, 0.25, 0.35, 0.35]
    )

    # surface loss
    mesh_loss_fn = MeshDeformLoss(loss_mode="mse")

    # displacement regularization loss
    reg_loss_fns = {
        "diffusion": DiffusionLoss(normalize=True, reduction="sum"),
        "bending": BendingEnergyLoss(normalize=True, reduction="sum"), 
        "jacobian": JacobianDeterminantLoss(jacobian_min=0.0, squared=True),
    }

    reg_weights = {
        "diffusion": 0.8,
        "bending": 0.0,
        "jacobian": 0.0,
    }

    reg_loss_wrapper = MultiLossWrapper(
        loss_fns=reg_loss_fns,
        weights=reg_weights,
    )

    displacements, inv_displacements = multi_scale_diffeomorphic_solver(
        fixed_images=fixed_images,
        moving_images=moving_images,
        fixed_labels=fixed_labels,
        moving_labels=moving_labels,
        fixed_surfs=fixed_surfs,
        moving_surfs=moving_surfs,
        iterations=args.iterations,
        image_loss_func=ncc_img_loss_fn,
        label_loss_func=dice_loss_fn,
        surf_loss_func=mesh_loss_fn,
        disp_loss_func=reg_loss_wrapper,
        convergence_eps=args.convergence_eps,
        learning_rate=args.learning_rate,
        lr_schedule='exp'
    )

    warps = displacements_to_warps(displacements)
    inv_warps = displacements_to_warps(inv_displacements)

    moved_image = F.grid_sample(
        moving_image.detach(),
        warps[-1],
        align_corners=align_corners
    )
    save_vol_nii(moved_image[0, 0, ...].detach().cpu().numpy(), trg_vol.affine, args.out_vol)

    if args.out_warp:
        save_vol_nii(warps[-1].permute(0, 4, 1, 2, 3)[0, 0, ...].detach().cpu().numpy(), trg_vol.affine, args.out_warp)
    
    if args.out_inv_warp:
        save_vol_nii(inv_warps[-1].permute(0, 4, 1, 2, 3)[0, 0, ...].detach().cpu().numpy(), trg_vol.affine, args.out_inv_warp)

    if args.src_lbl:
        moving_label = moving_label.to('cuda').squeeze(1).long()
        moving_label = F.one_hot(
            moving_label,
            num_classes=torch.unique(moving_label).numel()
        ).permute(0, 4, 1, 2, 3).float()
        moved_label = F.grid_sample(
            moving_label,
            warps[-1],
            mode='bilinear',
            padding_mode='border',
            align_corners=align_corners,
        ).argmax(dim=1)[0, ...].detach().cpu().numpy().astype(np.uint8)
        save_vol_nii(moved_label, trg_vol.affine, args.out_lbl)

    if args.src_surf:
        deformed_surf = deform_mesh(
            moving_surf,
            inv_warps[-1].permute(0, 4, 1, 2, 3),
            fixed_image.shape[2:],
        )
        deformed_surf = affine_mesh(
            deformed_surf,
            torch.tensor(trg_vol.affine)
        ).detach().cpu().numpy()
        save_surf_gii(deformed_surf, src_surf.faces_packed(), args.out_surf)


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

    # Output data arguments
    parser.add_argument('--out_vol', type=str, default=None, help="Output volume file name")
    parser.add_argument('--out_lbl', type=str, default=None, help="Output label file name")
    parser.add_argument('--out_surf', type=str, default=None, help="Output surface file name")
    parser.add_argument('--out_warp', type=str, default=None, help="Warp file name")
    parser.add_argument('--out_inv_warp', type=str, default=None, help="Inverse warp file name")

    # Configuration parameters
    parser.add_argument(
        '--iterations',
        type=int,
        nargs='+',
        default=[1500, 1200, 900],
        help="Number of iterations per scale"
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        nargs='+',
        default=[1e-2, 1e-2, 5e-3],
        help="Learning rate"
    )
    parser.add_argument(
        '--convergence_eps',
        type=float,
        nargs='+',
        default=[1e-5, 3e-6, 1e-6],
        help="Convergence epsilon"
    )

    args = parser.parse_args()
    main(args)
