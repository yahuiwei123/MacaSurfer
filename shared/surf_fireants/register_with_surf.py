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

from fireants.neuio.gifti import load_surf_gii, save_surf_gii, load_label_gii
from fireants.neuio.nifti import load_vol_nii, save_vol_nii
from fireants.utils.mesh import MeshDeformLoss, deform_mesh, affine_mesh, vox2phy
from fireants.utils.grid import displacements_to_warps, v2img_3d, img2v_3d, compute_inverse_displacement
from fireants.utils.schedule import build_scheduler
from fireants.utils.loss import MultiLossWrapper, JacobianDeterminantLoss
from fireants.utils.weights import MGDASolver

from fireants.io import Image, BatchedImages
from fireants.registration.affine import get_affine_transform
from fireants.registration.greedy import GreedyRegistration
from fireants.interpolator import fireants_interpolator
from fireants.utils.imageutils import jacobian

align_corners = True

# ==========================
#   main
# ==========================
def main(args):
    ###################
    # Read volume data
    ###################
    print("Loading images...")
    src_vol = nib.load(args.src_vol)
    trg_vol = nib.load(args.trg_vol)

    src_affine = src_vol.affine
    trg_affine = trg_vol.affine

    # Normalize
    src_vol_data, trg_vol_data = src_vol.get_fdata(), trg_vol.get_fdata()
    src_vol_mask, trg_vol_mask = src_vol_data > 0, trg_vol_data > 0

    src_mean, src_std = src_vol_data[src_vol_mask].mean(), max(src_vol_data[src_vol_mask].std(), 1e-6)
    src_vol_data = (src_vol_data - src_mean) / src_std

    trg_mean, trg_std = trg_vol_data[trg_vol_mask].mean(), max(trg_vol_data[trg_vol_mask].std(), 1e-6)
    trg_vol_data = (trg_vol_data - trg_mean) / trg_std

    moving_image = torch.from_numpy(src_vol_data).float()[None][None]  # [B, 1, D, H, W]
    fixed_image = torch.from_numpy(trg_vol_data).float()[None][None]   # [B, 1, D, H, W]
    fixed_image, moving_image = fixed_image.to('cuda'), moving_image.to('cuda')

    # args.src_surf = None

    
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
        src_surf = load_surf_gii(args.src_surf, src_affine)
        trg_surf = load_surf_gii(args.trg_surf, trg_affine)
        fixed_surf = trg_surf.verts_packed()[None].to('cuda') # [B, N, 3]
        moving_surf = src_surf.verts_packed()[None].to('cuda')
    else:
        fixed_surf, moving_surf = None, None
        
    if args.src_cort:
        fixed_roi = load_label_gii(args.trg_cort)[None].to('cuda') # [B, N]
        moving_roi = load_label_gii(args.src_cort)[None].to('cuda')

    ###################
    # Register
    ###################
    
    # image loss
    ncc_img_loss_fn = LocalNormalizedCrossCorrelationLoss(kernel_size=5)
    # ncc_img_loss_fn = SSIMLoss(spatial_dims=3, win_size=3)

    # surface loss
    mesh_loss_fn = MeshDeformLoss(loss_mode="mse")

    # displacement regularization loss
    reg_loss_fns = DiffusionLoss(normalize=True, reduction="mean")


    init_affine, _ = get_affine_transform(fixed_image, moving_image, fixed_surf, moving_surf)
    row = torch.zeros(init_affine.shape[0], 1, 3 + 1, device=init_affine.device)
    row[:, 0, -1] = 1.0
    init_affine = torch.cat([init_affine.detach(), row], dim=1)
    inv_init_affine = torch.linalg.inv(init_affine)
    # fixed_surf = affine_mesh(fixed_surf, init_affine, fixed_image.shape[2:], moving_image.shape[2:])
    # init_affine = None

    reg = GreedyRegistration(
        scales=args.scales,
        iterations=args.iterations,
        fixed_images=fixed_image,
        moving_images=moving_image,
        img_loss_fn=ncc_img_loss_fn,
        optimizer='Adam',
        optimizer_lr=args.learning_rate,
        init_affine=init_affine,
        fixed_surfs=fixed_surf,
        moving_surfs=moving_surf,
        fixed_roi=fixed_roi,
        moving_roi=moving_roi,
        surf_loss_func=mesh_loss_fn,
        displacement_reg=reg_loss_fns
    )
    reg.optimize()

    # get disp and affined warp
    fwd_disp = reg.fwd_warp.warp.data
    rev_disp = reg.rev_warp.warp.data
    fwd_warp_grid = F.affine_grid(init_affine[:, :-1, :], fixed_image.size(), align_corners=False) + fwd_disp
    rev_warp_grid = F.affine_grid(torch.eye(4)[None, :-1, :].to(moving_image.device), moving_image.size(), align_corners=False) + rev_disp
    rev_warp_grid = torch.bmm(
        inv_init_affine,
        torch.cat([rev_warp_grid, torch.ones(*rev_warp_grid.shape[:-1], 1, device=moving_image.device, dtype=fwd_warp_grid.dtype)], dim=-1).view(rev_warp_grid.shape[0], -1, 4).transpose(1, 2)
    ).transpose(1, 2)[..., :-1].view(*rev_warp_grid.shape[:-1], 3)


    # warp image
    moving_image_warped = F.grid_sample(
        moving_image,
        fwd_warp_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ) * src_std + src_mean
    save_vol_nii(moving_image_warped[0, 0, ...].detach().cpu().numpy(), trg_affine, args.out_vol)

    # fixed_image_warped = F.grid_sample(
    #     fixed_image,
    #     rev_warp_grid,
    #     mode="bilinear",
    #     padding_mode="zeros",
    #     align_corners=False,
    # ) * trg_std + trg_mean
    # save_vol_nii(fixed_image_warped[0, 0, ...].detach().cpu().numpy(), src_affine, args.out_vol)

    if args.out_warp:
        save_vol_nii(fwd_warp_grid.permute(0, 4, 1, 2, 3)[0, 0, ...].detach().cpu().numpy(), trg_affine, args.out_warp)
    
    if args.out_inv_warp:
        save_vol_nii(rev_warp_grid.permute(0, 4, 1, 2, 3)[0, 0, ...].detach().cpu().numpy(), trg_affine, args.out_inv_warp)

    if args.src_lbl:
        moving_label = moving_label.to('cuda').squeeze(1).long()
        moving_label = torch.where(moving_label == -1, 0, moving_label)
        moving_label = F.one_hot(
            moving_label,
            num_classes=torch.unique(moving_label).numel()
        ).permute(0, 4, 1, 2, 3).float()
        moved_label = F.grid_sample(
            moving_label,
            fwd_warp_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=align_corners,
        ).argmax(dim=1)[0, ...].detach().cpu().numpy().astype(np.uint8)
        save_vol_nii(moved_label, trg_affine, args.out_lbl)

    if args.src_surf:
        deformed_surf = deform_mesh(
            moving_surf,
            rev_warp_grid.permute(0, 4, 1, 2, 3),
            fixed_image.shape[2:],
        )
        deformed_surf = vox2phy(deformed_surf, torch.tensor(trg_affine)[None].to(deformed_surf.device))[0].detach().cpu().numpy()
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
    
    # Surface cortical region arguments
    parser.add_argument('--src_cort', type=str, default=None, help="Source surface ROI file name")
    parser.add_argument('--trg_cort', type=str, default=None, help="Target surface ROI file name")

    # Output data arguments
    parser.add_argument('--out_vol', type=str, default=None, help="Output volume file name")
    parser.add_argument('--out_lbl', type=str, default=None, help="Output label file name")
    parser.add_argument('--out_surf', type=str, default=None, help="Output surface file name")
    parser.add_argument('--out_warp', type=str, default=None, help="Warp file name")
    parser.add_argument('--out_inv_warp', type=str, default=None, help="Inverse warp file name")

    # Configuration parameters
    parser.add_argument(
        '--scales',
        type=int,
        nargs='+',
        default=[4, 2, 1],
        help="Downsample scales"
    )
    parser.add_argument(
        '--iterations',
        type=int,
        nargs='+',
        default=[600, 500, 400],
        help="Number of iterations per scale"
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.4,
        help="Learning rate"
    )
    parser.add_argument(
        '--convergence_eps',
        type=float,
        nargs='+',
        default=[1e-12, 1e-12, 1e-12],
        help="Convergence epsilon"
    )

    args = parser.parse_args()
    main(args)
