from fireants.solver.linear import AffineWarpField
from fireants.utils.mesh import MeshDeformLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import LocalNormalizedCrossCorrelationLoss
from tqdm import tqdm

def get_affine_transform(fixed_image, moving_image, fixed_surf, moving_surf):
    device = fixed_image.device

    affine_model = AffineWarpField(
        batch_size=1,
        n_dims=3,
        device=device,
    )

    affine_iters = 200
    affine_optimizer = torch.optim.AdamW(
        affine_model.parameters(),
        lr=1e-2,
        weight_decay=0.9,
    )
    ncc_img_loss_fn = LocalNormalizedCrossCorrelationLoss(kernel_size=11)
    surf_loss_fn = MeshDeformLoss(loss_mode="mse")
    pbar = tqdm(range(affine_iters), desc=f"Affine Transform")

    for _ in pbar:
        moved_image, affine_grid = affine_model(
            moving_image.detach(),
            fixed_image.shape
        )

        img_loss = ncc_img_loss_fn(moved_image, fixed_image.detach())

        surf_loss = torch.tensor(0.0, device=device)
        if moving_surf is not None:
            mv_surf = moving_surf.to(device)
            fx_surf = fixed_surf.to(device)
            _, surf_loss = surf_loss_fn(
                fx_surf,
                mv_surf,
                None,
                None,
                affine_grid.permute(0, 4, 1, 2, 3),
                moving_image.shape[2:]
            )

        loss_affine = 1e-4 * img_loss + 1.0 * surf_loss
        pbar.set_postfix(
            total=f"{loss_affine.item():.3e}",
            vol=f"{img_loss.item():.3e}",
            surf=f"{surf_loss.item():.3e}"
        )
        affine_optimizer.zero_grad()
        loss_affine.backward()
        affine_optimizer.step()

    return affine_model.matrix, moved_image