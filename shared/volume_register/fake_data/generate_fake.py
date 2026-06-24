import sys
sys.path.append('/home/weiyahui/software/macasurfer_v3.0/MacaSurfer/shared/volume_register')
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

# 是否和你之前一样使用 align_corners=False
ALIGN_CORNERS = False

def make_random_svf(
    shape,
    sigma=8.0,
    max_disp_vox=4.0,
    seed=None,
    device="cuda",
):
    """
    生成一个 3D 随机静态速度场（SVF）并映射到 [-1, 1] 坐标系。
    
    Args:
        shape: (D, H, W) 体数据空间大小（注意是 D,H,W）
        sigma: 高斯平滑的标准差（越大越平滑）
        max_disp_vox: 速度场在 voxel 单位下的最大位移幅度（大致上）
        seed: 随机种子
        device: 'cuda' 或 'cpu'
    
    Returns:
        v_norm: torch.Tensor, [1, D, H, W, 3]，在 [-1,1] 坐标系下的速度场（单位时间）
    """
    D, H, W = shape
    rng = np.random.RandomState(seed)

    # 先在 voxel 坐标系下生成一个随机向量场: [3, D, H, W]
    v_np = rng.randn(3, D, H, W).astype(np.float32)

    # 高斯平滑，使其更平滑、连续
    for c in range(3):
        v_np[c] = gaussian_filter(v_np[c], sigma=sigma)

    # 归一化每个点的向量长度，然后乘以最大位移 max_disp_vox（单位：voxel）
    v_flat = v_np.reshape(3, -1)                # [3, N]
    norms = np.linalg.norm(v_flat, axis=0) + 1e-8
    v_flat = v_flat / norms
    v_np = v_flat.reshape(3, D, H, W) * max_disp_vox
    print(v_np)

    # 把 voxel 位移转换到 [-1,1] 归一化坐标系（grid_sample 的坐标）
    # align_corners=False 时，步长大约是 2 / N
    if ALIGN_CORNERS:
        factors = np.array([
            2.0 / (D - 1),
            2.0 / (H - 1),
            2.0 / (W - 1),
        ], dtype=np.float32)
    else:
        factors = np.array([
            2.0 / D,
            2.0 / H,
            2.0 / W,
        ], dtype=np.float32)

    for c in range(3):
        v_np[c] *= factors[c]

    # 转成 Torch，形状改为 [1, D, H, W, 3]
    v = torch.from_numpy(v_np).to(device=device)
    v = v.permute(1, 2, 3, 0).unsqueeze(0)  # [D,H,W,3] -> [1,D,H,W,3]

    return v  # [1, D, H, W, 3]

def integrate_svf_scaling_squaring(v, n_steps=7):
    """
    对静态速度场 v 做 scaling-and-squaring，得到微分同胚形变场 phi。
    
    Args:
        v: [B, D, H, W, 3]，在 [-1,1] 坐标系下的速度场（单位时间）
        n_steps: scaling-and-squaring 步数，2^n_steps 越大，每小步越小，越接近理论 diffeo
    
    Returns:
        phi: [B, D, H, W, 3]，采样网格 (x -> x + disp(x))
    """
    device = v.device
    B, D, H, W, _ = v.shape

    # identity grid: [B, D, H, W, 3]
    theta = torch.eye(3, 4, device=device).unsqueeze(0).expand(B, -1, -1)
    size = (B, 1, D, H, W)
    grid = F.affine_grid(theta, size=size, align_corners=ALIGN_CORNERS)

    # 初始小位移场：v / 2^n_steps
    disp = v / (2.0 ** n_steps)

    for _ in range(n_steps):
        # disp 表示 x -> x + disp(x)
        # 计算 disp(x + disp(x))，即位移的自组合
        disp_img = disp.permute(0, 4, 1, 2, 3)  # [B, 3, D, H, W]
        sampled = F.grid_sample(
            disp_img,
            grid + disp,          # x + disp(x)
            mode="bilinear",
            padding_mode="border",
            align_corners=ALIGN_CORNERS,
        )
        sampled = sampled.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]
        disp = disp + sampled

    phi = grid + disp  # phi(x) = x + disp(x)
    return phi

def warp_image_3d(img_np, phi, mode="bilinear"):
    """
    用给定形变场 phi warp 一张 3D 图像。
    
    Args:
        img_np: numpy array, shape = [D, H, W] 或 [C, D, H, W]
        phi: torch.Tensor, [1, D, H, W, 3]，采样网格
        mode: 'bilinear' 或 'nearest'
    
    Returns:
        warped_np: numpy array，shape 同 img_np
    """
    device = phi.device

    img = torch.from_numpy(img_np).float()
    if img.ndim == 3:
        img = img.unsqueeze(0)     # [1, D, H, W]
    elif img.ndim == 4:
        pass                       # [C, D, H, W]
    else:
        raise ValueError("img_np should be [D,H,W] or [C,D,H,W]")

    img = img.unsqueeze(0).to(device)  # [1, C, D, H, W]

    warped = F.grid_sample(
        img,
        phi,
        mode=mode,
        padding_mode="border",
        align_corners=ALIGN_CORNERS,
    )  # [1, C, D, H, W]

    warped = warped.squeeze(0).cpu().numpy()  # [C, D, H, W]
    if img_np.ndim == 3:
        warped = warped[0]  # [D, H, W]

    return warped

import nibabel as nib
import numpy as np
from mesh_loss import deform_mesh, load_gii, save_gii

if __name__ == '__main__':
    img_path, lbl_path, surf_path = "/home/weiyahui/projects/Monkey_Surface/test/test_tissue_guide_reg/mebrain/mebrain.nii.gz", "/home/weiyahui/projects/Monkey_Surface/test/test_tissue_guide_reg/mebrain/mebrain_nbest.nii.gz", "/home/weiyahui/projects/Monkey_Surface/test/test_tissue_guide_reg/mebrain/all.surf.gii"
    out_path = "./warped.nii.gz"
    max_disp_vox = 4.0
    n_steps = int(4 * max_disp_vox)
    device = 'cuda'
    
    # 读入 NIfTI
    img = nib.load(img_path)
    img_np = img.get_fdata().astype(np.float32)  # [H, W, D] or [D, H, W]
    lbl = nib.load(lbl_path)
    lbl_np = lbl.get_fdata().astype(np.float32)  # [H, W, D] or [D, H, W]
    surf = load_gii(surf_path, img.affine)
    surf_np = surf.verts_packed().to(device)
    
    # 这里假设 nii.shape = (D, H, W)，如果是 (H,W,D) 就改下轴顺序
    if img_np.ndim != 3:
        raise ValueError("Only support 3D scalar image for this example.")
    D, H, W = img_np.shape

    # 归一化一下（可选）
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    # 1) 随机 SVF
    v = make_random_svf(
        shape=(D, H, W),
        sigma=17.0,
        max_disp_vox=max_disp_vox,
        seed=12,
        device=device,
    )  # [1, D, H, W, 3]

    # 2) 积分得到 diffeo 形变场
    phi = integrate_svf_scaling_squaring(v, n_steps=n_steps)  # [1, D, H, W, 3]
    inv_phi = integrate_svf_scaling_squaring(-v, n_steps=n_steps)  # [1, D, H, W, 3]

    # 3) warp 图像
    warped_img = warp_image_3d(img_np, phi, mode="bilinear")   # [D, H, W]
    warped_lbl = warp_image_3d(lbl_np, phi, mode="bilinear")   # [D, H, W]

    warped_surf = deform_mesh(surf_np, inv_phi.permute(0, 4, 1, 2, 3), img_np.shape).detach().cpu().numpy()
    warped_surf = img.affine @ np.hstack([warped_surf, np.ones((warped_surf.shape[0], 1))]).T
    warped_surf = warped_surf.T[:, :3]

    # 写回 NIfTI，使用原 affine
    warped_img = nib.Nifti1Image(warped_img, affine=img.affine)
    nib.save(warped_img, './warped_img.nii.gz')
    warped_lbl = nib.Nifti1Image(warped_lbl.astype(np.uint8), affine=img.affine)
    nib.save(warped_lbl, './warped_lbl.nii.gz')
    save_gii(
        warped_surf,
        surf.faces_packed(),
        './warped_all_surf.surf.gii'
    )

    # warp back to original
    surf = load_gii('./warped_all_surf.surf.gii', img.affine)
    surf_np = surf.verts_packed().to(device)
    warped_surf = deform_mesh(surf_np, phi.permute(0, 4, 1, 2, 3), img_np.shape).detach().cpu().numpy()
    warped_surf = img.affine @ np.hstack([warped_surf, np.ones((warped_surf.shape[0], 1))]).T
    warped_surf = warped_surf.T[:, :3]
    save_gii(
        warped_surf,
        surf.faces_packed(),
        './warped_back_all_surf.surf.gii'
    )

    print(f"Saved warped volume to: {out_path}")