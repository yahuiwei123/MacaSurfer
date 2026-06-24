import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def voxel_from_ras(coords, affine):
    """将 RAS 坐标转换为体素索引"""
    return nib.affines.apply_affine(np.linalg.inv(affine), coords)

def plot_surface_on_slices(mri_path, surf_paths, n_slices=5, out_file="overlay.svg"):
    # 1. 读取 MRI
    mri_img = nib.load(mri_path)
    mri_data = mri_img.get_fdata()
    affine = mri_img.affine

    # 2. 读取所有 surface 顶点，并区分 pial 和 white
    surf_labels = ['pial', 'pial', 'white', 'white']  # 顺序和传入顺序对应
    surf_colors = {'pial': 'red', 'white': 'blue'}   # 颜色定义
    surface_dict = {'pial': [], 'white': []}

    for path, label in zip(surf_paths, surf_labels):
        surf = nib.load(path)
        vertices = surf.darrays[0].data
        voxels = voxel_from_ras(vertices, affine)
        surface_dict[label].append(voxels)

    # 合并每类 surface 顶点
    for k in surface_dict:
        surface_dict[k] = np.vstack(surface_dict[k])

    # 3. 根据 MRI 强度确定非零体素的切片范围
    brain_mask = mri_data > 1e-8
    nonzero_coords = np.array(np.where(brain_mask))

    x_min, x_max = nonzero_coords[0].min(), nonzero_coords[0].max()
    y_min, y_max = nonzero_coords[1].min(), nonzero_coords[1].max()
    z_min, z_max = nonzero_coords[2].min(), nonzero_coords[2].max()
    
    bias = 10

    # 切片坐标改为基于非零范围内均匀选取
    x_slices = np.linspace(x_min + bias, x_max - bias, n_slices, dtype=int)
    y_slices = np.linspace(y_min + bias, y_max - bias, n_slices, dtype=int)
    z_slices = np.linspace(z_min + bias, z_max - bias, n_slices, dtype=int)

    # 4. 绘图
    fig, axes = plt.subplots(3, n_slices, figsize=(n_slices * 2.5, 9))
    axes = axes.reshape(3, n_slices)

    for i, x in enumerate(x_slices):
        ax = axes[0, i]
        img = mri_data[x, :, :].T
        ax.imshow(img, cmap="gray", origin="lower")
        for label in ['pial', 'white']:
            vertices = surface_dict[label]
            mask = np.abs(vertices[:, 0] - x) < 0.5
            pts = vertices[mask][:, [1, 2]]
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=0.1, c=surf_colors[label], label=label, edgecolors='none', linewidths=0)
        ax.set_title(f'Sagittal x={x}')
        ax.axis("off")

    for i, y in enumerate(y_slices):
        ax = axes[1, i]
        img = mri_data[:, y, :].T
        ax.imshow(img, cmap="gray", origin="lower")
        for label in ['pial', 'white']:
            vertices = surface_dict[label]
            mask = np.abs(vertices[:, 1] - y) < 0.5
            pts = vertices[mask][:, [0, 2]]
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=0.1, c=surf_colors[label], edgecolors='none', linewidths=0)
        ax.set_title(f'Coronal y={y}')
        ax.axis("off")

    for i, z in enumerate(z_slices):
        ax = axes[2, i]
        img = mri_data[:, :, z].T
        ax.imshow(img, cmap="gray", origin="lower")
        for label in ['pial', 'white']:
            vertices = surface_dict[label]
            mask = np.abs(vertices[:, 2] - z) < 0.5
            pts = vertices[mask][:, [0, 1]]
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=0.1, c=surf_colors[label], edgecolors='none', linewidths=0)
        ax.set_title(f'Axial z={z}')
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_file, format='svg')
    print(f"[✅] Saved overlay: {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mri", type=str, default='', help="MRI file")
    parser.add_argument("--lp", type=str, default='', help="left pial")
    parser.add_argument("--rp", type=str, default='', help="right pial")
    parser.add_argument("--lw", type=str, default='', help="left white")
    parser.add_argument("--rw", type=str, default='', help="right white")
    parser.add_argument("--slice_num", type=int, default=7, help="number of slices per view")
    parser.add_argument("--output", type=str, default='overlay.svg', help="output path")
    args = parser.parse_args()

    plot_surface_on_slices(
        args.mri,
        [args.lp, args.rp, args.lw, args.rw],
        args.slice_num,
        args.output
    )
