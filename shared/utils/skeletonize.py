import numpy as np
import nibabel as nib
import argparse
from scipy.ndimage import laplace

def main(args):
    # 读取符号距离场（SDF）
    sdf_img = nib.load(args.sdf)
    sdf = sdf_img.get_fdata()

    # 计算 SDF 的拉普拉斯（即二阶导数）
    laplacian_sdf = laplace(sdf)

    # 提取曲率大于0的点（即拉普拉斯值大于0的点）
    # 这些点通常是凸起的部分，表示曲率大于0
    curvature_points = np.where((laplacian_sdf > 0) & (sdf < 0))

    # 创建骨架（所有曲率大于0的点标记为 1）
    skeleton = np.zeros_like(sdf, dtype=bool)
    skeleton[curvature_points] = True

    # 保存骨架体积
    skeleton_img = nib.Nifti1Image(skeleton.astype(np.uint8), sdf_img.affine)
    nib.save(skeleton_img, args.skel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=str, default='', help="sdf to process")
    parser.add_argument("--skel", type=str, default='', help="skeleton mask path to save")
    args = parser.parse_args()
    
    main(args=args)