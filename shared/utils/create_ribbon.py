                                                                                                                                                                                                                                        
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import gc
import nibabel as nib
import numpy as np
from scipy.ndimage import label, binary_dilation, generate_binary_structure
import argparse
from pathlib import Path

def fix_thin_white_matter_pure_bridge(
    sdf_path: str,
    relax_thresh: float = 0.6,
    output_path: str = "white_mask_fixed.nii.gz",
    min_region_size: int = 5,
    connectivity: int = 6,
    bridge_tolerance: float = 0.1,
    pial_sdf_path: str = None,  # 新增：pial SDF路径
    pial_value: int = 3,        # 新增：pial区域灰度值
    white_value: int = 2        # 新增：white区域灰度值
) -> None:
    """
    精准修复白质细小断裂，可选合并pial mask（white完全覆盖pial）
    """
    if not Path(sdf_path).exists():
        raise FileNotFoundError(f"SDF文件不存在: {sdf_path}")

    img = nib.load(sdf_path)
    sdf_data = np.asarray(img.dataobj)
    affine = img.affine
    header = img.header
    voxel_size = img.header.get_zooms()[0]
    vol_shape = sdf_data.shape

    # ========== 1. 白质修复逻辑（和之前完全一致） ==========
    strict_mask = sdf_data < 0.0
    relaxed_mask = sdf_data <= relax_thresh
    candidate_mask = relaxed_mask & (~strict_mask)
    del relaxed_mask, sdf_data
    gc.collect()

    if not np.any(candidate_mask):
        white_final = strict_mask.astype(np.uint8)
        print("无需要填补的区域，使用原始白质mask")
    else:
        # 连通域结构矩阵
        if connectivity == 6:
            struct = np.zeros((3,3,3), dtype=bool)
            struct[1,1,:] = struct[1,:,1] = struct[:,1,1] = True
        elif connectivity == 18:
            struct = np.ones((3,3,3), dtype=bool)
            struct[0,0,0] = struct[0,0,2] = struct[0,2,0] = struct[0,2,2] = 0
            struct[2,0,0] = struct[2,0,2] = struct[2,2,0] = struct[2,2,2] = 0
        else:
            struct = np.ones((3,3,3), dtype=bool)

        strict_labels, n_strict = label(strict_mask, structure=struct)
        print(f"严格阈值下有 {n_strict} 个独立白质区域")
        sys.stdout.flush()

        # 过滤小噪声
        if min_region_size > 0 and n_strict > 1:
            region_sizes = np.bincount(strict_labels.flat)[1:]
            remove_labels = np.where(region_sizes < min_region_size)[0] + 1
            if len(remove_labels) > 0:
                mask = np.isin(strict_labels, remove_labels)
                strict_mask[mask] = False
                strict_labels[mask] = 0
                unique_labels = np.unique(strict_labels[strict_labels > 0])
                n_strict = len(unique_labels)
                print(f"过滤小噪声后剩余 {n_strict} 个有效白质区域")
                sys.stdout.flush()

        fill_mask = np.zeros_like(strict_mask, dtype=bool)
        if n_strict > 1 and np.any(candidate_mask):
            gc.collect()
            # ===== Memory-efficient bridge detection (NO EDT) =====
            # Strategy: find label boundaries in strict_labels, dilate to bridge zone
            # candidate_mask already encodes "within relax_thresh of white surface"
            # via the SDF (sdf <= relax_thresh), so no EDT is needed.

            print("检测标签边界...")
            sys.stdout.flush()
            boundary_mask = np.zeros_like(strict_mask, dtype=bool)
            for axis in range(3):
                for offset in [-1, 1]:
                    sl_c = [slice(None)] * 3
                    sl_n = [slice(None)] * 3
                    if offset == -1:
                        sl_c[axis] = slice(1, None)
                        sl_n[axis] = slice(0, -1)
                    else:
                        sl_c[axis] = slice(0, -1)
                        sl_n[axis] = slice(1, None)
                    center_lbl = strict_labels[tuple(sl_c)]
                    neighbor_lbl = strict_labels[tuple(sl_n)]
                    diff = ((center_lbl > 0) & (neighbor_lbl > 0) &
                            (center_lbl != neighbor_lbl))
                    boundary_mask[tuple(sl_c)] |= diff

            gc.collect()

            # Dilate boundary to bridge zone (relax_thresh mm → voxels)
            dilate_vox = max(1, int(np.ceil(relax_thresh / voxel_size)))
            print(f"膨胀边界 {dilate_vox} 体素以创建桥梁区域...")
            sys.stdout.flush()
            dil_struct = generate_binary_structure(3, 1)
            bridge_zone = binary_dilation(
                boundary_mask,
                structure=dil_struct,
                iterations=dilate_vox,
            )
            del boundary_mask, dil_struct
            gc.collect()

            # Bridge candidates: in relaxed zone AND near label boundary
            fill_mask = candidate_mask & bridge_zone
            del bridge_zone, candidate_mask
            gc.collect()

            # Validate: each bridge must touch at least 2 distinct white matter regions
            print("验证桥梁有效性...")
            sys.stdout.flush()
            bridge_labels, n_bridges = label(fill_mask, structure=struct)
            for bridge_id in range(1, n_bridges + 1):
                current_bridge = bridge_labels == bridge_id
                dilated_bridge = binary_dilation(
                    current_bridge, structure=struct, iterations=1
                )
                touching = np.unique(strict_labels[dilated_bridge])
                touching = touching[touching > 0]
                if len(touching) < 2:
                    fill_mask[current_bridge] = False
                else:
                    print(
                        f"发现纯桥梁：连接 {len(touching)} 个白质块，"
                        f"填补 {np.sum(current_bridge)} 个体素"
                    )
                    sys.stdout.flush()

            del bridge_labels
            gc.collect()
            print(f"总填补体素数：{np.sum(fill_mask)}")
            sys.stdout.flush()

        white_final = (strict_mask | fill_mask).astype(np.uint8)

    # ========== 2. 新增：合并pial mask逻辑 ==========
    if pial_sdf_path and Path(pial_sdf_path).exists():
        print(f"读取pial SDF: {pial_sdf_path}")
        pial_img = nib.load(pial_sdf_path)
        pial_data = np.asarray(pial_img.dataobj)
        if pial_data.shape != vol_shape:
            raise ValueError(f"pial尺寸和white不匹配：white={vol_shape}, pial={pial_data.shape}")

        # 生成pial mask
        pial_mask = pial_data < 0.0
        # 合并：先赋值pial，再覆盖white，实现white完全覆盖pial
        merged_vol = np.zeros(vol_shape, dtype=np.uint8)
        merged_vol[pial_mask] = pial_value
        merged_vol[white_final > 0] = white_value

        # 输出合并后的ribbon
        merged_output = Path(output_path)
        merged_img = nib.Nifti1Image(merged_vol, affine, header)
        merged_img.header.set_data_dtype(np.uint8)
        nib.save(merged_img, merged_output)
        print(f"合并结果已保存: {merged_output}")
        print(f"  白质区域灰度值: {white_value}, 灰质区域灰度值: {pial_value}")
    else:
        # ========== 3. 保存修复后的白质mask ==========
        output_img = nib.Nifti1Image(white_final, affine, header)
        output_img.header.set_data_dtype(np.uint8)
        nib.save(output_img, output_path)
        print(f"修复后白质mask已保存: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="精准修复白质细小断裂，可选合并pial mask")
    parser.add_argument("sdf_path", help="输入白质符号距离场文件(.nii.gz)")
    parser.add_argument("--relax_thresh", type=float, default=0.5, help="最大填补间隙(mm)，默认0.5")
    parser.add_argument("--output", default="white_mask_fixed.nii.gz", help="输出修复后白质mask路径")
    parser.add_argument("--min_size", type=int, default=0, help="忽略小于该值的孤立白质区域，默认0")
    parser.add_argument("--connectivity", type=int, default=6, choices=[6, 18, 26], help="连通性，默认6")
    parser.add_argument("--bridge_tol", type=float, default=0.1, help="桥梁容差(mm)，越小越精准，默认0.1")
    # 新增pial相关参数
    parser.add_argument("--pial_path", help="可选：输入pial符号距离场文件(.nii.gz)")
    parser.add_argument("--pial_value", type=int, default=3, help="pial区域灰度值，默认3")
    parser.add_argument("--white_value", type=int, default=2, help="white区域灰度值，默认2")

    args = parser.parse_args()
    fix_thin_white_matter_pure_bridge(
        sdf_path=args.sdf_path,
        relax_thresh=args.relax_thresh,
        output_path=args.output,
        min_region_size=args.min_size,
        connectivity=args.connectivity,
        bridge_tolerance=args.bridge_tol,
        pial_sdf_path=args.pial_path,
        pial_value=args.pial_value,
        white_value=args.white_value
    )