#!/usr/bin/env python3
import argparse
import numpy as np
import nibabel as nib
from itertools import product


def voxel_sizes(aff):
    # nibabel.affines.voxel_sizes 的等价实现，避免额外 import
    A = aff[:3, :3]
    return np.sqrt((A * A).sum(axis=0))


def is_perm_flip_matrix(R, tol=1e-4):
    """
    检查 3x3 矩阵是否接近“置换 + 翻转”（每列只有一个元素接近 ±1，其余接近 0）
    返回:
      ok, full_axis_of_crop_axis (len=3), flip_of_crop_axis (len=3)
    """
    full_axis_of_crop = [-1, -1, -1]
    flip_of_crop = [0, 0, 0]
    used_full_axes = set()

    for c in range(3):  # crop axis
        col = R[:, c]
        a = int(np.argmax(np.abs(col)))
        if np.abs(col[a]) < 1 - tol:
            return False, None, None

        # 其余元素必须接近 0
        for r in range(3):
            if r == a:
                continue
            if np.abs(col[r]) > tol:
                return False, None, None

        if a in used_full_axes:
            return False, None, None
        used_full_axes.add(a)

        full_axis_of_crop[c] = a
        flip_of_crop[c] = 1 if col[a] > 0 else -1

    return True, full_axis_of_crop, flip_of_crop


def pad_roi_to_full(crop_img, full_img, cval=0, tol=1e-4, verbose=False):
    A_full = np.asarray(full_img.affine, dtype=np.float64)
    A_crop = np.asarray(crop_img.affine, dtype=np.float64)

    crop_data = np.asanyarray(crop_img.dataobj)
    crop_shape_xyz = crop_data.shape[:3]
    extra_shape = crop_data.shape[3:]  # 允许 4D/5D：只 pad 前3维

    full_shape_xyz = full_img.shape[:3]

    # 先做一个必要的“同分辨率”检查（纯 padding 的前提）
    z_full = voxel_sizes(A_full)
    z_crop = voxel_sizes(A_crop)
    # if not np.allclose(z_full, z_crop, atol=tol, rtol=0):
    #     raise ValueError(
    #         f"Voxel sizes differ, cannot do pure padding.\n"
    #         f"full zooms={z_full}, crop zooms={z_crop}\n"
    #         f"Use resampling if you really need to support this."
    #     )

    # crop voxel -> full voxel 的仿射
    # v_full = inv(A_full) @ A_crop @ v_crop
    T = np.linalg.inv(A_full) @ A_crop
    R = T[:3, :3]
    t = T[:3, 3]

    ok, full_axis_of_crop, flip_of_crop = is_perm_flip_matrix(R, tol=tol)
    if not ok:
        raise ValueError(
            "Affine implies rotation/oblique/shear (not pure axis perm+flip). "
            "Pure padding (no resample) is not valid for this case."
        )

    # 检查平移是否接近整数（否则说明不在同一体素网格上，纯 padding 会错位）
    if not np.allclose(t, np.round(t), atol=tol, rtol=0):
        raise ValueError(
            f"Translation in voxel space is not near-integer: t={t}. "
            "This suggests different grids; pure padding is not safe."
        )

    # 构造：把 crop 数据重排到 full 的轴顺序
    # full_axis_of_crop[c] = a  表示 crop轴c 对齐到 full轴a
    # transpose 需要 full轴a 从哪个 crop轴来：perm[a] = c
    perm = [None, None, None]
    flip_full = [0, 0, 0]
    for c in range(3):
        a = full_axis_of_crop[c]
        perm[a] = c
        flip_full[a] = flip_of_crop[c]

    if any(p is None for p in perm):
        raise RuntimeError("Internal error: invalid axis mapping.")

    # 对数据做 transpose（把前三维变为 full 轴顺序），其余维度保持不动
    axes = perm + list(range(3, crop_data.ndim))
    data_full_axes = np.transpose(crop_data, axes=axes)

    # 再根据 flip_full 做翻转
    for a in range(3):
        if flip_full[a] < 0:
            data_full_axes = np.flip(data_full_axes, axis=a)

    # 用 8 个角点求在 full voxel 中的包围盒（更稳健，避免 “(0,0,0) 不是最小角” 的问题）
    nx, ny, nz = crop_shape_xyz
    corners = np.array(list(product([0, nx - 1], [0, ny - 1], [0, nz - 1])), dtype=np.float64)
    corners_h = np.concatenate([corners, np.ones((8, 1))], axis=1)  # 8x4
    full_corners = (T @ corners_h.T).T[:, :3]  # 8x3

    full_corners_round = np.round(full_corners).astype(int)
    if not np.allclose(full_corners, full_corners_round, atol=tol, rtol=0):
        raise ValueError(
            "Corner mapping not near-integer; grids are not identical. "
            "Pure padding is not safe."
        )

    start = full_corners_round.min(axis=0)
    end = full_corners_round.max(axis=0) + 1  # python slice end

    # 理论上 end-start 应该等于 data_full_axes.shape[:3]
    expected = np.array(data_full_axes.shape[:3], dtype=int)
    got = end - start
    if not np.array_equal(got, expected):
        raise ValueError(
            f"Extent mismatch after axis handling: expected {expected}, got {got}. "
            "This usually means the crop image affine does not correctly reflect the crop."
        )

    # 安全裁剪插入（允许 start/end 轻微越界时仍可运行）
    out_shape = tuple(full_shape_xyz) + extra_shape
    out = np.full(out_shape, cval, dtype=data_full_axes.dtype)

    src_slices = []
    dst_slices = []
    for a in range(3):
        s = int(start[a])
        e = int(end[a])
        ds = max(s, 0)
        de = min(e, full_shape_xyz[a])
        if ds >= de:
            raise ValueError(f"ROI is outside full FOV on axis {a}: start={s}, end={e}, full={full_shape_xyz[a]}")

        ss = ds - s
        se = ss + (de - ds)
        dst_slices.append(slice(ds, de))
        src_slices.append(slice(ss, se))

    # extra dims 全部取全
    for _ in range(len(extra_shape)):
        dst_slices.append(slice(None))
        src_slices.append(slice(None))

    if verbose:
        print("A_full:\n", A_full)
        print("A_crop:\n", A_crop)
        print("T (crop voxel -> full voxel):\n", T)
        print("perm (full axis <- crop axis):", perm)
        print("flip_full:", flip_full)
        print("start,end:", start, end)
        print("dst_slices:", dst_slices)
        print("src_slices:", src_slices)

    out[tuple(dst_slices)] = data_full_axes[tuple(src_slices)]
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Pure padding: place a cropped ROI NIfTI back into a full-size NIfTI grid using affine mapping (no resampling)."
    )
    ap.add_argument("--cropped", required=True, help="Cropped ROI NIfTI (must be aligned to full in world space).")
    ap.add_argument("--full", required=True, help="Original full-FOV NIfTI (defines output shape+affine).")
    ap.add_argument("--output", required=True, help="Output padded NIfTI (full size, full affine).")
    ap.add_argument("--cval", type=float, default=0.0, help="Constant value for padding background (default 0).")
    ap.add_argument("--tol", type=float, default=1e-1, help="Tolerance for integer/perm checks (default 1e-1).")
    ap.add_argument("--verbose", action="store_true", help="Print debug info.")
    args = ap.parse_args()

    crop_img = nib.load(args.cropped)
    full_img = nib.load(args.full)

    padded = pad_roi_to_full(crop_img, full_img, cval=args.cval, tol=args.tol, verbose=args.verbose)

    # 输出用 full 的 affine；header 也以 full 为基准
    out_hdr = full_img.header.copy()
    out_hdr.set_data_dtype(padded.dtype)
    out_img = nib.Nifti1Image(padded, full_img.affine, header=out_hdr)
    nib.save(out_img, args.output)


if __name__ == "__main__":
    main()
