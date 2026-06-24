import os
import shutil
import subprocess
import numpy as np
import nibabel as nib

# --------------------- 工具函数 ---------------------
def run_cmd(cmd, shell=False):
    """运行命令并打印日志"""
    cmd_str = " ".join(cmd) if not shell else cmd
    print("🔧 Running command:", cmd_str)
    subprocess.run(cmd, shell=shell, check=True)

def make_dirs(*paths):
    """创建多个目录，已存在时不报错"""
    for path in paths:
        os.makedirs(path, exist_ok=True)
        
def safe_move(src, dst):
    """移动文件前若目标存在则先删除"""
    if os.path.exists(dst):
        os.remove(dst)
    shutil.move(src, dst)
    
def safe_copy(src, dst):
    """复制文件前若目标存在则先删除"""
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copy(src, dst)

def correct_orientation(img_list: list, orientation: str, out_list: list):
    if len(img_list) != len(out_list):
        raise ValueError("length of img_list and out_list do not match!")
    
    for i, img in enumerate(img_list):
        run_cmd(["mri_convert", img, "--in_orientation", orientation, out_list[i]])
        
def get_orient(xform: np.ndarray = None) -> str:
    """
    Get orientation from xform matrix.

    Parameters
    ----------
    xform : np.ndarray
        3x3 linear part of a NIfTI affine, or a full 4x4 affine.

    Returns
    -------
    str
        Anatomical coordinates ('RAS', 'LIA', ...). Always a valid orient
        code: each anatomical axis appears exactly once.

    Notes
    -----
    Implemented via nibabel.aff2axcodes (-> io_orientation), which performs
    a global one-to-one assignment between voxel axes and anatomical axes.
    The earlier per-column np.argmax(|vec|) implementation could produce
    invalid codes like 'LIR' (LR axis represented twice, PA missing) when
    the input matrix was a non-axis-aligned rotation -- two columns might
    both be most-aligned with the same RAS axis. mri_convert
    --in_orientation rejects such codes with a hard error, which is the
    bug this fix resolves.
    """
    if xform.shape == (3, 3):
        aff = np.eye(4)
        aff[:3, :3] = xform
    else:
        aff = xform
    return "".join(nib.aff2axcodes(aff))


_SIGN_FLIP = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}


def _fsl_vox_to_fsl_mm(img: nib.spatialimages.SpatialImage) -> np.ndarray:
    """Return FSL's voxel-index -> scaled-mm coordinate transform."""
    if len(img.shape) < 3:
        raise ValueError(f"expected at least 3 image dimensions, got {img.shape}")

    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    if zooms.shape != (3,) or np.any(zooms <= 0):
        raise ValueError(f"invalid voxel sizes: {zooms}")

    vox_to_mm = np.diag([zooms[0], zooms[1], zooms[2], 1.0])

    # FLIRT uses a scaled-voxel mm coordinate system. For images with positive
    # voxel-to-world determinant, FSL inserts an x flip to maintain its internal
    # radiological convention.
    if np.linalg.det(img.affine[:3, :3]) > 0:
        nx = img.shape[0]
        flip = np.eye(4)
        flip[0, 0] = -1.0
        flip[0, 3] = (nx - 1) * zooms[0]
        vox_to_mm = flip @ vox_to_mm

    return vox_to_mm


def _flirt_mat_to_world_transform(
    mov_img: nib.spatialimages.SpatialImage,
    ref_img: nib.spatialimages.SpatialImage,
    flirt_mat: np.ndarray,
) -> np.ndarray:
    """Convert a FLIRT matrix into an RAS/world-space transform."""
    if flirt_mat.shape != (4, 4):
        raise ValueError(f"FLIRT matrix must be 4x4, got {flirt_mat.shape}")

    mov_vox_to_fsl = _fsl_vox_to_fsl_mm(mov_img)
    ref_vox_to_fsl = _fsl_vox_to_fsl_mm(ref_img)

    return (
        ref_img.affine
        @ np.linalg.inv(ref_vox_to_fsl)
        @ flirt_mat
        @ mov_vox_to_fsl
        @ np.linalg.inv(mov_img.affine)
    )


def _apply_flirt_header_only(mov: str, ref: str, flirt_mat: str, out: str):
    """Apply a FLIRT transform by changing only the NIfTI affine/header."""
    mov_img = nib.load(mov)
    ref_img = nib.load(ref)
    mat = np.loadtxt(flirt_mat)

    world_xform = _flirt_mat_to_world_transform(mov_img, ref_img, mat)
    new_affine = world_xform @ mov_img.affine

    data = np.asanyarray(mov_img.dataobj)
    header = mov_img.header.copy()
    out_img = nib.Nifti1Image(data, new_affine, header=header)
    out_img.set_qform(new_affine, code=1)
    out_img.set_sform(new_affine, code=1)

    if out_img.shape != mov_img.shape:
        raise RuntimeError(
            f"header-only FLIRT changed shape from {mov_img.shape} to {out_img.shape}"
        )

    nib.save(out_img, out)


def _affine_det_sign(path: str) -> int:
    """Return sign(det(affine[:3,:3])) of a NIfTI file."""
    return int(np.sign(np.linalg.det(nib.load(path).affine[:3, :3])))


def _flip_i_axis(in_path: str, out_path: str):
    """Physically flip the i voxel axis of a NIfTI (data + affine), to invert
    the sign of det(affine).

    We do this in nibabel rather than via `fslswapdim -x y z`. FSL's
    fslswapdim by default REFUSES handedness-changing swaps: it silently
    applies a compensating flip to preserve the input's neurological/
    radiological convention, leaving the output's affine and data identical
    to the input (only a "WARNING:: Flipping Left/Right orientation
    (as det < 0)" is printed). That defeats the purpose of preflight
    chirality alignment.
    """
    img = nib.load(in_path)
    data = np.asanyarray(img.dataobj)
    nx = data.shape[0]
    new_data = data[::-1, :, :].copy()
    A_old = img.affine
    A_new = A_old.copy()
    A_new[:, 0] = -A_old[:, 0]
    A_new[:, 3] = A_old[:, 3] + A_old[:, 0] * (nx - 1)
    new_img = nib.Nifti1Image(new_data, A_new, header=img.header)
    new_img.set_qform(A_new)
    new_img.set_sform(A_new)
    nib.save(new_img, out_path)


def get_correct_orientation(mov: str, ref: str, workdir: str):
    # ------------------ Reorientation 处理 ------------------
    make_dirs(workdir)

    # ---- Step 0: preflight chirality alignment ----
    # `flirt -dof 6` is rigid (det = +1) so it cannot resolve a chirality
    # mismatch between mov and ref. If the two affines have opposite det
    # signs, the FSL/FreeSurfer pipeline downstream can produce subtle L/R
    # flips that are very hard to undo at the orient-code level (the upstream
    # version of this function tried to patch this by string-replace on the
    # final orient letters, which is mathematically equivalent to negating
    # a single affine column and therefore IS itself a mirror reflection --
    # the very thing it was trying to avoid). Instead, physically flip mov's
    # i-axis (data + affine) up front so both inputs share the same
    # chirality bucket. The flip is undone analytically on the final orient
    # code below.
    mov_det = _affine_det_sign(mov)
    ref_det = _affine_det_sign(ref)
    chirality_flipped = (mov_det != ref_det)
    if chirality_flipped:
        mov_pipeline = os.path.join(workdir, "mov_chirality_matched.nii.gz")
        _flip_i_axis(mov, mov_pipeline)
        new_det = _affine_det_sign(mov_pipeline)
        if new_det != ref_det:
            raise RuntimeError(
                f"preflight chirality flip failed: det={new_det:+d} still "
                f"differs from ref det={ref_det:+d}"
            )
        print(f"[INFO] chirality mismatch (mov det={mov_det:+d}, "
              f"ref det={ref_det:+d}); flipped i-axis -> {mov_pipeline}")
    else:
        mov_pipeline = mov
        print(f"[INFO] chirality match (det={mov_det:+d}); no preflight flip")

    # 对 T1 模板影像做下采样
    template_down = os.path.join(workdir, "t1_template_06mm.nii.gz")
    run_cmd(["flirt", "-in", ref, "-ref", ref, "-out", template_down, "-applyisoxfm", "0.6"])

    # 对 brain 图像做下采样（chirality 不匹配时用翻转后的版本）
    brain_down = os.path.join(workdir, "mov_06mm.nii.gz")
    run_cmd(["flirt", "-in", mov_pipeline, "-ref", mov_pipeline, "-out", brain_down, "-applyisoxfm", "0.6"])

    # 执行空间对齐获取旋转矩阵
    reoriented_img = os.path.join(workdir, "reo.nii.gz")
    reorient_mat   = os.path.join(workdir, "reo.mat")
    run_cmd([
        "flirt", "-in", brain_down, "-ref", template_down,
        "-out", reoriented_img, "-omat", reorient_mat,
        "-dof", "6", "-searchrx", "-180", "180",
        "-cost", "corratio",
        "-searchcost", "corratio",
        "-searchry", "-180", "180", "-searchrz", "-180", "180",
        "-coarsesearch", "20", "-finesearch", "10"
    ])
    _apply_flirt_header_only(brain_down, template_down, reorient_mat, reoriented_img)

    # 获取旋转方向
    reo = nib.load(reoriented_img)
    real_orient = get_orient(reo.affine[:3, :3])

    # ---- Step 5: undo the preflight i-axis flip on the orient code ----
    # `real_orient` describes mov_pipeline's voxel grid -> RAS mapping. When
    # we did a chirality flip, mov_pipeline's voxel axis 0 is the FLIPPED
    # axis vs the original mov, so the original mov's voxel axis 0 points in
    # the opposite RAS direction. Sign-flip the first letter so the returned
    # code can be applied to the ORIGINAL mov (which is what `mri_convert
    # --in_orientation` will be called on by the caller).
    if chirality_flipped:
        real_orient = _SIGN_FLIP[real_orient[0]] + real_orient[1:]

    return real_orient