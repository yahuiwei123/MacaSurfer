#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import nibabel as nib


def load_mat(path: str) -> np.ndarray:
    m = np.loadtxt(path, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"Matrix at {path} is {m.shape}, expected 4x4.")
    return m


def save_mat(path: str, mat: np.ndarray):
    np.savetxt(path, mat, fmt="%.10f")


def pick_affine(img: nib.Nifti1Image, source: str) -> np.ndarray:
    """
    Choose which header affine to use:
      - 'auto' : nibabel img.affine (usually sform if present, else qform)
      - 'sform': sform only (must exist)
      - 'qform': qform only (must exist)
    """
    hdr = img.header
    if source == "auto":
        return np.array(img.affine, dtype=np.float64)

    if source == "sform":
        code = int(hdr["sform_code"])
        if code == 0:
            raise ValueError("Requested sform, but sform_code is 0 (invalid).")
        return np.array(img.get_sform(), dtype=np.float64)

    if source == "qform":
        code = int(hdr["qform_code"])
        if code == 0:
            raise ValueError("Requested qform, but qform_code is 0 (invalid).")
        return np.array(img.get_qform(), dtype=np.float64)

    raise ValueError(f"Unknown affine source: {source}")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compose a FLIRT matrix from ROI/LIA registration back to full-FOV original images.\n"
            "Given: rigid_reg.mat for (T2_reg -> T1_reg)\n"
            "Output: mat_full for (T2_full -> T1_full)\n"
            "Where reg images are your cropped+reoriented (e.g., LIA) versions."
        )
    )
    ap.add_argument("--t1-full", required=True, help="Full FOV T1 NIfTI (reference target in full space).")
    ap.add_argument("--t2-full", required=True, help="Full FOV T2 NIfTI (moving source in full space).")
    ap.add_argument("--t1-reg", required=True, help="T1 used in registration (cropped + LIA etc.).")
    ap.add_argument("--t2-reg", required=True, help="T2 used in registration (cropped + LIA etc.).")
    ap.add_argument("--rigid-reg", required=True, help="FLIRT rigid.mat computed from T2_reg to T1_reg.")
    ap.add_argument("--out-mat", required=True, help="Output FLIRT matrix: T2_full -> T1_full.")

    ap.add_argument(
        "--affine-source",
        choices=["auto", "sform", "qform"],
        default="auto",
        help="Which header affine to trust when building full<->reg mappings. Default: auto.",
    )
    ap.add_argument("--verbose", action="store_true", help="Print debug matrices and header codes.")
    args = ap.parse_args()

    # Import fslpy only when needed; fail with clear message
    try:
        from fsl.data.image import Image as FslImage
        from fsl.transform.flirt import toFlirt
    except Exception as e:
        print(
            "ERROR: Cannot import fslpy (module 'fsl').\n"
            "You need FSL's Python environment (fslpy). Try running under FSL's python, e.g.:\n"
            "  $FSLDIR/bin/python compose_flirt_rigid.py ...\n"
            f"Original import error: {e}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Load images (nibabel for affines; fslpy for FLIRT conversions)
    t1_full_img = nib.load(args.t1_full)
    t2_full_img = nib.load(args.t2_full)
    t1_reg_img  = nib.load(args.t1_reg)
    t2_reg_img  = nib.load(args.t2_reg)

    # Pick affines consistently
    A_t1_full = pick_affine(t1_full_img, args.affine_source)
    A_t2_full = pick_affine(t2_full_img, args.affine_source)
    A_t1_reg  = pick_affine(t1_reg_img,  args.affine_source)
    A_t2_reg  = pick_affine(t2_reg_img,  args.affine_source)

    if args.verbose:
        def codes(img):
            h = img.header
            return int(h["qform_code"]), int(h["sform_code"])
        print("Header codes (qform_code, sform_code):")
        print("  T1_full:", codes(t1_full_img))
        print("  T2_full:", codes(t2_full_img))
        print("  T1_reg :", codes(t1_reg_img))
        print("  T2_reg :", codes(t2_reg_img))

    # Build voxel-to-voxel mapping full -> reg using world affines:
    # world = A_full @ v_full
    # v_reg = inv(A_reg) @ world = inv(A_reg) @ A_full @ v_full
    vox_t1_full2reg = np.linalg.inv(A_t1_reg) @ A_t1_full
    vox_t2_full2reg = np.linalg.inv(A_t2_reg) @ A_t2_full

    # Convert voxel->voxel matrices to FLIRT coordinate convention
    fsl_t1_full = FslImage(args.t1_full)
    fsl_t1_reg  = FslImage(args.t1_reg)
    fsl_t2_full = FslImage(args.t2_full)
    fsl_t2_reg  = FslImage(args.t2_reg)

    # P1: T1_full -> T1_reg  (in FLIRT matrix convention)
    P1 = toFlirt(vox_t1_full2reg, fsl_t1_full, fsl_t1_reg, from_="voxel", to="voxel")
    # P2: T2_full -> T2_reg
    P2 = toFlirt(vox_t2_full2reg, fsl_t2_full, fsl_t2_reg, from_="voxel", to="voxel")

    # M_reg: T2_reg -> T1_reg  (your rigid.mat)
    M_reg = load_mat(args.rigid_reg)

    # Compose:
    # M_full = inv(P1) @ M_reg @ P2
    # so that: v_T1_full = M_full * v_T2_full   (in FLIRT coords)
    M_full = np.linalg.inv(P1) @ M_reg @ P2

    if args.verbose:
        np.set_printoptions(precision=6, suppress=True)
        print("\nP1 (T1_full -> T1_reg):\n", P1)
        print("\nP2 (T2_full -> T2_reg):\n", P2)
        print("\nM_reg (T2_reg -> T1_reg):\n", M_reg)
        print("\nM_full (T2_full -> T1_full):\n", M_full)

    save_mat(args.out_mat, M_full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
