#!/usr/bin/env python3
"""
Tissue segmentation via macaBrainNet predict_ensemble.py with --freesurfer.
Replaces nBEST / TissueSeg.sh in the MacaSurfer pipeline.

Outputs (in output_dir = enhance_dir/T1w/):
  {prefix}_desc-freesurfer_dseg.nii.gz       – FS-label aseg (from macaBrainNet)
  {prefix}_desc-nbest_dseg.nii.gz            – 4-class nbest (derived from FS)
  {prefix}_label-cerebrum_dseg.nii.gz         – cerebrum mask
  {prefix}_label-cerebellum-brainstem_dseg.nii.gz – cerebellum+brainstem mask
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import nibabel as nib


# ── FreeSurfer → nbest (4-class) mapping ──
# nbest: 0=BG, 1=CSF, 2=GM, 3=WM, 4=Cerebellum
# Build a lookup table covering all FS labels 0..255

def _build_fs_to_nbest():
    """Build FS label → nbest mapping array (size 256)."""
    lut = np.zeros(256, dtype=np.uint8)
    # CSF = 1
    csf_labels = {4, 43, 24}  # Lat-Vent (L/R), CSF
    for lbl in csf_labels:
        lut[lbl] = 1
    # GM = 2
    gm_labels = {3, 17, 18, 42}  # Cerebral Cortex (L/R)
    for lbl in gm_labels:
        lut[lbl] = 2
    # Cerebellum = 4
    cereb_labels = {7, 8, 46, 47}  # Cereb-WM/Cortex (L/R)
    for lbl in cereb_labels:
        lut[lbl] = 4
    # Brainstem = 5
    brainstem_labels = {16, 27}  # Brain-Stem
    for lbl in cereb_labels:
        lut[lbl] = 5
    # WM = 3 (cerebral WM + all subcortical)
    wm_labels = {2, 41, 10, 11, 12, 13, 26, 28,
                 49, 50, 51, 52, 53, 54, 58, 59, 60, 138, 139}
    for lbl in wm_labels:
        lut[lbl] = 3
    return lut

_FS_TO_NBEST = _build_fs_to_nbest()

# Cerebellum+brainstem FS labels (for mask generation)
_CEREB_FS = {7, 8, 46, 47, 16}

# Left → Right FreeSurfer label remapping
_FS_LEFT_TO_RIGHT = {
    2: 41, 3: 42, 4: 43,
    7: 46, 8: 47,
    10: 49, 11: 50, 12: 51, 13: 52,
    17: 53, 18: 54,
    26: 58, 27: 59, 28: 60,
    138: 139,
}
# Right → Left reverse mapping (for merged/hemi-agnostic aseg)
_FS_RIGHT_TO_LEFT = {v: k for k, v in _FS_LEFT_TO_RIGHT.items()}

# Midline labels (present in both hemispheres, no remapping needed)
_FS_MIDLINE = {16, 24, 140}

# Labels that can appear in both hemispheres (for building the complete aseg)
_FS_LEFT_HEMI_LABELS = set(_FS_LEFT_TO_RIGHT.keys()) | _FS_MIDLINE


def _merge_hemispheres(bilateral_data):
    """Merge left/right hemisphere labels into a single set.

    Remaps all right-hemisphere FS labels back to their left-hemisphere
    equivalents (e.g. 41→2, 42→3), producing a hemi-agnostic aseg
    suitable for bias field correction.

    Args:
        bilateral_data: 3D numpy array with bilateral FS labels

    Returns:
        merged: 3D numpy array with left-side-only FS labels
    """
    merged = bilateral_data.copy().astype(np.uint16)
    for right_lbl, left_lbl in _FS_RIGHT_TO_LEFT.items():
        merged[bilateral_data == right_lbl] = left_lbl
    return merged

def _generate_hemi_masks(brain_nifti_path, tmp_dir):
    """Generate left/right hemisphere masks using FreeSurfer make_hemi_mask.

    Uses make_hemi_mask to find the mid-sagittal plane accurately,
    then generates left and right hemisphere masks.

    Args:
        brain_nifti_path: Path to brain NIfTI file
        tmp_dir: Temporary directory for intermediate files

    Returns:
        left_mask, right_mask: boolean numpy arrays
    """
    import tempfile
    import nibabel as nib

    left_path = os.path.join(tmp_dir, 'Left-Hemi.nii.gz')
    right_path = os.path.join(tmp_dir, 'Right-Hemi.nii.gz')

    subprocess.run(
        ['make_hemi_mask', 'lh', brain_nifti_path, left_path],
        check=True, capture_output=True)
    subprocess.run(
        ['make_hemi_mask', 'rh', brain_nifti_path, right_path],
        check=True, capture_output=True)

    left_img = nib.load(left_path)
    right_img = nib.load(right_path)
    left_mask = np.asarray(left_img.dataobj) > 0
    right_mask = np.asarray(right_img.dataobj) > 0

    return left_mask, right_mask


def _build_bilateral_aseg(fs_data, brain_nifti_path, tmp_dir):
    """Remap left-hemisphere FS labels to bilateral labels.

    The macaBrainNet FS aseg contains only left-hemisphere labels.
    Voxels in the right hemisphere get remapped to their corresponding
    right-hemisphere FS labels.

    Args:
        fs_data: 3D numpy array of FS labels (left-hemisphere only)
        brain_nifti_path: Path to brain NIfTI (for make_hemi_mask)
        tmp_dir: Temporary directory

    Returns:
        bilateral: 3D numpy array with bilateral FS labels
    """
    left_mask, right_mask = _generate_hemi_masks(brain_nifti_path, tmp_dir)
    bilateral = fs_data.copy().astype(np.uint16)

    # For right hemisphere, remap left → right labels
    for left_lbl, right_lbl in _FS_LEFT_TO_RIGHT.items():
        condition = (fs_data == left_lbl) & right_mask
        bilateral[condition] = right_lbl

    # Midline labels are preserved in both hemispheres
    # No action needed — they're already in fs_data and get copied as-is

    return bilateral


def main():
    parser = argparse.ArgumentParser(
        description="BrainNet tissue segmentation for MacaSurfer")
    parser.add_argument("--input", required=True,
                        help="Input T1w NIfTI (t1w_init_corrected)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory (enhance_dir/T1w/)")
    parser.add_argument("--prefix", required=True,
                        help="BIDS prefix (e.g. sub-xxx_ses-xxx)")
    parser.add_argument("--model-dir", required=True,
                        help="macaBrainNet tissue_segmentation model dir")
    parser.add_argument("--predict-script", required=True,
                        help="Path to predict_ensemble.py")
    parser.add_argument("--python-inter", default="python3",
                        help="Python interpreter")
    parser.add_argument("--utils-path", required=True,
                        help="Path to shared/utils (for conform.py)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Create identity matrix (needed by downstream BiasFieldCorrect.sh) ──
    xfm_dir = os.path.join(args.output_dir, "xfms")
    os.makedirs(xfm_dir, exist_ok=True)
    identity_mat = os.path.join(xfm_dir, "identity.mat")
    with open(identity_mat, 'w') as f:
        f.write("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    # ── Output paths ──
    fs_out = os.path.join(args.output_dir,
                          f"{args.prefix}_desc-freesurfer_dseg.nii.gz")
    nbest_out = os.path.join(args.output_dir,
                             f"{args.prefix}_desc-nbest_dseg.nii.gz")
    cerebrum_out = os.path.join(args.output_dir,
                                f"{args.prefix}_label-cerebrum_dseg.nii.gz")
    cereb_out = os.path.join(
        args.output_dir,
        f"{args.prefix}_label-cerebellum-brainstem_dseg.nii.gz")

    # ── Step 1: Tissue segmentation via macaBrainNet ──
    print("=" * 60)
    print("Step 1: Tissue segmentation (predict_ensemble, 19-class, freesurfer)")
    print("=" * 60)

    cmd = [
        args.python_inter, args.predict_script,
        '--img', args.input,
        '--out', fs_out,
        '--ckpt-dir', args.model_dir,
        '--num-classes', '19',
        '--spacing', '0.4', '0.4', '0.4',
        '--freesurfer',
        '--device', args.device,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # ── Step 2: Load FS output ──
    print("=" * 60)
    print("Step 2: Load FS segmentation, map to nbest + masks")
    print("=" * 60)

    fs_img = nib.load(fs_out)
    fs_data = np.asarray(fs_img.dataobj, dtype=np.uint16)

    # ── Step 3: FS → nbest mapping ──
    nbest_data = _FS_TO_NBEST[fs_data]
    nbest_labels = np.unique(nbest_data)
    print(f"nbest unique labels: {nbest_labels}")

    # ── Step 4: Derive masks ──
    cereb_data = np.isin(fs_data, list(_CEREB_FS)).astype(np.uint8)
    cerebrum_data = ((fs_data > 0) & ~np.isin(fs_data, list(_CEREB_FS))).astype(np.uint8)

    # ── Step 5: Reorient to LIA ──
    print("Step 5: Reorient to LIA (matching TissueSeg.sh convention)")
    print("=" * 60)

    conform_py = os.path.join(args.utils_path, "conform.py")
    # Reorient FS output
    subprocess.run([
        args.python_inter, conform_py,
        '--input', fs_out,
        '--output', fs_out,
        '--reorient', 'LIA',
    ], check=True)
    # Save nbest
    nib.save(nib.Nifti1Image(nbest_data, fs_img.affine), nbest_out)
    subprocess.run([
        args.python_inter, conform_py,
        '--input', nbest_out,
        '--output', nbest_out,
        '--reorient', 'LIA',
    ], check=True)
    # Save cerebrum
    nib.save(nib.Nifti1Image(cerebrum_data, fs_img.affine), cerebrum_out)
    subprocess.run([
        args.python_inter, conform_py,
        '--input', cerebrum_out,
        '--output', cerebrum_out,
        '--reorient', 'LIA',
    ], check=True)
    # Save cerebellum-brainstem
    nib.save(nib.Nifti1Image(cereb_data, fs_img.affine), cereb_out)
    subprocess.run([
        args.python_inter, conform_py,
        '--input', cereb_out,
        '--output', cereb_out,
        '--reorient', 'LIA',
    ], check=True)

    # ── Step 6: Build bilateral complete aseg ──
    print("=" * 60)
    print("Step 6: Build bilateral complete aseg (hemisphere remapping)")
    print("=" * 60)

    complete_out = os.path.join(args.output_dir,
                                f"{args.prefix}_desc-completeaseg_dseg.nii.gz")
    merged_out = os.path.join(args.output_dir,
                              f"{args.prefix}_desc-completeaseg-merged_dseg.nii.gz")

    # Reload reoriented FS aseg
    fs_img_lia = nib.load(fs_out)
    fs_data_lia = np.asarray(fs_img_lia.dataobj, dtype=np.uint16)

    # Use make_hemi_mask on the T1w brain for accurate hemisphere splitting
    tmp_dir = os.path.join(args.output_dir, 'tmp_hemi')
    os.makedirs(tmp_dir, exist_ok=True)

    bilateral = _build_bilateral_aseg(fs_data_lia, args.input, tmp_dir)
    nib.save(nib.Nifti1Image(bilateral, fs_img_lia.affine), complete_out)
    subprocess.run([
        args.python_inter, conform_py,
        '--input', complete_out,
        '--output', complete_out,
        '--reorient', 'LIA',
    ], check=True)

    # Save merged (hemi-agnostic) version for bias field correction
    merged = _merge_hemispheres(bilateral)
    nib.save(nib.Nifti1Image(merged, fs_img_lia.affine), merged_out)
    subprocess.run([
        args.python_inter, conform_py,
        '--input', merged_out,
        '--output', merged_out,
        '--reorient', 'LIA',
    ], check=True)

    # Clean up temp hemisphere files
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"FreeSurfer aseg:  {fs_out}")
    print(f"nbest (4-class):  {nbest_out}")
    print(f"cerebrum mask:    {cerebrum_out}")
    print(f"cerebellum mask:  {cereb_out}")
    print(f"complete aseg:    {complete_out}")
    print(f"merged aseg:      {merged_out}")
    print("BrainNet tissue segmentation done.")


if __name__ == "__main__":
    main()
