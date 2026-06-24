#!/usr/bin/env python3
"""
Legacy nBEST tissue segmentation + subcortical fusion for MacaSurfer.
Replaces the old TissueSeg.sh bash script.

Does two things in one step:
  1. Run nBEST model → 4-class tissue segmentation (CSF=1, GM=2, WM=3, Cerebellum=4)
  2. Fuse subcortical structures from template aseg into nbest tissue → complete aseg

Outputs (in output_dir = enhance_dir/T1w/):
  {prefix}_desc-completeaseg_dseg.nii.gz      – bilateral FS-label aseg (fusion result)
  {prefix}_desc-nbest_dseg.nii.gz             – 4-class nbest
  {prefix}_label-cerebrum_dseg.nii.gz         – cerebrum mask
  {prefix}_label-cerebellum-brainstem_dseg.nii.gz – cerebellum+brainstem mask
"""

import os
import sys
import argparse
import subprocess
import shutil
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from skimage import morphology
from skimage import measure


# ── FreeSurfer label sets ──
# Left → Right label remapping
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

# Midline labels (present in both hemispheres)
_FS_MIDLINE = {16, 24, 140}

# nbest → left FS labels for GM/WM
_NBEST_GM_LEFT = 3
_NBEST_GM_RIGHT = 42
_NBEST_WM_LEFT = 2
_NBEST_WM_RIGHT = 41
_NBEST_CSF = 1
_NBEST_CEREB = 4


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
    os.makedirs(tmp_dir, exist_ok=True)
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


def _merge_hemispheres(bilateral_data):
    """Merge left/right hemisphere labels into a single set.

    Remaps all right-hemisphere FS labels back to their left-hemisphere
    equivalents (e.g. 41→2, 42→3), producing a hemi-agnostic aseg
    suitable for bias field correction.
    """
    merged = bilateral_data.copy().astype(np.uint16)
    for right_lbl, left_lbl in _FS_RIGHT_TO_LEFT.items():
        merged[bilateral_data == right_lbl] = left_lbl
    return merged


def _run_nbest(args):
    """Run nBEST model inference. Returns (workdir, prefix)."""
    nbest_root = os.path.dirname(os.path.dirname(
        os.path.abspath(args.nbest_model_path)))
    if not nbest_root:
        nbest_root = args.nbest_model_path
    scripts_dir = os.path.join(args.nbest_model_path, 'scripts')

    workdir = os.path.join(args.output_dir, 'nbest_work')
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    os.makedirs(os.path.join(workdir, 'brain_img'))
    os.makedirs(os.path.join(workdir, 'brain_mask'))

    conform_py = os.path.join(args.utils_path, 'conform.py')

    # Prepare images: reorient to LAS (required by nBEST)
    img_las = os.path.join(workdir, 'brain_img', 'T1w_conform_0000.nii.gz')
    subprocess.run([
        args.python_inter, conform_py,
        '--input', args.input,
        '--output', img_las,
        '--reorient', 'LAS',
    ], check=True)

    mask_las = os.path.join(workdir, 'brain_mask', 'T1w_conform.nii.gz')
    subprocess.run([
        args.python_inter, conform_py,
        '--input', args.mask,
        '--output', mask_las,
        '--reorient', 'LAS',
    ], check=True)

    # Run nBEST tissue segmentation
    print("Running nBEST tissue segmentation...")
    nbest_script = os.path.join(scripts_dir, 'nBEST_tissue.py')
    nbest_pythonpath = f"PYTHONPATH={nbest_root}:$PYTHONPATH"
    env = os.environ.copy()
    env['PYTHONPATH'] = nbest_root + ':' + env.get('PYTHONPATH', '')

    subprocess.run([
        args.python_inter, nbest_script,
        '--python_env', args.python_env,
        '--workdir', workdir,
    ], check=True, env=env)

    return workdir


def _fuse_nbest_aseg(workdir, args):
    """Load nbest results and template aseg, fuse into bilateral complete aseg."""
    nbest_tissue_path = os.path.join(
        workdir, 'brain_tissue', 'T1w_conform.nii.gz')
    cereb_mask_path = os.path.join(
        workdir, 'brain_cerebellum_brainstem_mask', 'T1w_conform.nii.gz')
    cerebrum_mask_path = os.path.join(
        workdir, 'brain_cerebrum_mask', 'T1w_conform.nii.gz')

    # Load nbest outputs (LAS orientation)
    nbest_img = nib.load(nbest_tissue_path)
    nbest_data = np.asarray(nbest_img.dataobj, dtype=np.uint8)

    cereb_img = nib.load(cereb_mask_path)
    cereb_data = np.asarray(cereb_img.dataobj, dtype=np.uint8)

    cerebrum_img = nib.load(cerebrum_mask_path)
    cerebrum_data = np.asarray(cerebrum_img.dataobj, dtype=np.uint8)

    # Load T1w brain (LAS) for hemisphere mask
    t1w_las_path = os.path.join(workdir, 'brain_img', 'T1w_conform_0000.nii.gz')
    t1w_img = nib.load(t1w_las_path)
    t1w_data = np.asarray(t1w_img.dataobj)

    tmp_hemi_dir = os.path.join(workdir, 'tmp_hemi')
    left_mask, right_mask = _generate_hemi_masks(t1w_las_path, tmp_hemi_dir)

    # Load template aseg
    aseg_img = nib.load(args.template_aseg)
    aseg_data = np.asarray(aseg_img.dataobj, dtype=np.uint16)

    print(f"nbest unique labels: {np.unique(nbest_data)}")
    print(f"template aseg unique labels: {np.unique(aseg_data)}")

    # ── Build bilateral complete aseg ──
    # Start with template aseg, remove GM/WM (they will be replaced by nbest)
    complete = aseg_data.copy().astype(np.uint16)
    complete[complete == _NBEST_GM_LEFT] = 0   # Left GM
    complete[complete == _NBEST_GM_RIGHT] = 0  # Right GM
    complete[complete == _NBEST_WM_LEFT] = 0   # Left WM
    complete[complete == _NBEST_WM_RIGHT] = 0  # Right WM

    # Overlay nbest GM
    gm_mask = nbest_data == 2  # nbest GM
    complete[gm_mask & left_mask] = _NBEST_GM_LEFT
    complete[gm_mask & right_mask] = _NBEST_GM_RIGHT

    # Overlay nbest WM
    wm_mask = nbest_data == 3  # nbest WM
    complete[wm_mask & left_mask] = _NBEST_WM_LEFT
    complete[wm_mask & right_mask] = _NBEST_WM_RIGHT

    # Overlay nbest CSF (overrides any template labels where nbest says CSF)
    csf_mask = nbest_data == 1
    complete[csf_mask] = 24

    # Cerebellum+brainstem: keep template aseg labels
    # nbest label 4 includes cerebellum WM/cortex and brainstem,
    # but template aseg has finer labels (7,46,8,47,16) — prefer template

    # Fill remaining brain voxels (not assigned any label) with CSF
    brain_mask = t1w_data > 0
    unassigned = brain_mask & (complete == 0)
    complete[unassigned] = 24

    print(f"complete aseg unique labels: {np.unique(complete)}")
    return complete, nbest_data, cereb_data, cerebrum_data, nbest_img.affine


def main():
    parser = argparse.ArgumentParser(
        description="nBEST tissue segmentation + subcortical fusion for MacaSurfer")
    parser.add_argument("--input", required=True,
                        help="Input T1w NIfTI (t1w_init_corrected)")
    parser.add_argument("--mask", required=True,
                        help="Brain mask (conform mask)")
    parser.add_argument("--template-aseg", required=True,
                        help="Template-based aseg from init_template_register")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory (enhance_dir/T1w/)")
    parser.add_argument("--prefix", required=True,
                        help="BIDS prefix (e.g. sub-xxx_ses-xxx)")
    parser.add_argument("--nbest-model-path", required=True,
                        help="Path to nBEST model directory")
    parser.add_argument("--python-env", required=True,
                        help="Python environment path (for nnUNet)")
    parser.add_argument("--python-inter", default="python3",
                        help="Python interpreter")
    parser.add_argument("--utils-path", required=True,
                        help="Path to shared/utils (for conform.py)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Create identity matrix (needed by downstream BiasFieldCorrect.sh)
    xfm_dir = os.path.join(args.output_dir, "xfms")
    os.makedirs(xfm_dir, exist_ok=True)
    identity_mat = os.path.join(xfm_dir, "identity.mat")
    with open(identity_mat, 'w') as f:
        f.write("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    conform_py = os.path.join(args.utils_path, 'conform.py')

    # ── Step 1: Run nBEST inference ──
    print("=" * 60)
    print("Step 1: nBEST tissue segmentation")
    print("=" * 60)
    workdir = _run_nbest(args)

    # ── Step 2: Fuse nbest + template aseg → complete aseg ──
    print("=" * 60)
    print("Step 2: Fuse nbest + template aseg → complete aseg")
    print("=" * 60)
    complete_data, nbest_data, cereb_data, cerebrum_data, las_affine = \
        _fuse_nbest_aseg(workdir, args)

    # ── Step 3: Save and reorient to LIA ──
    print("=" * 60)
    print("Step 3: Reorient to LIA and save outputs")
    print("=" * 60)

    complete_out = os.path.join(
        args.output_dir, f"{args.prefix}_desc-completeaseg_dseg.nii.gz")
    merged_out = os.path.join(
        args.output_dir, f"{args.prefix}_desc-completeaseg-merged_dseg.nii.gz")
    nbest_out = os.path.join(
        args.output_dir, f"{args.prefix}_desc-nbest_dseg.nii.gz")
    cerebrum_out = os.path.join(
        args.output_dir, f"{args.prefix}_label-cerebrum_dseg.nii.gz")
    cereb_out = os.path.join(
        args.output_dir, f"{args.prefix}_label-cerebellum-brainstem_dseg.nii.gz")

    # Save complete aseg
    nib.save(nib.Nifti1Image(complete_data, las_affine), complete_out)
    subprocess.run([
        args.python_inter, conform_py,
        '--input', complete_out,
        '--output', complete_out,
        '--reorient', 'LIA',
    ], check=True)

    # Save merged (hemi-agnostic) version for bias field correction
    merged_data = _merge_hemispheres(complete_data)
    nib.save(nib.Nifti1Image(merged_data, las_affine), merged_out)
    subprocess.run([
        args.python_inter, conform_py,
        '--input', merged_out,
        '--output', merged_out,
        '--reorient', 'LIA',
    ], check=True)

    # Save nbest (4-class)
    nib.save(nib.Nifti1Image(nbest_data.astype(np.uint8), las_affine), nbest_out)
    subprocess.run([
        args.python_inter, conform_py,
        '--input', nbest_out,
        '--output', nbest_out,
        '--reorient', 'LIA',
    ], check=True)

    # Save cerebrum mask
    nib.save(nib.Nifti1Image(cerebrum_data.astype(np.uint8), las_affine), cerebrum_out)
    subprocess.run([
        args.python_inter, conform_py,
        '--input', cerebrum_out,
        '--output', cerebrum_out,
        '--reorient', 'LIA',
    ], check=True)

    # Save cerebellum-brainstem mask
    nib.save(nib.Nifti1Image(cereb_data.astype(np.uint8), las_affine), cereb_out)
    subprocess.run([
        args.python_inter, conform_py,
        '--input', cereb_out,
        '--output', cereb_out,
        '--reorient', 'LIA',
    ], check=True)

    # Cleanup workdir
    shutil.rmtree(workdir, ignore_errors=True)

    print(f"complete aseg:    {complete_out}")
    print(f"merged aseg:      {merged_out}")
    print(f"nbest (4-class):  {nbest_out}")
    print(f"cerebrum mask:    {cerebrum_out}")
    print(f"cerebellum mask:  {cereb_out}")
    print("nBEST tissue segmentation + fusion done.")


if __name__ == "__main__":
    main()
