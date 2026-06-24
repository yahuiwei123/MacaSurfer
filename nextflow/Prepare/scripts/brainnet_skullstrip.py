#!/usr/bin/env python3
"""
Skull stripping via macaBrainNet predict_ensemble.py.
Replaces macaUNet / nBEST brain extraction in the MacaSurfer pipeline.

Outputs (per run, in output_dir):
  head.nii.gz       – N4 bias-field corrected head image
  brainmask.nii.gz  – binary brain mask
  brain.nii.gz      – masked brain image
"""

import os
import sys
import argparse
import subprocess
import shutil
import numpy as np
import nibabel as nib


def main():
    parser = argparse.ArgumentParser(
        description="BrainNet skull stripping for MacaSurfer")
    parser.add_argument("--input", required=True,
                        help="Input NIfTI (head image)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory")
    parser.add_argument("--model-dir", required=True,
                        help="macaBrainNet skull_stripping model dir")
    parser.add_argument("--predict-script", required=True,
                        help="Path to predict_ensemble.py")
    parser.add_argument("--python-inter", default="python3",
                        help="Python interpreter")
    parser.add_argument("--is-brain", default="false",
                        help="If input is already brain-extracted")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    head_img = os.path.join(args.output_dir, "head.nii.gz")
    brain_mask_img = os.path.join(args.output_dir, "brainmask.nii.gz")
    brain_img = os.path.join(args.output_dir, "brain.nii.gz")

    if args.is_brain == 'true':
        shutil.copy(args.input, brain_img)
        subprocess.run([
            'N4BiasFieldCorrection', '-i', brain_img, '-o', brain_img,
            '-d', '3', '-b', '[2x2x2,3]', '-s', '3',
            '-c', '[100x50x25x10,0]', '-t', '[0.15,0.01,200]'
        ], check=True)
        data = nib.load(brain_img).get_fdata()
        mask = (data > 0).astype(np.int16)
        nib.save(nib.Nifti1Image(mask, nib.load(brain_img).affine),
                 brain_mask_img)
        subprocess.run(['fslmaths', brain_img, '-mas',
                       brain_mask_img, brain_img], check=True)
    else:
        shutil.copy(args.input, head_img)
        subprocess.run([
            'N4BiasFieldCorrection', '-i', head_img, '-o', head_img,
            '-d', '3', '-b', '[2x2x2,3]', '-s', '3',
            '-c', '[100x50x25x10,0]', '-t', '[0.15,0.01,200]'
        ], check=True)
        print("N4 bias field correction done.")

        cmd = [
            args.python_inter, args.predict_script,
            '--img', head_img,
            '--out', brain_mask_img,
            '--ckpt-dir', args.model_dir,
            '--num-classes', '2',
            '--spacing', '0.5', '0.5', '0.5',
            '--device', args.device,
        ]
        print(f"Running predict_ensemble (skull strip): {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        subprocess.run(['fslmaths', head_img, '-mas',
                       brain_mask_img, brain_img], check=True)

    print(f"BrainNet skull strip done.")
    print(f"  head:       {head_img}")
    print(f"  brain mask: {brain_mask_img}")
    print(f"  brain:      {brain_img}")


if __name__ == "__main__":
    main()
