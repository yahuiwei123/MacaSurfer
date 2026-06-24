#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC visualization for multi-modality registration results.

Displays (per moving modality):
  - Moving image registered to target space
  - Target image
  - Checkerboard comparison (moving vs target)
  - Contour overlay (moving + target edges)

Inputs:
  --target      : target/reference image (e.g. T1w)
  --t2w         : T2w image registered to target space (optional)
  --flair       : FLAIR image registered to target space (optional)
  --output      : output path for QC figure
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def load_nifti(path):
    """Load NIfTI file and return data array and affine."""
    if path is None or not os.path.exists(path):
        return None, None
    img = nib.load(path)
    data = img.get_fdata()
    return data, img.affine


def reorient_to_ras(data, affine):
    """
    Reorient image data to standard RAS (Right-Anterior-Superior) orientation.
    """
    if data is None or affine is None:
        return None, None, None

    original_ornt = nib.io_orientation(affine)
    original_axcodes = nib.aff2axcodes(affine)
    target_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
    transform = nib.orientations.ornt_transform(original_ornt, target_ornt)
    reoriented_data = nib.orientations.apply_orientation(data, transform)

    if hasattr(nib.orientations, 'inv_ornt_affine'):
        ras_affine = nib.orientations.inv_ornt_affine(transform, data.shape) @ affine
    else:
        ras_affine = nib.orientations.inv_ornt_aff(transform, data.shape) @ affine

    return reoriented_data, original_axcodes, ras_affine


def normalize_image(img):
    """Normalize image to [0, 1] range."""
    if img is None:
        return None
    img_norm = img - np.min(img)
    if np.max(img_norm) > 0:
        img_norm = img_norm / np.max(img_norm)
    return img_norm


def create_checkerboard(img1, img2, block_size=8):
    """Create checkerboard pattern of two images."""
    if img1 is None or img2 is None:
        return None
    if img1.shape != img2.shape:
        print(f"Warning: Shape mismatch - img1: {img1.shape}, img2: {img2.shape}")
        return img1

    h, w = img1.shape
    checker = np.zeros((h, w))
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                checker[i:i+block_size, j:j+block_size] = img1[i:i+block_size, j:j+block_size]
            else:
                checker[i:i+block_size, j:j+block_size] = img2[i:i+block_size, j:j+block_size]
    return checker


def compute_center_of_mass(data):
    """Compute center of mass of a 3D volume."""
    total = np.sum(data)
    if total > 0:
        indices = np.indices(data.shape)
        cx = int(np.sum(indices[0] * data) / total)
        cy = int(np.sum(indices[1] * data) / total)
        cz = int(np.sum(indices[2] * data) / total)
    else:
        cx, cy, cz = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2
    return cx, cy, cz


def create_modality_qc_plot(target, moving, moving_label,
                             output_path, figsize=(22, 14), original_axcodes=None):
    """
    Create QC plot showing registration quality between a moving modality
    (registered to target space) and the target image.

    Layout (4 columns):
      Moving registered | Target | Checkerboard | Contour overlay

    Parameters:
        target: 3D array - target/reference image (RAS oriented)
        moving: 3D array - moving image registered to target space (RAS oriented)
        moving_label: str - label for the moving modality (e.g. "T2w", "FLAIR")
        output_path: path to save the figure
        figsize: figure size
        original_axcodes: original axis codes for display annotation
    """
    if target is None or moving is None:
        print("Error: Both target and moving images are required")
        return

    # Normalize
    target_norm = normalize_image(target)
    moving_norm = normalize_image(moving)

    # Resample moving to match target shape if needed
    if moving_norm.shape != target_norm.shape:
        from scipy.ndimage import zoom
        zoom_factors = [t/m for t, m in zip(target_norm.shape, moving_norm.shape)]
        moving_norm = zoom(moving_norm, zoom_factors, order=1)

    # Compute center of mass from target for slice selection
    cx, cy, cz = compute_center_of_mass(target)

    n_cols = 4
    fig, axes = plt.subplots(3, n_cols, figsize=figsize)
    if axes.ndim == 1:
        axes = axes.reshape(3, -1)

    col = 0

    # ========================================================================
    # Column 1: Moving image (registered to target space)
    # ========================================================================
    axes[0, col].imshow(moving_norm[:, :, cz].T, cmap='gray', origin='lower')
    axes[0, col].set_title(f'{moving_label} (registered)\nAxial (Z={cz})', fontsize=9)
    axes[0, col].set_xlabel('R → L', fontsize=7)
    axes[0, col].set_ylabel('A → P', fontsize=7)

    axes[1, col].imshow(moving_norm[:, cy, :].T, cmap='gray', origin='lower')
    axes[1, col].set_title(f'{moving_label}\nCoronal (Y={cy})', fontsize=9)
    axes[1, col].set_xlabel('R → L', fontsize=7)
    axes[1, col].set_ylabel('S → I', fontsize=7)

    axes[2, col].imshow(moving_norm[cx, :, :].T, cmap='gray', origin='lower')
    axes[2, col].set_title(f'{moving_label}\nSagittal (X={cx})', fontsize=9)
    axes[2, col].set_xlabel('A → P', fontsize=7)
    axes[2, col].set_ylabel('S → I', fontsize=7)
    col += 1

    # ========================================================================
    # Column 2: Target image
    # ========================================================================
    axes[0, col].imshow(target_norm[:, :, cz].T, cmap='gray', origin='lower')
    axes[0, col].set_title('Target (Axial)', fontsize=9)
    axes[1, col].imshow(target_norm[:, cy, :].T, cmap='gray', origin='lower')
    axes[1, col].set_title('Target (Coronal)', fontsize=9)
    axes[2, col].imshow(target_norm[cx, :, :].T, cmap='gray', origin='lower')
    axes[2, col].set_title('Target (Sagittal)', fontsize=9)
    for r in range(3):
        axes[r, col].set_xticks([])
        axes[r, col].set_yticks([])
    col += 1

    # ========================================================================
    # Column 3: Checkerboard (moving vs target)
    # ========================================================================
    checker_axial = create_checkerboard(moving_norm[:, :, cz], target_norm[:, :, cz])
    checker_coronal = create_checkerboard(moving_norm[:, cy, :], target_norm[:, cy, :])
    checker_sagittal = create_checkerboard(moving_norm[cx, :, :], target_norm[cx, :, :])

    axes[0, col].imshow(checker_axial.T, cmap='gray', origin='lower')
    axes[0, col].set_title(f'{moving_label} + Target\nCheckerboard (Axial)', fontsize=9)
    axes[1, col].imshow(checker_coronal.T, cmap='gray', origin='lower')
    axes[1, col].set_title('Checkerboard (Coronal)', fontsize=9)
    axes[2, col].imshow(checker_sagittal.T, cmap='gray', origin='lower')
    axes[2, col].set_title('Checkerboard (Sagittal)', fontsize=9)
    for r in range(3):
        axes[r, col].set_xticks([])
        axes[r, col].set_yticks([])
    col += 1

    # ========================================================================
    # Column 4: Overlay (moving + target contours)
    # ========================================================================
    try:
        from scipy import ndimage
        has_scipy = True
    except ImportError:
        has_scipy = False

    slice_specs = [
        (0, moving_norm[:, :, cz].T, target_norm[:, :, cz]),
        (1, moving_norm[:, cy, :].T, target_norm[:, cy, :]),
        (2, moving_norm[cx, :, :].T, target_norm[cx, :, :]),
    ]
    for row, moving_slice_t, target_slice in slice_specs:
        axes[row, col].imshow(moving_slice_t, cmap='gray', origin='lower')
        if has_scipy:
            axes[row, col].contour(target_slice.T, levels=[0.2, 0.4, 0.6],
                                   colors='r', linewidths=0.5, alpha=0.7)
        axes[row, col].set_title(
            f'{moving_label} + Target Contours' if row == 0 else '', fontsize=9)
    for r in range(3):
        axes[r, col].set_xticks([])
        axes[r, col].set_yticks([])
    col += 1

    # Title and save
    title = f'Modality Registration QC — {moving_label} → Target'
    if original_axcodes:
        title += f" | Original orientation: {'-'.join(original_axcodes)} | Display: RAS"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] QC modality registration figure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate QC visualization for multi-modality registration"
    )
    parser.add_argument("--target", type=str, required=True,
                        help="Path to target/reference image (e.g. T1w)")
    parser.add_argument("--t2w", type=str, default=None,
                        help="Path to T2w image registered to target space")
    parser.add_argument("--flair", type=str, default=None,
                        help="Path to FLAIR image registered to target space")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for QC figure (suffix added per modality)")

    args = parser.parse_args()

    # Validate target
    if not os.path.exists(args.target):
        print(f"Error: Target image not found: {args.target}")
        sys.exit(1)

    # Load and reorient: target
    print(f"Loading target: {args.target}")
    target_data, target_affine = load_nifti(args.target)
    target_ras, original_axcodes, _ = reorient_to_ras(target_data, target_affine)
    print(f"  Orientation: {'-'.join(original_axcodes) if original_axcodes else 'Unknown'}, "
          f"shape: {target_ras.shape}")

    # Determine which moving modalities are available
    moving_specs = []
    if args.t2w and os.path.exists(args.t2w):
        moving_specs.append(('t2w', args.t2w, 'T2w'))
    if args.flair and os.path.exists(args.flair):
        moving_specs.append(('flair', args.flair, 'FLAIR'))

    if not moving_specs:
        print("No moving modality (t2w, flair) is available")
        sys.exit(0)

    # Generate QC plot for each moving modality
    base_output = args.output
    if base_output.endswith('.png'):
        base_output = base_output[:-4]

    for suffix, moving_path, moving_label in moving_specs:
        print(f"Loading {moving_label}: {moving_path}")
        moving_data, moving_affine = load_nifti(moving_path)
        moving_ras, moving_axcodes, _ = reorient_to_ras(moving_data, moving_affine)
        print(f"  Orientation: {'-'.join(moving_axcodes) if moving_axcodes else 'Unknown'}, "
              f"shape: {moving_ras.shape}")

        output_path = f"{base_output}_{suffix}.png"
        create_modality_qc_plot(target_ras, moving_ras, moving_label,
                                 output_path, original_axcodes=original_axcodes)

    print("[DONE] QC visualization completed.")


if __name__ == "__main__":
    main()
