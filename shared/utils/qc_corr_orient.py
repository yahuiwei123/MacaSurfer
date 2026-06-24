#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC visualization for orientation correction results.
Display three orthogonal views of the corrected image with orientation labels.
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def load_nifti(path):
    """Load NIfTI file and return data array and header info."""
    img = nib.load(path)
    data = img.get_fdata()
    affine = img.affine
    return data, affine


def reorient_to_ras(data, affine):
    """
    Reorient image data to standard RAS (Right-Anterior-Superior) orientation.
    Returns:
        reoriented_data: RAS oriented data
        original_axcodes: Original axis codes
        ras_affine: Affine for the RAS oriented image
    """
    # Get original axis codes
    original_ornt = nib.io_orientation(affine)
    original_axcodes = nib.aff2axcodes(affine)

    # Target orientation is RAS
    target_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))

    # Calculate transformation
    transform = nib.orientations.ornt_transform(original_ornt, target_ornt)

    # Apply transformation
    reoriented_data = nib.orientations.apply_orientation(data, transform)

    # Calculate new affine
    # Support both old and new nibabel versions
    if hasattr(nib.orientations, 'inv_ornt_affine'):
        ras_affine = nib.orientations.inv_ornt_affine(transform, data.shape) @ affine
    else:
        ras_affine = nib.orientations.inv_ornt_aff(transform, data.shape) @ affine

    return reoriented_data, original_axcodes, ras_affine


def get_orientation_string(affine):
    """Get orientation string from affine matrix."""
    # Extract the orientation from affine
    ornt = nib.aff2axcodes(affine)
    return ''.join(ornt)


def create_three_view_plot(image, output_path, title="Orientation Corrected", figsize=(15, 5), original_axcodes=None):
    """
    Create three orthogonal views of the image.

    Parameters:
        image: 3D numpy array (already in RAS orientation)
        output_path: path to save the figure
        title: title for the plot
        figsize: figure size
        original_axcodes: Original axis codes of the input image, for display
    """
    # Normalize image
    img_norm = image - np.min(image)
    if np.max(img_norm) > 0:
        img_norm = img_norm / np.max(img_norm)

    # Find center (use intensity-weighted center)
    total = np.sum(image)
    if total > 0:
        indices = np.indices(image.shape)
        cx = int(np.sum(indices[0] * image) / total)  # R-L axis
        cy = int(np.sum(indices[1] * image) / total)  # A-P axis
        cz = int(np.sum(indices[2] * image) / total)  # S-I axis
    else:
        cx, cy, cz = image.shape[0]//2, image.shape[1]//2, image.shape[2]//2

    # Clip to valid range
    cx = np.clip(cx, 0, image.shape[0]-1)
    cy = np.clip(cy, 0, image.shape[1]-1)
    cz = np.clip(cz, 0, image.shape[2]-1)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Add orientation info to title if available
    full_title = title
    if original_axcodes:
        full_title += f"\nOriginal orientation: {'-'.join(original_axcodes)} | Display: RAS"

    # Axial view (top-down) - shows R-L and A-P axes
    im0 = axes[0].imshow(img_norm[:, :, cz].T, cmap='gray', origin='lower')
    axes[0].axhline(y=cy, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[0].axvline(x=cx, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[0].set_title(f'Axial (Z={cz}/{image.shape[2]-1})', fontsize=12)
    axes[0].set_xlabel('R → L (Right → Left)', fontsize=10)
    axes[0].set_ylabel('A → P (Anterior → Posterior)', fontsize=10)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Coronal view (front) - shows R-L and S-I axes
    im1 = axes[1].imshow(img_norm[:, cy, :].T, cmap='gray', origin='lower')
    axes[1].axhline(y=cz, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[1].axvline(x=cx, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[1].set_title(f'Coronal (Y={cy}/{image.shape[1]-1})', fontsize=12)
    axes[1].set_xlabel('R → L (Right → Left)', fontsize=10)
    axes[1].set_ylabel('S → I (Superior → Inferior)', fontsize=10)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Sagittal view (side) - shows A-P and S-I axes
    im2 = axes[2].imshow(img_norm[cx, :, :].T, cmap='gray', origin='lower')
    axes[2].axhline(y=cz, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[2].axvline(x=cy, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[2].set_title(f'Sagittal (X={cx}/{image.shape[0]-1})', fontsize=12)
    axes[2].set_xlabel('A → P (Anterior → Posterior)', fontsize=10)
    axes[2].set_ylabel('S → I (Superior → Inferior)', fontsize=10)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(full_title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] QC three-view figure saved to: {output_path}")


def create_before_after_plot(original, corrected, output_path, figsize=(15, 10), original_axcodes=None):
    """
    Create before/after comparison with three views each.

    Parameters:
        original: 3D numpy array of original image (already in RAS orientation)
        corrected: 3D numpy array of corrected image (already in RAS orientation)
        output_path: path to save the figure
        figsize: figure size
        original_axcodes: Original axis codes of the input image, for display
    """
    # Normalize images
    orig_norm = original - np.min(original)
    if np.max(orig_norm) > 0:
        orig_norm = orig_norm / np.max(orig_norm)

    corr_norm = corrected - np.min(corrected)
    if np.max(corr_norm) > 0:
        corr_norm = corr_norm / np.max(corr_norm)

    # Find center of corrected image
    total = np.sum(corrected)
    if total > 0:
        indices = np.indices(corrected.shape)
        cx = int(np.sum(indices[0] * corrected) / total)  # R-L axis
        cy = int(np.sum(indices[1] * corrected) / total)  # A-P axis
        cz = int(np.sum(indices[2] * corrected) / total)  # S-I axis
    else:
        cx, cy, cz = corrected.shape[0]//2, corrected.shape[1]//2, corrected.shape[2]//2

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    views = [
        ('Axial\n(R-L / A-P)', lambda img, x, y, z: img[:, :, z]),
        ('Coronal\n(R-L / S-I)', lambda img, x, y, z: img[:, y, :]),
        ('Sagittal\n(A-P / S-I)', lambda img, x, y, z: img[x, :, :])
    ]
    coords = [
        f'Z={cz}',
        f'Y={cy}',
        f'X={cx}'
    ]

    for col, (view_name, slice_fn) in enumerate(views):
        # Original
        axes[0, col].imshow(slice_fn(orig_norm, cx, cy, cz).T, cmap='gray', origin='lower')
        axes[0, col].set_title(f'Original - {view_name} ({coords[col]})', fontsize=10)
        axes[0, col].axis('off')

        # Corrected
        axes[1, col].imshow(slice_fn(corr_norm, cx, cy, cz).T, cmap='gray', origin='lower')
        axes[1, col].set_title(f'Corrected - {view_name} ({coords[col]})', fontsize=10)
        axes[1, col].axis('off')

    # Add orientation info
    title = 'Orientation Correction QC: Before vs After'
    if original_axcodes:
        title += f"\nOriginal orientation: {'-'.join(original_axcodes)} | Display: RAS"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] QC before/after figure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate QC visualization for orientation correction"
    )
    parser.add_argument("--corrected", type=str, required=True,
                        help="Path to orientation-corrected image")
    parser.add_argument("--original", type=str, default=None,
                        help="Path to original image (optional, for comparison)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for QC figure")

    args = parser.parse_args()

    # Check corrected file
    if not os.path.exists(args.corrected):
        print(f"Error: Corrected image not found: {args.corrected}")
        sys.exit(1)

    # Load corrected image and reorient to RAS
    print(f"Loading corrected image: {args.corrected}")
    corrected_data, corrected_affine = load_nifti(args.corrected)
    corrected_ras, original_axcodes, _ = reorient_to_ras(corrected_data, corrected_affine)
    orientation = '-'.join(original_axcodes)
    print(f"Original image orientation: {orientation}")
    print(f"Reoriented to RAS for display")
    print(f"Image shape (RAS): {corrected_ras.shape}")

    output_path = args.output
    if not output_path.endswith('.png'):
        output_path = output_path + '.png'

    # Generate three-view plot
    title = f"Orientation Corrected"
    create_three_view_plot(corrected_ras, output_path, title=title, original_axcodes=original_axcodes)

    # Generate before/after comparison if original is provided
    if args.original and os.path.exists(args.original):
        print(f"Loading original image: {args.original}")
        original_data, original_affine = load_nifti(args.original)
        original_ras, orig_axcodes, _ = reorient_to_ras(original_data, original_affine)
        orig_orientation = '-'.join(orig_axcodes)
        print(f"Original image orientation: {orig_orientation}")
        print(f"Reoriented to RAS for display")

        # Ensure both images have the same shape (interpolate if needed)
        if original_ras.shape != corrected_ras.shape:
            print(f"Warning: Original and corrected images have different shapes after reorientation")
            print(f"Original: {original_ras.shape}, Corrected: {corrected_ras.shape}")
            # Resample original to match corrected shape (simple nearest neighbor for display)
            from scipy.ndimage import zoom
            zoom_factors = [c/o for c, o in zip(corrected_ras.shape, original_ras.shape)]
            original_ras = zoom(original_ras, zoom_factors, order=0)
            print(f"Resampled original to: {original_ras.shape}")

        comparison_path = output_path.replace('.png', '_comparison.png')
        create_before_after_plot(original_ras, corrected_ras, comparison_path, original_axcodes=original_axcodes)

    print("[DONE] QC visualization completed.")


if __name__ == "__main__":
    main()
