#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC visualization for fixed brainmask results.
Compare original and fixed brainmask, overlay on head image.
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def load_nifti(path):
    """Load NIfTI file and return data array and affine."""
    img = nib.load(path)
    data = img.get_fdata()
    return data, img.affine


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


def create_comparison_plot(head, original_mask, fixed_mask, output_path, rows=4, cols=6, figsize=(18, 12), original_axcodes=None):
    """
    Create a comparison grid showing original vs fixed brainmask.

    Parameters:
        head: 3D numpy array of head image (already in RAS orientation)
        original_mask: 3D numpy array of original brain mask (already in RAS orientation)
        fixed_mask: 3D numpy array of fixed brain mask (already in RAS orientation)
        output_path: path to save the figure
        rows: number of rows (will be doubled for before/after comparison)
        cols: number of columns
        figsize: figure size
        original_axcodes: Original axis codes of the input image, for display
    """
    # Normalize head image
    head_norm = head - np.min(head)
    if np.max(head_norm) > 0:
        head_norm = head_norm / np.max(head_norm)

    # Calculate slice indices along S-I axis (Z)
    z_dim = head.shape[2]
    start_slice = int(z_dim * 0.1)
    end_slice = int(z_dim * 0.9)
    total_slices = rows * cols
    slice_indices = np.linspace(start_slice, end_slice, total_slices, dtype=int)

    # Create figure with 2 rows of images per slice (before/after)
    fig, axes = plt.subplots(rows * 2, cols, figsize=figsize)

    # Colormap for original mask (green) and fixed mask (blue)
    cmap_original = ListedColormap([(0, 0, 0, 0), (0, 1, 0, 0.5)])  # Green
    cmap_fixed = ListedColormap([(0, 0, 0, 0), (0, 0.5, 1, 0.5)])   # Blue

    for i, z in enumerate(slice_indices):
        row_idx = (i // cols) * 2
        col_idx = i % cols

        # Original mask row (axial slice - R-L / A-P)
        ax_orig = axes[row_idx, col_idx]
        ax_orig.imshow(head_norm[:, :, z].T, cmap='gray', origin='lower')
        ax_orig.imshow((original_mask[:, :, z] > 0).T, cmap=cmap_original,
                       origin='lower', vmin=0, vmax=1, interpolation='nearest')
        ax_orig.set_title(f'Original Z={z}\n(R-L / A-P)', fontsize=7)
        ax_orig.axis('off')

        # Fixed mask row (axial slice - R-L / A-P)
        ax_fixed = axes[row_idx + 1, col_idx]
        ax_fixed.imshow(head_norm[:, :, z].T, cmap='gray', origin='lower')
        ax_fixed.imshow((fixed_mask[:, :, z] > 0).T, cmap=cmap_fixed,
                       origin='lower', vmin=0, vmax=1, interpolation='nearest')
        ax_fixed.set_title(f'Fixed Z={z}\n(R-L / A-P)', fontsize=7)
        ax_fixed.axis('off')

    # Hide unused subplots
    for i in range(len(slice_indices) * 2, rows * 2 * cols):
        row_idx = i // cols
        col_idx = i % cols
        if row_idx < axes.shape[0] and col_idx < axes.shape[1]:
            axes[row_idx, col_idx].axis('off')

    # Add orientation info to title
    title = 'Brainmask Fix QC: Original (Green) vs Fixed (Blue)'
    if original_axcodes:
        title += f"\nOriginal orientation: {'-'.join(original_axcodes)} | Display: RAS"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] QC comparison figure saved to: {output_path}")


def create_diff_plot(head, original_mask, fixed_mask, output_path, figsize=(15, 5), original_axcodes=None):
    """
    Create three views showing the difference between original and fixed mask.

    Parameters:
        head: 3D numpy array (already in RAS orientation)
        original_mask: 3D numpy array (already in RAS orientation)
        fixed_mask: 3D numpy array (already in RAS orientation)
        output_path: path to save the figure
        figsize: figure size
        original_axcodes: Original axis codes of the input image, for display
    """
    head_norm = head - np.min(head)
    if np.max(head_norm) > 0:
        head_norm = head_norm / np.max(head_norm)

    # Calculate difference mask
    # Red: removed from original
    # Green: added in fixed
    orig_binary = (original_mask > 0).astype(int)
    fixed_binary = (fixed_mask > 0).astype(int)

    removed = (orig_binary - fixed_binary) > 0  # In original but not in fixed
    added = (fixed_binary - orig_binary) > 0    # In fixed but not in original

    # Find center of mass of fixed mask (RAS coordinates)
    coords = np.where(fixed_binary > 0)
    if len(coords[0]) > 0:
        cx = int(np.mean(coords[0]))  # R-L axis
        cy = int(np.mean(coords[1]))  # A-P axis
        cz = int(np.mean(coords[2]))  # S-I axis
    else:
        cx, cy, cz = head.shape[0]//2, head.shape[1]//2, head.shape[2]//2

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Axial view - R-L / A-P axes
    axes[0].imshow(head_norm[:, :, cz].T, cmap='gray', origin='lower')
    # Show removed in red, added in green
    axes[0].imshow(np.ma.masked_where(removed[:, :, cz] == 0, removed[:, :, cz]).T,
                   cmap='Reds', origin='lower', alpha=0.7, vmin=0, vmax=1)
    axes[0].imshow(np.ma.masked_where(added[:, :, cz] == 0, added[:, :, cz]).T,
                   cmap='Greens', origin='lower', alpha=0.7, vmin=0, vmax=1)
    axes[0].set_title(f'Axial (Z={cz})\nR-L / A-P\nRed=Removed, Green=Added', fontsize=9)
    axes[0].set_xlabel('R → L', fontsize=8)
    axes[0].set_ylabel('A → P', fontsize=8)

    # Coronal view - R-L / S-I axes
    axes[1].imshow(head_norm[:, cy, :].T, cmap='gray', origin='lower')
    axes[1].imshow(np.ma.masked_where(removed[:, cy, :] == 0, removed[:, cy, :]).T,
                   cmap='Reds', origin='lower', alpha=0.7, vmin=0, vmax=1)
    axes[1].imshow(np.ma.masked_where(added[:, cy, :] == 0, added[:, cy, :]).T,
                   cmap='Greens', origin='lower', alpha=0.7, vmin=0, vmax=1)
    axes[1].set_title(f'Coronal (Y={cy})\nR-L / S-I', fontsize=9)
    axes[1].set_xlabel('R → L', fontsize=8)
    axes[1].set_ylabel('S → I', fontsize=8)

    # Sagittal view - A-P / S-I axes
    axes[2].imshow(head_norm[cx, :, :].T, cmap='gray', origin='lower')
    axes[2].imshow(np.ma.masked_where(removed[cx, :, :] == 0, removed[cx, :, :]).T,
                   cmap='Reds', origin='lower', alpha=0.7, vmin=0, vmax=1)
    axes[2].imshow(np.ma.masked_where(added[cx, :, :] == 0, added[cx, :, :]).T,
                   cmap='Greens', origin='lower', alpha=0.7, vmin=0, vmax=1)
    axes[2].set_title(f'Sagittal (X={cx})\nA-P / S-I', fontsize=9)
    axes[2].set_xlabel('A → P', fontsize=8)
    axes[2].set_ylabel('S → I', fontsize=8)

    # Print statistics
    n_removed = np.sum(removed)
    n_added = np.sum(added)

    # Add orientation info to title
    title = f'Brainmask Difference: Removed={n_removed} voxels, Added={n_added} voxels'
    if original_axcodes:
        title += f"\nOriginal orientation: {'-'.join(original_axcodes)} | Display: RAS"
    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] QC difference figure saved to: {output_path}")
    print(f"     Removed voxels: {n_removed}")
    print(f"     Added voxels: {n_added}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate QC visualization for fixed brainmask results"
    )
    parser.add_argument("--head", type=str, required=True,
                        help="Path to head image")
    parser.add_argument("--original_mask", type=str, required=True,
                        help="Path to original brain mask")
    parser.add_argument("--fixed_mask", type=str, required=True,
                        help="Path to fixed brain mask")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for QC figure")
    parser.add_argument("--rows", type=int, default=4,
                        help="Number of rows (default: 4)")
    parser.add_argument("--cols", type=int, default=6,
                        help="Number of columns (default: 6)")

    args = parser.parse_args()

    # Check input files
    for f, name in [(args.head, "head"), (args.original_mask, "original_mask"),
                    (args.fixed_mask, "fixed_mask")]:
        if not os.path.exists(f):
            print(f"Error: {name} not found: {f}")
            sys.exit(1)

    # Load images and reorient to RAS
    print(f"Loading head: {args.head}")
    head_data, head_affine = load_nifti(args.head)
    head_ras, original_axcodes, _ = reorient_to_ras(head_data, head_affine)
    print(f"Original head orientation: {'-'.join(original_axcodes)}")
    print(f"Reoriented to RAS for display, shape: {head_ras.shape}")

    print(f"Loading original mask: {args.original_mask}")
    orig_mask_data, orig_mask_affine = load_nifti(args.original_mask)
    orig_mask_ras, orig_mask_axcodes, _ = reorient_to_ras(orig_mask_data, orig_mask_affine)
    print(f"Original mask orientation: {'-'.join(orig_mask_axcodes)}")
    print(f"Reoriented to RAS for display, shape: {orig_mask_ras.shape}")

    print(f"Loading fixed mask: {args.fixed_mask}")
    fixed_mask_data, fixed_mask_affine = load_nifti(args.fixed_mask)
    fixed_mask_ras, fixed_mask_axcodes, _ = reorient_to_ras(fixed_mask_data, fixed_mask_affine)
    print(f"Fixed mask orientation: {'-'.join(fixed_mask_axcodes)}")
    print(f"Reoriented to RAS for display, shape: {fixed_mask_ras.shape}")

    # Ensure all images have the same shape (resample masks to match head)
    from scipy.ndimage import zoom

    if orig_mask_ras.shape != head_ras.shape:
        print(f"Resampling original mask to match head shape: {head_ras.shape}")
        zoom_factors = [h/m for h, m in zip(head_ras.shape, orig_mask_ras.shape)]
        orig_mask_ras = zoom(orig_mask_ras, zoom_factors, order=0)

    if fixed_mask_ras.shape != head_ras.shape:
        print(f"Resampling fixed mask to match head shape: {head_ras.shape}")
        zoom_factors = [h/m for h, m in zip(head_ras.shape, fixed_mask_ras.shape)]
        fixed_mask_ras = zoom(fixed_mask_ras, zoom_factors, order=0)

    # Generate comparison plot
    output_path = args.output
    if not output_path.endswith('.png'):
        output_path = output_path + '.png'

    create_comparison_plot(head_ras, orig_mask_ras, fixed_mask_ras, output_path, args.rows, args.cols, original_axcodes=original_axcodes)

    # Generate difference plot
    diff_path = output_path.replace('.png', '_diff.png')
    create_diff_plot(head_ras, orig_mask_ras, fixed_mask_ras, diff_path, original_axcodes=original_axcodes)

    print("[DONE] QC visualization completed.")


if __name__ == "__main__":
    main()
