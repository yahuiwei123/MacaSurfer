#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC visualization for skullstrip results.
Overlay brainmask on head image and display as a grid of slices.
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


def create_overlay_plot(head, brainmask, output_path, rows=6, cols=6, figsize=(16, 16), original_axcodes=None):
    """
    Create a grid of slices with brainmask overlay on head image.

    Parameters:
        head: 3D numpy array of head image (already in RAS orientation)
        brainmask: 3D numpy array of brain mask (binary, already in RAS orientation)
        output_path: path to save the figure
        rows: number of rows in the grid
        cols: number of columns in the grid
        figsize: figure size
        original_axcodes: Original axis codes of the input image, for display
    """
    # Normalize head image for display
    head_norm = head - np.min(head)
    if np.max(head_norm) > 0:
        head_norm = head_norm / np.max(head_norm)

    # Create binary mask overlay
    mask_binary = (brainmask > 0).astype(float)

    # Calculate slice indices (evenly spaced, skipping edges) along S-I axis (Z)
    z_dim = head.shape[2]
    start_slice = int(z_dim * 0.1)
    end_slice = int(z_dim * 0.9)
    total_slices = rows * cols
    slice_indices = np.linspace(start_slice, end_slice, total_slices, dtype=int)

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    # Create colormap for overlay (transparent blue for mask)
    cmap_mask = ListedColormap([(0, 0, 0, 0), (0, 0.5, 1, 0.5)])

    for i, z in enumerate(slice_indices):
        ax = axes[i]

        # Display head image (axial slice - R-L / A-P axes)
        ax.imshow(head_norm[:, :, z].T, cmap='gray', origin='lower')

        # Overlay brainmask
        ax.imshow(mask_binary[:, :, z].T, cmap=cmap_mask, origin='lower',
                  vmin=0, vmax=1, interpolation='nearest')

        ax.set_title(f'Z={z}\n(R-L / A-P)', fontsize=7)
        ax.axis('off')

    # Hide unused subplots
    for i in range(len(slice_indices), len(axes)):
        axes[i].axis('off')

    # Add orientation info to title
    title = 'Skullstrip QC: Brainmask Overlay on Head (Axial Slices)'
    if original_axcodes:
        title += f"\nOriginal orientation: {'-'.join(original_axcodes)} | Display: RAS"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] QC figure saved to: {output_path}")


def create_three_view_plot(head, brainmask, output_path, figsize=(15, 5), original_axcodes=None):
    """
    Create three orthogonal views with brainmask overlay.

    Parameters:
        head: 3D numpy array of head image (already in RAS orientation)
        brainmask: 3D numpy array of brain mask (binary, already in RAS orientation)
        output_path: path to save the figure
        figsize: figure size
        original_axcodes: Original axis codes of the input image, for display
    """
    # Normalize head image
    head_norm = head - np.min(head)
    if np.max(head_norm) > 0:
        head_norm = head_norm / np.max(head_norm)

    mask_binary = (brainmask > 0).astype(float)

    # Find center of mass of brainmask (RAS coordinates)
    coords = np.where(mask_binary > 0)
    if len(coords[0]) > 0:
        cx = int(np.mean(coords[0]))  # R-L axis
        cy = int(np.mean(coords[1]))  # A-P axis
        cz = int(np.mean(coords[2]))  # S-I axis
    else:
        cx, cy, cz = head.shape[0]//2, head.shape[1]//2, head.shape[2]//2

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    cmap_mask = ListedColormap([(0, 0, 0, 0), (0, 0.5, 1, 0.5)])

    # Axial view - R-L / A-P axes
    axes[0].imshow(head_norm[:, :, cz].T, cmap='gray', origin='lower')
    axes[0].imshow(mask_binary[:, :, cz].T, cmap=cmap_mask, origin='lower', vmin=0, vmax=1)
    axes[0].set_title(f'Axial (Z={cz})\nR-L / A-P', fontsize=10)
    axes[0].set_xlabel('R → L', fontsize=8)
    axes[0].set_ylabel('A → P', fontsize=8)

    # Coronal view - R-L / S-I axes
    axes[1].imshow(head_norm[:, cy, :].T, cmap='gray', origin='lower')
    axes[1].imshow(mask_binary[:, cy, :].T, cmap=cmap_mask, origin='lower', vmin=0, vmax=1)
    axes[1].set_title(f'Coronal (Y={cy})\nR-L / S-I', fontsize=10)
    axes[1].set_xlabel('R → L', fontsize=8)
    axes[1].set_ylabel('S → I', fontsize=8)

    # Sagittal view - A-P / S-I axes
    axes[2].imshow(head_norm[cx, :, :].T, cmap='gray', origin='lower')
    axes[2].imshow(mask_binary[cx, :, :].T, cmap=cmap_mask, origin='lower', vmin=0, vmax=1)
    axes[2].set_title(f'Sagittal (X={cx})\nA-P / S-I', fontsize=10)
    axes[2].set_xlabel('A → P', fontsize=8)
    axes[2].set_ylabel('S → I', fontsize=8)

    # Add orientation info to title
    title = 'Skullstrip QC: Three Views'
    if original_axcodes:
        title += f"\nOriginal orientation: {'-'.join(original_axcodes)} | Display: RAS"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] QC three-view figure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate QC visualization for skullstrip results"
    )
    parser.add_argument("--head", type=str, required=True,
                        help="Path to head image (head.nii.gz)")
    parser.add_argument("--brainmask", type=str, required=True,
                        help="Path to brain mask (brainmask.nii.gz)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for QC figure")
    parser.add_argument("--rows", type=int, default=6,
                        help="Number of rows in the grid (default: 6)")
    parser.add_argument("--cols", type=int, default=6,
                        help="Number of columns in the grid (default: 6)")
    parser.add_argument("--three_view", action="store_true",
                        help="Also generate three-view plot")

    args = parser.parse_args()

    # Check input files
    if not os.path.exists(args.head):
        print(f"Error: Head image not found: {args.head}")
        sys.exit(1)
    if not os.path.exists(args.brainmask):
        print(f"Error: Brainmask not found: {args.brainmask}")
        sys.exit(1)

    # Load images and reorient to RAS
    print(f"Loading head image: {args.head}")
    head_data, head_affine = load_nifti(args.head)
    head_ras, original_axcodes, _ = reorient_to_ras(head_data, head_affine)
    print(f"Original head orientation: {'-'.join(original_axcodes)}")
    print(f"Reoriented to RAS for display, shape: {head_ras.shape}")

    print(f"Loading brainmask: {args.brainmask}")
    mask_data, mask_affine = load_nifti(args.brainmask)
    mask_ras, mask_axcodes, _ = reorient_to_ras(mask_data, mask_affine)
    print(f"Original mask orientation: {'-'.join(mask_axcodes)}")
    print(f"Reoriented to RAS for display, shape: {mask_ras.shape}")

    # Ensure mask has the same shape as head (resample if needed)
    if head_ras.shape != mask_ras.shape:
        print(f"Warning: Head and mask have different shapes after reorientation")
        print(f"Head: {head_ras.shape}, Mask: {mask_ras.shape}")
        # Resample mask to match head shape (nearest neighbor for segmentation)
        from scipy.ndimage import zoom
        zoom_factors = [h/m for h, m in zip(head_ras.shape, mask_ras.shape)]
        mask_ras = zoom(mask_ras, zoom_factors, order=0)
        print(f"Resampled mask to: {mask_ras.shape}")

    # Generate grid plot
    output_path = args.output
    if not output_path.endswith('.png'):
        output_path = output_path + '.png'

    create_overlay_plot(head_ras, mask_ras, output_path, args.rows, args.cols, original_axcodes=original_axcodes)

    # Generate three-view plot if requested
    if args.three_view:
        three_view_path = output_path.replace('.png', '_three_view.png')
        create_three_view_plot(head_ras, mask_ras, three_view_path, original_axcodes=original_axcodes)

    print("[DONE] QC visualization completed.")


if __name__ == "__main__":
    main()
