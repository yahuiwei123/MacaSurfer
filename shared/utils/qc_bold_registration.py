#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC visualization for BOLD-to-T1w registration.
Displays orthogonal views (axial, coronal, sagittal) with:
  - T1w reference
  - BOLD edges overlaid on T1w
  - Checkerboard composite
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage


def load_nifti(path):
    """Load NIfTI file and return data array and affine."""
    if path is None or not os.path.exists(path):
        return None, None
    img = nib.load(path)
    data = img.get_fdata()
    return data, img.affine


def reorient_to_ras(data, affine):
    """Reorient image data to standard RAS orientation."""
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


def compute_center_of_mass(data):
    """Compute center of mass of a 3D volume."""
    if data is None:
        return 0, 0, 0
    total = np.sum(data)
    if total > 0:
        indices = np.indices(data.shape)
        cx = int(np.sum(indices[0] * data) / total)
        cy = int(np.sum(indices[1] * data) / total)
        cz = int(np.sum(indices[2] * data) / total)
    else:
        cx, cy, cz = data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2
    return cx, cy, cz


def create_checkerboard(img1, img2, block_size=12):
    """Create checkerboard composite of two 2D images."""
    if img1 is None or img2 is None:
        return None
    if img1.shape != img2.shape:
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


def create_registration_qc_plot(bold_t1w, t1w_ref, output_path,
                                 figsize=(18, 10), original_axcodes=None):
    """
    Create a 2-row x 3-column QC plot for BOLD-to-T1w registration.
    Row 1: T1w reference (axial, coronal, sagittal)
    Row 2: BOLD edges overlaid on T1w (axial, coronal, sagittal)
    """
    if bold_t1w is None or t1w_ref is None:
        print("Error: Both BOLD (in T1w space) and T1w reference are required")
        return

    # Use time-averaged BOLD if 4D
    if bold_t1w.ndim == 4:
        bold_avg = np.mean(bold_t1w, axis=3)
    else:
        bold_avg = bold_t1w

    # Normalize
    t1w_norm = normalize_image(t1w_ref)
    bold_norm = normalize_image(bold_avg)

    # Resample BOLD to T1w grid if needed
    if bold_norm.shape != t1w_norm.shape:
        from scipy.ndimage import zoom
        zoom_factors = [t / b for t, b in zip(t1w_norm.shape, bold_norm.shape)]
        bold_norm = zoom(bold_norm, zoom_factors, order=1)
        print(f"Resampled BOLD to T1w grid: {bold_norm.shape}")

    # Compute edge map from BOLD
    bold_edges = np.zeros_like(bold_norm)
    for z in range(bold_norm.shape[2]):
        bold_edges[:, :, z] = np.abs(ndimage.sobel(bold_norm[:, :, z]))

    # Normalize edges for overlay
    if np.max(bold_edges) > 0:
        bold_edges = bold_edges / np.max(bold_edges)

    # Center slices using T1w
    cx, cy, cz = compute_center_of_mass(t1w_norm)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    views = [
        ('Axial', t1w_norm[:, :, cz], bold_edges[:, :, cz]),
        ('Coronal', t1w_norm[:, cy, :], bold_edges[:, cy, :]),
        ('Sagittal', t1w_norm[cx, :, :], bold_edges[cx, :, :]),
    ]

    for col, (view_name, t1w_slice, edge_slice) in enumerate(views):
        # Row 1: T1w reference
        axes[0, col].imshow(t1w_slice.T, cmap='gray', origin='lower')
        axes[0, col].set_title(f'T1w Ref — {view_name}', fontsize=10)
        axes[0, col].axis('off')

        # Row 2: BOLD edges on T1w
        axes[1, col].imshow(t1w_slice.T, cmap='gray', origin='lower')
        axes[1, col].imshow(edge_slice.T, cmap='hot', origin='lower',
                            alpha=0.6, vmin=0, vmax=1)
        axes[1, col].set_title(f'BOLD Edges on T1w — {view_name}', fontsize=10)
        axes[1, col].axis('off')

    # Orientation info
    axes[0, 0].set_xlabel('R → L', fontsize=8)
    axes[0, 0].set_ylabel('A → P', fontsize=8)
    axes[0, 1].set_xlabel('R → L', fontsize=8)
    axes[0, 1].set_ylabel('S → I', fontsize=8)
    axes[0, 2].set_xlabel('A → P', fontsize=8)
    axes[0, 2].set_ylabel('S → I', fontsize=8)

    title = 'BOLD-to-T1w Registration QC'
    if original_axcodes:
        title += f"  |  orig: {'-'.join(original_axcodes)}  |  display: RAS"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] QC BOLD registration figure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate QC visualization for BOLD-to-T1w registration"
    )
    parser.add_argument("--bold-t1w", type=str, required=True,
                        help="Path to BOLD in T1w space "
                             "(space-T1w_desc-preproc_bold.nii.gz)")
    parser.add_argument("--t1w-ref", type=str, required=True,
                        help="Path to T1w reference image")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for QC figure (.png)")

    args = parser.parse_args()

    # Validate inputs
    for name, path in [('BOLD-T1w', args.bold_t1w), ('T1w ref', args.t1w_ref)]:
        if not os.path.exists(path):
            print(f"Error: {name} not found: {path}")
            sys.exit(1)

    # Load and reorient
    print(f"Loading BOLD in T1w space: {args.bold_t1w}")
    bold_data, bold_affine = load_nifti(args.bold_t1w)
    bold_ras, bold_axcodes, _ = reorient_to_ras(bold_data, bold_affine)
    print(f"  Orientation: {'-'.join(bold_axcodes) if bold_axcodes else 'Unknown'}, "
          f"shape: {bold_ras.shape}")

    print(f"Loading T1w reference: {args.t1w_ref}")
    t1w_data, t1w_affine = load_nifti(args.t1w_ref)
    t1w_ras, original_axcodes, _ = reorient_to_ras(t1w_data, t1w_affine)
    print(f"  Orientation: {'-'.join(original_axcodes) if original_axcodes else 'Unknown'}, "
          f"shape: {t1w_ras.shape}")

    output_path = args.output
    if not output_path.endswith('.png'):
        output_path += '.png'

    create_registration_qc_plot(bold_ras, t1w_ras, output_path,
                                 original_axcodes=original_axcodes)

    print("[DONE] QC BOLD registration visualization completed.")


if __name__ == "__main__":
    main()
