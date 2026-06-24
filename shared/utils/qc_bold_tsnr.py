#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC visualization for BOLD TSNR (temporal signal-to-noise ratio).
Computes TSNR from preprocessed BOLD data and displays:
  - Multi-slice grid of TSNR map with colorbar
  - Histogram of TSNR values within brain mask
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def compute_tsnr(bold_data, mask=None):
    """
    Compute TSNR from 4D BOLD data.
    TSNR = mean / std along time axis (axis=3).
    """
    mean = np.mean(bold_data, axis=3)
    std = np.std(bold_data, axis=3)
    std[std < 1e-8] = 1e-8
    tsnr = mean / std
    return tsnr


def create_tsnr_qc_plot(tsnr, bold_mask, output_path,
                         rows=5, cols=5, figsize=(16, 12),
                         original_axcodes=None):
    """
    Create QC plot showing TSNR map on a grid of axial slices,
    plus a histogram of TSNR values within the brain mask.
    """
    if tsnr is None:
        print("Error: TSNR data is required")
        return

    # Mask TSNR
    if bold_mask is not None:
        bold_mask = bold_mask > 0
        if bold_mask.shape != tsnr.shape:
            from scipy.ndimage import zoom
            zoom_factors = [t / m for t, m in zip(tsnr.shape, bold_mask.shape)]
            bold_mask = zoom(bold_mask.astype(float), zoom_factors, order=0) > 0.5
        tsnr_masked = tsnr.copy()
        tsnr_masked[~bold_mask] = 0
    else:
        tsnr_masked = tsnr

    # Slice indices
    z_dim = tsnr.shape[2]
    start_slice = int(z_dim * 0.1)
    end_slice = int(z_dim * 0.9)
    total_slices = rows * cols
    slice_indices = np.linspace(start_slice, end_slice, total_slices, dtype=int)

    # TSNR display range (clip at 99th percentile within mask)
    if bold_mask is not None:
        brain_vals = tsnr[bold_mask]
        if len(brain_vals) > 0:
            vmax = np.percentile(brain_vals, 99)
        else:
            vmax = np.percentile(tsnr, 99)
    else:
        vmax = np.percentile(tsnr, 99)
    vmax = max(vmax, 10)  # minimum display range

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(rows, cols + 1, width_ratios=[1] * cols + [0.6])

    # TSNR slices
    axes_slices = []
    for i in range(total_slices):
        r, c = i // cols, i % cols
        ax = fig.add_subplot(gs[r, c])
        axes_slices.append(ax)

    for i, z in enumerate(slice_indices):
        ax = axes_slices[i]
        im = ax.imshow(tsnr_masked[:, :, z].T, cmap='hot', origin='lower',
                       vmin=0, vmax=vmax)
        ax.set_title(f'Z={z}', fontsize=7)
        ax.axis('off')

    for i in range(len(slice_indices), len(axes_slices)):
        axes_slices[i].axis('off')

    # Colorbar
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = plt.colorbar(im, cax=cbar_ax, shrink=0.8)
    cbar.set_label('TSNR', fontsize=10)

    # Report stats
    if bold_mask is not None:
        brain_vals = tsnr[bold_mask]
    else:
        brain_vals = tsnr[tsnr > 0]
    mean_tsnr = np.mean(brain_vals) if len(brain_vals) > 0 else 0
    median_tsnr = np.median(brain_vals) if len(brain_vals) > 0 else 0

    title = (f'BOLD TSNR QC — Mean: {mean_tsnr:.1f}  |  Median: {median_tsnr:.1f}')
    if original_axcodes:
        title += f"  |  orig: {'-'.join(original_axcodes)}  |  display: RAS"
    plt.suptitle(title, fontsize=14, fontweight='bold')

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] QC BOLD TSNR figure saved to: {output_path}")
    print(f"[INFO] Mean TSNR: {mean_tsnr:.2f}, Median TSNR: {median_tsnr:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate QC visualization for BOLD TSNR"
    )
    parser.add_argument("--bold-preproc", type=str, required=True,
                        help="Path to preprocessed BOLD (4D NIfTI)")
    parser.add_argument("--bold-mask", type=str, default=None,
                        help="Path to brain mask in EPI space (3D NIfTI)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for QC figure (.png)")
    parser.add_argument("--rows", type=int, default=5,
                        help="Number of rows in the grid (default: 5)")
    parser.add_argument("--cols", type=int, default=5,
                        help="Number of columns in the grid (default: 5)")

    args = parser.parse_args()

    # Validate
    if not os.path.exists(args.bold_preproc):
        print(f"Error: BOLD preproc not found: {args.bold_preproc}")
        sys.exit(1)

    # Load BOLD
    print(f"Loading BOLD preproc: {args.bold_preproc}")
    bold_data, bold_affine = load_nifti(args.bold_preproc)
    bold_ras, original_axcodes, _ = reorient_to_ras(bold_data, bold_affine)
    print(f"  Orientation: {'-'.join(original_axcodes) if original_axcodes else 'Unknown'}, "
          f"shape: {bold_ras.shape}")

    if bold_ras.ndim != 4:
        print(f"Error: Expected 4D BOLD data, got shape {bold_ras.shape}")
        sys.exit(1)

    # Load mask
    mask_ras = None
    if args.bold_mask:
        if not os.path.exists(args.bold_mask):
            print(f"[WARN] Brain mask not found: {args.bold_mask}")
        else:
            print(f"Loading brain mask: {args.bold_mask}")
            mask_data, mask_affine = load_nifti(args.bold_mask)
            mask_ras, _, _ = reorient_to_ras(mask_data, mask_affine)
            print(f"  Mask shape: {mask_ras.shape}")

    # Compute TSNR
    print("Computing TSNR...")
    tsnr = compute_tsnr(bold_ras, mask_ras)
    print(f"  TSNR shape: {tsnr.shape}")

    output_path = args.output
    if not output_path.endswith('.png'):
        output_path += '.png'

    create_tsnr_qc_plot(tsnr, mask_ras, output_path,
                         rows=args.rows, cols=args.cols,
                         original_axcodes=original_axcodes)

    print("[DONE] QC BOLD TSNR visualization completed.")


if __name__ == "__main__":
    main()
