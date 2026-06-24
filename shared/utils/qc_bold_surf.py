#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC visualization for BOLD volume-to-surface projection.
Reads GIFTI functional data and displays:
  - BOLD timeseries mean projected onto the cortical surface
  - Ribbon overlay on T1w for spatial context
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_gifti(path):
    """Load GIFTI file and return data arrays."""
    if path is None or not os.path.exists(path):
        return None
    img = nib.load(path)
    return [darray.data for darray in img.darrays]


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


def create_surf_qc_plot(bold_surf_arrays, ribbon, output_path,
                         figsize=(16, 8), original_axcodes=None):
    """
    Create QC plot for BOLD surface projection.

    Uses a ribbon overlay approach in volume space since flatmap/inflated surface
    rendering requires workbench tools. Shows:
      Left: BOLD surface timeseries mean as bar chart
      Right: Ribbon overlay on selected slices
    """
    if bold_surf_arrays is None:
        print("Error: BOLD surface data is required")
        return

    # Extract mean timeseries per vertex
    surf_data = bold_surf_arrays[0]  # shape: (n_verts, n_timepoints)
    if surf_data.ndim == 2:
        mean_signal = np.mean(surf_data, axis=1)
        std_signal = np.std(surf_data, axis=1)
    else:
        mean_signal = surf_data
        std_signal = np.zeros_like(mean_signal)

    fig = plt.figure(figsize=figsize)

    # --- Left panel: Surface signal distribution ---
    ax1 = fig.add_subplot(1, 2, 1)

    # Plot mean signal per vertex as a spatial "fingerprint"
    n_verts = len(mean_signal)
    x = np.arange(n_verts)
    ax1.fill_between(x, mean_signal - std_signal, mean_signal + std_signal,
                     alpha=0.3, color='steelblue')
    ax1.plot(x, mean_signal, 'b-', lw=0.5)
    ax1.axhline(y=0, color='k', lw=0.5, ls='--')
    ax1.set_xlabel('Vertex index')
    ax1.set_ylabel('Mean BOLD Signal')
    ax1.set_title(f'Surface BOLD Signal (n={n_verts} vertices)')

    # Stats annotation
    mean_val = np.mean(mean_signal)
    std_val = np.std(mean_signal)
    pct_nonzero = 100 * np.mean(np.abs(mean_signal) > 1e-6)
    ax1.text(0.02, 0.98,
             f'Mean: {mean_val:.4f}\n'
             f'Std:  {std_val:.4f}\n'
             f'Non-zero: {pct_nonzero:.1f}%',
             transform=ax1.transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.grid(True, alpha=0.3)

    # --- Right panel: Ribbon overlay on slices ---
    if ribbon is not None:
        # Show ribbon at 3 axial slices
        nz = ribbon.shape[2]
        z_slices = [nz // 4, nz // 2, 3 * nz // 4]

        for i, z in enumerate(z_slices):
            ax = fig.add_subplot(3, 3, 3 + i * 3 + 1)
            ribbon_slice = ribbon[:, :, z]
            ax.imshow(ribbon_slice.T, cmap='RdBu', origin='lower',
                      vmin=-1, vmax=1)
            ax.set_title(f'Ribbon Z={z}', fontsize=8)
            ax.axis('off')

        # Also show maximum intensity projection
        for i, z in enumerate(z_slices):
            ax = fig.add_subplot(3, 3, 3 + i * 3 + 2)
            # Show ribbon edges
            ribbon_bin = (ribbon[:, :, z] > 0).astype(float)
            ax.imshow(ribbon_bin.T, cmap='gray', origin='lower')
            ax.set_title(f'Ribbon Mask Z={z}', fontsize=8)
            ax.axis('off')
    else:
        # If no ribbon, show histogram of surface values
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.hist(mean_signal, bins=100, color='steelblue', edgecolor='none',
                 alpha=0.7)
        ax2.axvline(x=0, color='k', lw=0.5, ls='--')
        ax2.set_xlabel('Mean BOLD Signal')
        ax2.set_ylabel('Vertex count')
        ax2.set_title('Surface Signal Distribution')

    title = 'BOLD Surface Projection QC'
    if original_axcodes:
        title += f"  |  orig: {'-'.join(original_axcodes)}  |  display: RAS"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] QC BOLD surface figure saved to: {output_path}")
    print(f"[INFO] {n_verts} vertices, mean signal = {mean_val:.4f}, "
          f"{pct_nonzero:.1f}% non-zero")


def main():
    parser = argparse.ArgumentParser(
        description="Generate QC visualization for BOLD surface projection"
    )
    parser.add_argument("--surf-gii", type=str, required=True,
                        help="Path to BOLD surface GIFTI file (.func.gii)")
    parser.add_argument("--ribbon", type=str, default=None,
                        help="Path to cortical ribbon volume (optional)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for QC figure (.png)")

    args = parser.parse_args()

    # Validate
    if not os.path.exists(args.surf_gii):
        print(f"Error: Surface GIFTI not found: {args.surf_gii}")
        sys.exit(1)

    # Load GIFTI
    print(f"Loading BOLD surface: {args.surf_gii}")
    surf_arrays = load_gifti(args.surf_gii)
    if surf_arrays is None or len(surf_arrays) == 0:
        print("Error: Could not load GIFTI data")
        sys.exit(1)
    print(f"  Found {len(surf_arrays)} data arrays")
    for i, arr in enumerate(surf_arrays):
        print(f"  Array {i}: shape {arr.shape}")

    # Load ribbon
    ribbon_ras = None
    original_axcodes = None
    if args.ribbon:
        if not os.path.exists(args.ribbon):
            print(f"[WARN] Ribbon not found: {args.ribbon}")
        else:
            print(f"Loading ribbon: {args.ribbon}")
            ribbon_data, ribbon_affine = load_nifti(args.ribbon)
            ribbon_ras, original_axcodes, _ = reorient_to_ras(ribbon_data, ribbon_affine)
            print(f"  Orientation: {'-'.join(original_axcodes) if original_axcodes else 'Unknown'}, "
                  f"shape: {ribbon_ras.shape}")

    output_path = args.output
    if not output_path.endswith('.png'):
        output_path += '.png'

    create_surf_qc_plot(surf_arrays, ribbon_ras, output_path,
                         original_axcodes=original_axcodes)

    print("[DONE] QC BOLD surface visualization completed.")


if __name__ == "__main__":
    main()
