#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC visualization for BOLD signal: carpet plot.
Shows:
  - Carpet plot (voxel-wise timeseries, sorted by similarity)
  - Global signal (mean over brain voxels)
  - Framewise displacement (if provided)
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
    """Load NIfTI file and return data array."""
    if path is None or not os.path.exists(path):
        return None
    img = nib.load(path)
    return img.get_fdata()


def load_text(path):
    """Load single-column text file."""
    if path is None or not os.path.exists(path):
        return None
    return np.loadtxt(path)


def compute_fd_from_motion(motion_params):
    """Compute FD from 6-column motion parameters."""
    if motion_params is None:
        return None
    diff = np.diff(motion_params, axis=0)
    diff = np.vstack([np.zeros(6), diff])
    fd = np.abs(diff[:, 0]) + np.abs(diff[:, 1]) + np.abs(diff[:, 2]) + \
         50 * (np.abs(diff[:, 3]) + np.abs(diff[:, 4]) + np.abs(diff[:, 5]))
    return fd


def create_carpet_qc_plot(bold_data, bold_mask, output_path,
                           motion_params=None, rms_rel=None,
                           figsize=(14, 9), max_voxels=2000):
    """
    Create carpet plot for BOLD QC.

    Layout:
      Top panel: Carpet plot (voxels x time)
      Middle panel: Global signal
      Bottom panel: Framewise displacement (if available)
    """
    if bold_data is None:
        print("Error: BOLD data is required")
        return
    if bold_data.ndim != 4:
        print(f"Error: Expected 4D BOLD data, got {bold_data.ndim}D")
        return

    n_t = bold_data.shape[3]

    # Extract brain voxels using mask
    if bold_mask is not None:
        # Ensure mask matches data shape
        if bold_mask.shape != bold_data.shape[:3]:
            from scipy.ndimage import zoom
            zoom_factors = [d / m for d, m in zip(bold_data.shape[:3], bold_mask.shape)]
            bold_mask = zoom(bold_mask.astype(float), zoom_factors, order=0) > 0.5
        mask_3d = bold_mask > 0
    else:
        # Use simple threshold to find brain voxels
        mean_vol = np.mean(np.abs(bold_data), axis=3)
        mask_3d = mean_vol > np.percentile(mean_vol, 50)

    brain_ts = bold_data[mask_3d, :]
    n_vox = brain_ts.shape[0]
    print(f"[INFO] Brain voxels: {n_vox}")

    if n_vox == 0:
        print("[WARN] No brain voxels found")
        return

    # Subsample if needed
    if n_vox > max_voxels:
        idx = np.linspace(0, n_vox - 1, max_voxels, dtype=int)
        brain_ts = brain_ts[idx, :]
        n_vox = max_voxels

    # Normalize each voxel's time series (z-score)
    voxel_means = np.mean(brain_ts, axis=1, keepdims=True)
    voxel_stds = np.std(brain_ts, axis=1, keepdims=True)
    voxel_stds[voxel_stds < 1e-8] = 1e-8
    brain_ts_norm = (brain_ts - voxel_means) / voxel_stds

    # Sort voxels by correlation to global signal for better visual structure
    gs = np.mean(brain_ts_norm, axis=0)
    corr_with_gs = np.corrcoef(brain_ts_norm, gs[np.newaxis, :])[:-1, -1]
    sort_idx = np.argsort(corr_with_gs)
    brain_ts_sorted = brain_ts_norm[sort_idx, :]

    # Determine layout
    has_fd = motion_params is not None or rms_rel is not None
    n_panels = 2 + (1 if has_fd else 0)
    height_ratios = [4, 1] + ([1] if has_fd else [])

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize,
                              gridspec_kw={'height_ratios': height_ratios})
    if n_panels == 2:
        axes = [axes[0], axes[1]]

    # Panel 1: Carpet plot
    ax = axes[0]
    im = ax.imshow(brain_ts_sorted, aspect='auto', cmap='gray',
                   vmin=-3, vmax=3, interpolation='nearest')
    ax.set_ylabel('Brain voxels (sorted)')
    ax.set_title('BOLD Carpet Plot', fontsize=10)
    ax.set_xticks([])
    plt.colorbar(im, ax=ax, shrink=0.02, label='z-score')

    # Panel 2: Global signal
    ax = axes[1]
    ax.plot(gs, 'b-', lw=0.6)
    ax.set_ylabel('Global Signal')
    ax.set_xlabel('Volume')
    ax.set_xlim(0, n_t - 1)
    ax.grid(True, alpha=0.3)

    # Compute and display GS statistics
    gs_std = np.std(gs)
    ax.set_title(f'Global Signal (std = {gs_std:.3f})', fontsize=9)

    # Panel 3: FD / RMS
    if has_fd:
        ax = axes[2]
        if motion_params is not None:
            fd = compute_fd_from_motion(motion_params)
            mean_fd = np.mean(fd)
            ax.plot(fd, 'k-', lw=0.6, label=f'FD (mean={mean_fd:.3f} mm)')
            ax.axhline(y=0.5, color='red', lw=0.5, ls=':', label='0.5 mm')
            ax.legend(loc='upper right', fontsize=8)
        elif rms_rel is not None:
            ax.plot(rms_rel, 'b-', lw=0.6, label='Relative RMS')
            ax.legend(loc='upper right', fontsize=8)
        ax.set_ylabel('Displacement (mm)')
        ax.set_xlabel('Volume')
        ax.set_xlim(0, n_t - 1)
        ax.grid(True, alpha=0.3)

    plt.suptitle('BOLD Carpet Plot QC', fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] QC BOLD carpet figure saved to: {output_path}")
    print(f"[INFO] {n_vox} voxels x {n_t} volumes, GS std = {gs_std:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate carpet plot QC visualization for BOLD"
    )
    parser.add_argument("--bold-preproc", type=str, required=True,
                        help="Path to preprocessed BOLD (4D NIfTI)")
    parser.add_argument("--bold-mask", type=str, default=None,
                        help="Path to brain mask in EPI space (3D NIfTI)")
    parser.add_argument("--motion-params", type=str, default=None,
                        help="Path to MCFLIRT motion parameters (6-column text)")
    parser.add_argument("--rms-rel", type=str, default=None,
                        help="Path to relative RMS displacement file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for QC figure (.png)")

    args = parser.parse_args()

    # Validate
    if not os.path.exists(args.bold_preproc):
        print(f"Error: BOLD preproc not found: {args.bold_preproc}")
        sys.exit(1)

    # Load BOLD
    print(f"Loading BOLD preproc: {args.bold_preproc}")
    bold_data = load_nifti(args.bold_preproc)
    print(f"  Shape: {bold_data.shape}")

    # Load mask
    bold_mask = None
    if args.bold_mask:
        if os.path.exists(args.bold_mask):
            print(f"Loading brain mask: {args.bold_mask}")
            bold_mask = load_nifti(args.bold_mask)
            print(f"  Mask shape: {bold_mask.shape}")
        else:
            print(f"[WARN] Brain mask not found: {args.bold_mask}")

    # Load motion
    motion_params = None
    if args.motion_params:
        if os.path.exists(args.motion_params):
            motion_params = load_text(args.motion_params)
            print(f"Loaded motion params: {motion_params.shape}")
        else:
            print(f"[WARN] Motion params not found: {args.motion_params}")

    rms_rel = None
    if args.rms_rel:
        if os.path.exists(args.rms_rel):
            rms_rel = load_text(args.rms_rel)
            print(f"Loaded relative RMS: {len(rms_rel)} points")
        else:
            print(f"[WARN] RMS rel not found: {args.rms_rel}")

    output_path = args.output
    if not output_path.endswith('.png'):
        output_path += '.png'

    create_carpet_qc_plot(bold_data, bold_mask, output_path,
                           motion_params=motion_params, rms_rel=rms_rel)

    print("[DONE] QC BOLD carpet plot visualization completed.")


if __name__ == "__main__":
    main()
