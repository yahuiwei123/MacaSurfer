#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC visualization for BOLD motion correction.
Plots 6 motion parameters (3 translations + 3 rotations), framewise displacement,
and relative/absolute RMS displacement over time.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_motion_params(path):
    """Load MCFLIRT motion parameters (6-column text: rx,ry,rz,tx,ty,tz in radians/mm)."""
    if path is None or not os.path.exists(path):
        return None
    return np.loadtxt(path)


def load_rms(path):
    """Load single-column RMS displacement file."""
    if path is None or not os.path.exists(path):
        return None
    return np.loadtxt(path)


def compute_fd(motion_params):
    """
    Compute framewise displacement from motion parameters.
    FD = |Δtx| + |Δty| + |Δtz| + 50*|Δrx| + 50*|Δry| + 50*|Δrz|
    (approximating 50mm head radius for rotation-to-displacement conversion)
    """
    if motion_params is None:
        return None
    diff = np.diff(motion_params, axis=0)
    diff = np.vstack([np.zeros(6), diff])
    fd = np.abs(diff[:, 0]) + np.abs(diff[:, 1]) + np.abs(diff[:, 2]) + \
         50 * (np.abs(diff[:, 3]) + np.abs(diff[:, 4]) + np.abs(diff[:, 5]))
    return fd


def create_motion_qc_plot(motion_params, rms_rel, rms_abs, output_path,
                           figsize=(14, 10)):
    """
    Create a multi-panel QC plot for BOLD motion correction.

    Panel layout:
      Row 1: 3 translation parameters (tx, ty, tz)
      Row 2: 3 rotation parameters (rx, ry, rz)
      Row 3: Framewise displacement
      Row 4: Relative RMS displacement
      Row 5: Absolute RMS displacement
    """
    has_motion = motion_params is not None
    has_fd = has_motion
    has_rms_rel = rms_rel is not None
    has_rms_abs = rms_abs is not None

    # Count active panels
    n_panels = 0
    if has_motion:
        n_panels += 2  # translations + rotations
    if has_fd:
        n_panels += 1
    if has_rms_rel:
        n_panels += 1
    if has_rms_abs:
        n_panels += 1

    if n_panels == 0:
        print("[WARN] No motion data to plot")
        return

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    # --- Panel 1: Translations ---
    if has_motion:
        ax = axes[panel_idx]
        n_vols = motion_params.shape[0]
        x = np.arange(n_vols)
        ax.plot(x, motion_params[:, 0], 'r-', lw=0.8, label='tx (mm)')
        ax.plot(x, motion_params[:, 1], 'g-', lw=0.8, label='ty (mm)')
        ax.plot(x, motion_params[:, 2], 'b-', lw=0.8, label='tz (mm)')
        ax.axhline(y=0, color='gray', lw=0.5, ls='--')
        ax.set_ylabel('Translation (mm)')
        ax.set_xlabel('Volume')
        ax.legend(loc='upper right', fontsize=8, ncol=3)
        ax.set_title('Translations (tx, ty, tz)', fontsize=10)
        ax.set_xlim(0, n_vols - 1)
        ax.grid(True, alpha=0.3)
        panel_idx += 1

        # --- Panel 2: Rotations ---
        ax = axes[panel_idx]
        ax.plot(x, motion_params[:, 3], 'r-', lw=0.8, label='rx (rad)')
        ax.plot(x, motion_params[:, 4], 'g-', lw=0.8, label='ry (rad)')
        ax.plot(x, motion_params[:, 5], 'b-', lw=0.8, label='rz (rad)')
        ax.axhline(y=0, color='gray', lw=0.5, ls='--')
        ax.set_ylabel('Rotation (rad)')
        ax.set_xlabel('Volume')
        ax.legend(loc='upper right', fontsize=8, ncol=3)
        ax.set_title('Rotations (rx, ry, rz)', fontsize=10)
        ax.set_xlim(0, n_vols - 1)
        ax.grid(True, alpha=0.3)
        panel_idx += 1

    # --- Panel: Framewise Displacement ---
    if has_fd:
        fd = compute_fd(motion_params)
        ax = axes[panel_idx]
        ax.plot(fd, 'k-', lw=0.8)
        mean_fd = np.mean(fd)
        ax.axhline(y=mean_fd, color='orange', lw=1, ls='--',
                   label=f'Mean FD = {mean_fd:.3f} mm')
        ax.axhline(y=0.5, color='red', lw=0.5, ls=':', label='FD = 0.5 mm')
        ax.set_ylabel('FD (mm)')
        ax.set_xlabel('Volume')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(f'Framewise Displacement (mean = {mean_fd:.3f} mm)', fontsize=10)
        ax.set_xlim(0, len(fd) - 1)
        ax.grid(True, alpha=0.3)
        panel_idx += 1

    # --- Panel: Relative RMS ---
    if has_rms_rel:
        ax = axes[panel_idx]
        ax.plot(rms_rel, 'b-', lw=0.8)
        ax.set_ylabel('Rel. RMS (mm)')
        ax.set_xlabel('Volume')
        ax.set_title('Relative RMS Displacement', fontsize=10)
        ax.set_xlim(0, len(rms_rel) - 1)
        ax.grid(True, alpha=0.3)
        panel_idx += 1

    # --- Panel: Absolute RMS ---
    if has_rms_abs:
        ax = axes[panel_idx]
        ax.plot(rms_abs, 'b-', lw=0.8)
        ax.set_ylabel('Abs. RMS (mm)')
        ax.set_xlabel('Volume')
        ax.set_title('Absolute RMS Displacement', fontsize=10)
        ax.set_xlim(0, len(rms_abs) - 1)
        ax.grid(True, alpha=0.3)
        panel_idx += 1

    plt.suptitle('BOLD Motion Correction QC', fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] QC BOLD motion figure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate QC visualization for BOLD motion correction"
    )
    parser.add_argument("--motion-params", type=str, default=None,
                        help="Path to MCFLIRT motion parameters file "
                             "(6-column: rx,ry,rz,tx,ty,tz)")
    parser.add_argument("--rms-rel", type=str, default=None,
                        help="Path to relative RMS displacement file")
    parser.add_argument("--rms-abs", type=str, default=None,
                        help="Path to absolute RMS displacement file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for QC figure (.png)")

    args = parser.parse_args()

    # Load data
    motion_params = None
    if args.motion_params:
        if not os.path.exists(args.motion_params):
            print(f"[WARN] Motion params not found: {args.motion_params}")
        else:
            motion_params = load_motion_params(args.motion_params)
            print(f"Loaded motion params: {motion_params.shape}")

    rms_rel = None
    if args.rms_rel:
        if not os.path.exists(args.rms_rel):
            print(f"[WARN] Relative RMS not found: {args.rms_rel}")
        else:
            rms_rel = load_rms(args.rms_rel)
            print(f"Loaded relative RMS: {len(rms_rel)} points")

    rms_abs = None
    if args.rms_abs:
        if not os.path.exists(args.rms_abs):
            print(f"[WARN] Absolute RMS not found: {args.rms_abs}")
        else:
            rms_abs = load_rms(args.rms_abs)
            print(f"Loaded absolute RMS: {len(rms_abs)} points")

    output_path = args.output
    if not output_path.endswith('.png'):
        output_path += '.png'

    create_motion_qc_plot(motion_params, rms_rel, rms_abs, output_path)

    print("[DONE] QC BOLD motion visualization completed.")


if __name__ == "__main__":
    main()
