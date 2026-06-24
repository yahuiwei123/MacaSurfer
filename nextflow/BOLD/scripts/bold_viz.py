#!/usr/bin/env python3
"""
Generate PNG visualizations of BOLD preprocessing steps for debugging.
Called from bold_preprocess.sh after each major step.
"""
import argparse
import sys
from pathlib import Path

import nibabel as nb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def viz_single(image_path, output_png, title="", slice_z=None, vol_t=None,
               vmin=None, vmax=None, cmap='gray', overlay=None, overlay_alpha=0.4):
    """Visualize a single 3D/4D NIfTI image to PNG."""
    img = nb.load(image_path)
    data = img.get_fdata()

    if data.ndim == 4:
        nz, nt = data.shape[2], data.shape[3]
        z = slice_z if slice_z is not None else nz // 2
        t = vol_t if vol_t is not None else nt // 2
        img_slice = data[:, :, z, t]
        extra_info = f"vol={t}/{nt}"
    else:
        nz = data.shape[2]
        z = slice_z if slice_z is not None else nz // 2
        img_slice = data[:, :, z]
        extra_info = ""

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    vmin_val = vmin if vmin is not None else 0
    vmax_val = vmax if vmax is not None else np.percentile(img_slice, 99)
    ax.imshow(img_slice.T, cmap=cmap, origin='lower', vmin=vmin_val, vmax=vmax_val)

    if overlay is not None:
        ov = nb.load(overlay).get_fdata()
        if ov.ndim == 4 or ov.ndim == 3:
            z_idx = z if z < ov.shape[2] else ov.shape[2] // 2
            ov_slice = ov[:, :, z_idx] if ov.ndim == 3 else ov[:, :, z_idx, 0]
        mask = (ov_slice > 0).astype(float)
        ax.imshow(mask.T, cmap='Reds', origin='lower', alpha=overlay_alpha * mask.T)

    ax.set_title(f"{title}\nz={z} {extra_info}".strip(), fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [VIZ] {output_png}")


def viz_compare(pre_path, post_path, output_png, title="Pre vs Post",
                slice_z=None, vol_t=None, cmap='gray'):
    """Side-by-side comparison of two NIfTI images with difference map."""
    pre = nb.load(pre_path)
    post = nb.load(post_path)
    pre_data = pre.get_fdata()
    post_data = post.get_fdata()

    if pre_data.ndim == 4:
        nz, nt = pre_data.shape[2], pre_data.shape[3]
        z = slice_z if slice_z is not None else nz // 2
        t = vol_t if vol_t is not None else nt // 2
        pre_slice = pre_data[:, :, z, t]
        post_slice = post_data[:, :, z, t]
    else:
        nz = pre_data.shape[2]
        z = slice_z if slice_z is not None else nz // 2
        pre_slice = pre_data[:, :, z]
        post_slice = post_data[:, :, z]

    diff = post_slice - pre_slice
    vmax_pct = np.percentile(np.maximum(pre_slice, post_slice), 99)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(pre_slice.T, cmap=cmap, origin='lower', vmin=0, vmax=vmax_pct)
    axes[0].set_title(f'Before\nz={z}')
    axes[0].axis('off')

    axes[1].imshow(post_slice.T, cmap=cmap, origin='lower', vmin=0, vmax=vmax_pct)
    axes[1].set_title(f'After\nz={z}')
    axes[1].axis('off')

    diff_vmax = np.percentile(np.abs(diff), 99) or 1
    im = axes[2].imshow(diff.T, cmap='RdBu_r', origin='lower', vmin=-diff_vmax, vmax=diff_vmax)
    axes[2].set_title(f'Diff (max={diff_vmax:.1f})')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], shrink=0.8)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [VIZ] {output_png}")


def viz_motion(rms_rel_path, rms_abs_path, output_png, title="Motion"):
    """Plot motion parameters from RMS files (single-column text)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    for ax, rms_path, label in [(axes[0], rms_rel_path, 'Relative RMS (mm)'),
                                  (axes[1], rms_abs_path, 'Absolute RMS (mm)')]:
        if rms_path and Path(rms_path).exists():
            try:
                data = np.loadtxt(rms_path)
                ax.plot(data, 'b-', lw=0.5)
                ax.set_ylabel(label)
                ax.set_xlabel('Volume')
                ax.grid(True, alpha=0.3)
            except Exception:
                ax.text(0.5, 0.5, f'Failed to read {rms_path}',
                        ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, f'{rms_path}: not found',
                    ha='center', va='center', transform=ax.transAxes)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [VIZ] {output_png}")


def viz_align_check(bold_t1w_path, t1w_ref_path, output_png, title="BOLD-T1w alignment",
                    vol_t=None, n_slices=5):
    """Visualize alignment by showing corresponding physical slices side-by-side.

    Uses affine matrices to find matching slices across different voxel grids.
    Row 0 = T1w, Row 1 = BOLD (in T1w space via header transform).
    No overlay — different grids make overlay misleading.
    """
    if t1w_ref_path is None:
        print("[VIZ] align_check needs --reference <t1w>")
        return

    bold_img = nb.load(bold_t1w_path)
    t1w_img = nb.load(t1w_ref_path)
    bold_data = bold_img.get_fdata()
    t1w_data = t1w_img.get_fdata()

    nz_bold = bold_data.shape[2]
    nz_t1w = t1w_data.shape[2]
    t = vol_t if vol_t is not None else (bold_data.shape[3] // 2 if bold_data.ndim == 4 else 0)

    z_indices = np.linspace(nz_bold // 8, 7 * nz_bold // 8, n_slices, dtype=int)
    ncols = n_slices
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 7))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    t1w_vmax = np.percentile(t1w_data[t1w_data > 0], 99) if (t1w_data > 0).any() else t1w_data.max()
    bold_vmax = np.percentile(bold_data[bold_data > 0], 99) if bold_data.ndim == 4 else np.percentile(bold_data[bold_data > 0], 99)

    # Map BOLD→T1w voxel coords via affine (RAS physical space)
    bold2ras = bold_img.affine
    ras2t1w = np.linalg.inv(t1w_img.affine)

    for col, z_bold in enumerate(z_indices):
        # Physical RAS of this BOLD slice center → corresponding T1w voxel z
        center_ras = bold2ras @ np.array([bold_data.shape[0]/2, bold_data.shape[1]/2, z_bold, 1.0])
        center_t1w = ras2t1w @ center_ras
        z_t1w = int(np.clip(round(center_t1w[2]), 0, nz_t1w - 1))

        # Row 0: T1w at matching physical slice
        t1w_sl = t1w_data[:, :, z_t1w] if t1w_data.ndim >= 3 else t1w_data
        axes[0, col].imshow(t1w_sl.T, cmap='gray', origin='lower', vmin=0, vmax=t1w_vmax)
        axes[0, col].set_title(f'T1w (z={z_t1w})')
        axes[0, col].axis('off')

        # Row 1: BOLD (header in T1w space, native grid)
        if bold_data.ndim == 4:
            bold_sl = bold_data[:, :, z_bold, t]
        else:
            bold_sl = bold_data[:, :, z_bold]
        axes[1, col].imshow(bold_sl.T, cmap='hot', origin='lower', vmin=0, vmax=bold_vmax)
        axes[1, col].set_title(f'BOLD (z={z_bold})')
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('T1w ref', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('BOLD→T1w', fontsize=12, fontweight='bold')

    info = f"  T1w: {t1w_data.shape} | BOLD: {bold_data.shape[:3]}"
    if bold_data.ndim == 4:
        info += f"  vol={t}/{bold_data.shape[3]}"
    fig.suptitle(f"{title}\n{info}".strip(), fontsize=13)
    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [VIZ] {output_png}")


def main():
    parser = argparse.ArgumentParser(description='Visualize BOLD preprocessing step')
    parser.add_argument('--mode', required=True,
                        choices=['single', 'compare', 'motion', 'multi-slice', 'align_check'],
                        help='Visualization mode')
    parser.add_argument('--input', nargs='+', default=[])
    parser.add_argument('--output', required=True, help='Output PNG path')
    parser.add_argument('--title', default='', help='Plot title')
    parser.add_argument('--slice_z', type=int, default=None)
    parser.add_argument('--vol_t', type=int, default=None)
    parser.add_argument('--overlay', default=None, help='Overlay NIfTI for single mode')
    parser.add_argument('--reference', default=None, help='Reference underlay for align_check')
    parser.add_argument('--n_slices', type=int, default=4, help='Number of slices for multi-slice')
    args = parser.parse_args()

    if args.mode == 'single':
        viz_single(args.input[0], args.output, args.title,
                   slice_z=args.slice_z, vol_t=args.vol_t,
                   overlay=args.overlay)
    elif args.mode == 'compare':
        viz_compare(args.input[0], args.input[1], args.output, args.title,
                    slice_z=args.slice_z, vol_t=args.vol_t)
    elif args.mode == 'motion':
        viz_motion(args.input[0] if len(args.input) > 0 else None,
                   args.input[1] if len(args.input) > 1 else None,
                   args.output, args.title)
    elif args.mode == 'align_check':
        ref = args.reference if args.reference else (
            args.input[1] if len(args.input) > 1 else None)
        viz_align_check(args.input[0], ref,
                        args.output, args.title, args.vol_t, args.n_slices)
    elif args.mode == 'multi-slice':
        # Generate a multi-slice montage
        img = nb.load(args.input[0])
        data = img.get_fdata()
        nz = data.shape[2]
        t = args.vol_t if args.vol_t is not None else (data.shape[3] // 2 if data.ndim == 4 else 0)
        z_indices = np.linspace(nz // 8, 7 * nz // 8, args.n_slices, dtype=int)
        ncols = min(args.n_slices, 4)
        nrows = int(np.ceil(args.n_slices / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[np.newaxis, :]
        elif ncols == 1:
            axes = axes[:, np.newaxis]
        vmax_val = np.percentile(data[data > 0] if (data > 0).any() else data, 99)
        for idx, z in enumerate(z_indices):
            r, c = idx // ncols, idx % ncols
            sl = data[:, :, z, t] if data.ndim == 4 else data[:, :, z]
            axes[r, c].imshow(sl.T, cmap='gray', origin='lower', vmin=0, vmax=vmax_val)
            axes[r, c].set_title(f'z={z}')
            axes[r, c].axis('off')
        for idx in range(len(z_indices), nrows * ncols):
            r, c = idx // ncols, idx % ncols
            axes[r, c].axis('off')
        fig.suptitle(args.title, fontsize=13)
        plt.tight_layout()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  [VIZ] {args.output}")


if __name__ == '__main__':
    main()
