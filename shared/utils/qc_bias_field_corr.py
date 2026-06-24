#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC visualization for bias field correction results.
Display before/after comparison, per-class tissue statistics,
and Gaussian-fitted intensity distributions.
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# Tissue label definitions (label -> (name, color))
TISSUE_INFO = {
    1:  ('CSF',           (0.7, 0.7, 0.7)),
    2:  ('GM',            (0.9, 0.3, 0.3)),
    3:  ('WM',            (0.3, 0.3, 0.9)),
    4:  ('Deep GM',       (0.9, 0.6, 0.2)),
    5:  ('Cerebellum GM', (0.2, 0.8, 0.2)),
    6:  ('Cerebellum WM', (0.2, 0.6, 0.8)),
    7:  ('Brainstem',     (0.8, 0.2, 0.8)),
    8:  ('Ventricle',     (0.5, 0.5, 0.5)),
    9:  ('Thalamus',      (0.9, 0.5, 0.5)),
    10: ('Caudate',       (0.5, 0.9, 0.5)),
    11: ('Putamen',       (0.5, 0.5, 0.9)),
    12: ('Hippocampus',   (0.9, 0.9, 0.3)),
    13: ('Amygdala',      (0.9, 0.3, 0.9)),
    14: ('Pallidum',      (0.3, 0.9, 0.9)),
}


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


def normalize_image(img, percentile=99):
    """Normalize image to [0, 1] range using percentile clipping."""
    if img is None:
        return None
    vmin, vmax = np.percentile(img, 1), np.percentile(img, percentile)
    img_clipped = np.clip(img, vmin, vmax)
    img_norm = img_clipped - vmin
    if np.max(img_norm) > 0:
        img_norm = img_norm / np.max(img_norm)
    return img_norm


def gaussian_pdf(x, mean, std):
    """Gaussian (normal) probability density function."""
    if std <= 0:
        return np.zeros_like(x)
    return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)


def compute_class_stats(image, seg, labels):
    """
    Compute per-class mean and std of image intensities within each label mask.

    Returns:
        dict: label -> (mean, std, n_voxels)
    """
    stats = {}
    for lbl in labels:
        mask = seg == lbl
        n = np.sum(mask)
        if n < 50:  # skip tiny regions
            continue
        vals = image[mask]
        stats[lbl] = (float(np.mean(vals)), float(np.std(vals)), int(n))
    return stats


def create_bias_field_qc_plot(original, corrected, seg, white, pial,
                               output_path, figsize=(22, 14), original_axcodes=None):
    """
    Create QC plot showing bias field correction results.

    Parameters:
        original: 3D array - original image before BFC (RAS, optional)
        corrected: 3D array - corrected image after BFC (RAS, required)
        seg: 3D array - tissue segmentation labels (RAS, optional)
        white: 3D array - white matter surface mask (RAS, optional)
        pial: 3D array - pial surface mask (RAS, optional)
        output_path: path to save figure
        figsize: figure size
        original_axcodes: original axis codes for title annotation
    """
    if corrected is None:
        print("Error: Corrected image is required")
        return

    orig_norm = normalize_image(original) if original is not None else None
    corr_norm = normalize_image(corrected)
    has_original = orig_norm is not None
    has_seg = seg is not None
    has_white = white is not None
    has_pial = pial is not None

    # Center of mass from corrected image
    total = np.sum(corrected)
    if total > 0:
        indices = np.indices(corrected.shape)
        cx = int(np.sum(indices[0] * corrected) / total)
        cy = int(np.sum(indices[1] * corrected) / total)
        cz = int(np.sum(indices[2] * corrected) / total)
    else:
        cx, cy, cz = corrected.shape[0]//2, corrected.shape[1]//2, corrected.shape[2]//2

    # Resample seg to match corrected shape if needed
    seg_data = seg
    if has_seg and seg.shape != corrected.shape:
        from scipy.ndimage import zoom
        zoom_factors = [c/s for c, s in zip(corrected.shape, seg.shape)]
        seg_data = zoom(seg, zoom_factors, order=0)

    # Determine which tissue labels exist
    existing_labels = []
    if has_seg:
        existing_labels = sorted([int(l) for l in np.unique(seg_data) if l != 0])

    # Build tissue color map for existing labels
    tissue_colors = {}
    tissue_names = {}
    for lbl in existing_labels:
        if lbl in TISSUE_INFO:
            tissue_names[lbl] = TISSUE_INFO[lbl][0]
            tissue_colors[lbl] = TISSUE_INFO[lbl][1]
        else:
            tissue_names[lbl] = f'Label {lbl}'
            tissue_colors[lbl] = plt.cm.tab20(lbl % 20)

    # Column layout: Original | Corrected | Diff | Surface | Histogram
    n_cols = 0
    width_ratios = []
    col_map = {}  # logical name -> column index

    if has_original:
        col_map['original'] = n_cols
        n_cols += 1
        width_ratios.append(1.0)
    col_map['corrected'] = n_cols
    n_cols += 1
    width_ratios.append(1.0)
    if has_original:
        col_map['diff'] = n_cols
        n_cols += 1
        width_ratios.append(1.0)
    if has_white or has_pial:
        col_map['surface'] = n_cols
        n_cols += 1
        width_ratios.append(1.0)
    col_map['hist'] = n_cols
    n_cols += 1
    width_ratios.append(1.6)  # wider for histogram + stats, rightmost

    fig, axes = plt.subplots(3, n_cols, figsize=figsize,
                              gridspec_kw={'width_ratios': width_ratios})
    if axes.ndim == 1:
        axes = axes.reshape(3, -1)

    # ========================================================================
    # Column: Original image (before BFC)
    # ========================================================================
    if has_original:
        col = col_map['original']
        axes[0, col].imshow(orig_norm[:, :, cz].T, cmap='gray', origin='lower')
        axes[0, col].set_title('Original (Axial)', fontsize=9)
        axes[0, col].set_xlabel('R → L', fontsize=7)
        axes[0, col].set_ylabel('A → P', fontsize=7)
        axes[1, col].imshow(orig_norm[:, cy, :].T, cmap='gray', origin='lower')
        axes[1, col].set_title('Original (Coronal)', fontsize=9)
        axes[1, col].set_xlabel('R → L', fontsize=7)
        axes[1, col].set_ylabel('S → I', fontsize=7)
        axes[2, col].imshow(orig_norm[cx, :, :].T, cmap='gray', origin='lower')
        axes[2, col].set_title('Original (Sagittal)', fontsize=9)
        axes[2, col].set_xlabel('A → P', fontsize=7)
        axes[2, col].set_ylabel('S → I', fontsize=7)

    # ========================================================================
    # Column: Corrected image (after BFC)
    # ========================================================================
    col = col_map['corrected']
    axes[0, col].imshow(corr_norm[:, :, cz].T, cmap='gray', origin='lower')
    axes[0, col].set_title(f'Corrected (Axial Z={cz})', fontsize=9)
    axes[0, col].set_xlabel('R → L', fontsize=7)
    axes[0, col].set_ylabel('A → P', fontsize=7)
    axes[1, col].imshow(corr_norm[:, cy, :].T, cmap='gray', origin='lower')
    axes[1, col].set_title(f'Corrected (Coronal Y={cy})', fontsize=9)
    axes[1, col].set_xlabel('R → L', fontsize=7)
    axes[1, col].set_ylabel('S → I', fontsize=7)
    axes[2, col].imshow(corr_norm[cx, :, :].T, cmap='gray', origin='lower')
    axes[2, col].set_title(f'Corrected (Sagittal X={cx})', fontsize=9)
    axes[2, col].set_xlabel('A → P', fontsize=7)
    axes[2, col].set_ylabel('S → I', fontsize=7)

    # ========================================================================
    # Column: Difference (bias field estimate)
    # ========================================================================
    if has_original:
        col = col_map['diff']
        diff_axial = orig_norm[:, :, cz] - corr_norm[:, :, cz]
        diff_coronal = orig_norm[:, cy, :] - corr_norm[:, cy, :]
        diff_sagittal = orig_norm[cx, :, :] - corr_norm[cx, :, :]
        vmax = max(np.abs(diff_axial).max(), np.abs(diff_coronal).max(),
                   np.abs(diff_sagittal).max())
        if vmax > 0:
            diff_axial = diff_axial / vmax
            diff_coronal = diff_coronal / vmax
            diff_sagittal = diff_sagittal / vmax

        axes[0, col].imshow(diff_axial.T, cmap='RdBu_r', origin='lower', vmin=-1, vmax=1)
        axes[0, col].set_title('Bias Field Estimate (Axial)', fontsize=9)
        axes[1, col].imshow(diff_coronal.T, cmap='RdBu_r', origin='lower', vmin=-1, vmax=1)
        axes[1, col].set_title('Bias Field (Coronal)', fontsize=9)
        axes[2, col].imshow(diff_sagittal.T, cmap='RdBu_r', origin='lower', vmin=-1, vmax=1)
        axes[2, col].set_title('Bias Field (Sagittal)', fontsize=9)
        for r in range(3):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])

    # ========================================================================
    # Histogram / stats column (spanning all 3 rows)
    # ========================================================================
    hist_col = col_map['hist']

    # Extract brain-masked intensity values if seg available
    if has_seg:
        brain_mask = seg_data > 0
        orig_vals = original[brain_mask] if has_original else None
        corr_vals = corrected[brain_mask]
    else:
        orig_vals = original.flatten() if has_original else None
        corr_vals = corrected.flatten()

    # Compute per-class stats
    orig_class_stats = {}
    corr_class_stats = {}
    if has_seg:
        if has_original:
            orig_class_stats = compute_class_stats(original, seg_data, existing_labels)
        corr_class_stats = compute_class_stats(corrected, seg_data, existing_labels)

    # Unified x-range for consistent Gaussian overlay across histograms
    all_vals = corr_vals
    if has_original and orig_vals is not None:
        all_vals = np.concatenate([orig_vals, corr_vals])
    x_min = np.percentile(all_vals, 0.5)
    x_max = np.percentile(all_vals, 99.5)
    x_range = np.linspace(x_min, x_max, 400)

    bins = 120

    # Total brain voxel counts for scaling Gaussian by class prior probability
    n_total_orig = len(orig_vals) if has_original and orig_vals is not None else 0
    n_total_corr = len(corr_vals)

    if has_original and orig_vals is not None:
        # --- Row 0: Original intensity histogram + tissue Gaussians ---
        ax_hist_orig = axes[0, hist_col]
        ax_hist_orig.hist(orig_vals, bins=bins, alpha=0.5, color='blue',
                          density=True, edgecolor='none')
        if has_seg:
            for lbl in existing_labels:
                if lbl not in orig_class_stats:
                    continue
                color = tissue_colors.get(lbl, 'gray')
                mu, std, n = orig_class_stats[lbl]
                y = (n / n_total_orig) * gaussian_pdf(x_range, mu, std)
                ax_hist_orig.plot(x_range, y, '-', color=color, linewidth=1.5,
                                  label=tissue_names.get(lbl, f'L{lbl}'))
        ax_hist_orig.set_xlabel('Intensity (brain-masked)' if has_seg else 'Intensity',
                                fontsize=8)
        ax_hist_orig.set_ylabel('Density', fontsize=8)
        ax_hist_orig.set_title('Original Intensity Histogram', fontsize=9)
        if has_seg:
            ax_hist_orig.legend(fontsize=5, loc='upper right', ncol=2, framealpha=0.6)
        ax_hist_orig.grid(True, alpha=0.2)

        # --- Row 1: Corrected intensity histogram + tissue Gaussians ---
        ax_hist_corr = axes[1, hist_col]
        ax_hist_corr.hist(corr_vals, bins=bins, alpha=0.5, color='green',
                          density=True, edgecolor='none')
        if has_seg:
            for lbl in existing_labels:
                if lbl not in corr_class_stats:
                    continue
                color = tissue_colors.get(lbl, 'gray')
                mu, std, n = corr_class_stats[lbl]
                y = (n / n_total_corr) * gaussian_pdf(x_range, mu, std)
                ax_hist_corr.plot(x_range, y, '-', color=color, linewidth=1.5,
                                  label=tissue_names.get(lbl, f'L{lbl}'))
        ax_hist_corr.set_xlabel('Intensity (brain-masked)' if has_seg else 'Intensity',
                                fontsize=8)
        ax_hist_corr.set_ylabel('Density', fontsize=8)
        ax_hist_corr.set_title('Corrected Intensity Histogram', fontsize=9)
        if has_seg:
            ax_hist_corr.legend(fontsize=5, loc='upper right', ncol=2, framealpha=0.6)
        ax_hist_corr.grid(True, alpha=0.2)

        # --- Row 2: Per-class statistics table ---
        ax_stats = axes[2, hist_col]
    else:
        # No original: show corrected histogram in row 0, stats in rows 1-2
        ax_hist_corr = axes[0, hist_col]
        ax_hist_corr.hist(corr_vals, bins=bins, alpha=0.5, color='green',
                          density=True, edgecolor='none')
        if has_seg:
            for lbl in existing_labels:
                if lbl not in corr_class_stats:
                    continue
                color = tissue_colors.get(lbl, 'gray')
                mu, std, n = corr_class_stats[lbl]
                y = (n / n_total_corr) * gaussian_pdf(x_range, mu, std)
                ax_hist_corr.plot(x_range, y, '-', color=color, linewidth=1.5,
                                  label=tissue_names.get(lbl, f'L{lbl}'))
        ax_hist_corr.set_xlabel('Intensity (brain-masked)' if has_seg else 'Intensity',
                                fontsize=8)
        ax_hist_corr.set_ylabel('Density', fontsize=8)
        ax_hist_corr.set_title('Corrected Intensity Histogram', fontsize=9)
        if has_seg:
            ax_hist_corr.legend(fontsize=5, loc='upper right', ncol=2, framealpha=0.6)
        ax_hist_corr.grid(True, alpha=0.2)

        # Row 1: global stats
        ax_temp = axes[1, hist_col]
        ax_temp.axis('off')
        stats_text = (f"Corrected:\n  Mean: {np.mean(corrected):.2f}\n"
                      f"  Std:  {np.std(corrected):.2f}\n"
                      f"  Min:  {np.min(corrected):.2f}\n"
                      f"  Max:  {np.max(corrected):.2f}")
        ax_temp.text(0.1, 0.5, stats_text, fontsize=8, family='monospace',
                     verticalalignment='center', transform=ax_temp.transAxes)

        ax_stats = axes[2, hist_col]

    ax_stats.axis('off')

    if has_seg and existing_labels:
        lines = ["Class              Orig μ ± σ        Corr μ ± σ"]
        lines.append("-" * 55)
        for lbl in existing_labels:
            name = tissue_names.get(lbl, f'L{lbl}')
            orig_str = "      —       "
            corr_str = "      —       "
            if lbl in orig_class_stats:
                mu, std, _ = orig_class_stats[lbl]
                orig_str = f"{mu:7.1f} ± {std:5.1f}"
            if lbl in corr_class_stats:
                mu, std, _ = corr_class_stats[lbl]
                corr_str = f"{mu:7.1f} ± {std:5.1f}"
            lines.append(f" {name:<16s}  {orig_str}     {corr_str}")
        stats_text = "\n".join(lines)
        ax_stats.text(0.02, 0.95, stats_text, fontsize=7, family='monospace',
                      verticalalignment='top', transform=ax_stats.transAxes)
        ax_stats.set_title('Per-Class Intensity Statistics', fontsize=9)
    elif not has_original:
        # When no original and no seg, row 2 is empty
        pass
    else:
        stats_text = "Statistics:\n\n"
        stats_text += (f"Original:\n  Mean: {np.mean(original):.2f}\n"
                       f"  Std:  {np.std(original):.2f}\n"
                       f"  Min:  {np.min(original):.2f}\n"
                       f"  Max:  {np.max(original):.2f}\n\n")
        stats_text += (f"Corrected:\n  Mean: {np.mean(corrected):.2f}\n"
                       f"  Std:  {np.std(corrected):.2f}\n"
                       f"  Min:  {np.min(corrected):.2f}\n"
                       f"  Max:  {np.max(corrected):.2f}")
        ax_stats.text(0.1, 0.5, stats_text, fontsize=8, family='monospace',
                      verticalalignment='center', transform=ax_stats.transAxes)
    # ========================================================================
    # Surface overlay column
    # ========================================================================
    if has_white or has_pial:
        col = col_map['surface']
        axes[0, col].imshow(corr_norm[:, :, cz].T, cmap='gray', origin='lower')
        if has_white:
            axes[0, col].contour((white[:, :, cz] > 0).T, colors='green',
                                 linewidths=1, alpha=0.8)
        if has_pial:
            axes[0, col].contour((pial[:, :, cz] > 0).T, colors='red',
                                 linewidths=1, alpha=0.8)
        axes[0, col].set_title('Corrected + Surfaces (Axial)', fontsize=9)

        axes[1, col].imshow(corr_norm[:, cy, :].T, cmap='gray', origin='lower')
        if has_white:
            axes[1, col].contour((white[:, cy, :] > 0).T, colors='green',
                                 linewidths=1, alpha=0.8)
        if has_pial:
            axes[1, col].contour((pial[:, cy, :] > 0).T, colors='red',
                                 linewidths=1, alpha=0.8)
        axes[1, col].set_title('Green=WM, Red=Pial', fontsize=9)

        axes[2, col].imshow(corr_norm[cx, :, :].T, cmap='gray', origin='lower')
        if has_white:
            axes[2, col].contour((white[cx, :, :] > 0).T, colors='green',
                                 linewidths=1, alpha=0.8)
        if has_pial:
            axes[2, col].contour((pial[cx, :, :] > 0).T, colors='red',
                                 linewidths=1, alpha=0.8)
        axes[2, col].set_title('Surface Contours (Sagittal)', fontsize=9)
        for r in range(3):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])

    # Title
    title = 'Bias Field Correction QC'
    if original_axcodes:
        title += f" | Original orientation: {'-'.join(original_axcodes)} | Display: RAS"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] QC bias field correction figure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate QC visualization for bias field correction"
    )
    parser.add_argument("--original", type=str, default=None,
                        help="Path to original image (before BFC)")
    parser.add_argument("--corrected", type=str, required=True,
                        help="Path to corrected image (after BFC)")
    parser.add_argument("--seg", type=str, default=None,
                        help="Path to tissue segmentation label image")
    parser.add_argument("--white", type=str, default=None,
                        help="Path to white matter surface mask")
    parser.add_argument("--pial", type=str, default=None,
                        help="Path to pial surface mask")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for QC figure")

    args = parser.parse_args()

    if not os.path.exists(args.corrected):
        print(f"Error: Corrected image not found: {args.corrected}")
        sys.exit(1)

    # Load and reorient: corrected
    print(f"Loading corrected: {args.corrected}")
    corr_data, corr_affine = load_nifti(args.corrected)
    corr_ras, corrected_axcodes, _ = reorient_to_ras(corr_data, corr_affine)
    print(f"  Orientation: {'-'.join(corrected_axcodes) if corrected_axcodes else 'Unknown'}, "
          f"shape: {corr_ras.shape}")

    # Load and reorient: original
    original_ras = None
    original_axcodes = None
    if args.original and os.path.exists(args.original):
        print(f"Loading original: {args.original}")
        orig_data, orig_affine = load_nifti(args.original)
        original_ras, original_axcodes, _ = reorient_to_ras(orig_data, orig_affine)
        print(f"  Orientation: {'-'.join(original_axcodes) if original_axcodes else 'Unknown'}, "
              f"shape: {original_ras.shape}")

    # Load and reorient: segmentation
    seg_ras = None
    if args.seg and os.path.exists(args.seg):
        print(f"Loading segmentation: {args.seg}")
        seg_data, seg_affine = load_nifti(args.seg)
        seg_ras, seg_axcodes, _ = reorient_to_ras(seg_data, seg_affine)
        print(f"  Orientation: {'-'.join(seg_axcodes) if seg_axcodes else 'Unknown'}, "
              f"shape: {seg_ras.shape}")
        print(f"  Unique labels: {sorted(np.unique(seg_ras).astype(int))}")

    # Load and reorient: white surface
    white_ras = None
    if args.white and os.path.exists(args.white):
        print(f"Loading white mask: {args.white}")
        w_data, w_affine = load_nifti(args.white)
        white_ras, _, _ = reorient_to_ras(w_data, w_affine)
        if white_ras.shape != corr_ras.shape:
            from scipy.ndimage import zoom
            white_ras = zoom(white_ras,
                             [c/w for c, w in zip(corr_ras.shape, white_ras.shape)],
                             order=0)

    # Load and reorient: pial surface
    pial_ras = None
    if args.pial and os.path.exists(args.pial):
        print(f"Loading pial mask: {args.pial}")
        p_data, p_affine = load_nifti(args.pial)
        pial_ras, _, _ = reorient_to_ras(p_data, p_affine)
        if pial_ras.shape != corr_ras.shape:
            from scipy.ndimage import zoom
            pial_ras = zoom(pial_ras,
                            [c/p for c, p in zip(corr_ras.shape, pial_ras.shape)],
                            order=0)

    # Output
    output_path = args.output
    if not output_path.endswith('.png'):
        output_path += '.png'

    create_bias_field_qc_plot(original_ras, corr_ras, seg_ras, white_ras,
                               pial_ras, output_path,
                               original_axcodes=original_axcodes or corrected_axcodes)

    print("[DONE] QC visualization completed.")


if __name__ == "__main__":
    main()
