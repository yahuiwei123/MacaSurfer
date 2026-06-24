#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC visualization for tissue segmentation results.
Draw tissue contours on T1w image at evenly spaced slices (grid layout),
similar to qc_surface.py.
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Patch


# Tissue label definitions: label -> (name, color)
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


def normalize_image(img):
    """Normalize image to [0, 1] range."""
    if img is None:
        return None
    img_norm = img - np.min(img)
    if np.max(img_norm) > 0:
        img_norm = img_norm / np.max(img_norm)
    return img_norm


def get_contours(mask):
    """Extract contours from 2D binary mask using OpenCV."""
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contours_on_ax(ax, contours, color, linewidth):
    """Draw a list of OpenCV contours onto a matplotlib axes.

    OpenCV returns points as (x, y) = (col, row) in array-index convention.
    After imshow(img_slice.T, ...):
      display-x = original dim 0 = OpenCV y (row)
      display-y = original dim 1 = OpenCV x (col)
    No origin-based flip is needed — imshow's origin param already controls
    the display orientation via the .T transpose.
    """
    for cnt in contours:
        cnt = cnt.squeeze()
        if cnt.ndim != 2 or cnt.shape[0] < 2:
            continue
        x = cnt[:, 1]  # OpenCV y (row) = dim 0 → display x
        y = cnt[:, 0]  # OpenCV x (col) = dim 1 → display y
        ax.plot(x, y, color=color, linewidth=linewidth)


def create_tissue_qc_plot(t1w, nbest, cerebellum_brainstem, cerebrum,
                           output_path, axis='axial', n_rows=5, n_cols=5,
                           linewidth=1.0, figsize=None, origin='lower',
                           original_axcodes=None):
    """
    Create QC plot showing tissue segmentation contours on T1w image.

    Parameters:
        t1w: 3D array - T1w image (RAS oriented)
        nbest: 3D array - NBest tissue segmentation (RAS oriented)
        cerebellum_brainstem: 3D array - cerebellum+brainstem mask (optional)
        cerebrum: 3D array - cerebrum mask (optional)
        output_path: path to save the figure
        axis: viewing axis - 'axial', 'coronal', 'sagittal'
        n_rows: number of rows in the grid
        n_cols: number of columns in the grid
        linewidth: line width for contours
        figsize: figure size (auto-calculated if None)
        origin: 'lower' (radiological) or 'upper'
        original_axcodes: original axis codes for title annotation
    """
    if t1w is None:
        print("Error: T1w image is required")
        return
    if nbest is None:
        print("Error: NBest segmentation is required")
        return

    # Normalize T1w
    img_norm = normalize_image(t1w)

    # Determine which tissue labels exist in this segmentation
    existing_labels = sorted([int(lbl) for lbl in np.unique(nbest) if lbl != 0])
    if not existing_labels:
        print("Error: NBest contains no non-zero labels")
        return

    # Build tissue color/name map for existing labels
    tissue_colors = {}
    tissue_names = {}
    for lbl in existing_labels:
        if lbl in TISSUE_INFO:
            tissue_names[lbl] = TISSUE_INFO[lbl][0]
            tissue_colors[lbl] = TISSUE_INFO[lbl][1]
        else:
            tissue_names[lbl] = f'Label {lbl}'
            tissue_colors[lbl] = plt.cm.tab20(lbl % 20)

    # Build extra contour layers (binary masks)
    extra_layers = []
    if cerebellum_brainstem is not None:
        extra_layers.append(('Cerebellum+BS', (0.0, 1.0, 0.0), cerebellum_brainstem))
    if cerebrum is not None:
        extra_layers.append(('Cerebrum', (1.0, 0.0, 1.0), cerebrum))

    # Determine number of slices and spacing
    axis_map = {'axial': 2, 'coronal': 1, 'sagittal': 0}
    if axis not in axis_map:
        print(f"Error: Unknown axis '{axis}', use 'axial', 'coronal', or 'sagittal'")
        return
    axis_dim = axis_map[axis]
    n_slices = t1w.shape[axis_dim]

    total_plots = n_rows * n_cols
    slice_step = max(1, n_slices // total_plots)
    slices = [i * slice_step for i in range(total_plots)]

    # Auto figure size
    if figsize is None:
        figsize = (n_cols * 3, n_rows * 3)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for plot_idx, slice_idx in enumerate(slices):
        ax = axes[plot_idx]

        # Extract slice along the selected axis
        if axis == 'axial':
            img_slice = img_norm[:, :, slice_idx]
            seg_slice = nbest[:, :, slice_idx]
            title = f'Z={slice_idx}'
        elif axis == 'coronal':
            img_slice = img_norm[:, slice_idx, :]
            seg_slice = nbest[:, slice_idx, :]
            title = f'Y={slice_idx}'
        else:  # sagittal
            img_slice = img_norm[slice_idx, :, :]
            seg_slice = nbest[slice_idx, :, :]
            title = f'X={slice_idx}'

        # Draw T1w background
        ax.imshow(img_slice.T, cmap='gray', origin=origin)

        # Draw tissue contours for each label
        for lbl in existing_labels:
            mask = (seg_slice == lbl)
            if not np.any(mask):
                continue
            contours = get_contours(mask)
            draw_contours_on_ax(ax, contours,
                                tissue_colors[lbl], linewidth)

        # Draw extra mask contours
        for layer_name, layer_color, layer_data in extra_layers:
            if axis == 'axial':
                mask_slice = layer_data[:, :, slice_idx]
            elif axis == 'coronal':
                mask_slice = layer_data[:, slice_idx, :]
            else:
                mask_slice = layer_data[slice_idx, :, :]
            if np.any(mask_slice):
                contours = get_contours(mask_slice > 0)
                draw_contours_on_ax(ax, contours,
                                    layer_color, linewidth + 0.5)

        ax.set_title(title, fontsize=8)
        ax.axis('off')

    # Hide unused axes
    for ax in axes[len(slices):]:
        ax.axis('off')

    # Build legend
    legend_elements = []
    for lbl in existing_labels:
        legend_elements.append(
            Patch(facecolor='none', edgecolor=tissue_colors[lbl],
                  label=tissue_names[lbl], linewidth=2))
    for layer_name, layer_color, _ in extra_layers:
        legend_elements.append(
            Patch(facecolor='none', edgecolor=layer_color,
                  label=layer_name, linewidth=2))

    ncol_legend = min(4, len(legend_elements))
    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(0.99, 0.99), fontsize=7,
               ncol=ncol_legend, framealpha=0.8)

    # Title
    title = f'Tissue Segmentation QC - {axis.capitalize()} View'
    if original_axcodes:
        title += f"  |  orig: {'-'.join(original_axcodes)}  |  display: RAS"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] QC tissue segmentation figure saved to: {output_path}")
    print(f"[INFO] Displayed {len(slices)} slices from {n_slices} total "
          f"(axis={axis}, grid={n_rows}x{n_cols}, step={slice_step})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate QC visualization for tissue segmentation"
    )
    parser.add_argument("--t1w", type=str, required=True,
                        help="Path to T1w image")
    parser.add_argument("--nbest", type=str, required=True,
                        help="Path to NBest tissue segmentation")
    parser.add_argument("--cerebellum-brainstem", type=str, default=None,
                        help="Path to cerebellum+brainstem mask (optional)")
    parser.add_argument("--cerebrum", type=str, default=None,
                        help="Path to cerebrum mask (optional)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for QC figure (.png)")
    parser.add_argument("--axis", type=str, default='axial',
                        choices=['axial', 'coronal', 'sagittal'],
                        help="Viewing axis (default: axial)")
    parser.add_argument("--rows", type=int, default=5,
                        help="Number of rows in the grid (default: 5)")
    parser.add_argument("--cols", type=int, default=5,
                        help="Number of columns in the grid (default: 5)")
    parser.add_argument("--linewidth", type=float, default=1.0,
                        help="Line width for contours (default: 1.0)")
    parser.add_argument("--origin", type=str, default='lower',
                        choices=['lower', 'upper'],
                        help="Image origin: 'lower' (radiological, default) or 'upper'")

    args = parser.parse_args()

    # Validate required inputs
    for name, path in [('T1w', args.t1w), ('NBest', args.nbest)]:
        if not os.path.exists(path):
            print(f"Error: {name} image not found: {path}")
            sys.exit(1)

    # Load and reorient: T1w
    print(f"Loading T1w: {args.t1w}")
    t1w_data, t1w_affine = load_nifti(args.t1w)
    t1w_ras, original_axcodes, _ = reorient_to_ras(t1w_data, t1w_affine)
    print(f"  Orientation: {'-'.join(original_axcodes) if original_axcodes else 'Unknown'}, "
          f"shape: {t1w_ras.shape}")

    # Load and reorient: NBest
    print(f"Loading NBest: {args.nbest}")
    nbest_data, nbest_affine = load_nifti(args.nbest)
    nbest_ras, nbest_axcodes, _ = reorient_to_ras(nbest_data, nbest_affine)
    print(f"  Orientation: {'-'.join(nbest_axcodes) if nbest_axcodes else 'Unknown'}, "
          f"shape: {nbest_ras.shape}")
    print(f"  Unique labels: {sorted(np.unique(nbest_ras).astype(int))}")

    # Resample NBest to match T1w if needed
    if nbest_ras.shape != t1w_ras.shape:
        from scipy.ndimage import zoom
        zoom_factors = [t/n for t, n in zip(t1w_ras.shape, nbest_ras.shape)]
        nbest_ras = zoom(nbest_ras, zoom_factors, order=0)

    # Load and reorient: cerebellum+brainstem
    cerebellum_ras = None
    if args.cerebellum_brainstem and os.path.exists(args.cerebellum_brainstem):
        print(f"Loading cerebellum+brainstem: {args.cerebellum_brainstem}")
        cereb_data, cereb_affine = load_nifti(args.cerebellum_brainstem)
        cerebellum_ras, _, _ = reorient_to_ras(cereb_data, cereb_affine)
        if cerebellum_ras.shape != t1w_ras.shape:
            from scipy.ndimage import zoom
            zoom_factors = [t/c for t, c in zip(t1w_ras.shape, cerebellum_ras.shape)]
            cerebellum_ras = zoom(cerebellum_ras, zoom_factors, order=0)

    # Load and reorient: cerebrum
    cerebrum_ras = None
    if args.cerebrum and os.path.exists(args.cerebrum):
        print(f"Loading cerebrum: {args.cerebrum}")
        cerebrum_data, cerebrum_affine = load_nifti(args.cerebrum)
        cerebrum_ras, _, _ = reorient_to_ras(cerebrum_data, cerebrum_affine)
        if cerebrum_ras.shape != t1w_ras.shape:
            from scipy.ndimage import zoom
            zoom_factors = [t/c for t, c in zip(t1w_ras.shape, cerebrum_ras.shape)]
            cerebrum_ras = zoom(cerebrum_ras, zoom_factors, order=0)

    # Generate output
    output_path = args.output
    if not output_path.endswith('.png'):
        output_path += '.png'

    create_tissue_qc_plot(
        t1w=t1w_ras,
        nbest=nbest_ras,
        cerebellum_brainstem=cerebellum_ras,
        cerebrum=cerebrum_ras,
        output_path=output_path,
        axis=args.axis,
        n_rows=args.rows,
        n_cols=args.cols,
        linewidth=args.linewidth,
        origin=args.origin,
        original_axcodes=original_axcodes,
    )

    print("[DONE] QC visualization completed.")


if __name__ == "__main__":
    main()
