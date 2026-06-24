#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC visualization for surface ribbon files.
Draw white matter and gray matter contours from ribbon file onto original image.
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import ListedColormap


def load_nifti(path):
    """Load NIfTI file and return data array and affine."""
    if path is None or not os.path.exists(path):
        return None, None
    img = nib.load(path)
    data = img.get_fdata()
    affine = img.affine
    return data, affine


def normalize_image(img):
    """Normalize image to [0, 1] range."""
    if img is None:
        return None
    img_norm = img - np.min(img)
    if np.max(img_norm) > 0:
        img_norm = img_norm / np.max(img_norm)
    return img_norm


def get_contours(mask):
    """Extract contours from 2D mask using OpenCV."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contours_on_slice(ax, img_slice, contours, color='red', linewidth=1, origin='upper', alpha=1.0):
    """Draw contours on image slice."""
    ax.imshow(img_slice.T, cmap='gray', origin=origin)
    height, width = img_slice.shape
    for cnt in contours:
        cnt = cnt.squeeze()
        if len(cnt.shape) == 2:
            # Swap x and y to match transposed image
            x = cnt[:, 1]
            y = cnt[:, 0]
            if origin == 'lower':
                # Flip y coordinate for lower origin
                y = height - y
            ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha)



def create_ribbon_qc_plot(original_img, ribbon_data, output_path,
                         axis='axial', n_rows=5, n_cols=5,
                         wm_color='blue', gm_color='red', linewidth=1,
                         figsize=None, origin='upper',
                         draw_wm=True, draw_gm=True,
                         wm_alpha=1.0, gm_alpha=1.0,
                         single_contour=False):
    """
    Create QC plot showing ribbon contours on original image.

    Parameters:
        original_img: 3D numpy array of original image (e.g. T1w)
        ribbon_data: 3D numpy array of ribbon file (2=WM, 3=GM)
        output_path: path to save the figure
        axis: viewing axis - 'axial', 'coronal', 'sagittal'
        n_rows: number of rows in the grid
        n_cols: number of columns in the grid
        wm_color: color for white matter contours
        gm_color: color for gray matter contours
        linewidth: line width for contours
        figsize: figure size (optional, auto-calculated if not provided)
    """
    if original_img is None:
        print("Error: Original image is required")
        return

    if ribbon_data is None:
        print("Error: Ribbon file is required")
        return

    if original_img.shape != ribbon_data.shape:
        print(f"Error: Image shape {original_img.shape} does not match ribbon shape {ribbon_data.shape}")
        return

    # Normalize original image
    img_norm = normalize_image(original_img)

    # Get number of slices along the selected axis
    if axis == 'axial':
        n_slices = original_img.shape[2]
    elif axis == 'coronal':
        n_slices = original_img.shape[1]
    elif axis == 'sagittal':
        n_slices = original_img.shape[0]
    else:
        print(f"Error: Unknown axis {axis}, use 'axial', 'coronal', or 'sagittal'")
        return

    # Calculate slices to display (evenly spaced)
    total_plots = n_rows * n_cols
    slice_step = max(1, n_slices // total_plots)
    slices = [i * slice_step for i in range(total_plots)]
    # Adjust last slice if needed
    if slices[-1] >= n_slices:
        slices = [min(s, n_slices - 1) for s in slices]

    # Auto calculate figsize if not provided
    if figsize is None:
        figsize = (n_cols * 3, n_rows * 3)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for plot_idx, slice_idx in enumerate(slices):
        ax = axes[plot_idx]

        # Get the current slice
        if axis == 'axial':
            img_slice = img_norm[:, :, slice_idx]
            wm_mask = (ribbon_data[:, :, slice_idx] == 2) | (ribbon_data[:, :, slice_idx] == 41)
            gm_mask = (ribbon_data[:, :, slice_idx] == 3) | (ribbon_data[:, :, slice_idx] == 42)
            title = f'Axial Z={slice_idx}'
        elif axis == 'coronal':
            img_slice = img_norm[:, slice_idx, :]
            wm_mask = (ribbon_data[:, slice_idx, :] == 2) | (ribbon_data[:, slice_idx, :] == 41)
            gm_mask = (ribbon_data[:, slice_idx, :] == 3) | (ribbon_data[:, slice_idx, :] == 42)
            title = f'Coronal Y={slice_idx}'
        else: # sagittal
            img_slice = img_norm[slice_idx, :, :]
            wm_mask = (ribbon_data[slice_idx, :, :] == 2) | (ribbon_data[slice_idx, :, :] == 41)
            gm_mask = (ribbon_data[slice_idx, :, :] == 3) | (ribbon_data[slice_idx, :, :] == 42)
            title = f'Sagittal X={slice_idx}'

        # Extract contours
        if single_contour:
            # Single contour mode: only draw outer GM boundary and inner WM boundary
            ribbon_mask = wm_mask | gm_mask
            ribbon_contours = get_contours(ribbon_mask)  # Outer boundary of GM
            wm_contours = get_contours(wm_mask)          # Inner boundary of WM

            # Draw image
            ax.imshow(img_slice.T, cmap='gray', origin=origin)
            height, width = img_slice.shape

            # Draw outer ribbon boundary (GM outer edge)
            for cnt in ribbon_contours:
                cnt = cnt.squeeze()
                if len(cnt.shape) == 2:
                    x = cnt[:, 1]
                    y = cnt[:, 0]
                    if origin == 'lower':
                        y = height - y
                    ax.plot(x, y, color=gm_color, linewidth=linewidth, alpha=gm_alpha)

            # Draw inner WM boundary
            for cnt in wm_contours:
                cnt = cnt.squeeze()
                if len(cnt.shape) == 2:
                    x = cnt[:, 1]
                    y = cnt[:, 0]
                    if origin == 'lower':
                        y = height - y
                    ax.plot(x, y, color=wm_color, linewidth=linewidth, alpha=wm_alpha)
        else:
            # Normal mode: draw both WM and GM contours
            wm_contours = get_contours(wm_mask)
            gm_contours = get_contours(gm_mask)

            # Draw image and contours
            if draw_wm:
                draw_contours_on_slice(ax, img_slice, wm_contours, color=wm_color, linewidth=linewidth, origin=origin, alpha=wm_alpha)
            else:
                # Just draw the image
                ax.imshow(img_slice.T, cmap='gray', origin=origin)

            # Draw GM contours if enabled
            height, width = img_slice.shape
            if draw_gm:
                for cnt in gm_contours:
                    cnt = cnt.squeeze()
                    if len(cnt.shape) == 2:
                        # Swap x and y to match transposed image
                        x = cnt[:, 1]
                        y = cnt[:, 0]
                        if origin == 'lower':
                            y = height - y
                        ax.plot(x, y, color=gm_color, linewidth=linewidth, alpha=gm_alpha)

        ax.set_title(title, fontsize=8)
        ax.axis('off')

    # Hide any unused axes
    for ax in axes[len(slices):]:
        ax.axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor=wm_color, label='White Matter'),
        Patch(facecolor='none', edgecolor=gm_color, label='Gray Matter')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)

    plt.suptitle(f'Surface Ribbon QC - {axis.capitalize()} View', fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] QC surface ribbon figure saved to: {output_path}")
    print(f"[INFO] Displayed {len(slices)} slices from {n_slices} total slices")


def main():
    parser = argparse.ArgumentParser(
        description="Generate QC visualization for surface ribbon files"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to original input image (e.g. T1w NIfTI)")
    parser.add_argument("--ribbon", type=str, required=True,
                        help="Path to ribbon NIfTI file (2=WM, 3=GM)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for QC figure (png)")
    parser.add_argument("--axis", type=str, default='axial',
                        choices=['axial', 'coronal', 'sagittal'],
                        help="Viewing axis (default: axial)")
    parser.add_argument("--rows", type=int, default=5,
                        help="Number of rows in the grid (default: 5)")
    parser.add_argument("--cols", type=int, default=5,
                        help="Number of columns in the grid (default: 5)")
    parser.add_argument("--wm_color", type=str, default='blue',
                        help="Color for white matter contours (default: blue)")
    parser.add_argument("--gm_color", type=str, default='red',
                        help="Color for gray matter contours (default: red)")
    parser.add_argument("--linewidth", type=float, default=1.0,
                        help="Line width for contours (default: 1.0)")
    parser.add_argument("--origin", type=str, default='upper',
                        choices=['upper', 'lower'],
                        help="Image origin, 'upper' (top-left, default) or 'lower' (bottom-left)")
    parser.add_argument("--draw_wm", type=bool, default=True,
                        help="Whether to draw white matter contours (default: True)")
    parser.add_argument("--draw_gm", type=bool, default=True,
                        help="Whether to draw gray matter contours (default: True)")
    parser.add_argument("--wm_alpha", type=float, default=1.0,
                        help="Transparency for white matter contours (default: 1.0)")
    parser.add_argument("--gm_alpha", type=float, default=1.0,
                        help="Transparency for gray matter contours (default: 1.0)")
    parser.add_argument("--single_contour", action='store_true',
                        help="Only draw outer gray matter boundary and inner white matter boundary, no overlapping lines")

    args = parser.parse_args()

    # Check input files
    if not os.path.exists(args.input):
        print(f"Error: Input image not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.ribbon):
        print(f"Error: Ribbon file not found: {args.ribbon}")
        sys.exit(1)

    # Load images
    print(f"Loading original image: {args.input}")
    original_img, _ = load_nifti(args.input)

    print(f"Loading ribbon file: {args.ribbon}")
    ribbon_data, _ = load_nifti(args.ribbon)

    # Generate QC plot
    output_path = args.output
    if not output_path.endswith('.png'):
        output_path = output_path + '.png'

    create_ribbon_qc_plot(
        original_img=original_img,
        ribbon_data=ribbon_data,
        output_path=output_path,
        axis=args.axis,
        n_rows=args.rows,
        n_cols=args.cols,
        wm_color=args.wm_color,
        gm_color=args.gm_color,
        linewidth=args.linewidth,
        origin=args.origin,
        draw_wm=args.draw_wm,
        draw_gm=args.draw_gm,
        wm_alpha=args.wm_alpha,
        gm_alpha=args.gm_alpha,
        single_contour=args.single_contour
    )

    print("[DONE] QC surface visualization completed.")


if __name__ == "__main__":
    main()
