#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC visualization for template registration results.

Displays:
  - Original subject with atlas tissue boundary contours (aseg in native space)
  - Registered subject (aligned to template)
  - Template
  - Checkerboard comparison (registered subject vs template)
  - Contour overlay (registered subject + template edges)

Inputs:
  --subject-orig  : original subject image (before registration)
  --subject-reg   : subject image registered to template space
  --template      : template image
  --aseg          : atlas labels inverse-transformed to original subject space
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


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
    Returns:
        reoriented_data: RAS oriented data
        original_axcodes: Original axis codes
        ras_affine: Affine for the RAS oriented image
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


def create_checkerboard(img1, img2, block_size=12):
    """Create checkerboard pattern of two images."""
    if img1 is None or img2 is None:
        return None

    if img1.shape != img2.shape:
        print(f"Warning: Shape mismatch - img1: {img1.shape}, img2: {img2.shape}")
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


def compute_center_of_mass(data):
    """Compute center of mass of a 3D volume."""
    total = np.sum(data)
    if total > 0:
        indices = np.indices(data.shape)
        cx = int(np.sum(indices[0] * data) / total)
        cy = int(np.sum(indices[1] * data) / total)
        cz = int(np.sum(indices[2] * data) / total)
    else:
        cx, cy, cz = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2
    return cx, cy, cz


def draw_aseg_contours_on_axes(ax, img_slice_t, aseg_slice, label_colors):
    """
    Draw aseg tissue boundaries as colored contours on the original subject image.

    Parameters:
        ax: matplotlib axes
        img_slice_t: 2D image slice (already transposed for display)
        aseg_slice: 2D aseg slice (NOT transposed, matches img_slice_t before .T)
        label_colors: dict mapping label_id -> color

    Returns:
        set of label IDs present in this slice
    """
    ax.imshow(img_slice_t, cmap='gray', origin='lower')

    labels = np.unique(aseg_slice)
    present_labels = set()

    for label in labels:
        if label == 0:
            continue
        lbl = int(label)
        color = label_colors.get(lbl, 'white')
        mask = (aseg_slice == label).astype(np.float64)
        if np.sum(mask) > 0:
            ax.contour(mask.T, levels=[0.5], colors=[color], linewidths=1.0)
            present_labels.add(lbl)

    return present_labels


def create_template_qc_plot(subject_orig, subject_reg, template, aseg,
                             output_path, figsize=(24, 14), original_axcodes=None):
    """
    Create QC plot showing template registration quality.

    Parameters:
        subject_orig: 3D array - original subject image (before registration)
        subject_reg: 3D array - subject image registered to template space
        template: 3D array - template image
        aseg: 3D array - atlas labels inverse-transformed to original subject space
        output_path: path to save the figure
        figsize: figure size
        original_axcodes: original axis codes for display annotation
    """
    # Normalize images
    orig_norm = normalize_image(subject_orig) if subject_orig is not None else None
    reg_norm = normalize_image(subject_reg) if subject_reg is not None else None
    temp_norm = normalize_image(template) if template is not None else None

    has_orig = orig_norm is not None
    has_reg = reg_norm is not None
    has_temp = temp_norm is not None
    has_aseg = aseg is not None and has_orig

    # Compute centers of mass for slice selection
    if has_orig:
        cx_orig, cy_orig, cz_orig = compute_center_of_mass(subject_orig)
    if has_reg:
        cx_reg, cy_reg, cz_reg = compute_center_of_mass(subject_reg)
    elif has_temp:
        cx_reg, cy_reg, cz_reg = compute_center_of_mass(template)
    else:
        cx_reg = cy_reg = cz_reg = 0

    # Build column list dynamically
    columns = []
    if has_orig:
        columns.append('orig')
    if has_reg:
        columns.append('reg')
    if has_temp:
        columns.append('template')
        if has_reg:
            columns.append('checker')
            columns.append('overlay')

    n_cols = len(columns)
    if n_cols == 0:
        print("Error: No valid images to display")
        return

    fig, axes = plt.subplots(3, n_cols, figsize=figsize)
    if axes.ndim == 1:
        axes = axes.reshape(3, -1)

    # Precompute aseg label colors
    label_colors = {}
    if has_aseg:
        all_labels = np.unique(aseg)
        for lbl in all_labels:
            if lbl != 0:
                label_colors[int(lbl)] = plt.cm.tab20(int(lbl) % 20)

    # ========================================================================
    # Column: Original subject + atlas tissue contours
    # ========================================================================
    if 'orig' in columns:
        col = columns.index('orig')

        # Resample aseg to match original subject if needed
        aseg_data = aseg
        if has_aseg and aseg.shape != subject_orig.shape:
            from scipy.ndimage import zoom
            zoom_factors = [s/a for s, a in zip(subject_orig.shape, aseg.shape)]
            aseg_data = zoom(aseg, zoom_factors, order=0)

        all_present_labels = set()

        # Axial: data[:, :, z]
        if has_aseg:
            present = draw_aseg_contours_on_axes(
                axes[0, col], orig_norm[:, :, cz_orig].T,
                aseg_data[:, :, cz_orig], label_colors)
            all_present_labels |= present
        else:
            axes[0, col].imshow(orig_norm[:, :, cz_orig].T, cmap='gray', origin='lower')
        axes[0, col].set_title(f'Original Subject + Atlas Contours\nAxial (Z={cz_orig})', fontsize=9)
        axes[0, col].set_xlabel('R → L', fontsize=7)
        axes[0, col].set_ylabel('A → P', fontsize=7)

        # Coronal: data[:, y, :]
        if has_aseg:
            present = draw_aseg_contours_on_axes(
                axes[1, col], orig_norm[:, cy_orig, :].T,
                aseg_data[:, cy_orig, :], label_colors)
            all_present_labels |= present
        else:
            axes[1, col].imshow(orig_norm[:, cy_orig, :].T, cmap='gray', origin='lower')
        axes[1, col].set_title(f'Original + Atlas Contours\nCoronal (Y={cy_orig})', fontsize=9)
        axes[1, col].set_xlabel('R → L', fontsize=7)
        axes[1, col].set_ylabel('S → I', fontsize=7)

        # Sagittal: data[x, :, :]
        if has_aseg:
            present = draw_aseg_contours_on_axes(
                axes[2, col], orig_norm[cx_orig, :, :].T,
                aseg_data[cx_orig, :, :], label_colors)
            all_present_labels |= present
        else:
            axes[2, col].imshow(orig_norm[cx_orig, :, :].T, cmap='gray', origin='lower')
        axes[2, col].set_title(f'Original + Atlas Contours\nSagittal (X={cx_orig})', fontsize=9)
        axes[2, col].set_xlabel('A → P', fontsize=7)
        axes[2, col].set_ylabel('S → I', fontsize=7)

        # Add legend on the first row subplot
        if has_aseg and all_present_labels:
            from matplotlib.lines import Line2D
            handles = []
            for lbl in sorted(all_present_labels):
                handles.append(Line2D([0], [0], color=label_colors[lbl],
                                      linewidth=2, label=f'Label {lbl}'))
            axes[0, col].legend(handles=handles, fontsize=5,
                                loc='upper left', bbox_to_anchor=(1.02, 1.0),
                                title='Atlas Labels', title_fontsize=6,
                                framealpha=0.7, ncol=1)

    # ========================================================================
    # Column: Registered subject (aligned to template)
    # ========================================================================
    if 'reg' in columns:
        col = columns.index('reg')

        axes[0, col].imshow(reg_norm[:, :, cz_reg].T, cmap='gray', origin='lower')
        axes[0, col].set_title(f'Registered Subject\nAxial (Z={cz_reg})', fontsize=9)
        axes[0, col].set_xlabel('R → L', fontsize=7)
        axes[0, col].set_ylabel('A → P', fontsize=7)

        axes[1, col].imshow(reg_norm[:, cy_reg, :].T, cmap='gray', origin='lower')
        axes[1, col].set_title(f'Registered Subject\nCoronal (Y={cy_reg})', fontsize=9)
        axes[1, col].set_xlabel('R → L', fontsize=7)
        axes[1, col].set_ylabel('S → I', fontsize=7)

        axes[2, col].imshow(reg_norm[cx_reg, :, :].T, cmap='gray', origin='lower')
        axes[2, col].set_title(f'Registered Subject\nSagittal (X={cx_reg})', fontsize=9)
        axes[2, col].set_xlabel('A → P', fontsize=7)
        axes[2, col].set_ylabel('S → I', fontsize=7)

    # ========================================================================
    # Column: Template
    # ========================================================================
    if 'template' in columns:
        col = columns.index('template')

        # Resample template to match registered subject if needed
        temp_display = temp_norm
        if has_reg and temp_norm.shape != reg_norm.shape:
            from scipy.ndimage import zoom
            zoom_factors = [s/t for s, t in zip(reg_norm.shape, temp_norm.shape)]
            temp_display = zoom(temp_norm, zoom_factors, order=1)

        axes[0, col].imshow(temp_display[:, :, cz_reg].T, cmap='gray', origin='lower')
        axes[0, col].set_title('Template (Axial)', fontsize=9)
        axes[1, col].imshow(temp_display[:, cy_reg, :].T, cmap='gray', origin='lower')
        axes[1, col].set_title('Template (Coronal)', fontsize=9)
        axes[2, col].imshow(temp_display[cx_reg, :, :].T, cmap='gray', origin='lower')
        axes[2, col].set_title('Template (Sagittal)', fontsize=9)
        for r in range(3):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])

    # ========================================================================
    # Column: Checkerboard (registered subject vs template)
    # ========================================================================
    if 'checker' in columns:
        col = columns.index('checker')

        temp_display = temp_norm
        if reg_norm.shape != temp_norm.shape:
            from scipy.ndimage import zoom
            zoom_factors = [s/t for s, t in zip(reg_norm.shape, temp_norm.shape)]
            temp_display = zoom(temp_norm, zoom_factors, order=1)

        checker_axial = create_checkerboard(reg_norm[:, :, cz_reg], temp_display[:, :, cz_reg])
        checker_coronal = create_checkerboard(reg_norm[:, cy_reg, :], temp_display[:, cy_reg, :])
        checker_sagittal = create_checkerboard(reg_norm[cx_reg, :, :], temp_display[cx_reg, :, :])

        axes[0, col].imshow(checker_axial.T, cmap='gray', origin='lower')
        axes[0, col].set_title('Checkerboard\n(Reg.Subj + Template)', fontsize=9)
        axes[1, col].imshow(checker_coronal.T, cmap='gray', origin='lower')
        axes[1, col].set_title('Checkerboard (Coronal)', fontsize=9)
        axes[2, col].imshow(checker_sagittal.T, cmap='gray', origin='lower')
        axes[2, col].set_title('Checkerboard (Sagittal)', fontsize=9)
        for r in range(3):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])

    # ========================================================================
    # Column: Overlay (registered subject + template contours)
    # ========================================================================
    if 'overlay' in columns:
        col = columns.index('overlay')

        try:
            from scipy import ndimage
            has_scipy = True
        except ImportError:
            has_scipy = False

        temp_display = temp_norm
        if reg_norm.shape != temp_norm.shape:
            from scipy.ndimage import zoom
            zoom_factors = [s/t for s, t in zip(reg_norm.shape, temp_norm.shape)]
            temp_display = zoom(temp_norm, zoom_factors, order=1)

        slice_specs = [
            (0, reg_norm[:, :, cz_reg].T, temp_display[:, :, cz_reg]),
            (1, reg_norm[:, cy_reg, :].T, temp_display[:, cy_reg, :]),
            (2, reg_norm[cx_reg, :, :].T, temp_display[cx_reg, :, :]),
        ]
        for row, subj_slice_t, temp_slice in slice_specs:
            axes[row, col].imshow(subj_slice_t, cmap='gray', origin='lower')
            if has_scipy:
                axes[row, col].contour(temp_slice.T, levels=[0.2, 0.4, 0.6],
                                       colors='r', linewidths=0.5, alpha=0.7)
            axes[row, col].set_title(
                'Reg.Subj + Temp Contours' if row == 0 else '', fontsize=9)
        for r in range(3):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])

    # Title and save
    title = 'Template Registration QC'
    if original_axcodes:
        title += f" | Original orientation: {'-'.join(original_axcodes)} | Display: RAS"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] QC template registration figure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate QC visualization for template registration"
    )
    parser.add_argument("--subject-orig", type=str, default=None,
                        help="Path to original subject image (before registration)")
    parser.add_argument("--subject-reg", type=str, required=True,
                        help="Path to subject image registered to template space")
    parser.add_argument("--template", type=str, default=None,
                        help="Path to template image")
    parser.add_argument("--aseg", type=str, default=None,
                        help="Path to atlas segmentation inverse-transformed to "
                             "original subject space")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for QC figure")

    args = parser.parse_args()

    # Validate
    if not os.path.exists(args.subject_reg):
        print(f"Error: Registered subject image not found: {args.subject_reg}")
        sys.exit(1)

    # Load and reorient: registered subject
    print(f"Loading registered subject: {args.subject_reg}")
    reg_data, reg_affine = load_nifti(args.subject_reg)
    reg_ras, original_axcodes, _ = reorient_to_ras(reg_data, reg_affine)
    print(f"  Orientation: {'-'.join(original_axcodes) if original_axcodes else 'Unknown'}, "
          f"shape: {reg_ras.shape}")

    # Load and reorient: original subject
    orig_ras = None
    if args.subject_orig:
        if not os.path.exists(args.subject_orig):
            print(f"Error: Original subject image not found: {args.subject_orig}")
            sys.exit(1)
        print(f"Loading original subject: {args.subject_orig}")
        orig_data, orig_affine = load_nifti(args.subject_orig)
        orig_ras, orig_axcodes, _ = reorient_to_ras(orig_data, orig_affine)
        print(f"  Orientation: {'-'.join(orig_axcodes) if orig_axcodes else 'Unknown'}, "
              f"shape: {orig_ras.shape}")

    # Load and reorient: template
    temp_ras = None
    if args.template:
        if not os.path.exists(args.template):
            print(f"Warning: Template not found: {args.template}")
        else:
            print(f"Loading template: {args.template}")
            temp_data, temp_affine = load_nifti(args.template)
            temp_ras, temp_axcodes, _ = reorient_to_ras(temp_data, temp_affine)
            print(f"  Orientation: {'-'.join(temp_axcodes) if temp_axcodes else 'Unknown'}, "
                  f"shape: {temp_ras.shape}")

    # Load and reorient: aseg (in original subject space)
    aseg_ras = None
    if args.aseg:
        if not os.path.exists(args.aseg):
            print(f"Warning: Aseg not found: {args.aseg}")
        else:
            print(f"Loading aseg (original subject space): {args.aseg}")
            aseg_data, aseg_affine = load_nifti(args.aseg)
            aseg_ras, aseg_axcodes, _ = reorient_to_ras(aseg_data, aseg_affine)
            print(f"  Orientation: {'-'.join(aseg_axcodes) if aseg_axcodes else 'Unknown'}, "
                  f"shape: {aseg_ras.shape}")

    # Verify aseg and original subject shapes match
    if orig_ras is not None and aseg_ras is not None:
        if orig_ras.shape != aseg_ras.shape:
            print(f"Warning: Original subject shape {orig_ras.shape} != "
                  f"aseg shape {aseg_ras.shape}")
            print("  Aseg will be resampled during visualization")

    # Generate output
    output_path = args.output
    if not output_path.endswith('.png'):
        output_path += '.png'

    create_template_qc_plot(orig_ras, reg_ras, temp_ras, aseg_ras, output_path,
                             original_axcodes=original_axcodes)

    print("[DONE] QC visualization completed.")


if __name__ == "__main__":
    main()
