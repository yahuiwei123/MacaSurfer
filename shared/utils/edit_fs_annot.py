#!/usr/bin/env python3
"""
FreeSurfer Surface Scalar Modifier

This script reads FreeSurfer surface annot files and a medial wall mask file,
then sets the values at medial wall positions to NaN using nibabel.
"""

import os
import sys
import numpy as np
import nibabel as nib
import argparse

def find_label_indices(annot_data, ctab):
    ctab_labels = ctab[:, -1]
    label_to_index = {label: idx for idx, label in enumerate(ctab_labels)}
    label_indices = np.zeros_like(annot_data, dtype=int)
    
    for i, label_val in enumerate(annot_data):
        if label_val in label_to_index:
            label_indices[i] = label_to_index[label_val]
        else:
            label_indices[i] = -1

    unique_labels = np.unique(annot_data)    
    return label_indices, unique_labels, label_to_index


def apply_medial_mask_nan(annot_file, medial_mask_file, output_file, threshold=0.5, value=np.nan):
    """
    Apply medial wall mask to FreeSurfer annot file, setting medial wall positions to NaN
    """
    print("FreeSurfer Surface Scalar Modifier")
    print("=" * 50)
    print(f"Input annot file: {annot_file}")
    print(f"Medial mask file:  {medial_mask_file}")
    print(f"Output file:       {output_file}")
    print(f"Mask threshold:    {threshold}")
    print()
    

    # Load annot data using nibabel
    print("Loading annot data...")
    annot_data, ctab, names = nib.freesurfer.io.read_annot(annot_file, orig_ids=True)
    annot_data, _, _ = find_label_indices(annot_data, ctab)
    original_shape = annot_data.shape
    
    print(f"  Loaded {len(annot_data)} vertices")
    print(f"  Original data shape: {original_shape}")
    
    # Load medial mask using nibabel
    print("Loading medial mask...")
    medial_data = nib.load(medial_mask_file).darrays[0].data
    
    print(f"  Loaded {len(medial_data)} vertices")
    print(f"  Medial mask shape: {medial_data.shape}")
    
    # Check if dimensions match
    if len(annot_data) != len(medial_data):
        print(f"Warning: Dimension mismatch! Scalar: {len(annot_data)}, Mask: {len(medial_data)}")
        # Use the smaller dimension
        min_length = min(len(annot_data), len(medial_data))
        annot_data = annot_data[:min_length]
        medial_data = medial_data[:min_length]
        print(f"  Using first {min_length} vertices")
    
    # Apply medial mask - set medial wall positions to NaN
    print("Applying medial mask...")
    modified_data = annot_data.copy()
    
    # Identify medial wall vertices (where mask value > threshold)
    medial_indices = medial_data > threshold
    n_medial_vertices = np.sum(medial_indices)
    
    # Set medial wall positions to NaN
    modified_data[medial_indices] = value
    
    print(f"  {n_medial_vertices} vertices set to {value} "
            f"({n_medial_vertices/len(modified_data)*100:.2f}% of total)")
    
    # Save modified data using nibabel
    print("Saving modified data...")
    # Create new image with modified data, preserving affine and header
    nib.freesurfer.io.write_annot(output_file, modified_data, ctab, names, fill_ctab=True)
    
    print(f"Successfully created: {output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Original data range: [{np.nanmin(annot_data):.4f}, {np.nanmax(annot_data):.4f}]")
    print(f"  Modified data range: [{np.nanmin(modified_data):.4f}, {np.nanmax(modified_data):.4f}]")
    print(f"  NaN values: {np.sum(np.isnan(modified_data))} vertices")
    print(f"  Valid values: {np.sum(~np.isnan(modified_data))} vertices")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Apply medial wall mask to FreeSurfer surface annot files using nibabel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python apply_medial_mask.py \\
        lh.aparc.annot \\
        lh.medial_wall \\
        lh.aparc.edit.annot
        """
    )
    
    parser.add_argument('annot_file', help='Input annot file (?h.aparc.annot, etc.)')
    parser.add_argument('medial_mask', help='Medial wall mask file (1=medial wall, 0=other)')
    parser.add_argument('output_file', help='Output file path')
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                       help='Threshold for medial mask (default: 0.5)')
    parser.add_argument('-v', '--value', type=int, default=0,
                       help='Threshold for medial mask (default: 0.5)')
    
    args = parser.parse_args()
    
    success = apply_medial_mask_nan(
        args.annot_file, 
        args.medial_mask, 
        args.output_file, 
        args.threshold,
        args.value
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()