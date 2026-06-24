#!/usr/bin/env python3
"""
FreeSurfer Surface Scalar Modifier

This script reads FreeSurfer surface scalar files and a medial wall mask file,
then sets the values at medial wall positions to NaN using nibabel.
"""

import os
import sys
import numpy as np
import nibabel as nib
import argparse


def apply_medial_mask_nan(scalar_file, medial_mask_file, output_file, threshold=0.5, value=np.nan):
    """
    Apply medial wall mask to FreeSurfer scalar file, setting medial wall positions to NaN
    """
    print("FreeSurfer Surface Scalar Modifier")
    print("=" * 50)
    print(f"Input scalar file: {scalar_file}")
    print(f"Medial mask file:  {medial_mask_file}")
    print(f"Output file:       {output_file}")
    print(f"Mask threshold:    {threshold}")
    print()
    
    try:
        # Load scalar data using nibabel
        print("Loading scalar data...")
        scalar_data = nib.freesurfer.io.read_morph_data(scalar_file)
        original_shape = scalar_data.shape
        
        print(f"  Loaded {len(scalar_data)} vertices")
        print(f"  Original data shape: {original_shape}")
        
        # Load medial mask using nibabel
        print("Loading medial mask...")
        medial_data = nib.load(medial_mask_file).darrays[0].data
        
        print(f"  Loaded {len(medial_data)} vertices")
        print(f"  Medial mask shape: {medial_data.shape}")
        
        # Check if dimensions match
        if len(scalar_data) != len(medial_data):
            print(f"Warning: Dimension mismatch! Scalar: {len(scalar_data)}, Mask: {len(medial_data)}")
            # Use the smaller dimension
            min_length = min(len(scalar_data), len(medial_data))
            scalar_data = scalar_data[:min_length]
            medial_data = medial_data[:min_length]
            print(f"  Using first {min_length} vertices")
        
        # Apply medial mask - set medial wall positions to NaN
        print("Applying medial mask...")
        modified_data = scalar_data.copy().astype(np.float32)
        
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
        nib.freesurfer.io.write_morph_data(output_file, modified_data)
        
        print(f"Successfully created: {output_file}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"  Original data range: [{np.nanmin(scalar_data):.4f}, {np.nanmax(scalar_data):.4f}]")
        print(f"  Modified data range: [{np.nanmin(modified_data):.4f}, {np.nanmax(modified_data):.4f}]")
        print(f"  NaN values: {np.sum(np.isnan(modified_data))} vertices")
        print(f"  Valid values: {np.sum(~np.isnan(modified_data))} vertices")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Apply medial wall mask to FreeSurfer surface scalar files using nibabel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python apply_medial_mask.py \\
        lh.sulc \\
        lh.medial_wall \\
        lh.sulc.no_medial.mgz
        """
    )
    
    parser.add_argument('scalar_file', help='Input scalar file (sulc, thickness, curvature, etc.)')
    parser.add_argument('medial_mask', help='Medial wall mask file (1=medial wall, 0=other)')
    parser.add_argument('output_file', help='Output file path')
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                       help='Threshold for medial mask (default: 0.5)')
    parser.add_argument('-v', '--value', type=float, default=np.nan,
                       help='Threshold for medial mask (default: 0.5)')
    
    args = parser.parse_args()
    
    success = apply_medial_mask_nan(
        args.scalar_file, 
        args.medial_mask, 
        args.output_file, 
        args.threshold,
        args.value
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()