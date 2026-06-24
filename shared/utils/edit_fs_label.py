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

def write_label(filepath, label_array, scalar_array=None):
    """Write a Freesurfer .label file with vertex indices, coordinates, and scalar values.

    Parameters
    ----------
    filepath : str
        Path to save the label file.
    label_array : numpy array
        Array of vertex indices to be included in the label.
    scalar_array : numpy array (optional)
        Array of scalar data for each vertex. If None, only the vertex indices will be saved.
    """
    # If scalar_array is None, initialize it to zeros (placeholder)
    if scalar_array is None:
        scalar_array = np.zeros(len(label_array))
    
    # Example coordinates: here I just generate random coordinates for illustration
    # In practice, you will want to pass in actual coordinate data for each vertex
    coords = np.zeros((len(label_array), 3))  # Fake 3D coordinates

    # Open the file for writing
    with open(filepath, 'w') as f:
        # Write the header
        f.write("#!ascii label , from subject  vox2ras=TkReg\n")
        f.write(f"{len(label_array)}\n")  # Number of vertices
        f.write("0  0  0  0 0.0000000000\n")  # For the first line data, example format

        # Write each vertex index, coordinates and corresponding scalar (if available)
        for i, vertex_id in enumerate(label_array):
            x, y, z = coords[i]  # Coordinates for the vertex
            scalar_value = scalar_array[i]  # Scalar value for the vertex
            f.write(f"{vertex_id}  {x:.3f}  {y:.3f}  {z:.3f} {scalar_value:.10f}\n")


def apply_medial_mask_nan(annot_file, medial_mask_file, output_file, threshold=0.5):
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
    cortex_indices = nib.freesurfer.io.read_label(annot_file)
    original_shape = cortex_indices.shape
    
    # Load medial mask using nibabel
    print("Loading medial mask...")
    medial_data = nib.load(medial_mask_file).darrays[0].data
    
    print(f"  Loaded {len(medial_data)} vertices")
    print(f"  Medial mask shape: {medial_data.shape}")
    
    # Apply medial mask - set medial wall positions to NaN
    print("Applying medial mask...")
    medial_indices = np.where(medial_data > threshold)[0].astype(np.int64)
    
    # Set medial wall positions to NaN
    cortex_indices = cortex_indices[~np.isin(cortex_indices, medial_indices)]
    
    # Save modified data using nibabel
    print("Saving modified data...")
    # Create new image with modified data, preserving affine and header
    write_label(output_file, cortex_indices)
    
    print(f"Successfully created: {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Apply medial wall mask to FreeSurfer surface annot files using nibabel",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('label_file', help='Input annot file (?h.cortex.label, etc.)')
    parser.add_argument('medial_mask', help='Medial wall mask file (1=medial wall, 0=other)')
    parser.add_argument('output_file', help='Output file path')
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                       help='Threshold for medial mask (default: 0.5)')
    
    args = parser.parse_args()
    
    success = apply_medial_mask_nan(
        args.label_file, 
        args.medial_mask, 
        args.output_file, 
        args.threshold,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()