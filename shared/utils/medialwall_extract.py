#!/usr/bin/env python3
"""
Medial Wall Mapping - Python Implementation

This script extracts medial wall regions from filled volume and maps them to surface space.
It uses nibabel for volume processing and nibabel-gifti for surface mapping.
"""

import os
import sys
import argparse
import tempfile
import shutil
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import morphology, measure
import subprocess


class MedialWallMapper:
    def __init__(self, args):
        self.work_dir = args.work_dir
        self.mid_wall = args.mid_wall
        self.filled = args.filled
        self.lh_surface = args.lh_surface
        self.rh_surface = args.rh_surface
        self.out_dir = args.out_dir
        self.dilate_size = args.dilate_size
        self.morph_size = args.morph_size
        self.distance_threshold = args.distance_threshold
        self.clean_up = args.clean_up
        
        # Create directories
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Load data
        self.filled_data = None
        self.mid_wall_data = None
        self.filled_affine = None
        
    def load_volume_data(self):
        """Load volume data from files"""
        print("Loading volume data...")
        filled_img = nib.load(self.filled)
        self.filled_data = filled_img.get_fdata()
        self.filled_affine = filled_img.affine
        
        mid_wall_img = nib.load(self.mid_wall)
        self.mid_wall_data = mid_wall_img.get_fdata()
        
    def save_volume(self, data, filename):
        """Save volume data to file"""
        img = nib.Nifti1Image(data, self.filled_affine)
        nib.save(img, os.path.join(self.work_dir, filename))
        
    def dilate_mask(self, mask_data, size):
        """Dilate binary mask using spherical kernel"""
        print(f"Dilating mask with kernel size {size}...")
        # Create spherical kernel
        kernel = morphology.ball(size)
        dilated = ndimage.binary_dilation(mask_data, structure=kernel)
        return dilated.astype(np.float32)
    
    def erode_mask(self, mask_data, size):
        """Erode binary mask using spherical kernel"""
        print(f"Eroding mask with kernel size {size}...")
        # Create spherical kernel
        kernel = morphology.ball(size)
        eroded = ndimage.binary_erosion(mask_data, structure=kernel)
        return eroded.astype(np.float32)
    
    def skeletonize_mask(self, mask_data):
        """Extract skeleton from binary mask"""
        print("Extracting skeleton from mask...")
        # Use medial axis transform to skeletonize the mask
        skeleton = morphology.skeletonize(mask_data.astype(bool))
        return skeleton.astype(np.float32)
    
    def extract_largest_connected_component(self, mask_data, connectivity=6):
        """Extract largest connected component from binary mask"""
        print("Extracting largest connected component...")
        # Label connected components
        labeled_mask, num_features = ndimage.label(mask_data)
        
        if num_features == 0:
            print("Warning: No connected components found!")
            return mask_data
            
        # Find the largest component
        component_sizes = ndimage.sum(mask_data, labeled_mask, range(1, num_features + 1))
        if len(component_sizes) == 0:
            return mask_data
            
        largest_component = np.argmax(component_sizes) + 1
        largest_mask = (labeled_mask == largest_component).astype(np.float32)
        
        print(f"Found {num_features} components, keeping largest (size: {component_sizes[largest_component-1]:.0f} voxels)")
        return largest_mask
    
    def volume_to_surface_mapping(self, volume_file, surface_file, output_file):
        """Map volume data to surface using wb_command"""
        print(f"Mapping {volume_file} to surface {surface_file}...")
        
        cmd = [
            'wb_command', '-volume-to-surface-mapping',
            volume_file,
            surface_file,
            output_file,
            '-trilinear'
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Surface mapping completed: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error in volume-to-surface mapping: {e}")
            raise
        except FileNotFoundError:
            print("Error: wb_command not found. Please install Connectome Workbench.")
            raise
    
    def create_binary_surface_mask(self, input_file, output_file, threshold=0.5):
        """Create binary surface mask from continuous values"""
        print(f"Creating binary surface mask: {output_file}")
        
        cmd = [
            'wb_command', '-metric-math',
            f'x > {threshold}',
            output_file,
            '-var', 'x', input_file
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error creating binary surface mask: {e}")
            raise
    
    def load_surface_data(self, surface_file):
        """Load surface data from GIFTI file"""
        try:
            gifti_img = nib.load(surface_file)
            data = gifti_img.darrays[0].data
            return data, gifti_img
        except Exception as e:
            print(f"Error loading surface file {surface_file}: {e}")
            raise
    
    def save_surface_data(self, data, gifti_img, output_file):
        """Save surface data to GIFTI file"""
        try:
            # Create new GIFTI image with the same structure but new data
            new_gifti = nib.gifti.GiftiImage()
            
            # Copy header information
            new_gifti.header = gifti_img.header.copy()
            
            # Create new data array
            data_array = nib.gifti.GiftiDataArray(data=data.astype(np.float32))
            new_gifti.add_gifti_data_array(data_array)
            
            # Save the file
            nib.save(new_gifti, output_file)
            print(f"Surface data saved: {output_file}")
            
        except Exception as e:
            print(f"Error saving surface file {output_file}: {e}")
            raise
    
    def load_surface_faces(self, surface_file):
        """Load surface faces from GIFTI surface file"""
        try:
            gifti_img = nib.load(surface_file)
            for darry in gifti_img.darrays:
                # Look for face data (usually int32 data array)
                if darry.datatype == 'NIFTI_TYPE_INT32' and len(darry.data.shape) == 2:
                    if darry.data.shape[1] == 3:  # Triangle faces
                        return darry.data
            print("Warning: No face information found in surface file")
            return None
        except Exception as e:
            print(f"Error loading surface faces from {surface_file}: {e}")
            return None

    def process_hemisphere(self, hemisphere, surface_file):
        """Process one hemisphere using filled.nii.gz and initial middle wall mask"""
        print(f"Processing {hemisphere} hemisphere...")
        
        # Step 1: Dilate initial middle wall mask by 2 voxels
        print("Step 1: Dilating initial middle wall mask by 2 voxels...")
        medial_mask_dilated = self.dilate_mask(self.mid_wall_data, 2)
        
        # Step 2: Find intersection with filled volume
        print("Step 2: Finding intersection with filled volume...")
        # Create a rough mask from filled volume (non-zero regions)
        if hemisphere == 'lh':
            filled_mask = (self.filled_data == 255).astype(np.float32) # use left
        else:
            filled_mask = (self.filled_data == 127).astype(np.float32) # use right
        
        # Find intersection between dilated medial mask and filled volume
        intersection_mask = np.logical_and(medial_mask_dilated, filled_mask)
        
        # Step 3: Dilate the skeleton by 1 voxel
        print("Step 3: Filling hole by 1 voxel...")
        intersection_mask = self.dilate_mask(intersection_mask, 4)
        intersection_mask = self.erode_mask(intersection_mask, 3)
        
        # Step 4: Extract largest connected component (optional but recommended)
        print("Step 4: Extracting largest connected component...")
        medial_wall_final = self.extract_largest_connected_component(intersection_mask)
        self.save_volume(medial_wall_final, f"{hemisphere}_medial_wall_final.nii.gz")
        
        # Map to surface
        volume_file = os.path.join(self.work_dir, f"{hemisphere}_medial_wall_final.nii.gz")
        output_surface = os.path.join(self.out_dir, f"{hemisphere}_medial_wall.shape.gii")
        
        self.volume_to_surface_mapping(volume_file, surface_file, output_surface)
        
        # Create binary surface mask
        binary_output = os.path.join(self.out_dir, f"{hemisphere}_medial_wall_binary.shape.gii")
        self.create_binary_surface_mask(output_surface, binary_output, threshold=0.25)
        
        # Extract largest connected component on surface
        print("Processing surface mask to keep largest connected component...")
        
        # Load the binary surface mask
        surface_data, gifti_img = self.load_surface_data(binary_output)
        
        # Load surface faces for better connected component analysis
        surface_faces = self.load_surface_faces(surface_file)
        
        # Extract largest surface component using graph-based method
        if surface_faces is not None:
            import networkx as nx
            G = nx.Graph()
            
            # Add all vertices
            n_vertices = len(surface_data)
            G.add_nodes_from(range(n_vertices))
            
            # Add edges from faces
            for face in surface_faces:
                if len(face) == 3:  # Triangle faces
                    G.add_edge(face[0], face[1])
                    G.add_edge(face[1], face[2])
                    G.add_edge(face[2], face[0])
            
            # Find connected components in the binary mask
            binary_vertices = np.where(surface_data > 0.5)[0]
            if len(binary_vertices) > 0:
                subgraph = G.subgraph(binary_vertices)
                components = list(nx.connected_components(subgraph))
                
                if components:
                    # Find largest component
                    largest_component = max(components, key=len)
                    print(f"Found {len(components)} surface components, keeping largest (size: {len(largest_component)} vertices)")
                    
                    # Create output mask
                    surface_data_cleaned = np.zeros_like(surface_data)
                    surface_data_cleaned[list(largest_component)] = 1.0
                    
                    # Save the cleaned surface mask
                    self.save_surface_data(surface_data_cleaned, gifti_img, binary_output)
        
        # Create FreeSurfer .annot file
        print("Creating FreeSurfer .annot file...")
        annot_output = os.path.join(self.out_dir, f"{hemisphere}.medial_wall.annot")
        
        print(f"{hemisphere} hemisphere processing completed")
        return medial_wall_final
    
    def run(self):
        """Main processing pipeline"""
        print("Starting medial wall mapping...")
        
        # Load data
        self.load_volume_data()
        
        # Convert surface formats if needed
        lh_surface_gii = self.lh_surface
        rh_surface_gii = self.rh_surface
        
        # Process hemispheres
        lh_final = self.process_hemisphere("lh", lh_surface_gii)
        rh_final = self.process_hemisphere("rh", rh_surface_gii)
        
        # Create combined volume mask
        print("Creating combined volume mask...")
        combined_mask = np.logical_or(lh_final, rh_final).astype(np.float32)
        self.save_volume(combined_mask, "medial_wall_combined.nii.gz")
        
        # Move final combined mask to output directory
        shutil.move(
            os.path.join(self.work_dir, "medial_wall_combined.nii.gz"),
            os.path.join(self.out_dir, "medial_wall_combined.nii.gz")
        )
        
        # Generate summary report
        self.generate_summary_report(lh_final, rh_final)
        
        # Clean up if requested
        if self.clean_up:
            self.cleanup()
        
        print("Medial wall mapping completed successfully!")
    
    def generate_summary_report(self, lh_mask, rh_mask):
        """Generate processing summary report"""
        lh_voxels = np.sum(lh_mask > 0)
        rh_voxels = np.sum(rh_mask > 0)
        
        print("\n" + "="*50)
        print("MEDIAL WALL EXTRACTION SUMMARY")
        print("="*50)
        print(f"Left hemisphere:  {lh_voxels:>8} voxels")
        print(f"Right hemisphere: {rh_voxels:>8} voxels")
        print(f"Total:            {lh_voxels + rh_voxels:>8} voxels")
        print(f"Output directory: {self.out_dir}")
        print("="*50)
        
        # List output files
        print("\nOutput files:")
        for f in os.listdir(self.out_dir):
            if f.endswith(('.nii.gz', '.shape.gii', '.annot')):
                filepath = os.path.join(self.out_dir, f)
                filesize = os.path.getsize(filepath) / 1024  # KB
                print(f"  - {f} ({filesize:.1f} KB)")
    
    def cleanup(self):
        """Clean up intermediate files"""
        print("Cleaning up intermediate files...")
        for f in os.listdir(self.work_dir):
            if f.endswith('.nii.gz') and not f.startswith('medial_wall_combined'):
                os.remove(os.path.join(self.work_dir, f))


def main():
    parser = argparse.ArgumentParser(
        description="Medial Wall Mapping - Extract medial wall regions from filled volume and map to surface space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python medial_wall_mapping.py \\
        -w ./temp_work \\
        -e ./mid_wall.nii.gz \\
        -f ./filled.nii.gz \\
        -l ./lh.pial \\
        -r ./rh.pial \\
        -o ./output \\
        -d 3 -m 2 -t 0.5 -c
        """
    )
    
    # Required arguments
    parser.add_argument('-w', '--work-dir', required=True, help='Work directory')
    parser.add_argument('-e', '--mid-wall', required=True, help='Initial middle wall mask')
    parser.add_argument('-f', '--filled', required=True, help='Filled volume (filled.nii.gz)')
    parser.add_argument('-l', '--lh-surface', required=True, help='Left hemisphere surface file')
    parser.add_argument('-r', '--rh-surface', required=True, help='Right hemisphere surface file')
    parser.add_argument('-o', '--out-dir', required=True, help='Output directory')
    
    # Optional arguments
    parser.add_argument('-d', '--dilate-size', type=int, default=3, 
                       help='Dilation kernel size for hemisphere masks (default: 3)')
    parser.add_argument('-m', '--morph-size', type=int, default=2,
                       help='Morphological operation kernel size (default: 2)')
    parser.add_argument('-t', '--distance-threshold', type=float, default=0.5,
                       help='Cluster threshold for connected component analysis (default: 0.5)')
    parser.add_argument('-c', '--clean-up', action='store_true',
                       help='Clean up intermediate files')
    
    args = parser.parse_args()
    
    # Print parameters
    print("Running MedialWallMapping with the following parameters:")
    print(f"  Work directory:          {args.work_dir}")
    print(f"  Initial middle wall:     {args.mid_wall}")
    print(f"  Filled volume:           {args.filled}")
    print(f"  Left hemisphere surface: {args.lh_surface}")
    print(f"  Right hemisphere surface: {args.rh_surface}")
    print(f"  Output directory:        {args.out_dir}")
    print(f"  Dilation size:           {args.dilate_size}")
    print(f"  Morphology size:         {args.morph_size}")
    print(f"  Cluster threshold:       {args.distance_threshold}")
    print(f"  Clean up:                {'yes' if args.clean_up else 'no'}")
    print()
    
    # Run the medial wall mapper
    try:
        mapper = MedialWallMapper(args)
        mapper.run()
        print("\nEND: MedialWallMapping")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()