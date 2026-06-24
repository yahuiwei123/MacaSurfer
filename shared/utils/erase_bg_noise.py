import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, binary_opening, binary_closing
from sklearn.cluster import KMeans
import argparse

def process(args=None):
    # Parse arguments
    in_path = args.in_path
    out_path = args.out_path
    n_kmeans = args.n_kmeans
    modality = args.modality
    
    # Read MRI file
    mri_image = sitk.ReadImage(in_path)
    mri_array = sitk.GetArrayFromImage(mri_image)
    
    

    # Binary process & Use KMEANS to select approperiate threshold
    data = mri_array[mri_array > 0].flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_kmeans, random_state=0)
    kmeans.fit(data)
    cluster_centers = sorted(kmeans.cluster_centers_.flatten())
    
    if modality == 'T1':
        threshold = cluster_centers[0]
        binary_image = mri_array > threshold
    else:
        threshold = cluster_centers[-1]
        binary_image = (mri_array < threshold) & (mri_array > 0)

    # Find biggest region
    structure = np.ones((3, 3, 3), dtype=np.int16)
    labeled_image, num_features = label(binary_image, structure=structure)
    region_sizes = np.bincount(labeled_image.flatten())
    largest_region_label = region_sizes[1:].argmax() + 1  # oversee 0 value in background
    largest_region = (labeled_image == largest_region_label)
    

    # Open operation to erase boundaries
    structure = np.ones((3, 3, 3), dtype=np.int16)
    opened_image = binary_opening(largest_region, structure=structure)

    # Close operation to fill holes
    closed_image = binary_closing(opened_image, structure=structure)

    filled_image = closed_image

    # Save the file
    largest_region_image = sitk.GetImageFromArray(filled_image.astype(np.uint8))
    largest_region_image.CopyInformation(mri_image)
    sitk.WriteImage(largest_region_image, out_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default='', help="MRI file to be processed")
    parser.add_argument("--out_path", type=str, default='', help="MRI file to be output")
    parser.add_argument("--n_kmeans", type=int, default=5, help="centers of kmeans")
    parser.add_argument("--modality", type=str, default='T1', help="centers of kmeans")
    args = parser.parse_args()
    
    process(args=args)