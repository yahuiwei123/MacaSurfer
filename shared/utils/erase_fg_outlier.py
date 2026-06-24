import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, binary_opening, binary_closing, binary_erosion
from sklearn.cluster import KMeans
import argparse

def process(args=None):
    # Parse arguments
    in_path = args.in_path
    msk_path = args.msk_path
    out_path = args.out_path
    n_kmeans = args.n_kmeans
    modality = args.modality
    
    # Read MRI file
    mri_image = sitk.ReadImage(in_path)
    mri_array = sitk.GetArrayFromImage(mri_image)
    
    # Obtain foreground mask
    msk_image = sitk.ReadImage(msk_path)
    msk_array = sitk.GetArrayFromImage(msk_image)
    
    # Obtain foreground image
    fore_array = mri_array * msk_array

    # Binary process & Use KMEANS to select approperiate threshold
    data = fore_array.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_kmeans, random_state=0)
    kmeans.fit(data)
    cluster_centers = sorted(kmeans.cluster_centers_.flatten())
    
    # Find alternative noise points
    threshold = cluster_centers[-1]
    alter_noise = fore_array > threshold
    
    # Find isolated noise points
    structure = np.ones((3, 3, 3), dtype=np.int)
    isolated_noise = alter_noise - binary_erosion(alter_noise, structure)

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
    parser.add_argument("--msk_path", type=str, default='', help="foreground mask (brainmask)")
    parser.add_argument("--out_path", type=str, default='', help="MRI file to be output")
    parser.add_argument("--n_kmeans", type=int, default=5, help="centers of kmeans")
    parser.add_argument("--modality", type=str, default='T1', help="centers of kmeans")
    args = parser.parse_args()
    
    process(args=args)