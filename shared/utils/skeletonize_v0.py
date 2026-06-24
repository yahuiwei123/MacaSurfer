import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import argparse
import scipy.ndimage as ndi
from skimage.morphology import skeletonize

def main(args):
    msk_path = args.msk
    skel_path = args.skel
    width = args.width
    
    # read volumeinal image
    mask = nib.load(msk_path)
    mask_array = mask.get_fdata()
    
    if width > 1:
        struct = ndi.generate_binary_structure(rank=3, connectivity=1)
        mask_array = ndi.binary_dilation(mask_array, structure=struct, iterations=width - 1)
        
    # skeleton_mask = skeletonize(mask_array, method='lee')
    
    skeleton_mask = np.zeros_like(mask_array)
    
    # for i in range(mask_array.shape[0]):
    #     skeleton_mask[i, ...] += skeletonize(mask_array[i, ...], method='lee')
    
    for i in range(mask_array.shape[1]):
        skeleton_mask[:, i, ...] += skeletonize(mask_array[:, i, ...])
    
    # for i in range(mask_array.shape[2]):
    #     skeleton_mask[..., i] += skeletonize(mask_array[..., i], method='lee')
    
    # save vessel mask
    skeleton_mask = nib.Nifti1Image(skeleton_mask.astype(np.uint8), mask.affine)
    nib.save(skeleton_mask, skel_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--msk", type=str, default='', help="mask to process")
    parser.add_argument("--skel", type=str, default='', help="skeleton mask path to save")
    parser.add_argument("--width", type=int, default=1, help="width of skeleton")
    args = parser.parse_args()
    
    main(args=args)