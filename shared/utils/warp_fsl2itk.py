import nibabel as nib
import numpy as np
import argparse

def main(args):
    fsl_warp_path = args.in_warp
    ants_warp_path = args.out_warp
    
    fsl_warp = nib.load(fsl_warp_path)
    
    wx = fsl_warp.get_fdata()[..., 0]
    wy = fsl_warp.get_fdata()[..., 1]
    wz = fsl_warp.get_fdata()[..., 2]
    
    ants_warp = np.stack((wx, -1 * wy, wz), axis=-1)
    ants_warp = nib.Nifti1Image(ants_warp, fsl_warp.affine)
    nib.save(ants_warp, ants_warp_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_warp", type=str, default=None, help="input fsl warp")
    parser.add_argument("--out_warp", type=str, default=None, help="output ants warp")
    args = parser.parse_args()
    
    main(args=args)