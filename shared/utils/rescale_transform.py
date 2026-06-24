import nibabel as nib
import numpy as np
import argparse
from typing import Tuple

def main(args):
    '''
    This program help to generate a new affne or warp field, when vox2ras matrix have changed.
    To ensure this program is correct, you need to check if data array of movable and target are same.
    '''
    
    # parse param
    orig_mov = args.orig_mov
    orig_trg = args.orig_trg
    curr_mov = args.curr_mov
    curr_trg = args.curr_trg
    affine = args.affine
    warp = args.warp
    prefix = args.prefix
    
    # read necessary matrix    
    orig_mov_vox2ras = nib.load(orig_mov).affine
    orig_trg_vox2ras = nib.load(orig_trg).affine
    curr_mov_vox2ras = nib.load(curr_mov).affine
    curr_trg_vox2ras = nib.load(curr_trg).affine
    orig_mov_vox2ras = orig_mov_vox2ras.astype(np.float64)
    orig_trg_vox2ras = orig_trg_vox2ras.astype(np.float64)
    curr_mov_vox2ras = curr_mov_vox2ras.astype(np.float64)
    curr_trg_vox2ras = curr_trg_vox2ras.astype(np.float64)
    
    # inv(mov_vox2ras) @ trg_vox2ras @ affine_transform @ trg_point = mov_point
    # so inv(curr_mov_vox2ras) @ curr_trg_vox2ras @ curr_affine = inv(orig_mov_vox2ras) @ orig_trg_vox2ras @ orig_affine
    # then we have curr_affine = inv [ inv(curr_mov_vox2ras) @ curr_trg_vox2ras ] @ inv(orig_mov_vox2ras) @ orig_trg_vox2ras @ orig_affine
    if affine:
        orig_affine = np.loadtxt(affine)
        orig_affine = orig_affine.astype(np.float64)
        # curr_affine = np.linalg.inv(curr_mov_vox2ras) @ curr_trg_vox2ras @ np.linalg.inv(orig_trg_vox2ras) @ orig_mov_vox2ras @ orig_affine
        # curr_affine = np.linalg.inv(curr_trg_vox2ras) @ orig_mov_vox2ras @ np.linalg.inv(curr_mov_vox2ras) @ orig_trg_vox2ras @ orig_affine @ np.linalg.inv(orig_mov_vox2ras) @ curr_trg_vox2ras @ np.linalg.inv(orig_trg_vox2ras) @ curr_mov_vox2ras
        # curr_affine = np.linalg.inv(orig_trg_vox2ras) @ orig_affine @ orig_mov_vox2ras @ np.linalg.inv(curr_mov_vox2ras @ np.linalg.inv(orig_mov_vox2ras)) @ np.linalg.inv(orig_mov_vox2ras) @ curr_mov_vox2ras
        curr_affine = np.linalg.inv(orig_trg_vox2ras) @ orig_affine @ orig_mov_vox2ras
        print(f"new affine matrix:\n {curr_affine}")
        np.savetxt(f"{prefix}_affine.mat", curr_affine, delimiter=' ', fmt='%.6f')
    if warp:
        orig_warp = nib.load(warp)

    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_mov", type=str, default='', help="t1_1mm image which surface overlay now")
    parser.add_argument("--orig_trg", type=str, default='', help="t1 image")
    parser.add_argument("--curr_mov", type=str, default='', help="freesurfer surface file")
    parser.add_argument("--curr_trg", type=str, default='', help="freesurfer coord center offset")
    parser.add_argument("--affine", type=str, default='', help="freesurfer coord center offset")
    parser.add_argument("--warp", type=str, default='', help="freesurfer coord center offset")
    parser.add_argument("--prefix", type=str, default='', help="output image")
    args = parser.parse_args()
    
    main(args=args)