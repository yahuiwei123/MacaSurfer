import nibabel as nib
from nibabel.gifti import GiftiImage, GiftiDataArray
import numpy as np
import argparse

def main(args):
    # read necessary files
    orig_img = nib.load(args.orig_img)
    affine_transformed_img = nib.load(args.affine_img)
    surf = nib.load(args.in_surf)
    vertices = surf.darrays[0].data
    faces = surf.darrays[1].data

    # transform to original space
    vertices = np.concatenate((vertices, np.ones((vertices.shape[0], 1))), axis=1)
    vertices = vertices @ np.linalg.inv(affine_transformed_img.affine).T @ orig_img.affine.T
    vertices = vertices[:, 0:-1].astype(np.float32)

    # save as GIFTI file
    new_surf = nib.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(vertices, intent=nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']),
        nib.gifti.GiftiDataArray(faces, intent=nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])
    ])
    nib.save(new_surf, args.out_surf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_img", type=str, default='', help="original image")
    parser.add_argument("--affine_img", type=str, default='', help="affine transformed image")
    parser.add_argument("--in_surf", type=str, default='', help="input surface path")
    parser.add_argument("--out_surf", type=str, default='', help="output surface path")
    args = parser.parse_args()
    
    main(args=args)