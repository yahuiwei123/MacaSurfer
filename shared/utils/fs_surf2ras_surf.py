import os
import re
import subprocess
import nibabel as nib
import numpy as np
import argparse

RAS2LIA = np.array([[-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1]])


def xyzc_ras_cal(nifti_path):
    """read and calculate xyzc matrix"""
    nii = nib.load(nifti_path)
    affine = nii.affine
    data = nii.get_fdata()
    
    center = np.array(data.shape) / 2
    center = np.append(center, 1.0)
    c_ras = affine @ center.T
    
    xyzc_ras = np.zeros((4, 4))
    xyzc_ras[:, 3] = c_ras
    xyzc_ras[:3, 0] = affine[:3, 0] / np.linalg.norm(affine[:3, 0])
    xyzc_ras[:3, 1] = affine[:3, 1] / np.linalg.norm(affine[:3, 1])
    xyzc_ras[:3, 2] = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
    return xyzc_ras

def apply_affine_to_surf(vertices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """apply affine transform to coordinates of  vertices"""
    vertices = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    new_verts = vertices @ affine.T
    return new_verts[:, :3]


def main(args):
    surf_path = args.fs_surf
    nii_path = args.volume
    output_path = args.ras_surf
    
    # Calculate affine from RAS to FreeSurfer coordinates
    xyzc_ras = xyzc_ras_cal(nii_path)
    print("xyzc_ras:")
    print(xyzc_ras)

    # FreeSurfer use LIA as range of coordinates
    LIA2RAS = RAS2LIA.T

    # Transform FreeSurfer coordinates to RAS coordinates
    verts, faces = nib.freesurfer.read_geometry(surf_path)
    
    new_verts = apply_affine_to_surf(verts, LIA2RAS)
    new_verts = apply_affine_to_surf(new_verts, xyzc_ras)
    
    # Save result
    new_verts = new_verts.astype(np.float32)
    new_surf = nib.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(new_verts[:, :3], intent=nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']),
        nib.gifti.GiftiDataArray(faces, intent=nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])
    ])
    nib.save(new_surf, output_path)

    print(f"End surface format transform, saved at {output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", type=str, default='', help="input volume (aligned with surface)")
    parser.add_argument("--fs_surf", type=str, default='', help="input surface path (FreeSurfer format)")
    parser.add_argument("--ras_surf", type=str, default=None, help="output surface path (GIFTI format)")
    args = parser.parse_args()
    main(args=args)
    
