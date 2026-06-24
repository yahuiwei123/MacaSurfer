import nibabel as nib
import numpy as np
import argparse

def apply_affine_to_surf(vertices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """apply affine transform to coordinates of  vertices"""
    vertices = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    new_verts = vertices @ affine.T
    return new_verts[:, :3]


def main(args):
    
    orig_vol_path = args.orig_vol
    scaled_vol_path = args.scaled_vol
    in_surf_path = args.in_surf
    out_surf_path = args.out_surf
    
    # Transform scaled coordinates to original coordinates
    surf = nib.load(in_surf_path).darrays
    verts, faces = surf[0].data, surf[1].data
    
    scaled_vol_affine = nib.load(scaled_vol_path).affine    
    new_verts = apply_affine_to_surf(verts, np.linalg.inv(scaled_vol_affine))
    
    orig_vol_affine = nib.load(orig_vol_path).affine
    new_verts = apply_affine_to_surf(new_verts, orig_vol_affine)
    
    # Save result
    new_verts = new_verts.astype(np.float32)
    new_surf = nib.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(new_verts[:, :3], intent=nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']),
        nib.gifti.GiftiDataArray(faces, intent=nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])
    ])
    nib.save(new_surf, out_surf_path)

    print(f"End surface coordinate transform, saved at {out_surf_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_vol", type=str, default='', help="original input volume (only xform matrix different with scaled input volume)")
    parser.add_argument("--scaled_vol", type=str, default='', help="scaled input volume (aligned with surface)")
    parser.add_argument("--in_surf", type=str, default=None, help="output surface path (GIFTI format)")
    parser.add_argument("--out_surf", type=str, default='', help="output surface path (GIFTI format)")
    args = parser.parse_args()
    main(args=args)
    
