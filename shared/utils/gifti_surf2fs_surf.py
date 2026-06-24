#!/usr/bin/env python3
"""
Convert GIFTI surface (in scanner RAS space, as produced by fs_surf2ras_surf.py)
back to FreeSurfer native format (tkRAS space).

This is the inverse of fs_surf2ras_surf.py and is used to create a fake
FreeSurfer subject directory with correct-surface-space surfaces for BBRegister.

Usage:
    python gifti_surf2fs_surf.py \
        --gifti_surf /path/to/hemi-L_desc-white.surf.gii \
        --volume /path/to/space-orig_desc-brain_T1w.nii.gz \
        --output /tmp/fakesubject/surf/lh.white
"""

import os
import numpy as np
import nibabel as nib
import argparse

# RAS2LIA: FreeSurfer tkRAS → LIA-convention (from fs_surf2ras_surf.py)
RAS2LIA = np.array([[-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1]])

# LIA2RAS = RAS2LIA.T, which is its own inverse (rotation matrix)
LIA2RAS = RAS2LIA.T


def xyzc_ras_cal(nifti_path):
    """Calculate xyzc_ras matrix from a NIfTI volume (same as fs_surf2ras_surf.py)."""
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


def apply_affine_to_surf(vertices, affine):
    """Apply 4x4 affine to Nx3 vertices array."""
    vertices_h = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    new_verts = vertices_h @ affine.T
    return new_verts[:, :3]


def main():
    parser = argparse.ArgumentParser(
        description="Convert GIFTI surface (RAS) back to FreeSurfer format (tkRAS)"
    )
    parser.add_argument("--gifti_surf", required=True,
                        help="Input GIFTI surface (.surf.gii, in scanner RAS)")
    parser.add_argument("--volume", required=True,
                        help="Reference volume NIfTI (same as used for fs_surf2ras_surf.py)")
    parser.add_argument("--output", required=True,
                        help="Output FreeSurfer surface file (e.g. lh.white)")
    args = parser.parse_args()

    # Load GIFTI surface
    gii = nib.load(args.gifti_surf)
    gifti_verts = gii.darrays[0].data  # Nx3 vertices in scanner RAS
    gifti_faces = gii.darrays[1].data  # Mx3 faces

    # Calculate xyzc_ras from volume (same as forward transform)
    xyzc_ras = xyzc_ras_cal(args.volume)

    # Reverse: GIFTI scanner RAS → LIA-convention → FreeSurfer tkRAS
    # Step 1: scanner RAS → LIA-convention (inverse of xyzc_ras)
    inv_xyzc_ras = np.linalg.inv(xyzc_ras)
    verts_lia = apply_affine_to_surf(gifti_verts, inv_xyzc_ras)

    # Step 2: LIA-convention → FreeSurfer tkRAS (inverse of LIA2RAS = RAS2LIA)
    verts_tkras = apply_affine_to_surf(verts_lia, RAS2LIA)

    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Write FreeSurfer format surface
    verts_tkras = verts_tkras.astype(np.float32)
    gifti_faces = gifti_faces.astype(np.int32)
    nib.freesurfer.write_geometry(args.output, verts_tkras, gifti_faces)

    print(f"Converted {args.gifti_surf} → {args.output}")
    print(f"  Vertices: {verts_tkras.shape[0]}, Faces: {gifti_faces.shape[0]}")
    print(f"  Vert range RAS: [{gifti_verts.min(axis=0)} .. {gifti_verts.max(axis=0)}]")
    print(f"  Vert range tkRAS: [{verts_tkras.min(axis=0)} .. {verts_tkras.max(axis=0)}]")


if __name__ == "__main__":
    main()
