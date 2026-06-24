import numpy as np
import torch
import nibabel as nib
from pytorch3d.structures import Meshes

def save_surf_gii(vertices, faces, filename):
    """
    Save a surface (vertices + faces) into a GIFTI (.gii) file.

    Args:
        vertices (array-like): [N, 3] array of vertex coordinates.
        faces (array-like): [F, 3] array of triangular face indices.
        filename (str): Path to save the .gii file.

    Notes:
        - Vertices are stored using NIFTI_INTENT_POINTSET.
        - Faces are stored using NIFTI_INTENT_TRIANGLE.
    """

    # Ensure NumPy arrays with correct data types
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    # Create GIFTI data arrays
    verts_data = nib.gifti.GiftiDataArray(
        data=vertices,
        intent=nib.nifti1.intent_codes["NIFTI_INTENT_POINTSET"]
    )

    faces_data = nib.gifti.GiftiDataArray(
        data=faces,
        intent=nib.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"]
    )

    # Create GIFTI image
    gii = nib.gifti.GiftiImage()
    gii.add_gifti_data_array(verts_data)
    gii.add_gifti_data_array(faces_data)

    # Save
    nib.save(gii, filename)
    print(f"Surface file saved: {filename}")


def load_surf_gii(filename, volume_affine=None):
    """
    Load a GIFTI (.gii) surface file and return vertices and faces.

    Args:
        filename (str): Path to the .gii file.

    Returns:
        vertices (np.ndarray): [N, 3] vertex coordinates.
        faces (np.ndarray): [F, 3] triangular face indices.

    Notes:
        This assumes the GIFTI file contains:
        - DataArray #0: vertices (POINTSET)
        - DataArray #1: faces (TRIANGLE)
    """

    gii = nib.load(str(filename))

    verts = gii.agg_data('pointset')
    faces = gii.agg_data('triangle')
    faces = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("Invalid vertices array shape, expected (N, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Invalid faces array shape, expected (M, 3).")

    if volume_affine is not None:
        verts = nib.affines.apply_affine(np.linalg.inv(volume_affine), verts)

    return Meshes(
        verts=[torch.from_numpy(verts).float()],
        faces=[torch.from_numpy(faces).long()]
    )
