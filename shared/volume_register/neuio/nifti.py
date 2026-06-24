import numpy as np
import nibabel as nib


def save_vol_nii(data, affine, filename):
    """
    Save a 3D or 4D volume as a compressed NIfTI (.nii or .nii.gz) file.

    Args:
        data (array-like): 3D or 4D array containing the voxel intensities.
        affine (array-like): [4, 4] affine matrix mapping voxel → world coordinates.
        filename (str): Output filename (.nii or .nii.gz).

    Notes:
        - Data must be a NumPy array or convertible to NumPy.
        - Affine must be a valid 4×4 matrix.
    """

    data = np.asarray(data)
    affine = np.asarray(affine, dtype=np.float32)

    nii_img = nib.Nifti1Image(data, affine)
    nib.save(nii_img, filename)

    print(f"NIfTI volume saved: {filename}")


def load_vol_nii(filename):
    """
    Load a NIfTI (.nii or .nii.gz) file.

    Args:
        filename (str): Path to the NIfTI file.

    Returns:
        data (np.ndarray): The loaded volume data.
        affine (np.ndarray): [4, 4] affine matrix.
        header (nib.Nifti1Header): NIfTI header object for metadata.

    Notes:
        - Returned data is always a NumPy array.
        - Header contains important metadata (e.g., voxel size, qform, sform).
    """

    nii = nib.load(filename)
    data = nii.get_fdata(dtype=np.float32)  # ensures float32 output
    affine = nii.affine.astype(np.float32)
    header = nii.header

    print(f"NIfTI volume loaded: {filename}")
    return data, affine, header
