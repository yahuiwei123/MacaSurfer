#!/usr/bin/env python
import argparse
import os
from typing import List, Optional

import numpy as np
import torch
import nibabel as nib
from pytorch3d.structures import Meshes


def load_surface_file(surface_path: str,
                      volume_affine: Optional[np.ndarray] = None) -> Meshes:
    """
    Load a surface file (GIFTI format) and return it as a PyTorch3D Mesh.

    Parameters
    ----------
    surface_path : str
        Path to the GIFTI surface file (.surf.gii).
    volume_affine : np.ndarray, optional
        If not None, the inverse of this affine will be applied to the
        vertex coordinates to bring them into voxel space or another
        desired space.

    Returns
    -------
    Meshes
        A PyTorch3D Mesh object with a single mesh:
        - verts: FloatTensor of shape (N, 3)
        - faces: LongTensor of shape (F, 3)
    """
    gii = nib.load(str(surface_path))

    # GIFTI "pointset" = vertex coordinates, "triangle" = face indices
    verts = gii.agg_data('pointset')
    faces = gii.agg_data('triangle')
    faces = np.asarray(faces, dtype=np.int64)

    # Basic sanity checks
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(
            f"Invalid vertices array shape for {surface_path}, "
            f"expected (N, 3), got {verts.shape}."
        )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"Invalid faces array shape for {surface_path}, "
            f"expected (F, 3), got {faces.shape}."
        )

    # Optionally transform vertices by the inverse affine
    if volume_affine is not None:
        verts = nib.affines.apply_affine(np.linalg.inv(volume_affine), verts)

    # Wrap in a single-mesh PyTorch3D Meshes object
    mesh = Meshes(
        verts=[torch.from_numpy(verts).float()],
        faces=[torch.from_numpy(faces).long()]
    )
    return mesh


def save_surf_gii(vertices: np.ndarray,
                  faces: np.ndarray,
                  filename: str) -> None:
    """
    Save vertex and face data as a GIFTI surface file.

    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (N, 3) with vertex coordinates.
    faces : np.ndarray
        Array of shape (F, 3) with triangle indices.
    filename : str
        Output file path (.surf.gii).
    """
    # Ensure correct dtypes
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    # Create GIFTI data arrays for vertices and faces
    verts_data = nib.gifti.GiftiDataArray(
        data=vertices,
        intent=nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']
    )

    faces_data = nib.gifti.GiftiDataArray(
        data=faces,
        intent=nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']
    )

    # Build GIFTI image
    gii_image = nib.gifti.GiftiImage()
    gii_image.add_gifti_data_array(verts_data)
    gii_image.add_gifti_data_array(faces_data)

    # Save to disk
    nib.save(gii_image, filename)
    print(f"Surface file saved to: {filename}")


def parse_surfs_arg(surfs_str: str) -> List[str]:
    """
    Parse a bracket-style list of surface paths, e.g.:

        "[L.white.surf.gii,R.white.surf.gii,L.pial.surf.gii,R.pial.surf.gii]"

    into a Python list of strings.

    Parameters
    ----------
    surfs_str : str
        String containing surface file names in bracket/comma-separated form.

    Returns
    -------
    List[str]
        List of surface file names/paths.
    """
    if surfs_str is None:
        return []

    s = surfs_str.strip()
    # Remove surrounding brackets if present
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    if not s:
        return []

    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple GIFTI surfaces into a single surface "
            "by concatenating all vertices and faces."
        )
    )
    parser.add_argument('--surfs', help='Input annot file (?h.cortex.label, etc.)')
    parser.add_argument('--labels', help='Input label file (?h.aparc.label.gii)')
    parser.add_argument('--root', help='Surface directory')
    parser.add_argument('--surf_out', help='Output surface file path')
    parser.add_argument('--lbl_out', help='Output label file path')
    parser.add_argument(
        "--volume",
        default=None,
        help=(
            "Optional NIfTI volume used to provide an affine. If given, the "
            "inverse affine will be applied to surface vertices."
        )
    )
    return parser.parse_args()

def load_label_gii(label_path: str) -> np.ndarray:
    """
    Load a label file (GIFTI format) and return the label values for each vertex.

    Parameters
    ----------
    label_path : str
        Path to the label file (.label.gii).

    Returns
    -------
    np.ndarray
        Array of shape (N,) containing label values for each vertex.
    """
    gii = nib.load(str(label_path))

    # Assuming the label data is stored in a single GIFTI data array
    labels = gii.agg_data()  # First array typically contains the labels

    return np.asarray(labels, dtype=np.float32)

def save_label_gii(labels: np.ndarray, filename: str) -> None:
    """
    Save the label data as a GIFTI label file.

    Parameters
    ----------
    labels : np.ndarray
        Array of shape (N,) containing label values for each vertex.
    filename : str
        Output file path (.label.gii).
    """
    # Ensure the labels are in a suitable format
    labels = np.asarray(labels, dtype=np.int32)
    
    # Create a GIFTI DataArray for labels
    label_data = nib.gifti.GiftiDataArray(
        data=labels,
        intent=nib.nifti1.intent_codes['NIFTI_INTENT_LABEL']
    )
    
    label_table = nib.gifti.GiftiLabelTable()
    bg_label = nib.gifti.GiftiLabel(key=0, red=0.0, green=0.0, blue=0.0, alpha=0.0)
    bg_label.label = 'background'
    label_table.labels.append(bg_label)
    
    fg_label = nib.gifti.GiftiLabel(key=1, red=0.0, green=0.0, blue=1.0, alpha=1.0)
    fg_label.label = 'cortical'
    label_table.labels.append(fg_label)

    # Build the GIFTI image
    gii_image = nib.gifti.GiftiImage(darrays=[label_data], labeltable=label_table)
    
    # Save to disk as a GIFTI file
    gii_image.to_filename(filename)
    print(f"Label file saved to: {filename}")


def main() -> None:
    """
    Main entry point:

      1. Parse surf list string from command line.
      2. Optionally load a reference NIfTI volume for its affine.
      3. Load each surface and label as a PyTorch3D Mesh.
      4. Concatenate all meshes and labels into a single mesh and label array.
      5. Save the merged mesh and label as GIFTI surface and label files.
    """
    args = parse_args()

    # 1) Parse surface file names from the bracket-style list
    surf_names = parse_surfs_arg(args.surfs)
    if len(surf_names) == 0:
        raise RuntimeError(
            f"No surface names parsed from --surfs argument: {args.surfs}"
        )

    # 2) Parse label file names from the same argument
    label_names = parse_surfs_arg(args.labels)

    # Compose full paths (optionally with a root directory)
    if args.root:
        surf_paths = [os.path.join(args.root, name) for name in surf_names]
        label_paths = [os.path.join(args.root, name) for name in label_names]
    else:
        surf_paths = surf_names
        label_paths = label_names

    print("Surface files to be merged:")
    for p in surf_paths:
        print(f"  {p}")
    print("Label files to be merged:")
    for p in label_paths:
        print(f"  {p}")

    # 3) Load surfaces and labels into PyTorch3D meshes and label arrays
    verts_list = []
    faces_list = []
    labels_list = []

    for surf_path, label_path in zip(surf_paths, label_paths):
        if not os.path.exists(surf_path):
            raise FileNotFoundError(f"Surface file not found: {surf_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        # Load surface
        mesh = load_surface_file(surf_path, volume_affine=None)
        print(f"Loaded surface: {surf_path}")
        verts_list.append(mesh.verts_list()[0])   # (Ni, 3)
        faces_list.append(mesh.faces_list()[0])   # (Fi, 3)

        # Load labels and align them with surface
        labels = load_label_gii(label_path)
        labels_list.append(labels)

    # 4) Combine all surfaces into a single Meshes structure
    merged_mesh = Meshes(verts=verts_list, faces=faces_list)

    # Concatenate labels
    merged_labels = np.concatenate(labels_list)
    merged_labels = np.where(merged_labels > 0, 1, 0)

    # 5) Save merged surface and label files
    merged_vertices = merged_mesh.verts_packed().cpu().numpy()
    merged_faces = merged_mesh.faces_packed().cpu().numpy()

    print(f"Merged total vertices: {merged_vertices.shape[0]}")
    print(f"Merged total faces:    {merged_faces.shape[0]}")
    
    # Save the merged surface
    save_surf_gii(merged_vertices, merged_faces, args.surf_out)
    
    # Save the merged labels (ensure the label file has the same name structure)
    save_label_gii(merged_labels, args.lbl_out)

if __name__ == "__main__":
    main()