import nibabel as nib
import numpy as np
import argparse
from typing import Tuple

def switch(case: str) -> str:
    cases = {
        'R': 'L',
        'L': 'R',
        'A': 'P',
        'P': 'A',
        'S': 'I',
        'I': 'S',
    }
    return cases.get(case)


def reslice(vertices: np.ndarray = None,
            source: str = "LAS",
            target: str = "LIA") -> np.ndarray:
    """
    This function realize mri_convert --out_orientation in python, but do not conform the affine to eye.

    Parameters
    ----------
    vertices : np.ndarray
    source : str
    target : str
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
    """
    transpose = [0, 1, 2]
    for i, s in enumerate(source):
        idx = target.find(s)
        if idx == -1:
            # find the axis
            idx = target.find(switch(s))
            transpose[i] = idx

            # change the sign of coordiantes
            vertices[..., i] = -vertices[..., i]
        else:
            # find the axis
            transpose[i] = idx
    
    # change the axis
    vertices = vertices[..., transpose]
    return vertices

def get_orient(xform: np.ndarray = None) -> str:
    """
    Get orientation from xform matrix

    Parameters
    ----------
    xform : np.ndarray

    Returns
    -------
    str
        Anatomical coordinates ('RAS', 'LIA'...)
    """
    trans = xform[:3, :3]
    orient = ""
    for i in range(trans.shape[0]):
        vec = trans[..., i]
        idx = np.argmax(np.abs(vec))
        if idx == 0:
            if vec[idx] > 0:
                ax = "R"
                orient += ax
            else:
                ax = "L"
                orient += ax
        elif idx == 1:
            if vec[idx] > 0:
                ax = "A"
                orient += ax
            else:
                ax = "P"
                orient += ax
        else:
            if vec[idx] > 0:
                ax = "S"
                orient += ax
            else:
                ax = "I"
                orient += ax
    return orient


def main(args):
    # parse param
    t1_1mm = args.t1_1mm
    t1 = args.t1
    surf = args.surf
    t1_xform = args.t1_xform
    t1_1mm_xform = args.t1_1mm_xform
    output = args.output
    
    # get offset matrix
    t1_xform = np.loadtxt(t1_xform)
    t1_1mm_xform = np.loadtxt(t1_1mm_xform)

    # get t1 data & information
    t1_mri = nib.load(t1)
    t1_data = t1_mri.get_fdata()
    t1_mat = t1_mri.affine
    
    t1_shape = np.array(t1_data.shape)
    t1_orient = get_orient(t1_mat)
    t1_res = np.linalg.norm(t1_mat[:3, :3], axis=1)
    
    # get t1_1mm data & information
    t1_1mm_mri = nib.load(t1_1mm)
    t1_1mm_data = t1_1mm_mri.get_fdata()
    t1_1mm_mat = t1_1mm_mri.affine
    
    t1_1mm_shape = np.array(t1_1mm_data.shape)
    t1_1mm_orient = get_orient(t1_1mm_mat)
    t1_1mm_res = np.linalg.norm(t1_1mm_mat[:3, :3], axis=1)
    
    # get surface
    FS_vertices, faces = nib.freesurfer.read_geometry(surf)
    
    ones_pad = np.ones((FS_vertices.shape[0], 1))
    FS_vertices = np.hstack((FS_vertices, ones_pad))
    
    # switch FS space to 1mm RAS space through remove offset
    RAS_vertices = t1_1mm_xform @ FS_vertices.T
    RAS_vertices = RAS_vertices.T

    # transform 1mm RAS coord to ijk index
    IJK_vertices = np.linalg.inv(t1_1mm_mat) @ RAS_vertices.T
    IJK_vertices = IJK_vertices.T
    
    # transform ijk index to 0.xmm RAS coord
    RAS_vertices = t1_mat @ IJK_vertices.T
    RAS_vertices = RAS_vertices.T
    
    # switch 0.xmm space to FS space through add offset back
    FS_vertices = np.linalg.inv(t1_xform) @ RAS_vertices.T
    FS_vertices = FS_vertices.T
    FS_vertices = FS_vertices[..., :3]
    
    # save surface
    nib.freesurfer.write_geometry(output, FS_vertices, faces)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1_1mm", type=str, default='', help="t1_1mm image which surface overlay now")
    parser.add_argument("--t1", type=str, default='', help="t1 image")
    parser.add_argument("--surf", type=str, default='', help="freesurfer surface file")
    parser.add_argument("--t1_1mm_xform", type=str, default='', help="freesurfer coord center offset")
    parser.add_argument("--t1_xform", type=str, default='', help="freesurfer coord center offset")
    parser.add_argument("--output", type=str, default='', help="output image")
    args = parser.parse_args()
    
    main(args=args)