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


def reslice(img: np.ndarray = None,
            mat: np.ndarray = None,
            source: str = "LAS",
            target: str = "LIA") -> Tuple[np.ndarray, np.ndarray]:
    """
    This function realize mri_convert --out_orientation in python, but do not conform the affine to eye.

    Parameters
    ----------
    img : np.ndarray
    mat : np.ndarray
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
            
            # change the sign of affine
            mat[:3, 3] = mat[:3, i].T * np.array([img.shape[i] - 1] * 3).T + mat[:3, 3]
            mat[:3, i] = -mat[:3, i]
            
            # reverse slices in img
            slices = [slice(None)] * img.ndim
            slices[i] = slice(None, None, -1)
            img = img[tuple(slices)]
        else:
            # find the axis
            transpose[i] = idx
    
    # change the axis
    img = np.transpose(img, transpose)
    transpose.append(3)
    mat = mat[:, transpose]
    return img, mat

def pad(img: np.ndarray,
        mat: np.ndarray,
        size: list = [256, 256, 256]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Padding the image and ensure the ras coordinates

    Parameters
    ----------
    img : np.ndarray
    mat : np.ndarray
    size : int, optional
        size to pad, by default 256

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
    """
    # A and bias
    trans = mat[:3, :3]
    bias = mat[:3, 3]
    
    # padding image
    padxl = (size[0] - img.shape[0]) // 2
    padyl = (size[1] - img.shape[1]) // 2
    padzl = (size[2] - img.shape[2]) // 2
    
    padxr = size[0] - padxl - img.shape[0]
    padyr = size[1] - padyl - img.shape[1]
    padzr = size[2] - padzl - img.shape[2]
    
    img = np.pad(img, pad_width=[(padxl, padxr), (padyl, padyr), (padzl, padzr)], mode='constant', constant_values=0)
    
    new_bias = bias - trans @ np.array([padxl, padyl, padzl]).T
    mat[:3, 3] = new_bias
    return img, mat

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
    input = args.input
    orig = args.orig
    res_mat = args.res_mat
    output = args.output

    # get orig data & information
    orig_mri = nib.load(orig)
    orig_data = orig_mri.get_fdata()
    orig_mat = orig_mri.affine
    
    orig_shape = np.array(orig_data.shape)
    orig_orient = get_orient(orig_mat)
    
    # get input data & information
    input_mri = nib.load(input)
    input_data = input_mri.get_fdata()
    input_mat = input_mri.affine
    
    input_orient = get_orient(input_mat)
    
    # reorient the output_data
    print(input_orient, ' -> ', orig_orient)
    output_data, output_mat = reslice(input_data, input_mat, input_orient, orig_orient)
    
    # resize the input data array
    rescale = np.loadtxt(res_mat)
    output_mat = np.linalg.inv(rescale) @ output_mat
    # input_shape = np.array(output_data.shape)
    # l_rm_size = (input_shape - orig_shape) // 2
    # output_data = output_data[l_rm_size[0]: l_rm_size[0] + orig_shape[0],
    #                           l_rm_size[1]: l_rm_size[1] + orig_shape[1],
    #                           l_rm_size[2]: l_rm_size[2] + orig_shape[2]]

    # save mri
    new_mri = nib.Nifti1Image(output_data, output_mat)
    nib.save(new_mri, output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='', help="input image")
    parser.add_argument("--res_mat", type=str, default='', help="rescale matrix")
    parser.add_argument("--orig", type=str, default='', help="original image")
    parser.add_argument("--output", type=str, default='', help="output image")
    args = parser.parse_args()
    
    main(args=args)