import nibabel as nib
import numpy as np
import argparse
from typing import Tuple
import re
from scipy.stats import zscore
from scipy.ndimage import zoom
from scipy.optimize import minimize

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
    new_axis = ""
    for i, s in enumerate(source):
        idx = target.find(s)
        if idx == -1:
            # find the axis
            new_axis += switch(s)
            
            # change the sign of affine
            mat[:3, 3] = mat[:3, i].T * np.array([img.shape[i] - 1] * 3).T + mat[:3, 3]
            mat[:3, i] = -mat[:3, i]
            
            # reverse slices in img
            slices = [slice(None)] * img.ndim
            slices[i] = slice(None, None, -1)
            img = img[tuple(slices)]
        else:
            new_axis += s

    transpose = [0, 1, 2]
    for i, t in enumerate(target):
        idx = new_axis.find(t)
        transpose[i] = idx
    
    # change the axis
    img = np.transpose(img, transpose)
    transpose.append(3)
    mat = mat[:, transpose]
    return img, mat

def crop(img: np.ndarray,
         msk: np.ndarray,
         mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop to the valid area of image

    Parameters
    ----------
    img : np.ndarray
    msk : np.ndarray
        Mask of valid area in <img>, and must has the same fov and vox2ras matrix with <img>
    mat : np.ndarray

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
    """
    # store original dtype to ensure consistency
    orig_dtype = img.dtype
    orig_shape = img.shape
    
    # crop image array
    non_zero_indices = np.nonzero(msk)
    
    i_min, i_max = max(0, non_zero_indices[0].min() - 7), min(non_zero_indices[0].max() + 7, orig_shape[0])
    j_min, j_max = max(0, non_zero_indices[1].min() - 7), min(non_zero_indices[1].max() + 7, orig_shape[1])
    k_min, k_max = max(0, non_zero_indices[2].min() - 7), min(non_zero_indices[2].max() + 7, orig_shape[2])
    
    print(f"crop i from {i_min} to {i_max + 1}")
    print(f"crop j from {j_min} to {j_max + 1}")
    print(f"crop k from {k_min} to {k_max + 1}")
    
    img = img[i_min: i_max + 1,
              j_min: j_max + 1,
              k_min: k_max + 1].astype(orig_dtype, copy=False)
    
    # Calculate new bias
    trans = mat[:3, :3]
    bias = mat[:3, 3]
    new_bias = bias + trans @ np.array([i_min, j_min, k_min]).T
    mat[:3, 3] = new_bias
    
    return img, mat

def padding(img: np.ndarray,
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
    # padding image array
    padxl = (size[0] - img.shape[0]) // 2
    padyl = (size[1] - img.shape[1]) // 2
    padzl = (size[2] - img.shape[2]) // 2
    
    padxr = size[0] - padxl - img.shape[0]
    padyr = size[1] - padyl - img.shape[1]
    padzr = size[2] - padzl - img.shape[2]
    img = np.pad(img, pad_width=[(padxl, padxr), (padyl, padyr), (padzl, padzr)], mode='constant', constant_values=0)
    
    # Calculate new bias
    trans = mat[:3, :3]
    bias = mat[:3, 3]
    new_bias = bias - trans @ np.array([padxl, padyl, padzl]).T
    mat[:3, 3] = new_bias
    
    return img, mat

def isotropy(img: np.ndarray,
             mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    trans = mat[:3, :3]
    orig_res = np.linalg.norm(trans, axis=0)
    targ_res = np.array(3 * [np.min(orig_res)], dtype=np.float32) if np.min(orig_res) < 0.5 else np.array(3 * [0.5], dtype=np.float32)
    # raw scaling factor (voxel-size ratio), used for affine correction
    raw_zoom = np.array([orig / targ for orig, targ in zip(orig_res, targ_res)])
    raw_zoom = np.where(raw_zoom > 1.05, raw_zoom, 1)

    # compute target size from FOV to avoid align-corners rounding instability
    orig_size = np.array(img.shape)
    fov_mm = orig_size * orig_res
    targ_size = np.rint(fov_mm / targ_res).astype(int)
    zoom_factors = targ_size / orig_size

    # interpolate
    iso_img = zoom(img, zoom_factors, order=0) # B sample
    msk_img = zoom(img, zoom_factors, order=0) # nearest
    iso_img = np.where(msk_img, iso_img, 0)

    # fix affine
    scaling_factors = np.diag(np.append(1.0 / raw_zoom, 1))
    new_mat = mat @ scaling_factors
    return iso_img, new_mat

def resolution_to(mat: np.ndarray,
                  center: np.ndarray,
                  res: list = [1, 1, 1]) -> np.ndarray:
    """
    Change original resolution in <mat> to <res>

    Parameters
    ----------
    mat : np.ndarray
        sform or vox2ras matrix
    center : np.ndarray
        ijk of mri image center
    res : int, optional
        resolution, by default 1

    Returns
    -------
    np.ndarray
        new sform or vox2ras matrix
    """
    trans = mat[:3, :3]
    res = np.array(res)
    orig_res = np.linalg.norm(trans, axis=1)
    orig_center = np.append(center, 1) @ mat.T
    orig_center = orig_center[..., :3]
    factor = res / orig_res
    mat[:3, :3] = mat[:3, :3] * factor
    scaled_center = np.append(center, 1) @ mat.T
    scaled_center = scaled_center[..., :3]
    bias = scaled_center - orig_center
    mat[..., 3] = mat[..., 3] - np.append(bias, 0).T
    return mat

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

def normalize_t1(img: np.ndarray, norm: int = 255, gamma: float = 1.0) -> np.ndarray:
    """
    Normalize grayscale of img to 0-255 and change data type to usigned int

    Parameters
    ----------
    img : np.ndarray

    Returns
    -------
    np.ndarray
    """
    msk = img > 0
    
    upper_quantile = 0.995
    lower_quantile = 0.015
    # normalize to 0-255
    max_val = np.percentile(img[img > 0], upper_quantile * 100.0)
    min_val = np.percentile(img[img > 0], lower_quantile * 100.0)
    print(f"value range: [{min_val}, {max_val}] -> [0, {norm}]")
    
    # normalize to [0, 1]
    norm_img = (img - min_val) / (max_val - min_val)
    norm_img = np.clip(norm_img, 0, 1)
    
    # apply Gamma transform
    gamma_img = np.power(norm_img, gamma)
    
    # recover to original region
    out_img = np.where(msk, gamma_img * (norm - 5) + 5, 0)

    # change dtype to uint8
    uint8 = out_img.astype(np.uint8)
    
    return uint8

def normalize_t2(img: np.ndarray, norm: int = 255, gamma: float = 1.0) -> np.ndarray:
    """
    Normalize grayscale of img to 0-255 and change data type to usigned int

    Parameters
    ----------
    img : np.ndarray

    Returns
    -------
    np.ndarray
    """
    msk = img > 0
    
    upper_quantile = 0.985
    lower_quantile = 0.000
    # normalize to 0-255
    max_val = np.percentile(img, upper_quantile * 100.0)
    min_val = np.percentile(img, lower_quantile * 100.0)
    print(f"value range: [{min_val}, {max_val}] -> [0, {norm}]")
    
    # normalize to [0, 1]
    norm_img = (img - min_val) / (max_val - min_val)
    norm_img = np.clip(norm_img, 0, 1)
    
    # apply Gamma transform
    gamma_img = np.power(norm_img, gamma)
    
    # recover to original region
    out_img = np.where(msk, gamma_img * (norm - 20) + 20, 0)

    # change dtype to uint8
    uint8 = out_img.astype(np.uint8)
    
    return uint8

def main(args):
    # parse param
    input = args.input
    output = args.output

    # get data
    mri = nib.load(input)
    data_arr = mri.get_fdata()
    xform_mat = mri.affine
    data_type = mri.get_data_dtype()
        
    if args.crop_as:
        # Crop to ensure size of image array smaller than (256, 256, 256)
        mask = args.crop_as
        mask = nib.load(mask)
        mask = mask.get_fdata()
        print('croping image...')
        data_arr, xform_mat = crop(data_arr, mask, xform_mat)
        print('done')
        
    if args.isotropy:
        print('isotropy image...')
        data_arr, xform_mat = isotropy(data_arr, xform_mat)
        print('done')
    
    if args.pad:
        pad = list(map(int, re.sub(r"[\[\]]", "", args.pad).split(',')))
        # padding image
        size = pad
        print('padding to size of ', size, '...')
        data_arr, xform_mat = padding(data_arr, xform_mat, size=size)
        print('done')
        
    if args.rescale:
        
        if not args.hires:
            rescale = list(map(float, re.sub(r"[\[\]]", "", args.rescale).split(',')))
            # change resolution to 1mm
            res = rescale
            print('changing resolution to ', res, '...')
            orig_xform_mat = np.copy(xform_mat)
            xform_mat = resolution_to(xform_mat, np.array(data_arr.shape) / 2 ,res=res)
            print('done')
            
            # save transform
            if args.omat:
                omat = args.omat
                print(xform_mat, orig_xform_mat)
                transform = xform_mat @ np.linalg.inv(orig_xform_mat)
                np.savetxt(omat, transform, delimiter=' ', fmt='%.6f')
        else:
            rescale = list(map(float, re.sub(r"[\[\]]", "", args.rescale).split(',')))
            # change resolution to 1mm
            orig_xform_mat = np.copy(xform_mat)
            trans = orig_xform_mat[:3, :3]
            orig_res = np.linalg.norm(trans, axis=1)
            res = np.array([2.5, 2.5, 2.5]) * orig_res
            print('changing resolution to ', res, '...')
            xform_mat = resolution_to(xform_mat, np.array(data_arr.shape) / 2 ,res=res)
            print('done')
            
            # save transform
            if args.omat:
                omat = args.omat
                print(xform_mat, orig_xform_mat)
                transform = xform_mat @ np.linalg.inv(orig_xform_mat)
                np.savetxt(omat, transform, delimiter=' ', fmt='%.6f')
            
    if args.reorient:
        orient = args.reorient
        # reorient to <orient> and reslice
        targ_orient = orient
        orig_orient = get_orient(xform_mat)
        print('reorient: ', orig_orient, ' -> ', targ_orient)
        data_arr, xform_mat = reslice(data_arr, xform_mat, source=orig_orient, target=targ_orient)
        print('done')
        
    if args.norm and args.modal == 'T1':
        print('normalizing T1 image...')
        norm = float(args.norm)
        gamma = float(args.gamma)
        data_arr = normalize_t1(data_arr, norm, gamma)
        print('done')
    elif args.norm and args.modal == 'T2':
        print('normalizing T1 image...')
        norm = float(args.norm)
        gamma = float(args.gamma)
        data_arr = normalize_t2(data_arr, norm, gamma)
        print('done')
        
    # save mri
    new_mri = nib.Nifti1Image(data_arr.astype(data_type), xform_mat, header=mri.header)
    nib.save(new_mri, output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='', help="input image")
    parser.add_argument("--output", type=str, default='', help="output image")
    parser.add_argument("--crop_as", type=str, default=None, help="crop to nonzero area")
    parser.add_argument("--pad", type=str, default=None, help="change the size to 256x256x256")
    parser.add_argument("--isotropy", action='store_true', default=None, help="interpoalte the image to isotropy")
    parser.add_argument("--rescale", type=str, default=None, help="change the resolution to 1mm")
    parser.add_argument("--hires", action="store_true", help="hires change the resolution to 1mm (must with rescale)")
    parser.add_argument("--omat", type=str, default=None, help="transform between two resolution")
    parser.add_argument("--reorient", type=str, default=None, help="change the orientation to specified orientation")
    parser.add_argument("--norm", type=str, default=None, help="normalize grayscale to 0-255")
    parser.add_argument("--modal", type=str, default='T1', help="image modality (T1/T2)")
    parser.add_argument("--gamma", type=str, default=None, help="gamma transform intensity")
    args = parser.parse_args()
    
    main(args=args)