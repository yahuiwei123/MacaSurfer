import numpy as np
import nibabel as nib
import argparse
import scipy.ndimage as ndi
import collections
from skimage import io, filters, feature, measure
from sklearn.decomposition import PCA

def compute_pca_elongation(region):
    coords = region.coords
    if len(coords) < 10:
        return 1.0
    
    pca = PCA(n_components=2)
    pca.fit(coords)

    eigenvalues = pca.explained_variance_
    elongation_ratio = eigenvalues[0] / (eigenvalues[1] + 1e-5)
    
    return elongation_ratio


def segment_vessels_iterative(volume, label, valid, grad_thres: float = 10, num_sig: float = 3,
                              csf_labels=None, wm_labels=None):
    if csf_labels is None:
        csf_labels = [1]
    if wm_labels is None:
        wm_labels = [3]
    # assume tissue follow GMM
    csf_array = np.isin(label, csf_labels)
    csf_mean = np.mean(volume[csf_array > 0])
    csf_std = np.std(volume[csf_array > 0])
    
    print(f"\tclass: csf\tmean: {csf_mean}")
    print(f"\tclass: csf\tstd: {csf_std}")
    
    # obtain intensity threshold
    csf_thres = csf_mean + num_sig * csf_std
    intensity_thres = csf_thres
    
    # hessian
    sigma = 1.0
    H_elems = feature.hessian_matrix(volume, use_gaussian_derivatives=True, sigma=sigma, order='rc')
    eigenvals = feature.hessian_matrix_eigvals(H_elems)
    trace_hessian = np.sum(eigenvals, axis=0)
    hessian_threshold = -15
    
    # select seeds
    intensity_seeds = np.where(np.isin(label, [0] + list(csf_labels)), volume, 0) > intensity_thres
    hessian_seeds = np.where(np.isin(label, [0] + list(csf_labels)), trace_hessian, 0) < hessian_threshold
    seeds = intensity_seeds & hessian_seeds & (valid > 0)
    
    vessel_mask = np.zeros_like(volume, dtype=bool)
    seeds = np.argwhere(seeds > 0)
    queue = collections.deque(seeds)
    for seed in seeds:
        vessel_mask[tuple(seed)] = True

    # region extend in volumeinal image
    neighbors = [[i, j, k] 
             for i in [-1, 0, 1] 
             for j in [-1, 0, 1] 
             for k in [-1, 0, 1] 
             if not (i == 0 and j == 0 and k == 0)]
    neighbors = np.array(neighbors)
    
    shape = volume.shape
    
    wm_array = np.isin(label, wm_labels)
    wm_mean = np.mean(volume[wm_array > 0])
    wm_std = np.std(volume[wm_array > 0])

    while queue:
        current = queue.popleft()
        for offset in neighbors:
            neighbor = current + offset
            
            if np.any(neighbor < 0) or np.any(neighbor >= shape):
                continue
            
            neighbor = tuple(neighbor)
            current = tuple(current)
            
            if valid[neighbor] < 1 or vessel_mask[neighbor]:
                continue
            
            gradient = volume[neighbor] - volume[current]

            
            if gradient > 0 and volume[neighbor] > wm_mean - wm_std:
                vessel_mask[neighbor] = True
                queue.append(np.array(neighbor))
            else:
                intensity_thres = wm_mean + num_sig * wm_std
                hessian_threshold = -8
                if volume[neighbor] > intensity_thres and abs(gradient) < grad_thres and trace_hessian[neighbor] < hessian_threshold:
                    vessel_mask[neighbor] = True
                    queue.append(np.array(neighbor))
                    
    labeled_image = measure.label(vessel_mask, connectivity=1)
    new_mask = np.zeros_like(vessel_mask)
    elongation_threshold = 2
    for region in measure.regionprops(labeled_image):
        elongation = compute_pca_elongation(region)
        
        if elongation > elongation_threshold:
            new_mask[labeled_image == region.label] = 1
            
    new_mask = new_mask.astype(np.uint8) * np.where(np.isin(label, wm_labels), 0, 1)

    return new_mask.astype(np.uint8)

def prep(img: np.array = None, lab: np.array = None,
         csf_labels=None, wm_labels=None):
    if csf_labels is None:
        csf_labels = [1]
    if wm_labels is None:
        wm_labels = [3]
    # noramlize to 0-255
    img = 255 * (img - img.min()) / (img.max() - img.min())

    # extend csf
    csf_array = np.isin(lab, csf_labels)
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)
    dil_csf_array = ndi.binary_dilation(csf_array, structure=structure, iterations=2)
    csf_val = csf_labels[0]
    lab = np.where(dil_csf_array > 0, csf_val, lab)

    # extend wm
    wm_array = np.isin(lab, wm_labels)
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)
    dil_wm_array = ndi.binary_dilation(wm_array, structure=structure, iterations=1)
    wm_val = wm_labels[0]
    lab = np.where(dil_wm_array > 0, wm_val, lab)

    return img, lab
    
def main(args):
    img_path = args.img
    lab_path = args.seg
    val_path = args.val
    modal = args.modal
    msk_path = args.msk
    grad_thres = args.grad
    num_sig = args.sig

    # Parse comma-separated label lists
    csf_labels = [int(x.strip()) for x in args.csf_labels.split(',') if x.strip()]
    wm_labels = [int(x.strip()) for x in args.wm_labels.split(',') if x.strip()]

    # read volumeinal image
    img = nib.load(img_path)
    img_array = img.get_fdata()

    # read tissue
    lab = nib.load(lab_path)
    lab_array = lab.get_fdata()

    # read valid region
    val = nib.load(val_path)
    val_array = val.get_fdata()

    # check mask valid
    if not np.allclose(img.affine, lab.affine):
        raise ValueError(f"Affine of volumeinal image is not equal to affine of wm mask!")

    if img.shape != lab.shape:
        raise ValueError(f"Shape of volumeinal image and label is not equal!")

    # preprocess image
    img_array, lab_array = prep(img_array, lab_array,
                                csf_labels=csf_labels, wm_labels=wm_labels)

    vessel_mask = segment_vessels_iterative(
        img_array, lab_array, val_array, grad_thres=grad_thres, num_sig=num_sig,
        csf_labels=csf_labels, wm_labels=wm_labels)
    
    # save vessel mask
    msk = nib.Nifti1Image(vessel_mask, img.affine)
    nib.save(msk, msk_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default='', help="volumeinal image")
    parser.add_argument("--seg", type=str, default='', help="tissue segmentation in volumeinal image")
    parser.add_argument("--val", type=str, default='', help="valid blood vessel region")
    parser.add_argument("--modal", type=str, default='T1', help="modality of volumeinal image")
    parser.add_argument("--msk", type=str, default='', help="vessel mask path to save")
    parser.add_argument("--sig", type=float, default=2.0, help="vessel mask path to save")
    parser.add_argument("--grad", type=float, default=60, help="vessel mask path to save")
    parser.add_argument("--csf_labels", type=str, default="1",
                        help="Comma-separated CSF label IDs (default: 1)")
    parser.add_argument("--wm_labels", type=str, default="3",
                        help="Comma-separated WM label IDs (default: 3)")
    args = parser.parse_args()
    
    main(args=args)
