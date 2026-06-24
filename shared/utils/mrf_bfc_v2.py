import numpy as np
from scipy.ndimage import *
from scipy.stats import norm
import SimpleITK as sitk
from sklearn.cluster import KMeans
import argparse

# Load MRI Image
def load_mri_image(file_path):
    image = sitk.ReadImage(file_path)
    image_pad_array = sitk.GetArrayFromImage(image)
    return image, image_pad_array

# smooth hard label to soft label
def smooth_tissue_probabilities(tissue_types, num_classes=3, sigma=1.5):
    """
    Calculate smoothed probability distribution of tissue types based on neighborhood.
    
    Parameters:
    -----------
    tissue_types : numpy.ndarray
        3D array of shape [H, W, D] containing tissue type labels (0 to num_classes-1)
    num_classes : int
        Number of tissue type classes (default: 3)
    sigma : float or sequence of float
        Standard deviation for Gaussian kernel. Can be a single float or different values
        for each axis (default: 1.5)
        
    Returns:
    --------
    numpy.ndarray
        4D array of shape [num_classes, H, W, D] containing smoothed probabilities
        for each class at each position
    """
    
    # Get dimensions
    H, W, D = tissue_types.shape
    
    # Initialize output array
    smoothed_probs = np.zeros((num_classes, H, W, D))
    
    # Convert labels to one-hot encoding
    for class_idx in range(num_classes):
        # Create binary mask for current class
        class_mask = (tissue_types == class_idx).astype(float)
        
        # Apply Gaussian smoothing
        smoothed = gaussian_filter(class_mask, sigma=sigma, mode='constant', cval=0)
        
        # Store result
        smoothed_probs[class_idx] = smoothed
    
    # Normalize probabilities
    # Add small epsilon to avoid division by zero
    total_prob = np.sum(smoothed_probs, axis=0, keepdims=True) + 1e-10
    smoothed_probs /= total_prob
    
    return smoothed_probs

def smooth_bias_field(bias: np.ndarray, mask: np.ndarray, base: int = 10, kernel_size: float = 5) -> np.ndarray:
    """
    Smooth log space bias field in original space (huge bias in log space will cause very big effect).
    Also use SDF fill region outside of mask to ensure smooth result not affect by outside.
    """
    # transfer to original range
    orig_bias = bias # np.power(base, -bias)
    
    # fill through SDF
    mask = np.where(mask > 0, 1, 0)
    struct = generate_binary_structure(rank=3, connectivity=1)
    width = 2
    inner = binary_erosion(binary_dilation(mask, struct, iterations=1), struct, iterations=1 + width)
    orig_bias, _ = fill_outside_by_nearest(orig_bias, inner, width=width + int(kernel_size) + 5)
    
    corrected_image = sitk.GetImageFromArray(orig_bias.astype(np.float32))
    sitk.WriteImage(corrected_image, './1.nii.gz')

    # smooth
    orig_bias = mask * gaussian_filter(orig_bias, sigma=kernel_size, mode='nearest')
    return orig_bias

def estimate_bias_field(image, labels, mask, init_mean, kernel_size=3, soft_kernel=0.4):
    """
    estimate bias field
    """
    cls_num = np.unique(labels).shape[0]
    soft_labels = smooth_tissue_probabilities(labels, num_classes=cls_num, sigma=soft_kernel)
    
    # mask image
    # image = image * mask
    # mask = np.ones_like(mask)
    
    W = []
    for l in range(cls_num):
        mu = np.sum(image * soft_labels[l, ...]) / np.sum(soft_labels[l, ...])
        sigma = np.sqrt(np.sum((image - mu) ** 2 * soft_labels[l, ...]) / np.sum(soft_labels[l, ...])) # np.std(image * soft_labels[l, ...]) / np.mean(soft_labels[l, ...])
        print(f"\tclass: {l}\tmean: {mu}")
        print(f"\tclass: {l}\tstd: {sigma}")
        P_l = soft_labels[l, ...] # np.sum(labels == l) / np.shape(labels.flatten())
        W_l = P_l * norm.pdf(image, loc=mu, scale=sigma) #+ np.where(labels == l, 20, 0)
        
        W.append(W_l)
    
    W = np.array(W)
    W = W / np.sum(W, axis=0) # (cls, H, W, D)
    
    # compute residual
    R = np.zeros_like(image)
    phi = np.zeros_like(image)
    
    if modality == 'T1':
        consider = [0, 1, 2, 3]
    elif modality == 'T2':
        consider = [2]
    else:
        raise ValueError(f"Invalid modality {modality} indicate!")
    
    for l in range(cls_num):
        if l in consider:
            if init_mean:
                mean = init_mean[l] # always use initial means to keep contrast and original distribution
            else:
                mean = np.mean(image * soft_labels[l, ...]) / np.mean(soft_labels[l, ...])
                
            std = np.std(image * soft_labels[l, ...]) / np.std(soft_labels[l, ...])
            R += W[l, ...] * (image - mean) / std
            phi += W[l, ...] / std
    
    # smooth within the mask area
    residual = np.where(phi > 0, R / phi, 0)
    # mask = np.zeros_like(labels)
    # for i in consider:
    #     if i == 0:
    #         continue
    #     mask = np.logical_or(mask, np.where(labels == i, 1, 0))
    # mask = mask.astype(np.uint8)
    print(np.mean(mask), np.max(mask), np.min(mask))
    bias_field = smooth_bias_field(residual, mask, base, kernel_size)
    return bias_field

# Compute class statistics (mean and variance) for each class
def compute_class_statistics(image, labels, soft_kernel=0.4):
    num_classes = np.unique(labels).shape[0]
    soft_labels = smooth_tissue_probabilities(labels, num_classes=num_classes, sigma=soft_kernel)
    cls_mean, cls_std = [], []
    
    for l in range(num_classes):
        mu = np.sum(image * soft_labels[l, ...]) / np.sum(soft_labels[l, ...])
        sigma = np.sqrt(np.sum((image - mu) ** 2 * soft_labels[l, ...]) / np.sum(soft_labels[l, ...]))
        cls_mean.append(mu)
        cls_std.append(sigma)
    
    return cls_mean, cls_std

def refine_label(image, labels, soft_kernel=1.2):
    print(f"refine label with soft label kernel: {soft_kernel}")
    cls_num = np.unique(labels).shape[0]
    soft_labels = smooth_tissue_probabilities(labels, num_classes=cls_num, sigma=soft_kernel)
    
    W = []
    for l in range(cls_num):
        mu = np.sum(image * soft_labels[l, ...]) / np.sum(soft_labels[l, ...])
        sigma = np.sqrt(np.sum((image - mu) ** 2 * soft_labels[l, ...]) / np.sum(soft_labels[l, ...])) # np.std(image * soft_labels[l, ...]) / np.mean(soft_labels[l, ...])
        print(f"\tclass: {l}\tmean: {mu}")
        print(f"\tclass: {l}\tstd: {sigma}")
        P_l = soft_labels[l, ...] # np.sum(labels == l) / np.shape(labels.flatten())
        W_l = P_l * norm.pdf(image, loc=mu, scale=sigma) #+ np.where(labels == l, 20, 0)
        
        W.append(W_l)
    
    W = np.array(W)
    W = W / np.sum(W, axis=0) # (cls, H, W, D)
    new_label = np.argmax(W, axis=0)
    return new_label

# MRF Segmentation Algorithm
def mrf_segmentation(image_pad_array: np.ndarray,
                     label_pad_array: np.ndarray,
                     mask_pad_array: np.ndarray,
                     bias_max_iter: int = 10,
                     max_kernel: float = 9, min_kernel: float = 3,
                     refine_max_iter: int = 2,
                     label_soft_kernel: float = 0.4,
                     init_dist: list = None):
    
    # label preprocess (slightly refinement)
    print("Starting refine label ...")
    fake_bias = estimate_bias_field(image_pad_array, label_pad_array, mask_pad_array, init_mean=None, kernel_size=max_kernel, soft_kernel=label_soft_kernel)
    for iter in range(refine_max_iter):
        label_pad_array = refine_label(image_pad_array - fake_bias, label_pad_array, soft_kernel=1.5)
    
    # compute initial distribution (then always keep initial means of all classes)
    if init_dist == None:
        cls_mean, _ = compute_class_statistics(image_pad_array, label_pad_array, soft_kernel=label_soft_kernel)
    else:
        cls_mean = init_dist
    
    # real bfc process
    bias_array = np.zeros_like(image_pad_array)
    for iter in range(bias_max_iter):
        curr_array = image_pad_array - bias_array
        print(f"iteration: {iter} processing...")
        kernel = kernel_scheduler(iter, bias_max_iter, max_kernel=max_kernel, min_kernel=min_kernel)
        print(f"\tcurrent bias field smooth kernel size: {kernel}")
        high_freq_bias = estimate_bias_field(curr_array, label_pad_array, mask_pad_array, cls_mean, kernel_size=kernel, soft_kernel=label_soft_kernel)
        bias_array += high_freq_bias
    
    return bias_array

def kernel_scheduler(curr_epoch: int, total_epoch: int, max_kernel: int = 21, min_kernel: int = 3, method: str = 'exp'):
    if method == 'exp':
        A = max_kernel  # y(0) = 25
        B = np.log(min_kernel / max_kernel) / total_epoch
        y = A * np.exp(B * curr_epoch)
        return y
    elif method == 'linear':
        y = max_kernel - curr_epoch / total_epoch * (max_kernel - min_kernel)
        return y
    else:
        raise ValueError(f"method {method} is not valid!")

def add_noise_to_background(image, bg_msk, mu=1, sigma=1e-1):
    noisy_image = image.copy()
    noisy_image += np.random.normal(mu, sigma, size=image.shape)
    noisy_image = np.clip(noisy_image, a_min=0, a_max=None)
    noisy_image = np.where(bg_msk > 0, noisy_image, image)
    return noisy_image

def fill_outside_by_nearest(image, mask, width: int = 9):
    mask = np.where(mask > 0, 1, 0)
    distances, indices = distance_transform_edt(1 - mask, return_indices=True)
    
    filled_image = image.copy()

    struct = generate_binary_structure(rank=3, connectivity=1)
    dil_mask = binary_dilation(mask, structure=struct, iterations=width)
    fill_region = dil_mask - mask
    
    outside_indices = np.where(fill_region > 0)
    
    nearest_indices = indices[:, outside_indices[0], outside_indices[1], outside_indices[2]]
    filled_image[outside_indices[0], outside_indices[1], outside_indices[2]] = image[2 * nearest_indices[0] - outside_indices[0], 2 * nearest_indices[1] - outside_indices[1], 2 * nearest_indices[2] - outside_indices[2]]

    return filled_image, fill_region

def image_preprocess(image, label):
    # mask image through label
    image = np.where(label > 0, image, 0)
    
    # normalize image intensities
    min_scale = np.min(image)
    max_scale = np.max(image)
    
    norm = (image - min_scale) / (max_scale - min_scale)
    
    # obtain mean value of csf
    bg_mask = np.where(norm < 1e-3, 1, 0)
    norm = np.clip(norm, 1e-3, 1)
    
    # add noise to background
    noisy_image = add_noise_to_background(norm, bg_mask, mu=1e-3, sigma=1e-8)
    return noisy_image

# Main function
def main(args):
    # path to input image file
    input_path = args.input_img
    label_path = args.input_lab
    
    # path to output image file
    corrected_output_path = args.output_img
    bias_output_path = args.output_bias

    # load MRI image
    reference_image, image = load_mri_image(input_path)
    _, label = load_mri_image(label_path)
    
    # load parameters
    bias_iters = args.bias_max_iter
    max_kernel = args.max_kernel
    min_kernel = args.min_kernel
    refine_iters = args.refine_max_iter
    label_soft_kernel = args.label_soft_kernel
    
    # global variables
    global modality
    modality = args.modality
    global base
    base = 10
    global pad_size
    pad_size = max_kernel
    
    # process mask to 0-1
    mask = np.where(label > 0, 1, 0)
    
    # padding
    image_pad = np.pad(image, pad_width=pad_size)
    label_pad = np.pad(label, pad_width=pad_size)
    
    # preprocess
    image_pad = image_preprocess(image_pad, label_pad)
    mask_pad = np.where(label_pad > 0, 1, 0)
    
    # image to log
    image_pad = np.log(np.clip(image_pad, a_min=1e-3, a_max=1.0)) / np.log(base)

    # Perform segmentation
    if modality == 'T1':
        cls_mean = [np.log(intensity) / np.log(base) for intensity in [1e-3, 0.15, 0.55, 0.95]]
    elif modality == 'T2':
        cls_mean = [np.log(intensity) / np.log(base) for intensity in [1e-3, 0.95, 0.35, 0.15]]
    else:
        raise ValueError(f"Invalid modality {modality} indicate!")
    bias = mrf_segmentation(image_pad, label_pad, mask_pad, bias_max_iter=bias_iters, max_kernel=max_kernel, min_kernel=min_kernel, refine_max_iter=refine_iters, label_soft_kernel=label_soft_kernel, init_dist=cls_mean)
    
    bias, _ = fill_outside_by_nearest(bias, mask_pad, width=7)
    bias = bias[pad_size: -pad_size, pad_size: -pad_size, pad_size: -pad_size]
    
    # Save the bias field image
    bias_array = np.power(base, -bias)
    bias_image = sitk.GetImageFromArray(bias_array.astype(np.float32))
    bias_image.CopyInformation(reference_image)
    sitk.WriteImage(bias_image, bias_output_path)
    print(f"Bias image saved to {bias_output_path}")
    
    # save corrected image
    max_scale = np.max(image)
    min_scale = np.min(image)
    norm = (image - min_scale) / (max_scale - min_scale)
    corrected_array = np.power(base, np.log(np.clip(norm, a_min=1e-8, a_max=1.0)) / np.log(base) - bias)
    corrected_array = corrected_array * (max_scale - min_scale) + min_scale
    corrected_image = sitk.GetImageFromArray(corrected_array.astype(np.float32))
    corrected_image.CopyInformation(reference_image)
    sitk.WriteImage(corrected_image, corrected_output_path)
    print(f"Corrected image saved to {corrected_output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_img", type=str, default=None, help="input image")
    parser.add_argument("--input_lab", type=str, default=None, help="initial label (e.g. nbest segmentation)")
    parser.add_argument("--input_msk", type=str, default=None, help="input mask")
    
    parser.add_argument("--bias_max_iter", type=int, default=10, help="bias estimate max iterations")
    parser.add_argument("--max_kernel", type=int, default=9, help="max bias smooth kernel size")
    parser.add_argument("--min_kernel", type=int, default=3, help="min bias smooth kernel size")
    parser.add_argument("--refine_max_iter", type=int, default=2, help="refine label max iterations")
    parser.add_argument("--label_soft_kernel", type=float, default=0.8, help="label smooth kernel size")
    
    
    parser.add_argument("--output_img", type=str, default=None, help="corrected image")
    parser.add_argument("--output_bias", type=str, default=None, help="output bias")
    parser.add_argument("--modality", type=str, default='T1', help="T1 or T2")
    args = parser.parse_args()
    
    main(args=args)
