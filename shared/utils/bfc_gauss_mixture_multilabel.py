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
    """
    H, W, D = tissue_types.shape
    smoothed_probs = np.zeros((num_classes, H, W, D))

    for class_idx in range(num_classes):
        class_mask = (tissue_types == class_idx).astype(float)
        smoothed = gaussian_filter(class_mask, sigma=sigma, mode='constant', cval=0)
        smoothed_probs[class_idx] = smoothed

    total_prob = np.sum(smoothed_probs, axis=0, keepdims=True) + 1e-10
    smoothed_probs /= total_prob

    return smoothed_probs

def smooth_bias_field(bias: np.ndarray, mask: np.ndarray, base: int = 10, kernel_size: tuple = (3, 3, 3)) -> np.ndarray:
    """
    Smooth log space bias field in original space.
    """
    orig_bias = bias

    mask = np.where(mask > 0, 1, 0)
    struct = generate_binary_structure(rank=3, connectivity=1)
    width = 1
    inner = binary_erosion(binary_dilation(mask, struct, iterations=1), struct, iterations=1 + width)
    orig_bias, _ = fill_outside_by_nearest(orig_bias, inner, width=tuple(width + 3 * int(x) for x in kernel_size))

    orig_bias = gaussian_filter(orig_bias, sigma=kernel_size, mode='nearest')

    return orig_bias

def estimate_bias_field(image, labels, init_mean, kernel_size=(3, 3, 3), soft_kernel=0.4, consider=None):
    """
    estimate bias field using specified labels (or all non-background labels by default)
    """
    cls_num = np.unique(labels).shape[0]
    soft_labels = smooth_tissue_probabilities(labels, num_classes=cls_num, sigma=soft_kernel)

    # Use specified labels, or all non-background labels for bias estimation
    if consider is None:
        consider = list(range(1, cls_num))

    W = []
    for l in range(cls_num):
        mu = np.sum(image * soft_labels[l, ...]) / np.sum(soft_labels[l, ...])
        sigma = np.sqrt(np.sum((image - mu) ** 2 * soft_labels[l, ...]) / np.sum(soft_labels[l, ...]))
        print(f"\tclass: {l}\tmean: {mu}")
        print(f"\tclass: {l}\tstd: {sigma}")
        P_l = soft_labels[l, ...]
        W_l = P_l * norm.pdf(image, loc=mu, scale=sigma)

        W.append(W_l)

    W = np.array(W)
    W = W / np.sum(W, axis=0)

    # compute residual
    R = np.zeros_like(image)
    phi = np.zeros_like(image)

    for l in consider:
        if init_mean and l < len(init_mean):
            mean = init_mean[l]
        else:
            mean = np.mean(image * soft_labels[l, ...]) / np.mean(soft_labels[l, ...])

        std = np.std(image * soft_labels[l, ...]) / np.std(soft_labels[l, ...])
        R += W[l, ...] * (image - mean) / std
        phi += W[l, ...] / std

    # smooth within the mask area
    residual = np.where(phi > 0, R / phi, 0)
    mask = np.zeros_like(labels)
    for i in consider:
        if i == 0:
            continue
        mask = np.logical_or(mask, np.where(labels == i, 1, 0))
    mask = mask.astype(np.uint8)

    residual = np.clip(residual, a_min=-0.20, a_max=0.20)
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
        sigma = np.sqrt(np.sum((image - mu) ** 2 * soft_labels[l, ...]) / np.sum(soft_labels[l, ...]))
        print(f"\tclass: {l}\tmean: {mu}")
        print(f"\tclass: {l}\tstd: {sigma}")
        P_l = soft_labels[l, ...]
        W_l = P_l * norm.pdf(image, loc=mu, scale=sigma)

        W.append(W_l)

    W = np.array(W)
    W = W / np.sum(W, axis=0)
    new_label = np.argmax(W, axis=0)
    return new_label

# MRF Segmentation Algorithm
def mrf_segmentation(image_pad_array: np.ndarray,
                     label_pad_array: np.ndarray,
                     bias_max_iter: int = 10,
                     max_kernel: float = 9, min_kernel: float = 3,
                     refine_max_iter: int = 2,
                     label_soft_kernel: float = 0.4,
                     init_dist: list = None,
                     low_freq_kernel: tuple = None):

    if refine_max_iter > 0:
        print("Starting refine label ...")
        fake_bias = estimate_bias_field(image_pad_array, label_pad_array, init_mean=None, kernel_size=max_kernel, soft_kernel=1.2)
        for iter in range(refine_max_iter):
            label_pad_array = refine_label(image_pad_array - fake_bias, label_pad_array, soft_kernel=label_soft_kernel)

    # compute initial distribution from data (always compute from data for generic labels)
    if init_dist is None:
        cls_mean, _ = compute_class_statistics(image_pad_array, label_pad_array, soft_kernel=label_soft_kernel)
    else:
        cls_mean = init_dist

    # real bfc process - two stage:
    # Stage 1: low-frequency bias using top-2 most abundant tissue classes (most robust statistics)
    # Stage 2: high-frequency residual bias using all tissue classes
    bias_array = np.zeros_like(image_pad_array)

    # Find top 2 most abundant non-background classes
    unique_labels, counts = np.unique(label_pad_array, return_counts=True)
    non_bg = [(int(l), int(c)) for l, c in zip(unique_labels, counts) if l != 0]
    non_bg.sort(key=lambda x: x[1], reverse=True)
    top2_classes = [x[0] for x in non_bg[:2]]
    print(f"Top 2 classes by voxel count: {top2_classes} (counts: {[x[1] for x in non_bg[:2]]})")
    print(f"All non-bg classes: {[x[0] for x in non_bg]}")

    # Stage 1: single pass with fixed kernel, Stage 2: remaining iterations with dynamic kernel
    low_freq_iters = 1

    # Stage 1: Low-frequency bias using top 2 classes only, fixed 6mm Gaussian kernel
    if low_freq_kernel is None:
        low_freq_kernel = max_kernel
    print(f"Stage 1: Low-frequency BFC with top 2 classes ({low_freq_iters} iters), kernel={low_freq_kernel}")
    for iter in range(low_freq_iters):
        curr_array = image_pad_array - bias_array
        print(f"iteration: {iter} processing...")
        low_freq_bias = estimate_bias_field(curr_array, label_pad_array, cls_mean,
                                            kernel_size=low_freq_kernel, soft_kernel=label_soft_kernel,
                                            consider=top2_classes)
        bias_array += low_freq_bias

    # Stage 2: High-frequency residual bias using all tissue classes
    print(f"Stage 2: High-frequency BFC with all classes ({bias_max_iter - low_freq_iters} iters)")
    for iter in range(low_freq_iters, bias_max_iter):
        curr_array = image_pad_array - bias_array
        print(f"iteration: {iter} processing...")
        kernel = kernel_scheduler(iter, bias_max_iter, max_kernel=max_kernel, min_kernel=min_kernel)
        print(f"\tcurrent bias field smooth kernel size: {kernel}")
        high_freq_bias = estimate_bias_field(curr_array, label_pad_array, cls_mean,
                                             kernel_size=kernel, soft_kernel=label_soft_kernel,
                                             consider=None)
        bias_array += high_freq_bias

    return bias_array

def kernel_scheduler(curr_epoch: int,
                     total_epoch: int,
                     max_kernel: tuple = (21, 21, 21),
                     min_kernel: tuple = (3, 3, 3),
                     method: str = 'exp'):
    max_kernel = np.array(max_kernel, dtype=float)
    min_kernel = np.array(min_kernel, dtype=float)

    if method == 'exp':
        A = max_kernel
        B = np.log(min_kernel / max_kernel) / total_epoch
        y = A * np.exp(B * curr_epoch)
    elif method == 'linear':
        y = max_kernel - curr_epoch / total_epoch * (max_kernel - min_kernel)
    else:
        raise ValueError(f"method {method} is not valid!")
    return tuple(y)

def add_noise_to_background(image, bg_msk, mu=1, sigma=1e-1):
    noisy_image = image.copy()
    noisy_image += np.random.normal(mu, sigma, size=image.shape)
    noisy_image = np.clip(noisy_image, a_min=0, a_max=None)
    noisy_image = np.where(bg_msk > 0, noisy_image, image)
    return noisy_image

def fill_outside_by_nearest(image, mask, width: tuple = (9, 9, 9)):
    mask = np.where(mask > 0, 1, 0)
    distances, indices = distance_transform_edt(1 - mask, return_indices=True)

    filled_image = image.copy()

    struct = generate_binary_structure(rank=3, connectivity=1)
    dil_mask = binary_dilation(mask, structure=struct, iterations=max(width))
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

def compute_sigma(resolution_mm, base_resolution=0.4, base_sigma=3.2):
    resolution_mm = np.array(resolution_mm, dtype=float)
    target_fwhm = base_sigma * base_resolution
    sigma = target_fwhm / resolution_mm
    return tuple(sigma)

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
    _, label_raw = load_mri_image(label_path)

    # Remap non-consecutive labels to 0..N-1 (script assumes consecutive labels)
    unique_labels = np.unique(label_raw)
    label = np.zeros_like(label_raw, dtype=np.int32)
    for new_idx, old_val in enumerate(unique_labels):
        label[label_raw == old_val] = new_idx
    print(f"Label remapping: {dict(zip(range(len(unique_labels)), unique_labels))}")

    # Derive mask from label (label > 0) instead of requiring a separate mask
    if args.input_msk:
        _, region = load_mri_image(args.input_msk)
    else:
        region = np.where(label > 0, 1, 0).astype(np.uint8)
        print("No mask provided, using label>0 as mask.")

    # load parameters
    bias_iters = args.bias_max_iter
    max_kernel = (args.max_kernel,) * 3
    min_kernel = (args.min_kernel,) * 3 if args.min_kernel is not None else compute_sigma(reference_image.GetSpacing())
    refine_iters = args.refine_max_iter
    label_soft_kernel = args.label_soft_kernel

    # global variables
    global modality
    modality = args.modality
    global base
    base = 10
    global pad_size
    pad_size = int(args.max_kernel + 0.5)

    # padding
    image_pad = np.pad(image, pad_width=pad_size)
    label_pad = np.pad(label, pad_width=pad_size)
    region_pad = np.pad(region, pad_width=pad_size)

    # preprocess
    image_pad = image_preprocess(image_pad, label_pad)

    # image to log
    image_pad = np.log(np.clip(image_pad, a_min=1e-3, a_max=1.0)) / np.log(base)

    # Always compute cls_mean from data for generic multi-label input
    cls_mean = None
    print(f"Using data-driven class means (auto-computed from {np.unique(label).shape[0]} labels)")

    # Compute 6mm fixed kernel for Stage 1 low-frequency BFC
    low_freq_kernel = compute_sigma(reference_image.GetSpacing(), base_resolution=6.0, base_sigma=1.0)
    print(f"Stage 1 fixed low-freq kernel (6mm Gaussian): {low_freq_kernel}")

    bias = mrf_segmentation(image_pad, label_pad, bias_max_iter=bias_iters, max_kernel=max_kernel, min_kernel=min_kernel, refine_max_iter=refine_iters, label_soft_kernel=label_soft_kernel, init_dist=cls_mean, low_freq_kernel=low_freq_kernel)

    region_pad = np.where(region_pad > 0, 1, 0)
    bias, _ = fill_outside_by_nearest(bias, region_pad, width=(7,) * 3)
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
    corrected_array = np.where(region > 0, corrected_array, 0)
    corrected_image = sitk.GetImageFromArray(corrected_array.astype(np.float32))
    corrected_image.CopyInformation(reference_image)
    sitk.WriteImage(corrected_image, corrected_output_path)
    print(f"Corrected image saved to {corrected_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_img", type=str, default=None, help="input image")
    parser.add_argument("--input_lab", type=str, default=None, help="initial label (multi-label segmentation)")
    parser.add_argument("--input_msk", type=str, default=None, help="only calculate bias in this input mask (optional, derived from label if not given)")

    parser.add_argument("--bias_max_iter", type=int, default=10, help="bias estimate max iterations")
    parser.add_argument("--max_kernel", type=float, default=9, help="max bias smooth kernel size")
    parser.add_argument("--min_kernel", type=float, default=None, help="min bias smooth kernel size")
    parser.add_argument("--refine_max_iter", type=int, default=2, help="refine label max iterations")
    parser.add_argument("--label_soft_kernel", type=float, default=0.4, help="label smooth kernel size")

    parser.add_argument("--output_img", type=str, default=None, help="corrected image")
    parser.add_argument("--output_bias", type=str, default=None, help="output bias")
    parser.add_argument("--modality", type=str, default='None', help="ignored in multi-label mode")
    args = parser.parse_args()

    main(args=args)
