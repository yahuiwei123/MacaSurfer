#!/bin/bash
set -e
set -x

###### This script performs anatomical image conformation to standard space
###### It conforms brain, head, and brainmask images to the same space and reorients to LIA

# Help message
usage() {
echo "
Usage: $0 --modality <modality> --subj <subject> --ses <session> --prepare_dir <prepare_dir> --enhance_dir <enhance_dir> --python_inter <python_inter> --utils_path <utils_path>

Required arguments:
--modality          Modality (T1, T2, etc.)
--subj              Subject ID
--ses               Session ID
--prepare_dir       Prepare directory containing input images
--enhance_dir       Enhance directory for output
--python_inter      Python interpreter path
--utils_path        Utils scripts path
"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --modality)
      modality="$2"
      shift 2
      ;;
    --subj)
      subj="$2"
      shift 2
      ;;
    --ses)
      ses="$2"
      shift 2
      ;;
    --prepare_dir)
      prepare_dir="$2"
      shift 2
      ;;
    --enhance_dir)
      enhance_dir="$2"
      shift 2
      ;;
    --python_inter)
      python_inter="$2"
      shift 2
      ;;
    --utils_path)
      utils_path="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

# Validate required arguments
if [[ -z "$modality" || -z "$subj" || -z "$ses" || -z "$prepare_dir" || -z "$enhance_dir" || -z "$python_inter" || -z "$utils_path" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Set modal path
# Derive BIDS subject/session prefix from .../<sub>/<ses>/Enhance
subj_bids=$(basename "$(dirname "$(dirname "$enhance_dir")")")
ses_bids=$(basename "$(dirname "$enhance_dir")")
if [[ "${subj_bids}" != sub-* ]]; then
    subj_bids="sub-${subj_bids}"
fi
if [[ "${ses_bids}" != ses-* ]]; then
    ses_bids="ses-${ses_bids}"
fi
prefix="${subj_bids}_${ses_bids}"

modal_path="${enhance_dir}/${modality}w"
echo "Processing modality: $modality"
echo "Input directory: $prepare_dir"
echo "Output directory: $modal_path"

# Construct BIDS suffix and filenames for prepare_dir inputs
if [ "${modality}" == "FLAIR" ]; then
    bids_suffix="FLAIR"
else
    bids_suffix="${modality}w"
fi
subj_bids="${subj}"
ses_bids="${ses}"
if [[ "${subj_bids}" != sub-* ]]; then
    subj_bids="sub-${subj_bids}"
fi
if [[ "${ses_bids}" != ses-* ]]; then
    ses_bids="ses-${ses_bids}"
fi

brain_file="${prepare_dir}/${subj_bids}_${ses_bids}_desc-brain_${bids_suffix}.nii.gz"
head_file="${prepare_dir}/${subj_bids}_${ses_bids}_desc-head_${bids_suffix}.nii.gz"
mask_file="${prepare_dir}/${subj_bids}_${ses_bids}_desc-brain_mask_${bids_suffix}.nii.gz"
if [[ ! -f "${mask_file}" && "${modality}" == "T1" ]]; then
    mask_file="${prepare_dir}/${subj_bids}_${ses_bids}_desc-brain_mask.nii.gz"
fi

# Create output directory
mkdir -p ${modal_path}

# Conform brain image
echo "Conforming brain image..."
${python_inter} ${utils_path}/conform.py \
    --input ${brain_file} \
    --output ${modal_path}/${prefix}_desc-conform_${modality}w.nii.gz \
    --crop_as ${mask_file}

# Conform head image
echo "Conforming head image..."
${python_inter} ${utils_path}/conform.py \
    --input ${head_file} \
    --output ${modal_path}/${prefix}_desc-conform_head.nii.gz \
    --crop_as ${mask_file}

# Conform brainmask image
echo "Conforming brainmask image..."
${python_inter} ${utils_path}/conform.py \
    --input ${mask_file} \
    --output ${modal_path}/${prefix}_desc-conform_mask.nii.gz \
    --crop_as ${mask_file}

# Check voxel resolution and resample to 0.5mm if coarser
echo "Checking voxel resolution..."
for img_type in ${modality}w head mask; do
    if [ "${img_type}" = "${modality}w" ]; then
        img_file="${modal_path}/${prefix}_desc-conform_${modality}w.nii.gz"
    else
        img_file="${modal_path}/${prefix}_desc-conform_${img_type}.nii.gz"
    fi
    ${python_inter} -c "
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

img = nib.load('${img_file}')
data = img.get_fdata()
affine = img.affine

# Get voxel size from affine column norms
voxel_size = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
print(f'  ${img_type}: shape {data.shape[:3]}, voxel {voxel_size[0]:.2f}x{voxel_size[1]:.2f}x{voxel_size[2]:.2f}mm')

target_res = 0.4
over_limit = voxel_size > target_res + 0.01
if np.any(over_limit):
    resampled_axes = np.where(over_limit)[0]
    print(f'  Resampling ${img_type} axes {resampled_axes} to ~{target_res}mm (nn)...')
    old_shape = np.array(data.shape[:3])
    zoom_factors = np.where(over_limit, voxel_size / target_res, 1.0)
    new_shape = np.round(old_shape * zoom_factors).astype(int)
    zoom_factors = new_shape / old_shape

    if data.ndim == 4:
        resampled = np.zeros((*new_shape, data.shape[3]), dtype=data.dtype)
        for t in range(data.shape[3]):
            resampled[..., t] = zoom(data[..., t], zoom_factors, order=0)
    else:
        resampled = zoom(data, zoom_factors, order=0, prefilter=False)

    # Adjust affine: voxel columns scaled by 1/zoom, origin unchanged
    new_affine = affine.copy()
    new_affine[:3, 0] = affine[:3, 0] / zoom_factors[0]
    new_affine[:3, 1] = affine[:3, 1] / zoom_factors[1]
    new_affine[:3, 2] = affine[:3, 2] / zoom_factors[2]

    new_img = nib.Nifti1Image(resampled.astype(data.dtype), new_affine, header=img.header)
    nib.save(new_img, '${img_file}')
    new_voxel = np.sqrt(np.sum(new_affine[:3, :3] ** 2, axis=0))
    print(f'  Done: {data.shape[:3]} -> {resampled.shape[:3]}, voxel now {new_voxel[0]:.2f}x{new_voxel[1]:.2f}x{new_voxel[2]:.2f}mm')
else:
    print(f'  ${img_type}: resolution OK, skipping')
"
done

# Reorient all images to LIA orientation
echo "Reorienting images to LIA orientation..."

${python_inter} ${utils_path}/conform.py \
    --input ${modal_path}/${prefix}_desc-conform_${modality}w.nii.gz \
    --output ${modal_path}/${prefix}_desc-conform_${modality}w.nii.gz \
    --reorient LIA

${python_inter} ${utils_path}/conform.py \
    --input ${modal_path}/${prefix}_desc-conform_head.nii.gz \
    --output ${modal_path}/${prefix}_desc-conform_head.nii.gz \
    --reorient LIA

${python_inter} ${utils_path}/conform.py \
    --input ${modal_path}/${prefix}_desc-conform_mask.nii.gz \
    --output ${modal_path}/${prefix}_desc-conform_mask.nii.gz \
    --reorient LIA

# Threshold images to remove negative values
echo "Thresholding images to remove negative values..."
fslmaths ${modal_path}/${prefix}_desc-conform_${modality}w.nii.gz -thr 0 ${modal_path}/${prefix}_desc-conform_${modality}w.nii.gz
fslmaths ${modal_path}/${prefix}_desc-conform_head.nii.gz -thr 0 ${modal_path}/${prefix}_desc-conform_head.nii.gz

echo "Normalizing brain image to [0 ~ 255]..."
read rmin rmax <<< $(fslstats ${modal_path}/${prefix}_desc-conform_${modality}w.nii.gz -k ${modal_path}/${prefix}_desc-conform_mask.nii.gz -r)
range=$(awk -v a="$rmin" -v b="$rmax" 'BEGIN{print (b-a>0)?(b-a):1.0}')
fslmaths ${modal_path}/${prefix}_desc-conform_${modality}w.nii.gz -sub ${rmin} -thr 0.0 -div ${range} -mul 255 -mas ${modal_path}/${prefix}_desc-conform_mask.nii.gz ${modal_path}/${prefix}_desc-conform_${modality}w.nii.gz

echo "Anatomical conformation completed successfully for $modality"