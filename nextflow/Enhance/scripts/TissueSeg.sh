#!/bin/bash
set -e
set -x

###### This script performs tissue segmentation using nBEST
###### It segments cerebellum, brainstem, cerebrum and generates tissue probability maps

# Help message
usage() {
echo "
Usage: $0 --t1w_init_corrected <t1w_init_corrected> --enhance_dir <enhance_dir> --python_inter <python_inter> --utils_path <utils_path> --nbest_model_path <nbest_model_path> --python_env <python_env>

Required arguments:
--t1w_init_corrected    T1w initial corrected image
--enhance_dir           Enhance directory
--python_inter          Python interpreter path
--utils_path            Utils scripts path
--nbest_model_path      nBEST model path
--python_env            Python environment path
"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --t1w_init_corrected)
      t1w_init_corrected="$2"
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
    --nbest_model_path)
      nbest_model_path="$2"
      shift 2
      ;;
    --python_env)
      python_env="$2"
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
if [[ -z "$t1w_init_corrected" || -z "$enhance_dir" || -z "$python_inter" || -z "$utils_path" || -z "$nbest_model_path" || -z "$python_env" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Set paths
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

t1w_path="${enhance_dir}/T1w"
t1w_xfm_path="${t1w_path}/xfms"
nbest_path="${enhance_dir}/nbest"

echo "Starting tissue segmentation..."
cd ${t1w_path}

# Clean up existing nBEST directory if present
if [ -d ${nbest_path} ]; then
    rm -r ${nbest_path}
fi

# Create identity transformation matrix
echo "1 0 0 0" > ${t1w_xfm_path}/identity.mat
echo "0 1 0 0" >> ${t1w_xfm_path}/identity.mat
echo "0 0 1 0" >> ${t1w_xfm_path}/identity.mat
echo "0 0 0 1" >> ${t1w_xfm_path}/identity.mat

# Create nBEST directory structure
mkdir -p ${nbest_path}
mkdir -p ${nbest_path}/brain_img
mkdir -p ${nbest_path}/brain_mask

# Prepare images for nBEST processing (reorient to LAS)
echo "Preparing images for nBEST processing..."
${python_inter} ${utils_path}/conform.py --input ${t1w_init_corrected} --output ${nbest_path}/brain_img/T1w_conform_0000.nii.gz --reorient LAS
${python_inter} ${utils_path}/conform.py --input ${prefix}_desc-conform_mask.nii.gz --output ${nbest_path}/brain_mask/T1w_conform.nii.gz --reorient LAS

# Run nBEST tissue segmentation
echo "Running nBEST tissue segmentation..."
${python_inter} ${nbest_model_path}/scripts/nBEST_tissue.py --python_env ${python_env} --workdir ${nbest_path}

# Reorient results back to original space (LIA)
echo "Reorienting results to original space..."
${python_inter} ${utils_path}/conform.py --input ${nbest_path}/brain_cerebellum_brainstem_mask/T1w_conform.nii.gz --output ${prefix}_label-cerebellum-brainstem_dseg.nii.gz --reorient LIA
${python_inter} ${utils_path}/conform.py --input ${nbest_path}/brain_cerebrum_mask/T1w_conform.nii.gz --output ${prefix}_label-cerebrum_dseg.nii.gz --reorient LIA
${python_inter} ${utils_path}/conform.py --input ${nbest_path}/brain_tissue/T1w_conform.nii.gz --output ${prefix}_desc-nbest_dseg.nii.gz --reorient LIA

echo "Tissue segmentation completed successfully"