#!/bin/bash
set -e
set -x

###### This script detects blood vessels and creates vessel masks
###### It identifies vessels and applies correction to images

# Help message
usage() {
echo "
Usage: $0 --t1w_cerebellum_brainstem <t1w_cerebellum_brainstem> --t1w_aseg <t1w_aseg> --t1w_nbest <t1w_nbest> --t1w_final_corrected <t1w_final_corrected> --t2w_final_corrected <t2w_final_corrected> --enhance_dir <enhance_dir> --contain_t2 <contain_t2> --python_inter <python_inter> --utils_path <utils_path> --vessel_detect <vessel_detect>

Required arguments:
--t1w_cerebellum_brainstem  T1w cerebellum brainstem mask
--t1w_aseg                  T1w aseg segmentation
--t1w_nbest                 T1w nBEST segmentation
--t1w_final_corrected       T1w final corrected image
--t2w_final_corrected       T2w final corrected image
--enhance_dir               Enhance directory
--contain_t2                Contains T2 flag (True/False)
--python_inter              Python interpreter path
--utils_path                Utils scripts path
--vessel_detect             Vessel detection flag (True/False)
"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --t1w_cerebellum_brainstem)
      t1w_cerebellum_brainstem="$2"
      shift 2
      ;;
    --t1w_aseg)
      t1w_aseg="$2"
      shift 2
      ;;
    --t1w_nbest)
      t1w_nbest="$2"
      shift 2
      ;;
    --t1w_final_corrected)
      t1w_final_corrected="$2"
      shift 2
      ;;
    --t2w_final_corrected)
      t2w_final_corrected="$2"
      shift 2
      ;;
    --enhance_dir)
      enhance_dir="$2"
      shift 2
      ;;
    --contain_t2)
      contain_t2="$2"
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
    --vessel_detect)
      vessel_detect="$2"
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
if [[ -z "$t1w_cerebellum_brainstem" || -z "$t1w_aseg" || -z "$t1w_nbest" || -z "$t1w_final_corrected" || -z "$enhance_dir" || -z "$contain_t2" || -z "$python_inter" || -z "$utils_path" || -z "$vessel_detect" ]]; then
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

echo "Starting vessel detection..."
cd ${t1w_path}

if [[ ${vessel_detect} == "true" ]]; then
    echo "Performing vessel detection..."

    # Create vessel mask based on cerebellum/brainstem difference
    fslmaths ${t1w_final_corrected} -mas ${t1w_cerebellum_brainstem} -sub ${t1w_final_corrected} -abs ${prefix}_desc-bfc_label-vessel_T1w.nii.gz

    # Create valid tissue mask
    fslmaths ${t1w_aseg} -thr 42 -uthr 42 -bin ${prefix}_T1w_valid.nii.gz
    fslmaths ${t1w_aseg} -thr 3 -uthr 3 -bin -add ${prefix}_T1w_valid.nii.gz ${prefix}_T1w_valid.nii.gz
    fslmaths ${t1w_aseg} -thr 24 -uthr 24 -bin -add ${prefix}_T1w_valid.nii.gz ${prefix}_T1w_valid.nii.gz

    # Detect vessels using vessel_mask.py
    ${python_inter} ${utils_path}/vessel_mask.py \
        --img ${prefix}_desc-bfc_label-vessel_T1w.nii.gz \
        --seg ${t1w_nbest} \
        --val ${prefix}_T1w_valid.nii.gz \
        --msk ${prefix}_T1w_vessel_mask.nii.gz \
        --sig 2.0 \
        --grad 70

    # Obtain CSF mask
    fslmaths ${t1w_nbest} -thr 1 -uthr 1 -bin ${prefix}_T1w_csf_mask.nii.gz

    # Apply vessel correction to T1w
    cp ${t1w_final_corrected} ${prefix}_desc-bfc_label-vessel_T1w.nii.gz
    fslmaths ${prefix}_desc-bfc_label-vessel_T1w.nii.gz -mas ${prefix}_T1w_vessel_mask.nii.gz -sub ${prefix}_desc-bfc_label-vessel_T1w.nii.gz -abs ${prefix}_desc-bfc_label-vessel_T1w.nii.gz
    
    # Fill vessel regions with CSF mean intensity
    csf_mean=$(fslstats ${prefix}_desc-bfc_label-vessel_T1w.nii.gz -k ${prefix}_T1w_csf_mask.nii.gz -M)
    fslmaths ${prefix}_T1w_vessel_mask.nii.gz -mul ${csf_mean} -add ${prefix}_desc-bfc_label-vessel_T1w.nii.gz ${prefix}_desc-bfc_label-vessel_T1w.nii.gz

    if [[ ${contain_t2} == "True" ]]; then
        echo "Applying vessel correction to T2w..."
        cp ${t2w_final_corrected} ${prefix}_desc-bfc_label-vessel_T2w.nii.gz
        fslmaths ${prefix}_desc-bfc_label-vessel_T2w.nii.gz -mas ${prefix}_T1w_vessel_mask.nii.gz -sub ${prefix}_desc-bfc_label-vessel_T2w.nii.gz -abs ${prefix}_desc-bfc_label-vessel_T2w.nii.gz
        
        # Fill vessel regions with CSF mean intensity
        fslmaths ${prefix}_T1w_vessel_mask.nii.gz -mul ${csf_mean} -add ${prefix}_desc-bfc_label-vessel_T2w.nii.gz ${prefix}_desc-bfc_label-vessel_T2w.nii.gz
    fi

else
    echo "Skipping vessel detection, copying original images..."
    cp ${t1w_final_corrected} ${prefix}_desc-bfc_label-vessel_T1w.nii.gz
    if [[ ${contain_t2} == "True" ]]; then
        cp ${t2w_final_corrected} ${prefix}_desc-bfc_label-vessel_T2w.nii.gz
    fi
fi

echo "Vessel detection completed successfully"