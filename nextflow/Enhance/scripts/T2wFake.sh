#!/bin/bash
set -e
set -x

###### This script generates fake T2w images when real T2w is not available
###### It creates synthetic T2w images from T1w data

# Help message
usage() {
echo "
Usage: $0 --t1w_final_corrected <t1w_final_corrected> --t1w_pial <t1w_pial> --t1w_vessel <t1w_vessel> --enhance_dir <enhance_dir> --contain_t2 <contain_t2>

Required arguments:
--t1w_final_corrected  T1w final corrected image
--t1w_pial             T1w pial image
--t1w_vessel           T1w vessel corrected image
--enhance_dir          Enhance directory
--contain_t2           Contains T2 flag (True/False)
"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --t1w_final_corrected)
      t1w_final_corrected="$2"
      shift 2
      ;;
    --t1w_pial)
      t1w_pial="$2"
      shift 2
      ;;
    --t1w_vessel)
      t1w_vessel="$2"
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
if [[ -z "$t1w_final_corrected" || -z "$t1w_pial" || -z "$t1w_vessel" || -z "$enhance_dir" || -z "$contain_t2" ]]; then
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
brainmask="${t1w_path}/${prefix}_desc-conform_mask.nii.gz"

echo "Checking T2w availability..."
cd ${t1w_path}

if [[ ${contain_t2} == "False" ]]; then
    echo "Generating fake T2w images from T1w data..."
    
    # Calculate max intensity for normalization
    t1w_max=$(fslstats ${t1w_final_corrected} -k ${brainmask} -R | awk '{print $2}')
    t1w_pial_max=$(fslstats ${t1w_pial} -k ${brainmask} -R | awk '{print $2}')
    t1w_vessel_max=$(fslstats ${t1w_vessel} -k ${brainmask} -R | awk '{print $2}')
    
    # Generate fake T2w from final corrected T1w
    fslmaths ${t1w_final_corrected} -sub ${t1w_max} -abs ${prefix}_desc-bfc_T2w.nii.gz
    fslmaths ${prefix}_desc-bfc_T2w.nii.gz -mas ${brainmask} ${prefix}_desc-bfc_T2w.nii.gz

    # Generate fake T2w from pial T1w
    fslmaths ${t1w_pial} -sub ${t1w_pial_max} -abs ${prefix}_desc-pialbfc_T2w.nii.gz
    fslmaths ${prefix}_desc-pialbfc_T2w.nii.gz -mas ${brainmask} ${prefix}_desc-pialbfc_T2w.nii.gz

    # Generate fake T2w from vessel corrected T1w
    fslmaths ${t1w_vessel} -sub ${t1w_vessel_max} -abs ${prefix}_desc-bfc_label-vessel_T2w.nii.gz
    fslmaths ${prefix}_desc-bfc_label-vessel_T2w.nii.gz -mas ${brainmask} ${prefix}_desc-bfc_label-vessel_T2w.nii.gz

    echo "Fake T2w images generated successfully"
else
    echo "Real T2w images available, skipping fake T2w generation"
    # Ensure output files exist for Nextflow pipeline
    if [[ ! -f "${t1w_path}/${prefix}_desc-bfc_T2w.nii.gz" ]]; then
        echo "Warning: T2w output files not found but contain_t2=True"
    fi
fi

echo "Fake T2 generation step completed"