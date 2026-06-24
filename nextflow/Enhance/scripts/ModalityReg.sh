#!/bin/bash
set -e
set -x

###### This script performs T2w to T1w registration
###### It registers T2 images to T1 space and creates transformation matrices

# Help message
usage() {
echo "
Usage: $0 --t1w_conform <t1w_conform> --modality_list <modality_list> --enhance_dir <enhance_dir> --device <device> --script_dir <script_dir>

Required arguments:
--t1w_conform       T1w conformed image path
--modality_list     Modality list (comma separated)
--enhance_dir       Enhance directory for output
--device            Device for registration (cpu/gpu)
--script_dir        Enhancement scripts directory
"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --contain_t2)
      contain_t2="$2"
      shift 2
      ;;
    --contain_flair)
      contain_flair="$2"
      shift 2
      ;;
    --enhance_dir)
      enhance_dir="$2"
      shift 2
      ;;
    --device)
      device="$2"
      shift 2
      ;;
    --script_dir)
      script_dir="$2"
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
if [[ -z "$contain_t2" || -z "$enhance_dir" || -z "$contain_flair" || -z "$device" || -z "$script_dir" ]]; then
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
t2w_path="${enhance_dir}/T2w"
flair_path="${enhance_dir}/FLAIRw"
t1w_conform="${t1w_path}/${prefix}_desc-conform_T1w.nii.gz"
t2w_conform="${t2w_path}/${prefix}_desc-conform_T2w.nii.gz"
flair_conform="${flair_path}/${prefix}_desc-conform_FLAIR.nii.gz"
t1w_xfm_path="${t1w_path}/xfms"
t2w_xfm_path="${t2w_path}/xfms"
flair_xfm_path="${flair_path}/xfms"

# Create transformation directories
mkdir -p ${t1w_xfm_path}
mkdir -p ${t2w_xfm_path}
mkdir -p ${flair_xfm_path}

# Check if T2 modality is present
if [[ $contain_t2 == True ]]; then
    echo "T2 modality detected, performing T2w to T1w registration..."
    
    # Register T2w to T1w
    bash ${script_dir}/utils/T2wToT1wReg.sh \
        -w ${t2w_path}/T2w2T1wReg \
        -i ${t2w_conform} \
        -r ${t1w_conform} \
        -o ${t2w_xfm_path}/T2w2T1w \
        -d ${device}
    
    # Create inverse transformation matrix
    convert_xfm -omat ${t1w_xfm_path}/T1w2T2w.mat -inverse ${t2w_xfm_path}/T2w2T1w_linear.mat
    
    # Rename files
    mv ${t2w_xfm_path}/T2w2T1w_linear.mat ${t2w_xfm_path}/T2w2T1w.mat
    mv ${t2w_xfm_path}/T2w2T1w.nii.gz ${t1w_path}/${prefix}_desc-conform_T2w.nii.gz
    
    echo "T2w to T1w registration completed successfully"
else
    contain_t2="False"
    echo "No T2 modality detected, creating identity transformation matrices..."
    
    # Create identity transformation matrix
    echo "1 0 0 0" > ${t2w_xfm_path}/T2w2T1w.mat
    echo "0 1 0 0" >> ${t2w_xfm_path}/T2w2T1w.mat
    echo "0 0 1 0" >> ${t2w_xfm_path}/T2w2T1w.mat
    echo "0 0 0 1" >> ${t2w_xfm_path}/T2w2T1w.mat
    
    # Copy to T1w directory
    cp ${t2w_xfm_path}/T2w2T1w.mat ${t1w_xfm_path}/T1w2T2w.mat
fi

# Check if FLAIR modality is present
if [[ $contain_flair == True ]]; then
    echo "FLAIR modality detected, performing FLAIRw to T1w registration..."
    
    # Register FLAIR to T1w
    bash ${script_dir}/utils/T2wToT1wReg.sh \
        -w ${flair_path}/FLAIRw2T1wReg \
        -i ${flair_conform} \
        -r ${t1w_conform} \
        -o ${flair_xfm_path}/FLAIRw2T1w \
        -d ${device}
    
    # Create inverse transformation matrix
    convert_xfm -omat ${t1w_xfm_path}/T1w2FLAIRw.mat -inverse ${flair_xfm_path}/FLAIRw2T1w_linear.mat
    
    # Rename files
    mv ${flair_xfm_path}/FLAIRw2T1w_linear.mat ${flair_xfm_path}/FLAIRw2T1w.mat
    mv ${flair_xfm_path}/FLAIRw2T1w.nii.gz ${t1w_path}/${prefix}_desc-conform_FLAIR.nii.gz
    
    echo "FLAIRw to T1w registration completed successfully"
else
    contain_flair="False"
    echo "No FLAIR modality detected, creating identity transformation matrices..."
    
    # Create identity transformation matrix
    echo "1 0 0 0" > ${flair_xfm_path}/FLAIRw2T1w.mat
    echo "0 1 0 0" >> ${flair_xfm_path}/FLAIRw2T1w.mat
    echo "0 0 1 0" >> ${flair_xfm_path}/FLAIRw2T1w.mat
    echo "0 0 0 1" >> ${flair_xfm_path}/FLAIRw2T1w.mat
    
    # Copy to T1w directory
    cp ${flair_xfm_path}/FLAIRw2T1w.mat ${t1w_xfm_path}/T1w2FLAIRw.mat
fi

echo "Modality registration completed successfully"
echo "T2 present: $contain_t2"