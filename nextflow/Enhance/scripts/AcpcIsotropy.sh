#!/bin/bash
set -e
set -x

###### This script performs ACPC alignment and isotropic resampling
###### It aligns images to ACPC space and creates isotropic versions

# Help message
usage() {
echo "
Usage: $0 --t1w_cerebrum <t1w_cerebrum> --t1w_aseg <t1w_aseg> --t1w_nbest <t1w_nbest> --t1w_final_corrected <t1w_final_corrected> --t1w_white <t1w_white> --t1w_pial <t1w_pial> --t1w_vessel <t1w_vessel> --t2w_final_corrected <t2w_final_corrected> --t2w_pial <t2w_pial> --t2w_vessel <t2w_vessel> --wm_comp1 <wm_comp1> --wm_comp2 <wm_comp2> --gm_comp <gm_comp> --enhance_dir <enhance_dir> --contain_t2 <contain_t2> --python_inter <python_inter> --utils_path <utils_path> --t1w_template <t1w_template> --t1w_template_brain <t1w_template_brain>

Required arguments:
--t1w_cerebrum           T1w cerebrum mask
--t1w_aseg               T1w aseg segmentation
--t1w_nbest              T1w nBEST segmentation
--t1w_final_corrected    T1w final corrected image
--t1w_white              T1w white matter image
--t1w_pial               T1w pial image
--t1w_vessel             T1w vessel corrected image
--t2w_final_corrected    T2w final corrected image
--t2w_pial               T2w pial image
--t2w_vessel             T2w vessel corrected image
--wm_comp1               WM compliment 1
--wm_comp2               WM compliment 2
--gm_comp                GM compliment
--enhance_dir            Enhance directory
--contain_t2             Contains T2 flag (True/False)
--python_inter           Python interpreter path
--utils_path             Utils scripts path
--t1w_template           T1w template path
--t1w_template_brain     T1w template brain path
"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --t1w_conform)
      t1w_conform="$2"
      shift 2
      ;;
    --t1w_cerebrum)
      t1w_cerebrum="$2"
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
    --t1w_complete_aseg)
      t1w_complete_aseg="$2"
      shift 2
      ;;
    --t1w_final_corrected)
      t1w_final_corrected="$2"
      shift 2
      ;;
    --t1w_white)
      t1w_white="$2"
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
    --t2w_final_corrected)
      t2w_final_corrected="$2"
      shift 2
      ;;
    --t2w_pial)
      t2w_pial="$2"
      shift 2
      ;;
    --t2w_vessel)
      t2w_vessel="$2"
      shift 2
      ;;
    --wm_comp1)
      wm_comp1="$2"
      shift 2
      ;;
    --wm_comp2)
      wm_comp2="$2"
      shift 2
      ;;
    --gm_comp)
      gm_comp="$2"
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
    --t1w_template)
      t1w_template="$2"
      shift 2
      ;;
    --t1w_template_brain)
      t1w_template_brain="$2"
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
required_args=("t1w_cerebrum" "t1w_aseg" "t1w_nbest" "t1w_final_corrected" "t1w_white" "t1w_pial" "t1w_vessel" "enhance_dir" "contain_t2" "python_inter" "utils_path" "t1w_template" "t1w_template_brain")
for arg in "${required_args[@]}"; do
    if [ -z "${!arg}" ]; then
        echo "Error: Missing required argument $arg"
        usage
        exit 1
    fi
done

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
template_space_path="${enhance_dir}/MEBRAIN"

echo "Starting ACPC alignment and isotropic resampling..."
cd ${t1w_path}

# Extract rigid transformation from affine matrix
echo "Extracting rigid transformation..."
${python_inter} ${utils_path}/get_rigid_from_affine.py \
    --affine ${template_space_path}/xfms/orig2standardLinear.mat \
    --output_rigid xfms/acpc.mat \
    --output_scaling_shearing xfms/scaling.mat

# Register to ACPC space
tkregister2 --mov ${t1w_final_corrected} --targ ${t1w_template_brain} --fsl xfms/acpc.mat --reg xfms/acpc.dat --noedit

# Apply ACPC transformation to T1w images (header-only, no resampling)
echo "Applying ACPC transformation to T1w images (header-only)..."

apply_acpc() {
    ${python_inter} ${utils_path}/apply_affine_to_header.py --input "$1" --matrix xfms/acpc.mat --reference ${t1w_template_brain} --output "$2"
}

flirt -in ${t1w_final_corrected} -ref ${t1w_template_brain} -applyxfm -init xfms/acpc.mat -out ${prefix}_space-acpc_desc-resample_T1w.nii.gz
apply_acpc ${t1w_final_corrected} ${prefix}_space-acpc_desc-brain_T1w.nii.gz
apply_acpc ${t1w_white} ${prefix}_space-acpc_desc-whitebfc_T1w.nii.gz
apply_acpc ${t1w_pial} ${prefix}_space-acpc_desc-pialbfc_T1w.nii.gz
apply_acpc ${t1w_vessel} ${prefix}_space-acpc_desc-vessel_T1w.nii.gz
apply_acpc ${prefix}_desc-conform_mask.nii.gz ${prefix}_space-acpc_desc-brain_mask.nii.gz
apply_acpc ${prefix}_desc-conform_head.nii.gz ${prefix}_space-acpc_desc-head_T1w.nii.gz

if [[ ${contain_t2} == "True" ]]; then
    echo "Applying ACPC transformation to real T2w images (header-only)..."
else
    echo "Applying ACPC transformation to fake T2w images (header-only)..."
fi

flirt -in ${t2w_final_corrected} -ref ${t1w_template_brain} -applyxfm -init xfms/acpc.mat -out ${prefix}_space-acpc_desc-resample_T2w.nii.gz
apply_acpc ${t2w_final_corrected} ${prefix}_space-acpc_desc-brain_T2w.nii.gz
apply_acpc ${t2w_pial} ${prefix}_space-acpc_desc-pialbfc_T2w.nii.gz
apply_acpc ${t2w_vessel} ${prefix}_space-acpc_desc-vessel_T2w.nii.gz

# Transform segmentation results to ACPC space (header-only)
echo "Transforming segmentation results to ACPC space (header-only)..."
apply_acpc ${t1w_cerebrum} ${prefix}_space-acpc_label-cerebrum_dseg.nii.gz
apply_acpc ${t1w_nbest} ${prefix}_space-acpc_desc-nbest_dseg.nii.gz
apply_acpc ${t1w_aseg} ${prefix}_space-acpc_desc-aseg_dseg.nii.gz

# Transform complete aseg to ACPC space
if [[ -n "${t1w_complete_aseg:-}" && -f "${t1w_complete_aseg}" ]]; then
    apply_acpc ${t1w_complete_aseg} ${prefix}_space-acpc_desc-completeaseg_dseg.nii.gz
fi

# Transform FS aseg (from macaBrainNet) to ACPC space
t1w_fs_aseg="${t1w_path}/${prefix}_desc-freesurfer_dseg.nii.gz"
if [[ -f "${t1w_fs_aseg}" ]]; then
    apply_acpc ${t1w_fs_aseg} ${prefix}_space-acpc_desc-freesurfer_dseg.nii.gz
fi
apply_acpc ${wm_comp1} ${prefix}_space-acpc_desc-wmcomp1_mask.nii.gz
apply_acpc ${wm_comp2} ${prefix}_space-acpc_desc-wmcomp2_mask.nii.gz
apply_acpc ${gm_comp} ${prefix}_space-acpc_desc-gmcomp_mask.nii.gz


# Create isotropic versions (0.4mm)
# Use t1w_conform as reference for ALL 0.4mm resamplings to ensure
# consistent output grid dimensions regardless of ACPC affine origin.
echo "Creating isotropic versions..."
flirt -in ${t1w_final_corrected} -ref ${t1w_conform} -out ${prefix}_res-04mm_desc-brain_T1w.nii.gz -applyisoxfm 0.4 -interp trilinear
apply_acpc ${prefix}_res-04mm_desc-brain_T1w.nii.gz ${prefix}_space-acpc_res-04mm_desc-brain_T1w.nii.gz
flirt -in ${t1w_white} -ref ${t1w_conform} -out ${prefix}_res-04mm_desc-whitebfc_T1w.nii.gz -applyisoxfm 0.4 -interp trilinear
apply_acpc ${prefix}_res-04mm_desc-whitebfc_T1w.nii.gz ${prefix}_space-acpc_res-04mm_desc-whitebfc_T1w.nii.gz
flirt -in ${t1w_pial} -ref ${t1w_conform} -out ${prefix}_res-04mm_desc-pialbfc_T1w.nii.gz -applyisoxfm 0.4 -interp trilinear
apply_acpc ${prefix}_res-04mm_desc-pialbfc_T1w.nii.gz ${prefix}_space-acpc_res-04mm_desc-pialbfc_T1w.nii.gz
flirt -in ${t1w_vessel} -ref ${t1w_conform} -out ${prefix}_res-04mm_desc-vessel_T1w.nii.gz -applyisoxfm 0.4 -interp trilinear
apply_acpc ${prefix}_res-04mm_desc-vessel_T1w.nii.gz ${prefix}_space-acpc_res-04mm_desc-vessel_T1w.nii.gz

# Create isotropic brainmask (native mask → t1w_conform-ref → ACPC)
flirt -in ${prefix}_desc-conform_mask.nii.gz -ref ${t1w_conform} -out ${prefix}_res-04mm_desc-brain_mask.nii.gz -applyisoxfm 0.4 -interp nearestneighbour
apply_acpc ${prefix}_res-04mm_desc-brain_mask.nii.gz ${prefix}_space-acpc_res-04mm_desc-brain_mask.nii.gz

flirt -in ${t2w_final_corrected} -ref ${t1w_conform} -out ${prefix}_res-04mm_desc-brain_T2w.nii.gz -applyisoxfm 0.4 -interp trilinear
apply_acpc ${prefix}_res-04mm_desc-brain_T2w.nii.gz ${prefix}_space-acpc_res-04mm_desc-brain_T2w.nii.gz
flirt -in ${t2w_pial} -ref ${t1w_conform} -out ${prefix}_res-04mm_desc-pialbfc_T2w.nii.gz -applyisoxfm 0.4 -interp trilinear
apply_acpc ${prefix}_res-04mm_desc-pialbfc_T2w.nii.gz ${prefix}_space-acpc_res-04mm_desc-pialbfc_T2w.nii.gz
flirt -in ${t2w_vessel} -ref ${t1w_conform} -out ${prefix}_res-04mm_desc-vessel_T2w.nii.gz -applyisoxfm 0.4 -interp trilinear
apply_acpc ${prefix}_res-04mm_desc-vessel_T2w.nii.gz ${prefix}_space-acpc_res-04mm_desc-vessel_T2w.nii.gz

# Create isotropic segmentations (native → t1w_conform-ref → ACPC)
flirt -in ${t1w_cerebrum} -ref ${t1w_conform} -out ${prefix}_res-04mm_label-cerebrum_dseg.nii.gz -applyisoxfm 0.4 -interp nearestneighbour
apply_acpc ${prefix}_res-04mm_label-cerebrum_dseg.nii.gz ${prefix}_space-acpc_res-04mm_label-cerebrum_dseg.nii.gz

# Additional processing for high-resolution nBEST results
flirt -in ${t1w_conform} -ref ${t1w_conform} -out ${prefix}_res-04mm_desc-conform_T1w.nii.gz -applyisoxfm 0.4 -interp trilinear
tkregister2 --mov ${prefix}_res-04mm_desc-conform_T1w.nii.gz --targ ${t1w_template_brain} --fsl xfms/acpc.mat --reg xfms/acpc_04mm.dat --noedit

flirt -in ${t1w_nbest} -ref ${t1w_conform} -out ${prefix}_space-acpc_res-04mm_desc-nbest_dseg.nii.gz -applyisoxfm 0.4 -interp nearestneighbour
${python_inter} ${utils_path}/conform.py --input ${prefix}_space-acpc_res-04mm_desc-nbest_dseg.nii.gz --output ${prefix}_space-acpc_res-04mm_desc-nbest_dseg.nii.gz --reorient LIA

apply_acpc ${prefix}_space-acpc_res-04mm_desc-nbest_dseg.nii.gz ${prefix}_space-acpc_res-04mm_desc-nbest_dseg.tmp.nii.gz
mv ${prefix}_space-acpc_res-04mm_desc-nbest_dseg.tmp.nii.gz ${prefix}_space-acpc_res-04mm_desc-nbest_dseg.nii.gz

# Create isotropic FS aseg (from macaBrainNet)
if [[ -f "${t1w_fs_aseg}" ]]; then
    flirt -in ${t1w_fs_aseg} -ref ${t1w_conform} -out ${prefix}_space-acpc_res-04mm_desc-freesurfer_dseg.nii.gz -applyisoxfm 0.4 -interp nearestneighbour
    ${python_inter} ${utils_path}/conform.py --input ${prefix}_space-acpc_res-04mm_desc-freesurfer_dseg.nii.gz --output ${prefix}_space-acpc_res-04mm_desc-freesurfer_dseg.nii.gz --reorient LIA
    apply_acpc ${prefix}_space-acpc_res-04mm_desc-freesurfer_dseg.nii.gz ${prefix}_space-acpc_res-04mm_desc-freesurfer_dseg.tmp.nii.gz
    mv ${prefix}_space-acpc_res-04mm_desc-freesurfer_dseg.tmp.nii.gz ${prefix}_space-acpc_res-04mm_desc-freesurfer_dseg.nii.gz
fi

# Create isotropic complete aseg (from nbest or macaBrainNet)
if [[ -n "${t1w_complete_aseg:-}" && -f "${t1w_complete_aseg}" ]]; then
    flirt -in ${t1w_complete_aseg} -ref ${t1w_conform} -out ${prefix}_space-acpc_res-04mm_desc-completeaseg_dseg.nii.gz -applyisoxfm 0.4 -interp nearestneighbour
    ${python_inter} ${utils_path}/conform.py --input ${prefix}_space-acpc_res-04mm_desc-completeaseg_dseg.nii.gz --output ${prefix}_space-acpc_res-04mm_desc-completeaseg_dseg.nii.gz --reorient LIA
    apply_acpc ${prefix}_space-acpc_res-04mm_desc-completeaseg_dseg.nii.gz ${prefix}_space-acpc_res-04mm_desc-completeaseg_dseg.tmp.nii.gz
    mv ${prefix}_space-acpc_res-04mm_desc-completeaseg_dseg.tmp.nii.gz ${prefix}_space-acpc_res-04mm_desc-completeaseg_dseg.nii.gz
fi

# Create isotropic versions of compliment masks (native → t1w_conform-ref → ACPC)
flirt -in ${t1w_aseg} -ref ${t1w_conform} -out ${prefix}_res-04mm_desc-aseg_dseg.nii.gz -applyisoxfm 0.4 -interp nearestneighbour
apply_acpc ${prefix}_res-04mm_desc-aseg_dseg.nii.gz ${prefix}_space-acpc_res-04mm_desc-aseg_dseg.nii.gz
flirt -in ${wm_comp1} -ref ${t1w_conform} -out ${prefix}_res-04mm_desc-wmcomp1_mask.nii.gz -applyisoxfm 0.4 -interp nearestneighbour
apply_acpc ${prefix}_res-04mm_desc-wmcomp1_mask.nii.gz ${prefix}_space-acpc_res-04mm_desc-wmcomp1_mask.nii.gz
flirt -in ${wm_comp2} -ref ${t1w_conform} -out ${prefix}_res-04mm_desc-wmcomp2_mask.nii.gz -applyisoxfm 0.4 -interp nearestneighbour
apply_acpc ${prefix}_res-04mm_desc-wmcomp2_mask.nii.gz ${prefix}_space-acpc_res-04mm_desc-wmcomp2_mask.nii.gz
flirt -in ${gm_comp} -ref ${t1w_conform} -out ${prefix}_res-04mm_desc-gmcomp_mask.nii.gz -applyisoxfm 0.4 -interp nearestneighbour
apply_acpc ${prefix}_res-04mm_desc-gmcomp_mask.nii.gz ${prefix}_space-acpc_res-04mm_desc-gmcomp_mask.nii.gz

echo "ACPC alignment and isotropic resampling completed successfully"