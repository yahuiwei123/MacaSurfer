#!/bin/bash
set -e
set -x

###### This script processes Original space surface generation

# Help message
usage() {
echo "
Usage: $0 --preprocess_dir <preprocess_dir> --resample_dir <freesurfer_dir> --python_inter <python_inter> --surf_reg_dir <surf_reg_dir> --template_dir <template_dir> --utils_path <utils_path> --prefix <subj_ses_prefix> 

Required arguments:
--preprocess_dir           Preprocessing directory path
--resample_dir             Resample directory path
--python_inter             Python interpreter path
--surf_reg_dir             Surface based registration toolbox path
--template_dir             MEBRAIN template directory path
--utils_path               Utilities scripts path
--prefix                   Subject/session prefix (e.g. sub-032144_ses-004)
"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --preprocess_dir)
      PreProcessDIR="$2"
      shift 2
      ;;
    --resample_dir)
      ResampleDIR="$2"
      shift 2
      ;;
    --python_inter)
      PYTHON_INTER="$2"
      shift 2
      ;;
    --surf_reg_dir)
      SURF_REG_DIR="$2"
      shift 2
      ;;
    --template_dir)
      TEMPLATE_DIR="$2"
      shift 2
      ;;
    --utils_path)
      UTILS_PATH="$2"
      shift 2
      ;;
    --prefix)
      Prefix="$2"
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
if [[ -z "$PreProcessDIR" || -z "$ResampleDIR" || -z "$PYTHON_INTER" || -z "$SURF_REG_DIR" || -z "$TEMPLATE_DIR" || -z "$UTILS_PATH" || -z "$Prefix" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Log message function
log_Msg() {
    echo "$(date): $*"
}

# Main execution
log_Msg "Starting surface-based registration"

InverseAtlasTransform=${PreProcessDIR}/MEBRAIN/xfms/from-MEBRAIN_to-T1w_mode-image_xfm.nii.gz
AtlasTransform=${PreProcessDIR}/MEBRAIN/xfms/from-T1w_to-MEBRAIN_mode-image_xfm.nii.gz
AtlasSpaceFolder=${ResampleDIR}/Atlas

# create parcellation on acpc ribbon
${PYTHON_INTER} ${UTILS_PATH}/surf2vol.py \
    --ribbon ${ResampleDIR}/ACPC/Volume/${Prefix}_desc-ribbon_dseg.nii.gz \
    --lh-white ${ResampleDIR}/ACPC/Native/${Prefix}_hemi-L_desc-white.surf.gii --lh-pial ${ResampleDIR}/ACPC/Native/${Prefix}_hemi-L_desc-pial.surf.gii --lh-label ${ResampleDIR}/ACPC/Native/${Prefix}_hemi-L_desc-aparc.label.gii \
    --rh-white ${ResampleDIR}/ACPC/Native/${Prefix}_hemi-R_desc-white.surf.gii --rh-pial ${ResampleDIR}/ACPC/Native/${Prefix}_hemi-R_desc-pial.surf.gii --rh-label ${ResampleDIR}/ACPC/Native/${Prefix}_hemi-R_desc-aparc.label.gii \
    --n-steps 30 --num-threads 16 --fill-iters 5 --out ${ResampleDIR}/ACPC/Volume/${Prefix}_desc-corticalaparc_dseg.nii.gz

# create parcellation on original ribbon
${PYTHON_INTER} ${UTILS_PATH}/surf2vol.py \
    --ribbon ${ResampleDIR}/Original/Volume/${Prefix}_desc-ribbon_dseg.nii.gz \
    --lh-white ${ResampleDIR}/Original/Native/${Prefix}_hemi-L_desc-white.surf.gii --lh-pial ${ResampleDIR}/Original/Native/${Prefix}_hemi-L_desc-pial.surf.gii --lh-label ${ResampleDIR}/Original/Native/${Prefix}_hemi-L_desc-aparc.label.gii \
    --rh-white ${ResampleDIR}/Original/Native/${Prefix}_hemi-R_desc-white.surf.gii --rh-pial ${ResampleDIR}/Original/Native/${Prefix}_hemi-R_desc-pial.surf.gii --rh-label ${ResampleDIR}/Original/Native/${Prefix}_hemi-R_desc-aparc.label.gii \
    --n-steps 30 --num-threads 16 --fill-iters 5 --out ${ResampleDIR}/Original/Volume/${Prefix}_desc-corticalaparc_dseg.nii.gz

# create parcellation on atlas ribbon for whole results validation
applywarp --rel --interp=nn -i ${ResampleDIR}/Original/Volume/${Prefix}_desc-corticalaparc_dseg.nii.gz -r ${ResampleDIR}/Atlas/Volume/${Prefix}_space-MEBRAIN_desc-brain_T1w.nii.gz -w ${PreProcessDIR}/MEBRAIN/xfms/from-T1w_to-MEBRAIN_mode-image_xfm.nii.gz -o ${ResampleDIR}/Atlas/Volume/${Prefix}_space-MEBRAIN_desc-vol_only_aparc_dseg.nii.gz
${PYTHON_INTER} ${UTILS_PATH}/conform.py --input ${ResampleDIR}/Atlas/Volume/${Prefix}_space-MEBRAIN_desc-vol_only_aparc_dseg.nii.gz --output ${ResampleDIR}/Atlas/Volume/${Prefix}_space-MEBRAIN_desc-vol_only_aparc_dseg.nii.gz --crop_as ${TEMPLATE_DIR}/mebrain_T1w_04mm_brain_LIA.nii.gz

${PYTHON_INTER} ${UTILS_PATH}/combine_gii.py \
    --surfs [${Prefix}_hemi-L_desc-white_res-32k.surf.gii,${Prefix}_hemi-R_desc-white_res-32k.surf.gii,${Prefix}_hemi-L_desc-pial_res-32k.surf.gii,${Prefix}_hemi-R_desc-pial_res-32k.surf.gii] \
    --labels [${Prefix}_hemi-L_desc-thickness_res-32k.shape.gii,${Prefix}_hemi-R_desc-thickness_res-32k.shape.gii,${Prefix}_hemi-L_desc-thickness_res-32k.shape.gii,${Prefix}_hemi-R_desc-thickness_res-32k.shape.gii] \
    --root ${ResampleDIR}/Original/fsaverage_LR32k \
    --surf_out ${ResampleDIR}/Original/fsaverage_LR32k/${Prefix}_combined.surf.gii \
    --lbl_out ${ResampleDIR}/Original/fsaverage_LR32k/${Prefix}_combined.label.gii

${PYTHON_INTER} ${UTILS_PATH}/combine_gii.py \
    --surfs [${Prefix}_hemi-L_desc-white_res-32k.surf.gii,${Prefix}_hemi-R_desc-white_res-32k.surf.gii,${Prefix}_hemi-L_desc-pial_res-32k.surf.gii,${Prefix}_hemi-R_desc-pial_res-32k.surf.gii] \
    --labels [${Prefix}_hemi-L_desc-thickness_res-32k.shape.gii,${Prefix}_hemi-R_desc-thickness_res-32k.shape.gii,${Prefix}_hemi-L_desc-thickness_res-32k.shape.gii,${Prefix}_hemi-R_desc-thickness_res-32k.shape.gii] \
    --root ${ResampleDIR}/ACPC/fsaverage_LR32k \
    --surf_out ${ResampleDIR}/ACPC/fsaverage_LR32k/${Prefix}_combined.surf.gii \
    --lbl_out ${ResampleDIR}/ACPC/fsaverage_LR32k/${Prefix}_combined.label.gii

${PYTHON_INTER} ${UTILS_PATH}/conform.py --input ${ResampleDIR}/ACPC/Volume/${Prefix}_space-acpc_desc-brain_T1w.nii.gz --output ${ResampleDIR}/Atlas/Volume/${Prefix}_space-acpc_desc-brain_T1w.nii.gz --reorient LIA
${PYTHON_INTER} ${UTILS_PATH}/conform.py --input ${ResampleDIR}/Atlas/Volume/${Prefix}_space-acpc_desc-brain_T1w.nii.gz --output ${ResampleDIR}/Atlas/Volume/${Prefix}_space-acpc_desc-brain_T1w.nii.gz --crop_as ${ResampleDIR}/Atlas/Volume/${Prefix}_space-acpc_desc-brain_T1w.nii.gz

${PYTHON_INTER} ${UTILS_PATH}/conform.py --input ${TEMPLATE_DIR}/mebrain_T1w_04mm_brain_LIA.nii.gz --output ${ResampleDIR}/Atlas/Volume/space-MEBRAIN_res-04mm_desc-brain_T1w.nii.gz --reorient LIA
${PYTHON_INTER} ${UTILS_PATH}/conform.py --input ${TEMPLATE_DIR}/mebrain_04mm_ribbon_aparc_LIA.nii.gz --output ${ResampleDIR}/Atlas/Volume/space-MEBRAIN_res-04mm_desc-ribbonaparc_dseg.nii.gz --reorient LIA
${PYTHON_INTER} ${UTILS_PATH}/conform.py --input ${ResampleDIR}/Atlas/Volume/space-MEBRAIN_res-04mm_desc-ribbonaparc_dseg.nii.gz --output ${ResampleDIR}/Atlas/Volume/space-MEBRAIN_res-04mm_desc-ribbonaparc_dseg.nii.gz --crop_as ${ResampleDIR}/Atlas/Volume/space-MEBRAIN_res-04mm_desc-brain_T1w.nii.gz
${PYTHON_INTER} ${UTILS_PATH}/conform.py --input ${ResampleDIR}/Atlas/Volume/space-MEBRAIN_res-04mm_desc-brain_T1w.nii.gz --output ${ResampleDIR}/Atlas/Volume/space-MEBRAIN_res-04mm_desc-brain_T1w.nii.gz --crop_as ${ResampleDIR}/Atlas/Volume/space-MEBRAIN_res-04mm_desc-brain_T1w.nii.gz



${PYTHON_INTER} ${SURF_REG_DIR}/register_with_surf.py \
    --src_vol ${ResampleDIR}/Atlas/Volume/${Prefix}_space-acpc_desc-brain_T1w.nii.gz --trg_vol ${ResampleDIR}/Atlas/Volume/space-MEBRAIN_res-04mm_desc-brain_T1w.nii.gz \
    --src_lbl ${ResampleDIR}/ACPC/Volume/${Prefix}_desc-corticalaparc_dseg.nii.gz --trg_lbl ${ResampleDIR}/Atlas/Volume/space-MEBRAIN_res-04mm_desc-ribbonaparc_dseg.nii.gz \
    --src_surf ${ResampleDIR}/ACPC/fsaverage_LR32k/${Prefix}_combined.surf.gii --trg_surf ${TEMPLATE_DIR}/fsaverage_LR32k/combined.surf.gii \
    --src_cort ${ResampleDIR}/ACPC/fsaverage_LR32k/${Prefix}_combined.label.gii --trg_cort ${TEMPLATE_DIR}/fsaverage_LR32k/combined.label.gii \
    --out_vol ${ResampleDIR}/Atlas/Volume/${Prefix}_space-MEBRAIN_desc-volumemoved_T1w.nii.gz \
    --out_lbl ${ResampleDIR}/Atlas/Volume/${Prefix}_space-MEBRAIN_desc-surf_aware_aparc_dseg.nii.gz \
    --out_surf ${ResampleDIR}/Atlas/Volume/${Prefix}_space-MEBRAIN_desc-surfacemoved.surf.gii \
    --out_warp ${ResampleDIR}/Atlas/Volume/${Prefix}_from-T1w_to-MEBRAIN_mode-image_xfm.nii.gz --out_inv_warp ${ResampleDIR}/Atlas/Volume/${Prefix}_from-MEBRAIN_to-T1w_mode-image_xfm.nii.gz