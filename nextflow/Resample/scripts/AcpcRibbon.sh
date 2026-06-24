#!/bin/bash
set -e
set -x

###### This script processes ACPC space surface generation

# Help message
usage() {
echo "
Usage: $0 --preprocess_dir <preprocess_dir> --freesurfer_dir <freesurfer_dir> --resample_dir <freesurfer_dir> --t1w_image_0mm <t1w_image_0mm> --t1w_image_1mm <t1w_image_1mm> --caret7_dir <caret7_dir> --python_inter <python_inter> --utils_path <utils_path> --prefix <subj_ses_prefix> 

Required arguments:
--preprocess_dir           Preprocessing directory path
--freesurfer_dir           Surface directory path
--resample_dir             Resample directory path
--t1w_image_0mm            T1w original resolution
--t1w_image_1mm            T1w 1mm resolution
--caret7_dir               Connectome Workbench directory path
--python_inter             Python interpreter path
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
    --freesurfer_dir)
      FreeSurferDIR="$2"
      shift 2
      ;;
    --resample_dir)
      ResampleDIR="$2"
      shift 2
      ;;
    --t1w_image_0mm)
      T1wImage_orig_res="$2"
      shift 2
      ;;
    --t1w_image_1mm)
      T1wImage_1mm_res="$2"
      shift 2
      ;;
    --caret7_dir)
      CARET7DIR="$2"
      shift 2
      ;;
    --python_inter)
      PYTHON_INTER="$2"
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
if [[ -z "$PreProcessDIR" || -z "$FreeSurferDIR" || -z "$ResampleDIR" || -z "$T1wImage_orig_res" || -z "$T1wImage_1mm_res" || -z "$CARET7DIR" || -z "$PYTHON_INTER" || -z "$UTILS_PATH" || -z "$Prefix" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Log message function
log_Msg() {
    echo "$(date): $*"
}

# Main execution
log_Msg "Starting ACPC space processing"

mkdir -p ${ResampleDIR}/ACPC
mkdir -p ${ResampleDIR}/ACPC/Volume
mkdir -p ${ResampleDIR}/ACPC/Native

cp ${PreProcessDIR}/T1w/${Prefix}_space-acpc_desc-head_T1w.nii.gz ${ResampleDIR}/ACPC/Volume/${Prefix}_space-acpc_desc-brain_T1w.nii.gz
cp ${PreProcessDIR}/T1w/${Prefix}_space-acpc_desc-brain_T2w.nii.gz ${ResampleDIR}/ACPC/Volume/${Prefix}_space-acpc_desc-brain_T2w.nii.gz

${CARET7DIR}/wb_command -add-to-spec-file ${ResampleDIR}/ACPC/wb.spec INVALID ${ResampleDIR}/ACPC/Volume/${Prefix}_space-acpc_desc-brain_T1w.nii.gz
${CARET7DIR}/wb_command -add-to-spec-file ${ResampleDIR}/ACPC/wb.spec INVALID ${ResampleDIR}/ACPC/Volume/${Prefix}_space-acpc_desc-brain_T2w.nii.gz

for Hemisphere in L R ; do
    if [ $Hemisphere = "L" ] ; then
      hemisphere="lh"
      Structure="CORTEX_LEFT"
    elif [ $Hemisphere = "R" ] ; then
      hemisphere="rh"
      Structure="CORTEX_RIGHT"
    fi

    Types="ANATOMICAL@GRAY_WHITE ANATOMICAL@PIAL"
    i=1
    for surf_type in white pial ; do
        Type=$(echo "$Types" | cut -d " " -f $i)
        Secondary=$(echo "$Type" | cut -d "@" -f 2)
        Type=$(echo "$Type" | cut -d "@" -f 1)
        if [ ! $Secondary = $Type ] ; then
          Secondary=$(echo " -surface-secondary-type ""$Secondary")
        else
          Secondary=""
        fi

        ## rescale 1mm surface to acpc space with orig resolution
        ${PYTHON_INTER} ${UTILS_PATH}/fs_surf2ras_surf.py --volume ${T1wImage_1mm_res} --fs_surf ${FreeSurferDIR}/surf/${hemisphere}.${surf_type} --ras_surf ${ResampleDIR}/ACPC/Native/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}.surf.gii
        ${PYTHON_INTER} ${UTILS_PATH}/ras_surf_rescale.py \
        --orig_vol ${T1wImage_orig_res} \
        --scaled_vol ${T1wImage_1mm_res} \
        --in_surf ${ResampleDIR}/ACPC/Native/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}.surf.gii \
        --out_surf ${ResampleDIR}/ACPC/Native/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}.surf.gii
        ${CARET7DIR}/wb_command -set-structure ${ResampleDIR}/ACPC/Native/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}.surf.gii ${Structure} -surface-type $Type$Secondary
        ${CARET7DIR}/wb_command -add-to-spec-file ${ResampleDIR}/ACPC/wb.spec $Structure ${ResampleDIR}/ACPC/Native/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}.surf.gii

        # surface to volume
        ${CARET7DIR}/wb_command -create-signed-distance-volume ${ResampleDIR}/ACPC/Native/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}.surf.gii ${T1wImage_orig_res} ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}_dseg.nii.gz
    done
done


# create ribbon
${PYTHON_INTER} ${UTILS_PATH}/create_ribbon.py \
    ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-L_desc-white_dseg.nii.gz \
    --output ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-L_desc-ribbon_dseg.nii.gz \
    --pial_path ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-L_desc-pial_dseg.nii.gz \
    --pial_value 3 --white_value 2

${PYTHON_INTER} ${UTILS_PATH}/create_ribbon.py \
    ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-R_desc-white_dseg.nii.gz \
    --output ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-R_desc-ribbon_dseg.nii.gz \
    --pial_path ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-R_desc-pial_dseg.nii.gz \
    --pial_value 42 --white_value 41

fslmaths ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-L_desc-ribbon_dseg.nii.gz -max ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-R_desc-ribbon_dseg.nii.gz ${ResampleDIR}/ACPC/Volume/${Prefix}_desc-ribbon_dseg.nii.gz


# fslmaths ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-L_desc-pial_dseg.nii.gz -uthr 0 -abs -mul ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-L_desc-white_dseg.nii.gz -thr 0 -bin ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-L_desc-ribbon_dseg.nii.gz
# fslmaths ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-R_desc-pial_dseg.nii.gz -uthr 0 -abs -mul ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-R_desc-white_dseg.nii.gz -thr 0 -bin ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-R_desc-ribbon_dseg.nii.gz

# fslmaths ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-L_desc-white_dseg.nii.gz -uthr 0 -abs -bin -mul 2 ${ResampleDIR}/ACPC/Volume/${Prefix}_desc-ribbon_dseg.nii.gz
# fslmaths ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-R_desc-white_dseg.nii.gz -uthr 0 -abs -bin -mul 41 -add ${ResampleDIR}/ACPC/Volume/${Prefix}_desc-ribbon_dseg.nii.gz ${ResampleDIR}/ACPC/Volume/${Prefix}_desc-ribbon_dseg.nii.gz
# fslmaths ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-L_desc-ribbon_dseg.nii.gz -bin -mul 3 -max ${ResampleDIR}/ACPC/Volume/${Prefix}_desc-ribbon_dseg.nii.gz ${ResampleDIR}/ACPC/Volume/${Prefix}_desc-ribbon_dseg.nii.gz
# fslmaths ${ResampleDIR}/ACPC/Volume/${Prefix}_hemi-R_desc-ribbon_dseg.nii.gz -bin -mul 42 -max ${ResampleDIR}/ACPC/Volume/${Prefix}_desc-ribbon_dseg.nii.gz ${ResampleDIR}/ACPC/Volume/${Prefix}_desc-ribbon_dseg.nii.gz
# fslcpgeom ${T1wImage_orig_res} ${ResampleDIR}/ACPC/Volume/${Prefix}_desc-ribbon_dseg.nii.gz

# create report
mri_convert ${PreProcessDIR}/T1w/${Prefix}_space-acpc_desc-resample_T1w.nii.gz --out_orientation RAS ${ResampleDIR}/ACPC/Volume/${Prefix}_space-acpc_desc-resample_T1w.nii.gz
# ${PYTHON_INTER} ${UTILS_PATH}/qc_surface.py \
#     --input ${ResampleDIR}/ACPC/Volume/${Prefix}_space-acpc_desc-brain_T1w.nii.gz \
#     --ribbon ${ResampleDIR}/ACPC/Volume/${Prefix}_desc-ribbon_dseg.nii.gz \
#     --output ${ResampleDIR}/ACPC/qc_surface.png \
#     --single_contour