#!/bin/bash
set -e
set -x

###### This script processes Original space surface generation

# Help message
usage() {
echo "
Usage: $0 --preprocess_dir <preprocess_dir> --freesurfer_dir <freesurfer_dir> --resample_dir <freesurfer_dir> --original_vol <original_vol> --acpc_vol <acpc_vol> --caret7_dir <caret7_dir> --python_inter <python_inter> --utils_path <utils_path> --prefix <subj_ses_prefix> 

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
    --original_vol)
      original_vol="$2"
      shift 2
      ;;
    --acpc_vol)
      acpc_vol="$2"
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
if [[ -z "$PreProcessDIR" || -z "$FreeSurferDIR" || -z "$ResampleDIR" || -z "$original_vol" || -z "$acpc_vol" || -z "$CARET7DIR" || -z "$PYTHON_INTER" || -z "$UTILS_PATH" || -z "$Prefix" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Log message function
log_Msg() {
    echo "$(date): $*"
}

# Main execution
log_Msg "Starting Original space processing"

mkdir -p ${ResampleDIR}/Original
mkdir -p ${ResampleDIR}/Original/Volume
mkdir -p ${ResampleDIR}/Original/Native

cp ${original_vol} ${ResampleDIR}/Original/Volume/${Prefix}_space-orig_desc-brain_T1w.nii.gz

${CARET7DIR}/wb_command -add-to-spec-file ${ResampleDIR}/Original/wb.spec INVALID ${ResampleDIR}/Original/Volume/${Prefix}_space-orig_desc-brain_T1w.nii.gz

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

        # Transform ACPC surface to original space
        ${PYTHON_INTER} ${UTILS_PATH}/surf_transform.py \
          --orig_img ${original_vol} \
          --affine_img ${acpc_vol} \
          --in_surf ${ResampleDIR}/ACPC/Native/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}.surf.gii \
          --out_surf ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}.surf.gii
        ${CARET7DIR}/wb_command -set-structure ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}.surf.gii ${Structure} -surface-type $Type$Secondary
        ${CARET7DIR}/wb_command -add-to-spec-file ${ResampleDIR}/Original/wb.spec $Structure ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}.surf.gii

        # surface to volume
        ${CARET7DIR}/wb_command -create-signed-distance-volume ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}.surf.gii ${original_vol} ${ResampleDIR}/Original/Volume/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}_dseg.nii.gz
    done
done


# create ribbon (will slight fix topo)
${PYTHON_INTER} ${UTILS_PATH}/create_ribbon.py \
    ${ResampleDIR}/Original/Volume/${Prefix}_hemi-L_desc-white_dseg.nii.gz \
    --output ${ResampleDIR}/Original/Volume/${Prefix}_hemi-L_desc-ribbon_dseg.nii.gz \
    --pial_path ${ResampleDIR}/Original/Volume/${Prefix}_hemi-L_desc-pial_dseg.nii.gz \
    --pial_value 3 --white_value 2

${PYTHON_INTER} ${UTILS_PATH}/create_ribbon.py \
    ${ResampleDIR}/Original/Volume/${Prefix}_hemi-R_desc-white_dseg.nii.gz \
    --output ${ResampleDIR}/Original/Volume/${Prefix}_hemi-R_desc-ribbon_dseg.nii.gz \
    --pial_path ${ResampleDIR}/Original/Volume/${Prefix}_hemi-R_desc-pial_dseg.nii.gz \
    --pial_value 42 --white_value 41

fslmaths ${ResampleDIR}/Original/Volume/${Prefix}_hemi-L_desc-ribbon_dseg.nii.gz -max ${ResampleDIR}/Original/Volume/${Prefix}_hemi-R_desc-ribbon_dseg.nii.gz ${ResampleDIR}/Original/Volume/${Prefix}_desc-ribbon_dseg.nii.gz

# fslmaths ${ResampleDIR}/Original/Volume/${Prefix}_hemi-L_desc-pial_dseg.nii.gz -uthr 0 -abs -mul ${ResampleDIR}/Original/Volume/${Prefix}_hemi-L_desc-white_dseg.nii.gz -thr 0 -bin -mul 3 ${ResampleDIR}/Original/Volume/${Prefix}_hemi-L_desc-ribbon_dseg.nii.gz
# fslmaths ${ResampleDIR}/Original/Volume/${Prefix}_hemi-R_desc-pial_dseg.nii.gz -uthr 0 -abs -mul ${ResampleDIR}/Original/Volume/${Prefix}_hemi-R_desc-white_dseg.nii.gz -thr 0 -bin -mul 42 ${ResampleDIR}/Original/Volume/${Prefix}_hemi-R_desc-ribbon_dseg.nii.gz

# fslmaths ${ResampleDIR}/Original/Volume/${Prefix}_hemi-L_desc-white_dseg.nii.gz -uthr 0 -abs -bin -mul 2 ${ResampleDIR}/Original/Volume/${Prefix}_desc-ribbon_dseg.nii.gz
# fslmaths ${ResampleDIR}/Original/Volume/${Prefix}_hemi-R_desc-white_dseg.nii.gz -uthr 0 -abs -bin -mul 41 -add ${ResampleDIR}/Original/Volume/${Prefix}_desc-ribbon_dseg.nii.gz ${ResampleDIR}/Original/Volume/${Prefix}_desc-ribbon_dseg.nii.gz

# fslmaths ${ResampleDIR}/Original/Volume/${Prefix}_hemi-L_desc-ribbon_dseg.nii.gz -max ${ResampleDIR}/Original/Volume/${Prefix}_desc-ribbon_dseg.nii.gz ${ResampleDIR}/Original/Volume/${Prefix}_desc-ribbon_dseg.nii.gz
# fslmaths ${ResampleDIR}/Original/Volume/${Prefix}_hemi-R_desc-ribbon_dseg.nii.gz -max ${ResampleDIR}/Original/Volume/${Prefix}_desc-ribbon_dseg.nii.gz ${ResampleDIR}/Original/Volume/${Prefix}_desc-ribbon_dseg.nii.gz
# fslcpgeom ${original_vol} ${ResampleDIR}/Original/Volume/${Prefix}_desc-ribbon_dseg.nii.gz