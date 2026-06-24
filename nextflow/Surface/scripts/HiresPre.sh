#!/bin/bash
set -e
set -x

###### This script prepares necessary files for high resolution surface generation

# Help message
usage() {
echo "
Usage: $0 --subject_dir <subject_dir> --subject_id <subject_id> --preprocess_dir <preprocess_dir> --deep_white <deep_white> --caret7_dir <caret7_dir> --utils_path <utils_path> --python_inter <python_inter>

Required arguments:
--subject_dir              Subject directory path
--subject_id               Subject ID
--preprocess_dir           Preprocessing directory path
--deep_white               Deep-based white matter
--caret7_dir               Connectome Workbench directory path
--utils_path               Utilities scripts path
--python_inter             Python interpreter path
"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --subject_dir)
      SubjectDIR="$2"
      shift 2
      ;;
    --subject_id)
      SubjectID="$2"
      shift 2
      ;;
    --preprocess_dir)
      PreprocessDIR="$2"
      shift 2
      ;;
    --deep_white)
      deep_white="$2"
      shift 2
      ;;
    --caret7_dir)
      CARET7DIR="$2"
      shift 2
      ;;
    --utils_path)
      UTILS_PATH="$2"
      shift 2
      ;;
    --python_inter)
      PYTHON_INTER="$2"
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
if [[ -z "$SubjectDIR" || -z "$SubjectID" || -z "$PreprocessDIR" || -z "$deep_white" || -z "$CARET7DIR" || -z "$UTILS_PATH" || -z "$PYTHON_INTER" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Resolve subject/session-prefixed Enhance/T1w outputs
T1wPreprocessDIR="${PreprocessDIR}/T1w"
T1wAcpcFile=(${T1wPreprocessDIR}/*_space-acpc_desc-brain_T1w.nii.gz)
if [[ ${#T1wAcpcFile[@]} -ne 1 || ! -f "${T1wAcpcFile[0]}" ]]; then
    echo "Error: Expected exactly one *_space-acpc_desc-brain_T1w.nii.gz in ${T1wPreprocessDIR}, found ${#T1wAcpcFile[@]}" >&2
    exit 1
fi
T1wPrefix=$(basename "${T1wAcpcFile[0]}")
T1wPrefix=${T1wPrefix%_space-acpc_desc-brain_T1w.nii.gz}

# Log message function
log_Msg() {
    echo "$(date): $*"
}

# Main execution
log_Msg "Starting pre-hires preparation"

#####
##### Prepare some necessary file for making high resolution surface
#####

DIR=`pwd`
cd ${SubjectDIR}/${SubjectID}/mri

# High resolution for white surface
${PYTHON_INTER} ${UTILS_PATH}/conform.py \
    --input ${T1wPreprocessDIR}/${T1wPrefix}_space-acpc_desc-brain_T1w.nii.gz \
    --output T1w_hires.nii.gz \
    --rescale [1.0,1.0,1.0] --hires

${PYTHON_INTER} ${UTILS_PATH}/conform.py \
    --input ${T1wPreprocessDIR}/${T1wPrefix}_space-acpc_desc-brain_T2w.nii.gz \
    --output T2w_hires.nii.gz \
    --rescale [1.0,1.0,1.0] --hires

${PYTHON_INTER} ${UTILS_PATH}/conform.py \
    --input ${T1wPreprocessDIR}/${T1wPrefix}_space-acpc_desc-whitebfc_T1w.nii.gz \
    --output T1w_hires_white.nii.gz \
    --rescale [1.0,1.0,1.0] --hires

${PYTHON_INTER} ${UTILS_PATH}/conform.py \
    --input ${T1wPreprocessDIR}/${T1wPrefix}_space-acpc_desc-wmcomp2_mask.nii.gz \
    --output wm.compliment2.orig.nii.gz \
    --rescale [1.0,1.0,1.0] --hires

${PYTHON_INTER} ${UTILS_PATH}/conform.py \
    --input ${T1wPreprocessDIR}/${T1wPrefix}_space-acpc_desc-nbest_dseg.nii.gz \
    --output T1w_hires_nbest.nii.gz \
    --rescale [1.0,1.0,1.0] --hires

# Transform some necessary image (csf, wm, cerebrum) to high resolution space
mri_convert aseg+claustrum.mgz aseg+claustrum.nii.gz
mri_convert aseg.mgz aseg.nii.gz
mri_convert filled.mgz filled.nii.gz
mri_convert wm.mgz wm.nii.gz

flirt -in aseg+claustrum.nii.gz \
    -ref T1w_hires_white.nii.gz \
    -out aseg+claustrum.orig.nii.gz \
    -applyxfm -init ${T1wPreprocessDIR}/xfms/identity.mat \
    -interp nearestneighbour

flirt -in aseg.nii.gz \
    -ref T1w_hires_white.nii.gz \
    -out aseg.orig.nii.gz \
    -applyxfm -init ${T1wPreprocessDIR}/xfms/identity.mat \
    -interp nearestneighbour

flirt -in filled.nii.gz \
    -ref T1w_hires_white.nii.gz \
    -out filled.orig.nii.gz \
    -applyxfm -init ${T1wPreprocessDIR}/xfms/identity.mat \
    -interp nearestneighbour

flirt -in wm.nii.gz \
    -ref T1w_hires_white.nii.gz \
    -out wm.orig.nii.gz \
    -applyxfm -init ${T1wPreprocessDIR}/xfms/identity.mat \
    -interp nearestneighbour

flirt -in nbest.cerebrum.nii.gz \
    -ref T1w_hires_white.nii.gz \
    -out nbest.cerebrum.orig.nii.gz \
    -applyxfm -init ${T1wPreprocessDIR}/xfms/identity.mat \
    -interp nearestneighbour

mri_convert aseg.orig.nii.gz aseg.orig.mgz
mri_convert filled.orig.nii.gz filled.orig.mgz
# Caution: mris_autodet_stats & mris_make_surfaces need wm is of 250 value, or it will detect nothing
fslmaths wm.orig.nii.gz -bin -mul 250 wm.orig.nii.gz
mri_convert wm.orig.nii.gz wm.orig.mgz

####################################
# High resolution for white surface
####################################

# normalize
fslmaths aseg+claustrum.orig.nii.gz -thr 41 -uthr 41 -bin -mul -39 -add aseg+claustrum.orig.nii.gz -thr 2 -uthr 2 -bin wm.mask.orig.nii.gz
wmMean=`fslstats T1w_hires_white.nii.gz -k wm.mask.orig.nii.gz -M`
rm wm.mask.orig.nii.gz

if [[ $deep_white == "true" ]]; then
    ${PYTHON_INTER} ${UTILS_PATH}/fs_surf2ras_surf.py \
        --volume T1w_hires_white.nii.gz \
        --fs_surf ${SubjectDIR}/${SubjectID}/surf/lh.orig \
        --ras_surf ${SubjectDIR}/${SubjectID}/surf/lh.orig.surf.gii
        
    ${PYTHON_INTER} ${UTILS_PATH}/fs_surf2ras_surf.py \
        --volume T1w_hires_white.nii.gz \
        --fs_surf ${SubjectDIR}/${SubjectID}/surf/rh.orig \
        --ras_surf ${SubjectDIR}/${SubjectID}/surf/rh.orig.surf.gii

    ${CARET7DIR}/wb_command -create-signed-distance-volume \
        ${SubjectDIR}/${SubjectID}/surf/lh.orig.surf.gii \
        T1w_hires_white.nii.gz \
        lh.orig.nii.gz

    ${CARET7DIR}/wb_command -create-signed-distance-volume \
        ${SubjectDIR}/${SubjectID}/surf/rh.orig.surf.gii \
        T1w_hires_white.nii.gz \
        rh.orig.nii.gz

    ${PYTHON_INTER} ${UTILS_PATH}/create_ribbon.py \
        lh.orig.nii.gz \
        --output lh.wm.nii.gz \
        --white_value 1

    ${PYTHON_INTER} ${UTILS_PATH}/create_ribbon.py \
        rh.orig.nii.gz \
        --output rh.wm.nii.gz \
        --white_value 1
    
    fslmaths lh.wm.nii.gz -add rh.wm.nii.gz -bin deep_white.orig.nii.gz

    # Use wm segmented by DL model light up original image
    fslmaths T1w_hires_white.nii.gz -div $wmMean -mul 90 T1w_hires_white.nii.gz
    fslmaths deep_white.orig.nii.gz -mul 110 -max T1w_hires_white.nii.gz T1w_hires_white.nii.gz
else
    fslmaths T1w_hires_white.nii.gz -div $wmMean -mul 110 T1w_hires_white.nii.gz    
fi
# compliment
fslmaths wm.compliment2.orig.nii.gz -bin -mul 110 -max T1w_hires_white.nii.gz T1w_hires_white.nii.gz
fslmaths aseg+claustrum.orig.nii.gz -thr 51 -uthr 51 -bin -mul -39 -add aseg+claustrum.orig.nii.gz -thr 12 -uthr 12 -bin -kernel sphere 0.5 -dilD -mul 112 \
    -max T1w_hires_white.nii.gz T1w_hires_white.nii.gz
fslmaths aseg+claustrum.orig.nii.gz -thr 49 -uthr 49 -bin -mul -39 -add aseg+claustrum.orig.nii.gz -thr 10 -uthr 10 -bin -kernel sphere 0.5 -dilD -mul 112 \
    -max T1w_hires_white.nii.gz T1w_hires_white.nii.gz
fslmaths aseg+claustrum.orig.nii.gz -thr 50 -uthr 50 -bin -mul -39 -add aseg+claustrum.orig.nii.gz -thr 11 -uthr 11 -bin -kernel sphere 0.5 -dilD -mul 112 \
    -max T1w_hires_white.nii.gz T1w_hires_white.nii.gz
fslmaths aseg+claustrum.orig.nii.gz -thr 52 -uthr 52 -bin -mul -39 -add aseg+claustrum.orig.nii.gz -thr 13 -uthr 13 -bin -kernel sphere 0.5 -dilD -mul 112 \
    -max T1w_hires_white.nii.gz T1w_hires_white.nii.gz
fslmaths aseg+claustrum.orig.nii.gz -thr 58 -uthr 58 -bin -mul -32 -add aseg+claustrum.orig.nii.gz -thr 26 -uthr 26 -bin -kernel sphere 0.5 -dilD -mul 112 \
    -max T1w_hires_white.nii.gz T1w_hires_white.nii.gz
fslmaths aseg+claustrum.orig.nii.gz -thr 60 -uthr 60 -bin -mul -32 -add aseg+claustrum.orig.nii.gz -thr 28 -uthr 28 -bin -kernel sphere 0.5 -dilD -mul 112 \
    -max T1w_hires_white.nii.gz T1w_hires_white.nii.gz
fslmaths aseg+claustrum.orig.nii.gz -thr 43 -uthr 43 -bin -mul -39 -add aseg+claustrum.orig.nii.gz -thr 4 -uthr 4 -bin -kernel sphere 0.5 -dilD -mul 112 \
    -max T1w_hires_white.nii.gz T1w_hires_white.nii.gz

# clip intensity of white matter
fslmaths T1w_hires_white.nii.gz -thr 125 -bin -mul 125 clip_white.nii.gz
fslmaths T1w_hires_white.nii.gz -uthr 125 -add clip_white.nii.gz T1w_hires_white.nii.gz
rm clip_white.nii.gz

# convert fs format
mri_convert -ns 1 -odt uchar T1w_hires_white.nii.gz T1w_hires_white.mgz


###################################
# High resolution for pial surface
###################################
${PYTHON_INTER} ${UTILS_PATH}/conform.py \
    --input ${T1wPreprocessDIR}/${T1wPrefix}_space-acpc_desc-pialbfc_T1w.nii.gz \
    --output T1w_hires_pial.nii.gz \
    --rescale [1.0,1.0,1.0] --hires

${PYTHON_INTER} ${UTILS_PATH}/conform.py \
    --input ${T1wPreprocessDIR}/${T1wPrefix}_space-acpc_desc-pialbfc_T2w.nii.gz \
    --output T2w_hires_pial.nii.gz \
    --rescale [1.0,1.0,1.0] --hires

# prepare necessary files
mri_convert aseg.orig.mgz aseg.orig.nii.gz
fslmaths aseg.orig.nii.gz -thr 41 -uthr 41 -bin wm_mask.orig.nii.gz
fslmaths aseg.orig.nii.gz -thr 2 -uthr 2 -bin -add wm_mask.orig.nii.gz wm_mask.orig.nii.gz
fslmaths aseg.orig.nii.gz -thr 42 -uthr 42 -bin gm_mask.orig.nii.gz
fslmaths aseg.orig.nii.gz -thr 3 -uthr 3 -bin -add gm_mask.orig.nii.gz gm_mask.orig.nii.gz
fslmaths aseg.orig.nii.gz -thr 24 -uthr 24 -bin csf_mask.orig.nii.gz

# norm white matter region of T1_hires.norm
wmMean=`fslstats T1w_hires_pial.nii.gz -k wm_mask.orig.nii.gz -M`
fslmaths T1w_hires_pial.nii.gz -div $wmMean -mul 95 T1w_hires.norm.nii.gz
fslmaths filled.orig.nii.gz -bin -mul 110 -max T1w_hires.norm.nii.gz T1w_hires.norm.nii.gz

# rescale T2 gm intensity bigger than T1 1.6 times gm intensity
T1_gm_mean=`fslstats T1w_hires.norm.nii.gz -k gm_mask.orig.nii.gz -M`
# T2_gm_mean=`fslstats T2w_hires_pial.nii.gz -k gm_mask.orig.nii.gz -M`
# T2_times=$(echo "1.2 * ${T1_gm_mean}" | bc -l)
# fslmaths T2w_hires_pial.nii.gz -div $T2_gm_mean -mul $T2_times T2w_hires.norm.nii.gz
cp T2w_hires_pial.nii.gz T2w_hires.norm.nii.gz
fslmaths wm_mask.orig.nii.gz -bin -mul 0.10 -sub 1 -abs -mul T2w_hires.norm.nii.gz T2w_hires.norm.nii.gz
T2_wm_mean=`fslstats T2w_hires.norm.nii.gz -k wm_mask.orig.nii.gz -M`
T2_gm_mean=`fslstats T2w_hires.norm.nii.gz -k gm_mask.orig.nii.gz -M`
fslmaths T2w_hires.norm.nii.gz -div $T2_wm_mean -mul 110 T2w_hires.norm.nii.gz

# remove csf in T1 for first time pial recon
fslmaths T1w_hires.norm.nii.gz -mas csf_mask.orig.nii.gz -sub T1w_hires.norm.nii.gz -abs T1w_hires.norm.one.nii.gz

# downgrade csf (T1 * 0.90, T2 * 1.10) for second time pial recon
fslmaths csf_mask.orig.nii.gz -kernel sphere 0.8 -ero -bin -s 0.6 csf_mask_ero.orig.nii.gz

fslmaths csf_mask_ero.orig.nii.gz -mul 0.10 -sub 1 -abs -mul T1w_hires.norm.nii.gz T1w_hires.norm.two.nii.gz
fslmaths csf_mask_ero.orig.nii.gz -mul 0.10 -add 1 -abs -mul T2w_hires.norm.nii.gz -max 5.0 -mas T1w_hires.norm.nii.gz T2w_hires.norm.nii.gz

##### Convert
mri_convert -odt float T1w_hires.norm.one.nii.gz T1w_hires.norm.one.mgz
mri_convert -odt float T1w_hires.norm.two.nii.gz T1w_hires.norm.two.mgz
mri_convert -odt float T1w_hires.norm.nii.gz T1w_hires.norm.mgz
mri_convert -odt float T2w_hires.norm.nii.gz T2w_hires.norm.mgz

cd $DIR

log_Msg "Pre-hires preparation completed successfully"