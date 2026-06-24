#!/bin/bash
set -e
set -x

###### This script performs FreeSurfer reconstruction pipeline
###### It includes initialization, normalization, atlas generation, nucleus extraction, presurface processing and tessellation

# Help message
usage() {
echo "
Usage: $0 --subject_dir <subject_dir> --subject_id <subject_id> --t1w_image_file <t1w_image_file> --num_cores <num_cores> --preprocess_dir <preprocess_dir> --fake_talairch_transform <fake_talairch_transform> --utils_path <utils_path> --python_inter <python_inter> --pipeline_scripts <pipeline_scripts>

Required arguments:
--subject_dir              Subject directory path
--subject_id               Subject ID
--t1w_image_file           T1w image file path (without .nii.gz extension)
--num_cores                Number of cores for parallel processing
--preprocess_dir           Preprocessing directory path
--fake_talairch_transform  Fake Talairch transform file path
--utils_path               Utilities scripts path
--python_inter             Python interpreter path
--pipeline_scripts         Pipeline scripts directory
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
    --t1w_image_file)
      T1wImageFile="$2"
      shift 2
      ;;
    --num_cores)
      num_cores="$2"
      shift 2
      ;;
    --preprocess_dir)
      PreprocessDIR="$2"
      shift 2
      ;;
    --complete_aseg)
      CompleteAseg="$2"
      shift 2
      ;;
    --fake_talairch_transform)
      FakeTalairchTransform="$2"
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
    --pipeline_scripts)
      PipelineScripts="$2"
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
if [[ -z "$SubjectDIR" || -z "$SubjectID" || -z "$T1wImageFile" || -z "$num_cores" || -z "$PreprocessDIR" || -z "$CompleteAseg" || -z "$FakeTalairchTransform" || -z "$UTILS_PATH" || -z "$PYTHON_INTER" || -z "$PipelineScripts" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Derive Enhance/T1w filename prefix from the ACPC T1w input
T1wPrefix=$(basename "$T1wImageFile")
T1wPrefix=${T1wPrefix%_space-acpc_res-04mm_desc-brain_T1w.nii.gz}
T1wPreprocessDIR="${PreprocessDIR}/T1w"

# Log message function
log_Msg() {
    echo "$(date): $*"
}

# Initialize function
function initialize() {
    #####
    ##### Initial Recon-all Steps
    #####
    if [ -e "$SubjectDIR"/"$SubjectID" ] ; then
        log_Msg "Removing previous FS directory"
        rm -rf "$SubjectDIR"/"$SubjectID"
    fi

    log_Msg "Initial recon-all steps"

    mkdir -p ${SubjectDIR}/${SubjectID}/label \
        ${SubjectDIR}/${SubjectID}/mri \
        ${SubjectDIR}/${SubjectID}/scripts \
        ${SubjectDIR}/${SubjectID}/stats \
        ${SubjectDIR}/${SubjectID}/surf \
        ${SubjectDIR}/${SubjectID}/tmp \
        ${SubjectDIR}/${SubjectID}/touch \
        ${SubjectDIR}/${SubjectID}/trash \
        ${SubjectDIR}/${SubjectID}/mri/orig \
        ${SubjectDIR}/${SubjectID}/mri/transforms
    
    mri_convert "$T1wImageFile" ${SubjectDIR}/${SubjectID}/mri/orig/001.mgz
    cp ${SubjectDIR}/${SubjectID}/mri/orig/001.mgz ${SubjectDIR}/${SubjectID}/mri/rawavg.mgz

    mri_convert ${SubjectDIR}/${SubjectID}/mri/rawavg.mgz ${SubjectDIR}/${SubjectID}/mri/rawavg.nii.gz
    ${PYTHON_INTER} ${UTILS_PATH}/conform.py --input ${SubjectDIR}/${SubjectID}/mri/rawavg.nii.gz --output ${SubjectDIR}/${SubjectID}/mri/orig.nii.gz --norm 255 --gamma 1.0
    ${PYTHON_INTER} ${UTILS_PATH}/conform.py --input ${SubjectDIR}/${SubjectID}/mri/orig.nii.gz --output ${SubjectDIR}/${SubjectID}/mri/orig.nii.gz --rescale [1.0,1.0,1.0]

    mri_convert ${SubjectDIR}/${SubjectID}/mri/rawavg.nii.gz ${SubjectDIR}/${SubjectID}/mri/rawavg.mgz
    mri_convert ${SubjectDIR}/${SubjectID}/mri/orig.nii.gz ${SubjectDIR}/${SubjectID}/mri/orig.mgz
    mri_add_xform_to_header -c ${SubjectDIR}/${SubjectID}/mri/transforms/talairach.xfm ${SubjectDIR}/${SubjectID}/mri/orig.mgz ${SubjectDIR}/${SubjectID}/mri/orig.mgz

    cp ${FakeTalairchTransform} ${SubjectDIR}/${SubjectID}/mri/transforms/talairach.xfm
}

# Normalize function
function normalize() {
    #####
    ##### Normalize
    ##### input [orig.mgz]
    ##### output [nu.mgz, norm.mgz, brainmask.mgz, brain.mgz, brain.finalsurfs.mgz]
    #####
    cp ${SubjectDIR}/${SubjectID}/mri/orig.mgz ${SubjectDIR}/${SubjectID}/mri/nu.mgz
    cp ${SubjectDIR}/${SubjectID}/mri/nu.mgz ${SubjectDIR}/${SubjectID}/mri/norm.mgz
}

# Atlas function
function atlas() {
    #####
    ##### Set up FS-space aseg from pre-computed complete aseg (produced in Enhance stage)
    ##### input [space-acpc complete aseg from --complete_aseg, cerebrum mask, norm.mgz]
    ##### output [aseg.auto.nii.gz, aseg.mgz, aseg+claustrum.mgz, aseg.presurf.mgz]
    #####
    DIR=`pwd`
    cd ${SubjectDIR}/${SubjectID}/mri

    # Conform pre-computed complete aseg from ACPC space to FS 1mm space
    ${PYTHON_INTER} ${UTILS_PATH}/conform.py \
        --input ${CompleteAseg} \
        --output ${SubjectDIR}/${SubjectID}/mri/aseg.auto.nii.gz \
        --rescale [1.0,1.0,1.0]

    # Conform cerebrum mask from ACPC space (needed by presurf for wm/filled masking)
    ${PYTHON_INTER} ${UTILS_PATH}/conform.py \
        --input ${T1wPreprocessDIR}/${T1wPrefix}_space-acpc_res-04mm_label-cerebrum_dseg.nii.gz \
        --output ${SubjectDIR}/${SubjectID}/mri/nbest.cerebrum.nii.gz \
        --rescale [1.0,1.0,1.0]

    ## Split left and right hemispheres (generates middle/ masks used by SurfReg.sh etc.)
    mri_convert norm.mgz norm.nii.gz
    bash "$PipelineScripts"/utils/SplitSphere.sh -w ${SubjectDIR}/${SubjectID}/mri/middle -i norm.nii.gz -o norm.nouse.nii.gz -a aseg.auto.nii.gz

    mri_convert aseg.auto.nii.gz aseg.mgz
    cp aseg.mgz aseg+claustrum.mgz
    cp aseg.mgz aseg.presurf.mgz

    cd $DIR
}

# Presurf function
function presurf() {
    #####
    ##### Generate brain.finalsurfs.mgz
    ##### input [aseg.presurf.mgz, norm.mgz]
    ##### output [brain.mgz, brain.finalsurfs.mgz]
    #####
    DIR=`pwd`
    cd ${SubjectDIR}/${SubjectID}/mri
    cp norm.mgz brainmask.mgz
    # recon-all -subjid $SubjectID -sd $SubjectDIR -normalization2 -maskbfs
    mri_normalize -mprage -monkey -noskull -gentle -aseg aseg.mgz -mask brainmask.mgz norm.mgz brain.mgz
    mri_mask -T 5 brain.mgz brainmask.mgz brain.finalsurfs.mgz
    cd $DIR

    #####
    ##### Fill white matter
    ##### input [aseg.mgz, aseg+claustrum.mgz, brain.finalsurfs.mgz, prefixed wm compliment mask]
    ##### output [wm.mgz, filled.mgz]
    #####
    DIR=`pwd`
    cd ${SubjectDIR}/${SubjectID}/mri
    mri_convert aseg.mgz aseg.nii.gz
    mri_convert brain.finalsurfs.mgz brain.finalsurfs.nii.gz
    fslmaths aseg.nii.gz -thr 41 -uthr 41 -bin wm.nii.gz
    fslmaths aseg.nii.gz -thr 2 -uthr 2 -bin -add wm.nii.gz wm.nii.gz
    fslmaths brain.finalsurfs.nii.gz -mas wm.nii.gz wm.asegedit.nii.gz

    mri_convert aseg+claustrum.mgz aseg+claustrum.nii.gz

    # Paste nucleus to wm.mgz
    fslmaths aseg+claustrum.nii.gz -thr 43 -uthr 43 -bin -mul -39 -add aseg+claustrum.nii.gz -thr 4 -uthr 4 -bin -mul 250 \
        -max wm.asegedit.nii.gz wm.asegedit.nii.gz
    fslmaths aseg+claustrum.nii.gz -thr 51 -uthr 51 -bin -mul -39 -add aseg+claustrum.nii.gz -thr 12 -uthr 12 -bin -mul 250 \
        -max wm.asegedit.nii.gz wm.asegedit.nii.gz
    fslmaths aseg+claustrum.nii.gz -thr 49 -uthr 49 -bin -mul -39 -add aseg+claustrum.nii.gz -thr 10 -uthr 10 -bin -mul 250 \
        -max wm.asegedit.nii.gz wm.asegedit.nii.gz
    fslmaths aseg+claustrum.nii.gz -thr 50 -uthr 50 -bin -mul -39 -add aseg+claustrum.nii.gz -thr 11 -uthr 11 -bin -mul 250 \
        -max wm.asegedit.nii.gz wm.asegedit.nii.gz
    fslmaths aseg+claustrum.nii.gz -thr 52 -uthr 52 -bin -mul -39 -add aseg+claustrum.nii.gz -thr 13 -uthr 13 -bin -mul 250 \
        -max wm.asegedit.nii.gz wm.asegedit.nii.gz
    fslmaths aseg+claustrum.nii.gz -thr 58 -uthr 58 -bin -mul -32 -add aseg+claustrum.nii.gz -thr 26 -uthr 26 -bin -mul 250 \
        -max wm.asegedit.nii.gz wm.asegedit.nii.gz
    fslmaths aseg+claustrum.nii.gz -thr 60 -uthr 60 -bin -mul -32 -add aseg+claustrum.nii.gz -thr 28 -uthr 28 -bin -mul 250 \
        -max wm.asegedit.nii.gz wm.asegedit.nii.gz

    # pasting claustrum to wm.mgz
    fslmaths aseg+claustrum.nii.gz -thr 138 -uthr 138 -bin -add aseg+claustrum.nii.gz -thr 139 -uthr 139 -bin -mul 250 \
        -max wm.asegedit.nii.gz wm.asegedit.nii.gz

    ## Paste the compliment wm caused by the difference of white matter between nbest and gca
    ${PYTHON_INTER} ${UTILS_PATH}/conform.py --input ${T1wPreprocessDIR}/${T1wPrefix}_space-acpc_res-04mm_desc-wmcomp1_mask.nii.gz --output wm.compliment1.nii.gz --rescale [1.0,1.0,1.0]
    fslmaths wm.compliment1.nii.gz -bin -mul 250 -max wm.asegedit.nii.gz wm.asegedit.nii.gz

    ## Mask out ventricleIDC
    fslmaths aseg.auto.nii.gz -thr 140 -uthr 140 ventricleIDC.nii.gz
    fslmaths wm.asegedit.nii.gz -mas ventricleIDC.nii.gz -sub wm.asegedit.nii.gz -abs wm.asegedit.nii.gz

    # ## Paste wm skeleton from aseg.auto.nii.gz to wm.nii.gz
    ${PYTHON_INTER} ${UTILS_PATH}/conform.py --input ${T1wPreprocessDIR}/${T1wPrefix}_space-acpc_res-04mm_desc-wmskeleton_mask.nii.gz --output wm_fix_skeleton.nii.gz --rescale [1.0,1.0,1.0]
    fslmaths wm_fix_skeleton.nii.gz -bin -mul 250 -max wm.asegedit.nii.gz wm.asegedit.nii.gz

    mri_convert -ns 1 -odt uchar wm.asegedit.nii.gz wm.asegedit.mgz

    ## convert back to mgz format
    mri_convert -ns 1 -odt uchar wm.asegedit.nii.gz wm.asegedit.mgz # save in 8-bit
    mri_pretess wm.asegedit.mgz wm brain.finalsurfs.mgz wm.mgz

    # fill hole and genearte filled.mgz
    fslmaths wm.asegedit.nii.gz -kernel sphere 0.8 -fillh -fillh filled.nii.gz
    fslmaths filled.nii.gz -kernel sphere 0.5 -ero filled.nii.gz
    fslmaths filled.nii.gz -mas middle/Left-Hemi.nii.gz -bin -mul 255 left_filled.nii.gz
    fslmaths filled.nii.gz -mas middle/Right-Hemi.nii.gz -bin -mul 127 right_filled.nii.gz
    fslmaths left_filled.nii.gz -add right_filled.nii.gz filled.nii.gz
    rm left_filled.nii.gz right_filled.nii.gz

    mri_convert wm.nii.gz wm.mgz
    mri_convert filled.nii.gz filled.mgz

    # make the white matter in brain.finalsurfs.mgz more brighter than before.
    mri_convert brain.finalsurfs.mgz brain.finalsurfs.nii.gz
    mri_convert filled.mgz filled.nii.gz
    fslmaths wm.asegedit.nii.gz -bin -mul 110 wm_threshold.nii.gz

    wmMean=`fslstats brain.finalsurfs.nii.gz -k wm.nii.gz -M`
    fslmaths brain.finalsurfs.nii.gz -div $wmMean -mul 70 brain.finalsurfs.nii.gz
    fslmaths brain.finalsurfs.nii.gz -mas wm_threshold.nii.gz -sub brain.finalsurfs.nii.gz -abs -max wm_threshold.nii.gz brain.finalsurfs.nii.gz

    cp brain.finalsurfs.mgz brain.finalsurfs.orig.mgz
    mri_convert brain.finalsurfs.nii.gz brain.finalsurfs.mgz

    cd $DIR
}

# Main execution
log_Msg "Starting FreeSurfer reconstruction pipeline"

log_Msg "Step 1: Initialize"
initialize

log_Msg "Step 2: Normalize"
normalize

log_Msg "Step 3: Atlas generation"
atlas

log_Msg "Step 4: Presurface processing"
presurf

log_Msg "FreeSurfer prepare completed successfully"