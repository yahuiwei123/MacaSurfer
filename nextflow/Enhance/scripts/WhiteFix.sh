#!/bin/bash
set -e
set -x

###### This script performs white matter fixation and skeletonization
###### It enhances white matter segmentation and creates WM skeleton

# Help message
usage() {
echo "
Usage: $0 --t1w_nbest <t1w_nbest> --t1w_aseg <t1w_aseg> --enhance_dir <enhance_dir> --atlas_folder <atlas_folder> --device <device> --python_inter <python_inter> --utils_path <utils_path> --fix_white <fix_white>

Required arguments:
--t1w_nbest          T1w nBEST segmentation
--t1w_aseg           T1w aseg segmentation
--enhance_dir        Enhance directory
--atlas_folder       Atlas folder contains mebrain surface
--device             Device for processing
--python_inter       Python interpreter path
--utils_path         Utils scripts path
--fix_white          Fix white matter flag (True/False)
"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --t1w_nbest)
      t1w_nbest="$2"
      shift 2
      ;;
    --t1w_aseg)
      t1w_aseg="$2"
      shift 2
      ;;
    --t1w_complete_aseg)
      t1w_complete_aseg="$2"
      shift 2
      ;;
    --enhance_dir)
      enhance_dir="$2"
      shift 2
      ;;
    --atlas_folder)
      atlas_folder="$2"
      shift 2
      ;;
    --device)
      device="$2"
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
    --fix_white)
      fix_white="$2"
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
if [[ -z "$t1w_nbest" || -z "$t1w_aseg" || -z "$enhance_dir" || -z "$atlas_folder" || -z "$device" || -z "$python_inter" || -z "$utils_path" || -z "$fix_white" ]]; then
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
atlas_path="${enhance_dir}/MEBRAIN"
workdir="${t1w_path}/fix"

echo "Starting white matter fixation..."
cd ${t1w_path}

# Create working directories
mkdir -p ${workdir}
mkdir -p ${workdir}/xfms/

if [[ ${fix_white} == "true" ]]; then
    echo "Performing white matter fixation..."

    orig_path="${t1w_path}/${prefix}_desc-conform_T1w.nii.gz"
    acpc_path="${t1w_path}/${prefix}_space-acpc_res-04mm_desc-brain_T1w.nii.gz"
    acpc_mat_path="${t1w_path}/xfms/acpc.mat"

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
        for surf_type in wm pial ; do
            Type=$(echo "$Types" | cut -d " " -f $i)
            Secondary=$(echo "$Type" | cut -d "@" -f 2)
            Type=$(echo "$Type" | cut -d "@" -f 1)
            if [ ! $Secondary = $Type ] ; then
              Secondary=$(echo " -surface-secondary-type ""$Secondary")
            else
              Secondary=""
            fi

            ${CARET7DIR}/wb_command -surface-apply-warpfield ${atlas_folder}/${Hemisphere}.MEBRAINS.${surf_type}.surf.gii ${atlas_path}/xfms/from-T1w_to-MEBRAIN_mode-image_xfm.nii.gz ${workdir}/${Hemisphere}.persudo.${surf_type}.surf.gii -fnirt ${atlas_path}/xfms/from-MEBRAIN_to-T1w_mode-image_xfm.nii.gz
            ${CARET7DIR}/wb_command -surface-apply-affine ${workdir}/${Hemisphere}.persudo.${surf_type}.surf.gii ${acpc_mat_path} ${workdir}/${Hemisphere}.persudo.${surf_type}.surf.gii -flirt ${orig_path} ${atlas_folder}/mebrain_T1w_04mm_LIA.nii.gz
            ${CARET7DIR}/wb_command -set-structure ${workdir}/${Hemisphere}.persudo.${surf_type}.surf.gii ${Structure} -surface-type ${Type}${Secondary}
            ${CARET7DIR}/wb_command -create-signed-distance-volume ${workdir}/${Hemisphere}.persudo.${surf_type}.surf.gii ${acpc_path} ${workdir}/${Hemisphere}.${surf_type}.nii.gz
        done
    done

    # Skeletonize WM
    ${python_inter} ${utils_path}/skeletonize.py \
        --sdf ${workdir}/L.wm.nii.gz \
        --skel ${workdir}/left_wm_skeleton.nii.gz

    ${python_inter} ${utils_path}/skeletonize.py \
        --sdf ${workdir}/R.wm.nii.gz \
        --skel ${workdir}/right_wm_skeleton.nii.gz

    fslmaths ${workdir}/left_wm_skeleton.nii.gz -bin -add ${workdir}/right_wm_skeleton.nii.gz ${workdir}/${prefix}_space-acpc_res-04mm_desc-wmskeleton_mask.nii.gz
    cp ${workdir}/${prefix}_space-acpc_res-04mm_desc-wmskeleton_mask.nii.gz ${t1w_path}/${prefix}_space-acpc_res-04mm_desc-wmskeleton_mask.nii.gz
else
    echo "Skipping white matter fixation, extracting WM from aseg..."

    # Prefer complete aseg (always available in new pipeline), fallback to legacy paths
    if [[ -n "${t1w_complete_aseg:-}" && -f "${t1w_complete_aseg}" ]]; then
        fslmaths ${t1w_complete_aseg} -thr 2 -uthr 2 -bin ${workdir}/wm_l.nii.gz
        fslmaths ${t1w_complete_aseg} -thr 41 -uthr 41 -bin ${workdir}/wm_r.nii.gz
        fslmaths ${workdir}/wm_l.nii.gz -add ${workdir}/wm_r.nii.gz -bin ${t1w_path}/${prefix}_space-acpc_res-04mm_desc-wmskeleton_mask.nii.gz
        rm -f ${workdir}/wm_l.nii.gz ${workdir}/wm_r.nii.gz
    elif [[ -f "${t1w_path}/${prefix}_space-acpc_res-04mm_desc-freesurfer_dseg.nii.gz" ]]; then
        fslmaths ${t1w_path}/${prefix}_space-acpc_res-04mm_desc-freesurfer_dseg.nii.gz -thr 2 -uthr 2 -bin ${workdir}/wm_l.nii.gz
        fslmaths ${t1w_path}/${prefix}_space-acpc_res-04mm_desc-freesurfer_dseg.nii.gz -thr 41 -uthr 41 -bin ${workdir}/wm_r.nii.gz
        fslmaths ${workdir}/wm_l.nii.gz -add ${workdir}/wm_r.nii.gz -bin ${t1w_path}/${prefix}_space-acpc_res-04mm_desc-wmskeleton_mask.nii.gz
        rm -f ${workdir}/wm_l.nii.gz ${workdir}/wm_r.nii.gz
    else
        fslmaths ${t1w_nbest} -thr 3 -uthr 3 -bin ${t1w_path}/${prefix}_space-acpc_res-04mm_desc-wmskeleton_mask.nii.gz
    fi
fi

echo "White matter fixation completed successfully"