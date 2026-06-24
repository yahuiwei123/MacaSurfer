#!/bin/bash
set -e
set -x

###### This script performs initial template registration and bias field correction
###### It registers images to template space and performs initial tissue segmentation

# Help message
usage() {
echo "
Usage: $0 --t1w_conform <t1w_conform> --t2w_conform <t2w_conform> --enhance_dir <enhance_dir> --device <device> --script_dir <script_dir> --t1w_template <t1w_template> --t1w_template_brain <t1w_template_brain> --template_mask <template_mask> --t1w_template_2mm <t1w_template_2mm> --template_2mm_mask <template_2mm_mask> --fnirt_config <fnirt_config> --t1w_template_atlas <t1w_template_atlas> --wm_compliment1 <wm_compliment1> --wm_compliment2 <wm_compliment2> --gm_compliment <gm_compliment>

Required arguments:
--t1w_conform           T1w conformed image
--t2w_conform           T2w conformed image
--enhance_dir           Enhance directory
--device                Device for processing
--script_dir            Enhancement scripts directory
--t1w_template          T1w template path
--t1w_template_brain    T1w template brain path
--template_mask         Template mask path
--t1w_template_2mm      T1w 2mm template path
--template_2mm_mask     Template 2mm mask path
--fnirt_config          FNIRT configuration file
--t1w_template_atlas    T1w template atlas path
--wm_compliment1        WM compliment 1 path
--wm_compliment2        WM compliment 2 path
--gm_compliment         GM compliment path
"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --t1w_conform)
      t1w_conform="$2"
      shift 2
      ;;
    --t2w_conform)
      t2w_conform="$2"
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
    --t1w_template)
      t1w_template="$2"
      shift 2
      ;;
    --t1w_template_brain)
      t1w_template_brain="$2"
      shift 2
      ;;
    --template_mask)
      template_mask="$2"
      shift 2
      ;;
    --t1w_template_2mm)
      t1w_template_2mm="$2"
      shift 2
      ;;
    --template_2mm_mask)
      template_2mm_mask="$2"
      shift 2
      ;;
    --fnirt_config)
      fnirt_config="$2"
      shift 2
      ;;
    --t1w_template_atlas)
      t1w_template_atlas="$2"
      shift 2
      ;;
    --wm_compliment1)
      wm_compliment1="$2"
      shift 2
      ;;
    --wm_compliment2)
      wm_compliment2="$2"
      shift 2
      ;;
    --gm_compliment)
      gm_compliment="$2"
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
required_args=("t1w_conform" "t2w_conform" "enhance_dir" "device" "script_dir" "t1w_template" "t1w_template_brain" "template_mask" "t1w_template_2mm" "template_2mm_mask" "fnirt_config" "t1w_template_atlas" "wm_compliment1" "wm_compliment2" "gm_compliment")
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
t1w_init_corrected="${t1w_path}/${prefix}_desc-initcorrected_T1w.nii.gz"
t2w_init_corrected="${t1w_path}/${prefix}_desc-initcorrected_T2w.nii.gz"

echo "Starting template registration..."
cd ${t1w_path}

# Register to MEBRAIN template
sh ${script_dir}/utils/RegisterMEBRAIN.sh \
    --workingdir=${template_space_path} \
    --t1=${t1w_conform} \
    --t1rest=${t1w_conform} \
    --t1restbrain=${t1w_conform} \
    --t2=${t2w_conform} \
    --t2rest=${t2w_conform} \
    --t2restbrain=${t2w_conform} \
    --ref=${t1w_template} \
    --refbrain=${t1w_template_brain} \
    --refmask=${template_mask} \
    --ref2mm=${t1w_template_2mm} \
    --ref2mmmask=${template_2mm_mask} \
    --owarp=${template_space_path}/xfms/from-T1w_to-MEBRAIN_mode-image_xfm.nii.gz \
    --oinvwarp=${template_space_path}/xfms/from-MEBRAIN_to-T1w_mode-image_xfm.nii.gz \
    --ot1=${template_space_path}/T1w \
    --ot1rest=${template_space_path}/${prefix}_space-MEBRAIN_desc-restore_T1w \
    --ot1restbrain=${template_space_path}/${prefix}_space-MEBRAIN_desc-restorebrain_T1w \
    --ot2=${template_space_path}/T2w \
    --ot2rest=${template_space_path}/${prefix}_space-MEBRAIN_desc-restore_T2w \
    --ot2restbrain=${template_space_path}/${prefix}_space-MEBRAIN_desc-restorebrain_T2w \
    --fnirtconfig=${fnirt_config} \
    --device=${device}

# Apply warps for subcortical tissue segmentation
echo "Applying warps for tissue segmentation..."
applywarp -r ${t1w_conform} -i ${t1w_template_atlas} -o ${prefix}_desc-aseg_dseg.nii.gz -w ${template_space_path}/xfms/from-MEBRAIN_to-T1w_mode-image_xfm.nii.gz --interp=nn
applywarp -r ${t1w_conform} -i ${wm_compliment1} -o ${prefix}_desc-wmcomp1_mask.nii.gz -w ${template_space_path}/xfms/from-MEBRAIN_to-T1w_mode-image_xfm.nii.gz --interp=nn
applywarp -r ${t1w_conform} -i ${wm_compliment2} -o ${prefix}_desc-wmcomp2_mask.nii.gz -w ${template_space_path}/xfms/from-MEBRAIN_to-T1w_mode-image_xfm.nii.gz --interp=nn
applywarp -r ${t1w_conform} -i ${gm_compliment} -o ${prefix}_desc-gmcomp_mask.nii.gz -w ${template_space_path}/xfms/from-MEBRAIN_to-T1w_mode-image_xfm.nii.gz --interp=nn

# Create white matter mask for bias field correction
echo "Creating white matter mask for bias field correction..."
fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 2 -uthr 2 -bin ${prefix}_desc-aseg_label-WM_dseg.nii.gz
fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 41 -uthr 41 -bin -add ${prefix}_desc-aseg_label-WM_dseg.nii.gz ${prefix}_desc-aseg_label-WM_dseg.nii.gz
fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 7 -uthr 7 -bin -add ${prefix}_desc-aseg_label-WM_dseg.nii.gz ${prefix}_desc-aseg_label-WM_dseg.nii.gz
fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 46 -uthr 46 -bin -add ${prefix}_desc-aseg_label-WM_dseg.nii.gz ${prefix}_desc-aseg_label-WM_dseg.nii.gz

# Copy geometry and perform initial bias field correction
fslcpgeom ${t1w_conform} ${prefix}_desc-aseg_label-WM_dseg.nii.gz
N4BiasFieldCorrection -d 3 -i ${t1w_conform} -o ${t1w_init_corrected} -w ${prefix}_desc-aseg_label-WM_dseg.nii.gz -s 1
if [[ -e ${t2w_conform} ]]; then
  N4BiasFieldCorrection -d 3 -i ${t2w_conform} -o ${t2w_init_corrected} -w ${prefix}_desc-aseg_label-WM_dseg.nii.gz -s 1
fi
echo "Initial template registration and bias field correction completed successfully"