#!/bin/bash
set -e
set -x

###### This script performs high resolution pial surface generation

# Help message
usage() {
echo "
Usage: $0 --subject_dir <subject_dir> --subject_id <subject_id> --hemi <hemisphere> --enable_t2 <enable_t2>

Required arguments:
--subject_dir              Subject directory path
--subject_id               Subject ID
--hemi                     hemisphere to process
--enable_t2                whether use t2 refine pial
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
    --hemi)
      hemi="$2"
      shift 2
      ;;
    --enable_t2)
      enable_t2="$2"
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
if [[ -z "$SubjectDIR" || -z "$SubjectID" || -z "$hemi" || -z "$enable_t2" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Log message function
log_Msg() {
    echo "$(date): $*"
}

# Main execution
log_Msg "Starting high resolution pial surface generation"

#####
##### High resolution pial
#####

VARIABLESIGMA="-variablesigma 9"
MAXTHICKNESS="-max 15"
PSIGMA="-psigma 4"

mri_dir=$SubjectDIR/$SubjectID/mri
surf_dir=$SubjectDIR/$SubjectID/surf
label_dir=$SubjectDIR/$SubjectID/label

# define inside threshold and outside threshold for T2
T2_wm_mean=`fslstats ${mri_dir}/T2w_hires.norm.nii.gz -k ${mri_dir}/wm_mask.orig.nii.gz -M`
T2_gm_mean=`fslstats ${mri_dir}/T2w_hires.norm.nii.gz -k ${mri_dir}/gm_mask.orig.nii.gz -M`
T2_csf_mean=`fslstats ${mri_dir}/T2w_hires.norm.nii.gz -k ${mri_dir}/csf_mask.orig.nii.gz -M`
T2_max=`fslstats ${mri_dir}/T2w_hires.norm.nii.gz -k ${mri_dir}/aseg.orig.nii.gz -R | awk '{print $2}'`

T2_min_inside=$(echo "scale=0; $T2_wm_mean * 1 / 6" | bc)
T2_max_inside=$(echo "scale=0; $T2_max * 1 / 1" | bc)
# T2_min_outside=$(echo "scale=0; $T2_gm_mean * 4 / 5 + $T2_csf_mean * 1 / 5" | bc)
# T2_min_outside=$(echo "scale=0; $T2_gm_mean * 1 / 5 + $T2_wm_mean * 4 / 5" | bc)
T2_min_outside=$(echo "scale=0; $T2_wm_mean * 2 / 6" | bc)
T2_max_outside=$(echo "scale=0; $T2_max * 1 / 1" | bc)

##############################################
#    first loop for good sulci preformance   #
##############################################
log_Msg "mris_make_surface 1 using T1w hires one"
T1wHires="T1w_hires.norm.one";
##### Re-detect statistics
if command -v mris_autodet_gwstats > /dev/null 2>&1; then
	# if FreeSurfer version is above 7.0.0, need to do <mris_autodet_gwstats> additionally.
	mris_autodet_gwstats --o "$surf_dir"/autodet.gw.stats.${hemi}.dat --i "${mri_dir}/${T1wHires}.mgz" --wm "$mri_dir"/wm.orig.mgz  --lh-surf "$surf_dir"/${hemi}.orig
fi

mris_make_surfaces $VARIABLESIGMA $PSIGMA $MAXTHICKNESS $MINGRAY $MAXGRAY -white NOWRITE -aseg aseg.orig -orig white.preaparc -filled filled.orig -wm wm.orig -sdir ${SubjectDIR} -T1 ${T1wHires} ${SubjectID} ${hemi} -border-vals-hires
cp ${surf_dir}/${hemi}.pial ${surf_dir}/${hemi}.pial.preT2


#############################################
#       second loop for good boundary       # 
#############################################
log_Msg "mris_make_surface 1 using T1w hires two"
T1wHires="T1w_hires.norm.two"
##### Re-detect statistics
if command -v mris_autodet_gwstats > /dev/null 2>&1; then
	# if FreeSurfer version is above 7.0.0, need to do <mris_autodet_gwstats> additionally.
  mris_autodet_gwstats --o "${surf_dir}/autodet.gw.stats.${hemi}.dat" --i "${mri_dir}/${T1wHires}.mgz" --wm "${mri_dir}/wm.orig.mgz" --${hemi}-surf "${surf_dir}/${hemi}.white.preaparc"
fi

if [[ $enable_t2 == 'true' ]]; then
    log_Msg "mris_make_surface 2 using T2w_hires"
    mris_place_surface \
      --adgws-in "${surf_dir}/autodet.gw.stats.${hemi}.dat" \
      --invol "${mri_dir}/${T1wHires}.nii.gz" \
      --${hemi} --pial \
      --i "${surf_dir}/${hemi}.pial.preT2" --o "${surf_dir}/${hemi}.pial.T2" \
      --wm "${mri_dir}/wm.orig.mgz" \
      --seg "${mri_dir}/aseg.orig.mgz" \
      --aparc "${surf_dir}/../label/${hemi}.aparc.annot" \
      --nsmooth 1 \
      --white-surf "${surf_dir}/${hemi}.white.preaparc" \
      --repulse-surf "${surf_dir}/${hemi}.white.preaparc" \
      --pin-medial-wall "${label_dir}/${hemi}.cortex.label" \
      --mmvol "${mri_dir}/T2w_hires.norm.mgz" T2 \
      --mm_min_inside $T2_min_inside --mm_max_inside $T2_max_inside --mm_min_outside $T2_min_outside --mm_max_outside $T2_max_outside \
      --tspring 0.25 --nspring 0.25 --surf-repulse 25
else
    log_Msg "mris_make_surface 2 using T1w_hires.norm.two"
    mris_place_surface \
      --adgws-in "${surf_dir}/autodet.gw.stats.${hemi}.dat" \
      --invol "${mri_dir}/${T1wHires}.nii.gz" \
      --${hemi} --pial \
      --i "${surf_dir}/${hemi}.pial.preT2" --o "${surf_dir}/${hemi}.pial.T2" \
      --wm "${mri_dir}/wm.orig.mgz" \
      --seg "${mri_dir}/aseg.orig.mgz" \
      --aparc "${surf_dir}/../label/${hemi}.aparc.annot" \
      --nsmooth 1 \
      --white-surf "${surf_dir}/${hemi}.white.preaparc" \
      --repulse-surf "${surf_dir}/${hemi}.white.preaparc" \
      --pin-medial-wall "${label_dir}/${hemi}.cortex.label" \
      --tspring 0.25 --nspring 0.25 --surf-repulse 25
fi

mris_place_surface --area-map ${surf_dir}/${hemi}.pial.T2 ${surf_dir}/${hemi}.area.pial.T2
mris_place_surface --curv-map ${surf_dir}/${hemi}.pial.T2 2 10 ${surf_dir}/${hemi}.curv.pial.T2
mris_place_surface --thickness ${surf_dir}/${hemi}.white ${surf_dir}/${hemi}.pial.T2 20 7 ${surf_dir}/${hemi}.thickness.T2

cp $SubjectDIR/$SubjectID/surf/${hemi}.pial.T2 $SubjectDIR/$SubjectID/surf/${hemi}.pial
cp $SubjectDIR/$SubjectID/surf/${hemi}.thickness $SubjectDIR/$SubjectID/surf/${hemi}.thickness.preT2
cp $SubjectDIR/$SubjectID/surf/${hemi}.thickness.T2 $SubjectDIR/$SubjectID/surf/${hemi}.thickness
cp $SubjectDIR/$SubjectID/surf/${hemi}.area.pial.T2 $SubjectDIR/$SubjectID/surf/${hemi}.area.pial
cp $SubjectDIR/$SubjectID/surf/${hemi}.curv.pial.T2 $SubjectDIR/$SubjectID/surf/${hemi}.curv.pial
log_Msg "High resolution pial surface generation completed successfully"