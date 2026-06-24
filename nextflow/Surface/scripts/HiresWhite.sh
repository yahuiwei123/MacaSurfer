#!/bin/bash
set -e
set -x

###### This script performs high resolution white surface generation

# Help message
usage() {
echo "
Usage: $0 --subject_dir <subject_dir> --subject_id <subject_id> --t1w_image_file <t1w_image_file> --t2w_image_file <t2w_image_file> --hemi <hemisphere> --deep_white <deep_white> --pipeline_scripts <surface_script_dir> --utils_path <utils_path> --python_inter <python_inter> --fully_deep <fully_deep> --species <species>

Required arguments:
--subject_dir              Subject directory path
--subject_id               Subject ID
--t1w_image_file           T1w image file path (without .nii.gz extension)
--t2w_image_file           T2w image file path (without .nii.gz extension)
--hemi                     Hemisphere to process
--deep_white               Fully deeplearning white matter
--pipeline_scripts         Surface script dir
--utils_path               Utilities scripts path
--python_inter             Python interpreter path
--species                  Species specification
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
      T1wImage="$2"
      shift 2
      ;;
    --t2w_image_file)
      T2wImage="$2"
      shift 2
      ;;
    --hemi)
      hemi="$2"
      shift 2
      ;;
    --deep_white)
      deep_white="$2"
      shift 2
      ;;
    --pipeline_scripts)
      pipeline_scripts="$2"
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
    --species)
      SPECIES="$2"
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
if [[ -z "$SubjectDIR" || -z "$SubjectID" || -z "$T1wImage" || -z "$T2wImage" || -z "$hemi" || -z "$deep_white" || -z "$pipeline_scripts" || -z "$UTILS_PATH" || -z "$PYTHON_INTER" || -z "$SPECIES" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Log message function
log_Msg() {
    echo "$(date): $*"
}

# Main execution
log_Msg "Starting high resolution white surface generation"

#####
##### High resolution white
#####
T1wImageName="${T1wImage%.nii.gz*}"
T2wImageName="${T2wImage%.nii.gz*}"
BrainFinalSurf=T1w_hires_white

mri_dir=$SubjectDIR/$SubjectID/mri
surf_dir=$SubjectDIR/$SubjectID/surf

# check original resolution
all_res=$(mri_info ${mri_dir}/${BrainFinalSurf}.mgz | grep "voxel sizes" | awk '{print $3, $4, $5}' | sed 's/,//g')
IFS=' ' read -r x_res y_res z_res <<< ${all_res}

# Save copies of the "prehires" versions
cp --preserve=timestamps $SubjectDIR/$SubjectID/surf/${hemi}.white $SubjectDIR/$SubjectID/surf/${hemi}.white.prehires
cp --preserve=timestamps $SubjectDIR/$SubjectID/surf/${hemi}.curv $SubjectDIR/$SubjectID/surf/${hemi}.curv.prehires
cp --preserve=timestamps $SubjectDIR/$SubjectID/surf/${hemi}.area $SubjectDIR/$SubjectID/surf/${hemi}.area.prehires
cp --preserve=timestamps $SubjectDIR/$SubjectID/label/${hemi}.cortex.label $SubjectDIR/$SubjectID/label/${hemi}.cortex.prehires.label

if (( $(echo "$x_res <= 2.05" | bc -l) )) && (( $(echo "$y_res <= 2.05" | bc -l) )) && (( $(echo "$z_res <= 2.05" | bc -l) )); then
    FIRSTWMPEAK="-min_border_white 85 -repulse 0.4 -border-vals-hires"
    mris_make_surfaces ${FIRSTWMPEAK} -noaparc -aseg aseg.orig -orig white -filled filled.orig -wm wm.orig -sdir $SubjectDIR -T1 $BrainFinalSurf -orig_white white -output .deformed -w 0 $SubjectID ${hemi} -hires

    # Fine Tune T2w to T1w Registration
    echo "$SubjectID" > "$mri_dir"/transforms/eye.dat
    echo "1" >> "$mri_dir"/transforms/eye.dat
    echo "1" >> "$mri_dir"/transforms/eye.dat
    echo "1" >> "$mri_dir"/transforms/eye.dat
    echo "1 0 0 0" >> "$mri_dir"/transforms/eye.dat
    echo "0 1 0 0" >> "$mri_dir"/transforms/eye.dat
    echo "0 0 1 0" >> "$mri_dir"/transforms/eye.dat
    echo "0 0 0 1" >> "$mri_dir"/transforms/eye.dat
    echo "round" >> "$mri_dir"/transforms/eye.dat

    # # bbregister does not work well for marmoset data even having good initialization and correct white surface for marmoset. - Takuya Hayashi Dec 2017
    # cp "$SubjectDIR"/"$SubjectID"/surf/${hemi}.thickness.deformed "$SubjectDIR"/"$SubjectID"/surf/${hemi}.thickness
    # bbregister --s "$SubjectID" --mov "$T2wImage" --surf white.deformed --init-reg "$mri_dir"/transforms/eye.dat --t2 \
    #  --reg "$mri_dir"/transforms/T2wtoT1w.dat --o "$T2wImage"

    # tkregister2 --noedit --reg "$mri_dir"/transforms/T2wtoT1w.dat --mov "$T2wImage" --targ "$T1wImage" --fslregout "$mri_dir"/transforms/T2wtoT1w.mat
    # applywarp --interp=spline -i "$T2wImage" -r "$T1wImage" --premat="$mri_dir"/transforms/T2wtoT1w.mat -o "$T2wImage"
    # fslmaths "$T2wImage" -abs -add 1 "$T2wImage"
    # fslmaths "$T1wImage" -mul "$T2wImage" -sqrt "$mri_dir"/T1wMulT2w_hires.nii.gz

    cp --preserve=timestamps $SubjectDIR/$SubjectID/surf/${hemi}.white.deformed $SubjectDIR/$SubjectID/surf/${hemi}.white
    cp --preserve=timestamps $SubjectDIR/$SubjectID/surf/${hemi}.curv.deformed $SubjectDIR/$SubjectID/surf/${hemi}.curv
    cp --preserve=timestamps $SubjectDIR/$SubjectID/surf/${hemi}.area.deformed  $SubjectDIR/$SubjectID/surf/${hemi}.area
    cp --preserve=timestamps $SubjectDIR/$SubjectID/label/${hemi}.cortex.deformed.label $SubjectDIR/$SubjectID/label/${hemi}.cortex.label
else
    cp --preserve=timestamps $SubjectDIR/$SubjectID/surf/${hemi}.white.prehires $SubjectDIR/$SubjectID/surf/${hemi}.white.deformed
    cp --preserve=timestamps $SubjectDIR/$SubjectID/surf/${hemi}.curv.prehires $SubjectDIR/$SubjectID/surf/${hemi}.curv.deformed
    cp --preserve=timestamps $SubjectDIR/$SubjectID/surf/${hemi}.area.prehires  $SubjectDIR/$SubjectID/surf/${hemi}.area.deformed
    cp --preserve=timestamps $SubjectDIR/$SubjectID/label/${hemi}.cortex.prehires.label $SubjectDIR/$SubjectID/label/${hemi}.cortex.deformed.label

    cp --preserve=timestamps $SubjectDIR/$SubjectID/surf/${hemi}.white.deformed $SubjectDIR/$SubjectID/surf/${hemi}.white
    cp --preserve=timestamps $SubjectDIR/$SubjectID/surf/${hemi}.curv.deformed $SubjectDIR/$SubjectID/surf/${hemi}.curv
    cp --preserve=timestamps $SubjectDIR/$SubjectID/surf/${hemi}.area.deformed  $SubjectDIR/$SubjectID/surf/${hemi}.area
    cp --preserve=timestamps $SubjectDIR/$SubjectID/label/${hemi}.cortex.deformed.label $SubjectDIR/$SubjectID/label/${hemi}.cortex.label
fi

cd ${surf_dir}
mris_smooth -n 3 -nw ${hemi}.white ${hemi}.smoothwm

# generate a normal ${hemi}.inflated.nofix for showing data
mris_inflate -n 10 ${hemi}.smoothwm ${hemi}.inflated
mv ${hemi}.inflated ${hemi}.inflated.10

# generate a more inflated one for dealing with fix error
mris_inflate -n 35 ${hemi}.smoothwm ${hemi}.inflated
mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 ${hemi}.inflated
mris_sphere ${hemi}.inflated ${hemi}.sphere

log_Msg "High resolution white surface generation completed successfully"