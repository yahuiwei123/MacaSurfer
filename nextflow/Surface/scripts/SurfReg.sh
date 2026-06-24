#!/bin/bash
set -e
set -x

###### This script performs surface registration in FreeSurfer pipeline

# Help message
usage() {
echo "
Usage: $0 --subject_dir <subject_dir> --subject_id <subject_id> --hemi <hemisphere> --gca_dir <GCAdir>

Required arguments:
--subject_dir              Subject directory path
--subject_id               Subject ID
--hemi                     hemisphere
--gca_dir                  gca directory
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
    --gca_dir)
      GCAdir="$2"
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
if [[ -z "$SubjectDIR" || -z "$SubjectID" || -z "$hemi" || -z "$GCAdir" || -z "$UTILS_PATH" || -z "$PYTHON_INTER" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Log message function
log_Msg() {
    echo "$(date): $*"
}

# Main execution
log_Msg "Starting surface registration"

#####
##### Surface registration
#####
surf_dir=${SubjectDIR}/${SubjectID}/surf
label_dir=${SubjectDIR}/${SubjectID}/label
mri_dir=${SubjectDIR}/${SubjectID}/mri

cd ${surf_dir}
mris_smooth -n 3 -nw ${hemi}.white ${hemi}.smoothwm
mris_inflate ${hemi}.smoothwm ${hemi}.inflated
mris_sphere ${hemi}.inflated ${hemi}.sphere

dist="-dist 20"
max_degrees="-max_degrees 50"

mris_register -curv $dist $max_degrees ${hemi}.sphere $GCAdir/${hemi}.average.curvature.filled.buckner40.tif ${hemi}.sphere.reg

mris_jacobian ${hemi}.white ${hemi}.sphere.reg ${hemi}.jacobian_white


# remove the medial wall part from cortex.label to prevent label get into medial wall
mv ${label_dir}/${hemi}.cortex.label ${label_dir}/${hemi}.cortex.nofix.label
${PYTHON_INTER} ${UTILS_PATH}/edit_fs_label.py \
    ${label_dir}/${hemi}.cortex.nofix.label \
    ${SubjectDIR}/${SubjectID}/mri/middle/${hemi}_medial_wall_binary.shape.gii \
    ${label_dir}/${hemi}.cortex.label

mris_ca_label -sdir $SubjectDIR -l ${label_dir}/${hemi}.cortex.label -aseg ${mri_dir}/aseg.presurf.mgz ${SubjectID} ${hemi} ${hemi}.sphere.reg $GCAdir/${hemi}.MBNA.gcs ${hemi}.aparc.annot

log_Msg "Surface registration completed successfully"