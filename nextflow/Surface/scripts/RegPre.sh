#!/bin/bash
set -e
set -x

###### This script performs high resolution white surface generation

# Help message
usage() {
echo "
Usage: $0 --subject_dir <subject_dir> --subject_id <subject_id> --utils_path <utils_path> --python_inter <python_inter>

Required arguments:
--subject_dir              Subject directory path
--subject_id               Subject ID
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
if [[ -z "$SubjectDIR" || -z "$SubjectID" || -z "$UTILS_PATH" || -z "$PYTHON_INTER" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Log message function
log_Msg() {
    echo "$(date): $*"
}

FreeSurferDIR=${SubjectDIR}/${SubjectID}
WorkDIR=${FreeSurferDIR}/mri/middle/

# Main execution
log_Msg "Starting medial wall generation"

dist="-dist 20"
max_degrees="-max_degrees 50"

for Hemisphere in L R ; do
    if [ $Hemisphere = "L" ] ; then
      hemisphere="lh"
    elif [ $Hemisphere = "R" ] ; then
      hemisphere="rh"
    fi

    ${PYTHON_INTER} ${UTILS_PATH}/fs_surf2ras_surf.py \
        --volume ${FreeSurferDIR}/mri/filled.nii.gz \
        --fs_surf ${FreeSurferDIR}/surf/${hemisphere}.white \
        --ras_surf ${WorkDIR}/${Hemisphere}.white.surf.gii
done

${PYTHON_INTER} ${UTILS_PATH}/medialwall_extract.py \
    -w ${WorkDIR}/ \
    -e ${WorkDIR}/Middle-Wall-Mask.nii.gz \
    -f ${FreeSurferDIR}/mri/filled.nii.gz \
    -l ${WorkDIR}/L.white.surf.gii \
    -r ${WorkDIR}/R.white.surf.gii \
    -o ${WorkDIR}/

for hemisphere in lh rh ; do
    for scalar in sulc curv ; do
        cp ${FreeSurferDIR}/surf/${hemisphere}.${scalar} ${FreeSurferDIR}/surf/${hemisphere}.${scalar}.all

        if [ "$scalar" = "thickness" ]; then
            value=0.0
        else
            value=-2.0
        fi

        ${PYTHON_INTER} ${UTILS_PATH}/edit_fs_morph.py \
            ${FreeSurferDIR}/surf/${hemisphere}.${scalar} \
            ${WorkDIR}/${hemisphere}_medial_wall_binary.shape.gii \
            ${FreeSurferDIR}/surf/${hemisphere}.${scalar} \
            -t 0.5 -v ${value}
    done
done

log_Msg "Medial wall generation completed successfully"