#!/bin/bash
set -e
set -x

###### This script performs annotation in FreeSurfer pipeline

# Help message
usage() {
echo "
Usage: $0 --subject_dir <subject_dir> --subject_id <subject_id> --fake_talairch_transform <fake_talairch_transform> --utils_path <utils_path> --python_inter <python_inter>

Required arguments:
--subject_dir              Subject directory path
--subject_id               Subject ID
--fake_talairch_transform  Fake Talairch transform file path
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
if [[ -z "$SubjectDIR" || -z "$SubjectID" || -z "$FakeTalairchTransform" || -z "$UTILS_PATH" || -z "$PYTHON_INTER" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Log message function
log_Msg() {
    echo "$(date): $*"
}

# Main execution
log_Msg "Starting annotation"

#####
##### Annotation
#####
if [ ! -e "$SubjectDIR"/"$SubjectID"/mri/transform/talairach.xfm ] ; then
    cp ${FakeTalairchTransform} "$SubjectDIR"/"$SubjectID"/mri/transforms/talairach.xfm
fi

recon-all -subjid $SubjectID -sd $SubjectDIR -cortribbon -parcstats

DIR=`pwd`
cd "$SubjectDIR"/"$SubjectID"/mri
mri_relabel_hypointensities aseg.presurf.mgz ../surf aseg.presurf.hypos.mgz
set +e
mri_aparc2aseg --s $SubjectID --sd $SubjectDIR --new-ribbon --annot aparc || mri_aparc2aseg --s $SubjectID --sd $SubjectDIR --old-ribbon --annot aparc
set -e

apas2aseg --i aparc+aseg.mgz --o aseg.aparc.mgz

cp aseg.aparc.mgz wmparc.mgz
cd $DIR

log_Msg "Annotation completed successfully"