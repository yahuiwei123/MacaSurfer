#!/bin/bash
set -e
set -x
echo " START: T2w2T1Reg"

# Some default parameters
CLEAN_UP=0

# Help message
usage () {
echo "
=== T2wToT1wReg ===

This script provides an registration from T2w to T1w.

Usage:
sh T2wToT1wReg.sh -w [work directory] -i [movable] -r [target] -o [output] -a [output xfm file] [-c]

Required arguments:
-w    work directory.
-i    input volume (to be registered).
-r    target volume (the space to be registered to).
-o  output file.

Optional arguments
-a  transform matrix (FSL format .mat). Default: off.
-c    clean up intermediate files. Default: off.
-h     display this help message.
"

}

# Parse arguments
while getopts ":w:r:i:o:d:ch" opt; do
  case $opt in
    w) WORK_DIR=${OPTARG};;
    r) TRG=${OPTARG};;
    i) MOV=${OPTARG};;
    o) OUTPUT_PREFIX=${OPTARG};;
    d) DEVICE=${OPTARG};;
    c) CLEAN_UP=1;;
    h)
      usage
      exit 1
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
      :)
      echo "Option -$OPTARG requires an argument" >&2
      exit 1
      ;;
  esac
done

# Check that required parameters, paths, and folders are set
if ((OPTIND == 1))
then
    usage; exit 1
elif [ "x" == "x$WORK_DIR" ]; then
    echo "-r [WORK_DIR] input is required"
    exit 1
elif [ "x" == "x$MOV" ]; then
    echo "-i [T2w] input is required"
    exit 1
elif [ "x" == "x$TRG" ]; then
    echo "-r [T1w] input is required"
    exit 1
elif [ "x" == "x$OUTPUT_PREFIX" ]; then
    echo "-o [OUTPUT_PREFIX] input is required"
    exit 1
elif [ "x" == "x$DEVICE" ]; then
    echo "-d [DEVICE] input is required"
    exit 1
fi

echo "
Running T2wToT1wReg with the following parameters:
- work directory:            ${WORK_DIR}
- T2w image:                 ${MOV}
- T1w image:                ${TRG}
- output directory:         ${OUTPUT_PREFIX}
- device:                     ${DEVICE}
"

if [[ CLEAN_UP -eq 1 ]]
then
    echo "- CLEAN_UP:                yes"
else
    echo "- CLEAN_UP:                no"
fi

# Setup directory
mkdir -p $OUTPUT_PREFIX
mkdir -p $WORK_DIR

# Copy necessary file into work directory
cp ${MOV} ${WORK_DIR}
cp ${TRG} ${WORK_DIR}

# Prepare some file path
MOV=${WORK_DIR}/$(basename "$MOV")
TRG=${WORK_DIR}/$(basename "$TRG")
MidMat=${WORK_DIR}/T2w2T1w.mat
MidWarp=${WORK_DIR}/T2w2T1w.nii.gz
MidImg=${WORK_DIR}/T2w_in_T1w.nii.gz
OutMat=${OUTPUT_PREFIX}_linear.mat
OutWarp=${OUTPUT_PREFIX}_nonlinear.nii.gz
OutImg=${OUTPUT_PREFIX}.nii.gz

# dof
dof=rigid # affine / nonlinear

# T2w register to T1w
bash ${HCPPIPEDIR_Shared}/utils/generalRegister.sh -i $MOV -o $WORK_DIR -w $WORK_DIR -r $TRG -m $dof -l MI -f aladin -d $DEVICE

# Rename linear transform and output file
if [[ $dof == 'nonlinear' ]]; then
    mv -f ${WORK_DIR}/linear.mat $MidMat
    mv -f ${WORK_DIR}/nonlinear.nii.gz $MidWarp
    mv -f ${WORK_DIR}/final.nii.gz $MidImg

    cp $MidMat $OutMat
    cp $MidWarp $OutWarp
    cp $MidImg $OutImg
else
    mv -f ${WORK_DIR}/linear.mat $MidMat
    mv -f ${WORK_DIR}/final.nii.gz $MidImg

    cp $MidMat $OutMat
    cp $MidImg $OutImg
fi

# Clean up work dir
if [[ "${CLEAN_UP}" -eq 1 ]]; then
    rm -rv ${WORK_DIR}
fi

echo " "
echo "END: T2w2T1Reg"
