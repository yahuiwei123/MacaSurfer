#!/bin/bash
set -e

echo " START: Remove Middle Wall"

# Some default parameters
binary=0
clean_up=0

# Help message
usage () {
echo "
=== Remove Middle Wall ===

This script provides two optional ways to gengerate mask of left and right hemispheres.
And finally remove or decrease the intensity of middle wall.

Usage:
sh RemoveMiddleWall.sh -w [WD] -i [img] -o [output] [-a aseg file] [-b] [-c]

Required arguments:
-w  work directory
-i	input volume (to be registered, needs to be skullstripped).
-o	output directory (e.g. ${subject}/mri/transforms).
-a	aseg volume (optional). If aseg is offered, use aseg. Otherwise mri_robust_register. 

Optional arguments
-b  if img is a binary file
-c	clean up intermediate files. Include this flag to remove some 
	intermediate files (saves disk space). Default: off.
-h 	display this help message.
"

}

# Parse arguments
while getopts ":w:i:o:a:bch" opt; do
  case $opt in
	w) WORK_DIR=${OPTARG};;
    i) img=${OPTARG};;
    o) out=${OPTARG};;
    a) aseg=${OPTARG};;
    b) binary=1;;
    c) clean_up=1;;
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
elif [ "x" == "x$img" ]; then
    echo "-i [T1w] input is required"
    exit 1
elif [ "x" == "x$out" ]; then
    echo "-o [output] input is required"
    exit 1
fi

echo "
Running RemoveMiddleWall with the following parameters:
- work directory:    		${WORK_DIR}
- T1w image: 			    ${img}
- output image:	    		${out}
- aseg img: 		        ${aseg}
"

if [[ binary -eq 1 ]]
then
	echo "- binary:				yes"
else
	echo "- binary:				no"
fi

if [[ clean_up -eq 1 ]]
then
	echo "- clean-up:				yes"
else
	echo "- clean-up:				no"
fi

# Setup directory
mkdir -p $WORK_DIR

make_hemi_mask rh $img ${WORK_DIR}/Right-Hemi.nii.gz
make_hemi_mask lh $img ${WORK_DIR}/Left-Hemi.nii.gz

fslmaths ${WORK_DIR}/Right-Hemi.nii.gz -dilD ${WORK_DIR}/Right-Hemi-Dil.nii.gz
cp ${WORK_DIR}/Left-Hemi.nii.gz ${WORK_DIR}/Left-Hemi-Dil.nii.gz

fslmaths ${WORK_DIR}/Left-Hemi-Dil.nii.gz -sub ${WORK_DIR}/Right-Hemi-Dil.nii.gz -mas ${WORK_DIR}/Right-Hemi-Dil.nii.gz -abs ${WORK_DIR}/Right-Complement.nii.gz
fslmaths ${WORK_DIR}/Right-Hemi-Dil.nii.gz -sub ${WORK_DIR}/Right-Complement.nii.gz -fillh ${WORK_DIR}/Middle-Wall-Mask.nii.gz


if [[ binary -eq 1 ]]
then
	fslmaths $img -mas ${WORK_DIR}/Middle-Wall-Mask.nii.gz ${WORK_DIR}/Middle-Wall-Eliminate.nii.gz
    fslmaths $img -sub ${WORK_DIR}/Middle-Wall-Eliminate.nii.gz $out
else
	fslmaths ${WORK_DIR}/Middle-Wall-Mask.nii.gz -s 0.8 ${WORK_DIR}/Middle-Wall-Mask-Smooth.nii.gz
    fslmaths $img -mul ${WORK_DIR}/Middle-Wall-Mask-Smooth.nii.gz -mul 1.0 ${WORK_DIR}/Middle-Wall-Eliminate.nii.gz
    fslmaths $img -sub ${WORK_DIR}/Middle-Wall-Eliminate.nii.gz $out
fi

if [[ clean_up -eq 1 ]]
then
	rm -r ${WORK_DIR}
fi

echo " "
echo "END: Remove Middle Wall"
