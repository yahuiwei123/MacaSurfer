#!/bin/bash
set -e
set -x
# Some default parameters
clean_up=0

# Help message
usage () {
echo "
=== registerTalairach ===

This script provides an alternative to FreeSurfer's -talairach, 
-gcareg, and -careg steps. It uses antsRegistration to create a linear
affine (talairach.xfm and talairach.lta) and a nonlinear 
(talairach.m3z) warp to Talairach space.

Usage:
sh MacaqueReg.sh -i [movable] -o [output directory] -g [gca file path] [-c]

Required arguments:
-i	input volume (to be registered to Talairach space, needs to be skullstripped).
-o	output directory (e.g. ${subject}/mri/transforms).
-g	gca file path

Optional arguments
-c	clean up intermediate files. Include this flag to remove some 
	intermediate files (saves disk space). Default: off.
-h 	display this help message.

For the two- and three-step registrations, pre-calculated warps can be
used (e.g. from chimpanzee to Talairach or from NMT to Talairach) to
save computational time. The script will automatically skip these
steps if existing warps are found in the output directory.
"

}

# Parse arguments
#!/bin/bash

# 定义选项
SHORTOPTS=chl::n::o::g::
LONGOPTS=clean-up,help,orig_mov::,orig_trg::,curr_mov::,curr_trg::,linear::,nonlinear::,output-dir::,gca::

# 解析选项
PARSED=`getopt -o $SHORTOPTS --long $LONGOPTS -n "$0" -- "$@"`

if [[ $? -ne 0 ]]; then 
    echo "Error parsing options." >&2
    exit 1
fi

# 重新设置位置参数
eval set -- "$PARSED"

# 初始化变量
clean_up=0

# 解析选项
while true; do
  case "$1" in
    -l|--linear)
        linear="$2"
        shift 2
        ;;
    -n|--nonlinear)
        nonlinear="$2"
        shift 2
        ;;
    --orig_mov)
        orig_mov="$2"
        shift 2
        ;;
    --orig_trg)
        orig_trg="$2"
        shift 2
        ;;
    --curr_mov)
        curr_mov="$2"
        shift 2
        ;;
    --curr_trg)
        curr_trg="$2"
        shift 2
        ;;
    -o|--output-dir)
        OUTPUT_DIR="$2"
        shift 2
        ;;
    -g|--gca)
        GCA="$2"
        shift 2
        ;;
    -c|--clean-up)
        clean_up=1
        shift
        ;;
    -h|--help)
        usage
        exit 1
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "Invalid option: $1" >&2
        exit 1
        ;;
  esac
done

cd ${OUTPUT_DIR}

echo
echo Converting ANTs warps to FreeSurfer xfm, lta, and m3z formats.
echo


outname=orig2standard

# Need to rescale affine and warp because voxras matrix already rescaled to 1mm
# inv(mov_vox2ras) @ trg_vox2ras @ affine_transform @ trg_point = mov_point
# affine_transform need to be updated by new mov_vox2ras and trg_vox2ras
${PYTHON_INTER} ${UTILS_PATH}/rescale_transform.py \
  --orig_mov $orig_mov \
  --orig_trg $orig_trg \
  --curr_mov $curr_mov \
  --curr_trg $curr_trg \
  --affine $linear \
  --warp $nonlinear \
  --prefix $outname

# check fsl mat
flirt -in $curr_mov -ref $curr_trg -out test_fsl.nii.gz -applyxfm -init ${outname}_affine.mat

# Convert the FSL warps -> ANTS -> FS formats
${PYTHON_INTER} ${UTILS_PATH}/fsl2ras.py --fsl_mat ${outname}_affine.mat --ras_mat talairach.mat --mov ${curr_mov} --trg ${curr_trg}
# wb_command -convert-warpfield -from-fnirt ${outname}_warp.nii.gz ${curr_mov} -to-itk talairach.nii.gz

# check ants mat
antsApplyTransforms -i $curr_mov -r $curr_trg -o test_ants.nii.gz -t talairach.mat

ConvertTransformFile 3 talairach.mat talairach.txt
lta_convert --initk talairach.txt --outmni talairach.xfm --src ${curr_mov} --trg ${curr_trg} # xfm
lta_convert --ltavox2vox --initk talairach.txt --outlta talairach.lta --src ${curr_mov} --trg ${GCA} # lta
mri_warp_convert --initk talairach.nii.gz --outm3z talairach.m3z --insrcgeom ${curr_mov} # m3z

# Optional: clean up intermediate files
#if [ "${clean_up}" -eq 1 ]
#then

#	echo Removed:
#	rm -v ${outBase0}*
#	rm -v ${outBase1}*
#	rm -v ${outBase2}*
#	rm -v ${outBase3}*
#	rm -v {finalTransformLin}*
#	rm -v {finalTransformNonLin}*

#fi

echo
echo registerTalairach done.
echo

exit 0

