#!/bin/bash
set -e
set -x
# Some default parameters
func="aladin"
device="cpu"
clean_up=0

# Help message
usage () {
echo "
=== generalRegister ===

This script provides a general robust registration function for 
rigid, affine, nonlinear transform. The registration combines ANTs
and FSL for best performance.

Usage:
sh generalRegister.sh -i [movable] -o [output directory] -w [work directory] -r [target] -m [mode] -l [loss function] [-c]

Required arguments:
-i	input volume (to be registered, needs to be skullstripped).
-o	output directory (e.g. ${subject}/mri/transforms).
-w  work directory
-r	target volume (the space to be registered to).
-m	mode (rigid, affine, nonlinear transform to be apply).
-l	loss function (MI, CC).

Optional arguments
-f  linear registration function (ants or fsl). Default: fsl.
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
while getopts ":i:o:w:r:m:l:f:d:ch" opt; do
  case $opt in
    i) mov=${OPTARG};;
    o) OUTPUT_DIR=${OPTARG};;
	w) WORK_DIR=${OPTARG};;
    r) trg=${OPTARG};;
    m) mode=${OPTARG};;
	l) loss=${OPTARG};;
	f) func=${OPTARG};;
	d) device=${OPTARG};;
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
elif [ "x" == "x$mov" ]; then
    echo "-i [movable] input is required"
    exit 1
elif [ "x" == "x$OUTPUT_DIR" ]; then
    echo "-o [OUTPUT_DIR] input is required"
    exit 
elif [ "x" == "x$WORK_DIR" ]; then
    echo "-w [WORK_DIR] input is required"
    exit 1
elif [ "x" == "x$trg" ]; then
    echo "-r [target] input is required"
    exit 1
elif [ "x" == "x$mode" ]; then
    echo "-m [mode] input is required"
    exit 1
elif [ "x" == "x$loss" ]; then
    echo "-l [loss function] input is required"
    exit 1
elif [ "x" == "x$func" ]; then
    echo "-f [register func] input is required"
    exit 1
elif [ "x" == "x$device" ]; then
    echo "-d [which device] input is required"
    exit 1
elif [ "x" == "x$(which antsRegistration)" ]; then
  echo "Could not find ANTs"
  exit 1
elif [ "x" == "x$(which flirt)" ]; then
  echo "Could not find FSL"
  exit 1
fi

echo "
Running registerTalairach with the following parameters:
- input volume: 			${mov}
- output directory: 		${OUTPUT_DIR}
- work directory: 			${WORK_DIR}
- reference volume: 		${trg}
- register mode:            ${mode}
- loss function:			${loss}
- register function:		${func}
- device:					${device}
"

if [[ clean_up -eq 1 ]]
then
	echo "- clean-up:				yes"
else
	echo "- clean-up:				no"
fi

# Loss function
if [[ $loss = 'MI' ]]; then
	if [[ "$func" == "ants" ]]; then
		loss="--metric MI[${trg}, ${mov}, 1, 32, Regular, 0.2]"
	elif [[ "$func" == "fsl" ]]; then
		loss="-cost normmi"
	fi
elif [[ $loss = 'MSE' ]]; then
	if [[ "$func" == "ants" ]]; then
		loss="-m NMSE[${trg}, ${mov}, 1, 32, Regular, 0.2]"
	elif [[ "$func" == "fsl" ]]; then
		loss="-cost leastsq"
	fi
fi

# Rigid register functions
doRigid() {
	if [[ "$func" == "ants" ]]
	then
		# Initialization (ANTs)
		antsRegistration -d 3 --float 1 -r [${trg}, ${mov} , 0] -t Rigid[0.1] \
		--winsorize-image-intensities [0.005, 0.995] \
		--interpolation BSpline \
		-m MI[${trg}, ${mov}, 1, 32] -c 0 -f 4 -s 2 \
		-o [${regBase}_init, ${regBase}_init.nii.gz] -v

		# Rigid registration. using the initialization (ANTs)
		antsRegistration --dimensionality 3 --float 1 \
		--winsorize-image-intensities [0.005, 0.995] \
		--interpolation BSpline \
		--use-histogram-matching 1 \
		--transform Rigid[0.2] \
		$loss \
		--convergence [1000x500x250x100, 1e-7, 10] \
		--shrink-factors 8x4x2x1 \
		--smoothing-sigmas 3x2x1x0vox \
		--output [${regBase}_rigid, ${regBase}_rigid.nii.gz] \
		--v
	elif [[ "$func" == "fsl" ]]
	then
		# Initialization (FLIRT)
		flirt -in ${mov} \
		-ref ${trg} \
		-out ${regBase}_init.nii.gz \
		-omat ${regBase}_init0GenericAffine.mat \
		-interp spline \
		-cost corratio \
		-dof 6 \
		# -searchcost normmi \
		# -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -coarsesearch 45 -finesearch 15

		# Rigid registration. using the initialization (FLIRT)
		flirt -in ${regBase}_init.nii.gz \
		-ref ${trg} \
		-out ${regBase}_rigid.nii.gz \
		-omat ${regBase}_rigid0GenericAffine.mat \
		-interp spline \
		$loss \
		-dof 6
	else
		echo "Registration function ${func} is not specified! Use reg_aladin by default. "

		# Initialization (Identity)
		echo "1 0 0 0" > ${regBase}_init0GenericAffine.mat
		echo "0 1 0 0" >> ${regBase}_init0GenericAffine.mat
		echo "0 0 1 0" >> ${regBase}_init0GenericAffine.mat
		echo "0 0 0 1" >> ${regBase}_init0GenericAffine.mat
		${PYTHON_INTER} ${UTILS_PATH}/affine_fsl2niftyreg.py --fsl_mat ${regBase}_init0GenericAffine.mat --aladin_mat ${regBase}_init0GenericAffine.mat --mov ${mov} --trg ${trg}
		cp ${mov} ${regBase}_init.nii.gz

		# Rigid registration. 
		reg_aladin \
			-ref ${trg} \
			-flo ${regBase}_init.nii.gz \
			-res ${regBase}_rigid.nii.gz \
			-aff ${regBase}_rigid0GenericAffine.mat \
			-rigOnly \
			-omp 64
	fi
}

doAffine() {
	if [[ "$func" == "ants" ]]
	then
		# Affine registration using affine obtain by {doRidid} as initial transform (ANTs)
		antsRegistration --dimensionality 3 --float 1 \
		--interpolation Linear \
		--winsorize-image-intensities [0.005, 0.995] \
		--interpolation BSpline \
		--use-histogram-matching 1 \
		--transform Affine[0.2] \
		$loss \
		--convergence [1000x500x250x100, 1e-7, 10] \
		--shrink-factors 8x4x2x1 \
		--smoothing-sigmas 3x2x1x0vox \
		--output [${regBase}_affine, ${regBase}_affine.nii.gz] \
		--v
	elif [[ "$func" == "fsl" ]]
	then
		# Affine registration using affine obtain by {doRidid} as initial transform (FLIRT)
		flirt -in ${regBase}_rigid.nii.gz \
		-ref ${trg} \
		-out ${regBase}_affine.nii.gz \
		-omat ${regBase}_affine0GenericAffine.mat \
		-interp spline \
		$loss \
		-dof 12
	else
		echo "Registration function ${func} is not specified! Use reg_aladin by default. "

		reg_aladin \
			-ref ${trg} \
			-flo ${regBase}_rigid.nii.gz \
			-res ${regBase}_affine.nii.gz \
			-aff ${regBase}_affine0GenericAffine.mat \
			-affDirect \
			-omp 64
	fi
}

canUseFireantsGpu() {
	${PYTHON_INTER} - <<'PY' >/dev/null 2>&1
import fireants
import torch
torch.empty(1, device='cuda')
PY
}

doNonlinear() {
    # Nonlinear registration using affine obtain by {doRidid, doAffine} as initial transform
	if [[ "$device" == "cpu" ]] || { [[ "$device" == "gpu" ]] && ! canUseFireantsGpu; }
	then
		antsRegistration --dimensionality 3 --float 1 \
		--winsorize-image-intensities [0.005, 0.995] \
		--interpolation BSpline \
		--use-histogram-matching 1 \
		--initial-moving-transform ${regBase}_linear.mat \
		--transform SyN[0.15,3,0] \
		--metric CC[${trg}, ${mov}, 1, 4] \
		--convergence [360x240x180x90,5e-7,10] \
		--shrink-factors 8x4x2x1 \
		--smoothing-sigmas 3x2x1x0vox \
		--output ${regBase}_nonlinear \
		--v
	elif [[ "$device" == "gpu" ]]
	then
		antsApplyTransforms -d 3 \
		-i ${mov} \
		-r ${trg} \
		-t ${regBase}_linear.mat \
		-o ${regBase}_nonlinearInit.nii.gz

		${PYTHON_INTER} ${UTILS_PATH}/fireants_nonlinear.py \
		--fixed ${trg} --moving ${regBase}_nonlinearInit.nii.gz \
		--moved ${regBase}_nonlinearRes.nii.gz \
		--warp ${regBase}_nonlinear1Warp.nii.gz

		cp ${regBase}_linear.mat ${regBase}_nonlinear0GenericAffine.mat
		fslmaths ${regBase}_nonlinear1Warp.nii.gz -mul -1 ${regBase}_nonlinear1InverseWarp.nii.gz
	else
		echo "Invalid device specify!"
      	exit 1
	fi

	antsApplyTransforms --dimensionality 3 --float 1 \
	--input ${mov} --reference-image ${trg} \
	--output ${WORK_DIR}/check.nii.gz \
	--interpolation BSpline \
	--transform ${regBase}_nonlinear1Warp.nii.gz \
	--transform ${regBase}_nonlinear0GenericAffine.mat \
	--v
}

# Registration functions
doRegister() {
    if [[ "$mode" = "rigid" ]] ; then
        doRigid;
		if [[ "$func" == "ants" ]]
		then
			${PYTHON_INTER} ${UTILS_PATH}/affine_itk2fsl.py --ras_mat ${regBase}_init0GenericAffine.mat --fsl_mat ${regBase}_init0GenericAffine.mat --mov ${mov} --trg ${trg}
			${PYTHON_INTER} ${UTILS_PATH}/affine_itk2fsl.py --ras_mat ${regBase}_rigid0GenericAffine.mat --fsl_mat ${regBase}_rigid0GenericAffine.mat --mov ${regBase}_init.nii.gz --trg ${trg}
		elif [[ "$func" == "aladin" ]]
		then
			${PYTHON_INTER} ${UTILS_PATH}/affine_niftyreg2fsl.py --aladin_mat ${regBase}_init0GenericAffine.mat --fsl_mat ${regBase}_init0GenericAffine.mat --mov ${mov} --trg ${trg}
			${PYTHON_INTER} ${UTILS_PATH}/affine_niftyreg2fsl.py --aladin_mat ${regBase}_rigid0GenericAffine.mat --fsl_mat ${regBase}_rigid0GenericAffine.mat --mov ${regBase}_init.nii.gz --trg ${trg}
		fi

        # Combine initial and rigid transforms
		convert_xfm -omat ${regBase}_linear.mat -concat ${regBase}_rigid0GenericAffine.mat ${regBase}_init0GenericAffine.mat 
    fi

    if [[ "$mode" = "affine" ]] ; then
        doRigid;
		doAffine;

		if [[ "$func" == "ants" ]]
		then
			${PYTHON_INTER} ${UTILS_PATH}/affine_itk2fsl.py --ras_mat ${regBase}_init0GenericAffine.mat --fsl_mat ${regBase}_init0GenericAffine.mat --mov ${mov} --trg ${trg}
			${PYTHON_INTER} ${UTILS_PATH}/affine_itk2fsl.py --ras_mat ${regBase}_rigid0GenericAffine.mat --fsl_mat ${regBase}_rigid0GenericAffine.mat --mov ${regBase}_init.nii.gz --trg ${trg}
			${PYTHON_INTER} ${UTILS_PATH}/affine_itk2fsl.py --ras_mat ${regBase}_affine0GenericAffine.mat --fsl_mat ${regBase}_affine0GenericAffine.mat --mov ${regBase}_rigid.nii.gz --trg ${trg}
		elif [[ "$func" == "aladin" ]]
		then
			${PYTHON_INTER} ${UTILS_PATH}/affine_niftyreg2fsl.py --aladin_mat ${regBase}_init0GenericAffine.mat --fsl_mat ${regBase}_init0GenericAffine.mat --mov ${mov} --trg ${trg}
			${PYTHON_INTER} ${UTILS_PATH}/affine_niftyreg2fsl.py --aladin_mat ${regBase}_rigid0GenericAffine.mat --fsl_mat ${regBase}_rigid0GenericAffine.mat --mov ${regBase}_init.nii.gz --trg ${trg}
			${PYTHON_INTER} ${UTILS_PATH}/affine_niftyreg2fsl.py --aladin_mat ${regBase}_affine0GenericAffine.mat --fsl_mat ${regBase}_affine0GenericAffine.mat --mov ${regBase}_rigid.nii.gz --trg ${trg}
		fi

        # Combine all transforms so far to create an initial transform for the nonlinear registration
		convert_xfm -omat ${regBase}_linear.mat -concat ${regBase}_rigid0GenericAffine.mat ${regBase}_init0GenericAffine.mat 
		convert_xfm -omat ${regBase}_linear.mat -concat ${regBase}_affine0GenericAffine.mat ${regBase}_linear.mat 
    fi


    if [[ "$mode" = "nonlinear" ]] ; then
		doRigid;
        doAffine;

		if [[ "$func" == "fsl" ]]
		then
			convert_xfm -omat ${regBase}_linear.mat -concat ${regBase}_rigid0GenericAffine.mat ${regBase}_init0GenericAffine.mat
			convert_xfm -omat ${regBase}_linear.mat -concat ${regBase}_affine0GenericAffine.mat ${regBase}_linear.mat
			${PYTHON_INTER} ${UTILS_PATH}/affine_fsl2itk.py --fsl_mat ${regBase}_linear.mat --ras_mat ${regBase}_linear.mat --mov ${mov} --trg ${trg}
		elif [[ "$func" == "aladin" ]]
		then
			${PYTHON_INTER} ${UTILS_PATH}/affine_niftyreg2fsl.py --aladin_mat ${regBase}_init0GenericAffine.mat --fsl_mat ${regBase}_init0GenericAffine.mat --mov ${mov} --trg ${trg}
			${PYTHON_INTER} ${UTILS_PATH}/affine_niftyreg2fsl.py --aladin_mat ${regBase}_rigid0GenericAffine.mat --fsl_mat ${regBase}_rigid0GenericAffine.mat --mov ${regBase}_init.nii.gz --trg ${trg}
			${PYTHON_INTER} ${UTILS_PATH}/affine_niftyreg2fsl.py --aladin_mat ${regBase}_affine0GenericAffine.mat --fsl_mat ${regBase}_affine0GenericAffine.mat --mov ${regBase}_rigid.nii.gz --trg ${trg}
			convert_xfm -omat ${regBase}_linear.mat -concat ${regBase}_rigid0GenericAffine.mat ${regBase}_init0GenericAffine.mat
			convert_xfm -omat ${regBase}_linear.mat -concat ${regBase}_affine0GenericAffine.mat ${regBase}_linear.mat
			${PYTHON_INTER} ${UTILS_PATH}/affine_fsl2itk.py --fsl_mat ${regBase}_linear.mat --ras_mat ${regBase}_linear.mat --mov ${mov} --trg ${trg}

		fi

		doNonlinear;

		# ANTs format to FSL format
		${PYTHON_INTER} ${UTILS_PATH}/affine_itk2fsl.py --ras_mat ${regBase}_nonlinear0GenericAffine.mat --fsl_mat ${regBase}_nonlinear0GenericAffine.mat --mov ${mov} --trg ${trg}
		${PYTHON_INTER} ${UTILS_PATH}/affine_itk2fsl.py --ras_mat ${regBase}_linear.mat --fsl_mat ${regBase}_linear.mat --mov ${mov} --trg ${trg}
		wb_command -convert-warpfield -from-itk ${regBase}_nonlinear1Warp.nii.gz -to-fnirt ${regBase}_nonlinear1Warp.nii.gz ${regBase}_affine.nii.gz
		wb_command -convert-warpfield -from-itk ${regBase}_nonlinear1InverseWarp.nii.gz -to-fnirt ${regBase}_nonlinear1InverseWarp.nii.gz ${regBase}_affine.nii.gz

        # Combine all transforms so far to create an initial transform for the nonlinear registration
		convertwarp --rel --ref=${trg} --premat=${regBase}_nonlinear0GenericAffine.mat --warp1=${regBase}_nonlinear1Warp.nii.gz --out=${regBase}.nii.gz
    fi
}

register_one_step() {

	echo
	echo Starting one-step registration.
	echo
	
	# Get sizes of the volumes
	mri_binarize --count ${WORK_DIR}/size_mov.txt --i ${mov} --min 0.001 
	mri_binarize --count ${WORK_DIR}/size_trg.txt --i ${trg} --min 0.001 
	vol_mov=$(awk '{print $(NF-2)}' ${WORK_DIR}/size_mov.txt)
	vol_trg=$(awk '{print $(NF-2)}' ${WORK_DIR}/size_trg.txt)
	rm ${WORK_DIR}/size_mov.txt
	rm ${WORK_DIR}/size_trg.txt
	
	# Register movable volume to talairach volume directly.
	# This may work well if both volumes have approximately the same size, resolution, and anatomy.
	echo
	echo Step 1:
	echo Registering movable volume to talairach volume.
	echo
	
	if awk "BEGIN {exit !(${vol_trg} > ${vol_mov})}" # if trg is larger than mov
	then
		mov=${mov}
		trg=${trg}

		# Obtain name of registration file
		movBase=`basename ${mov}`
		movBase=${movBase%%.*}
		trgBase=`basename ${trg}`
		trgBase=${trgBase%%.*}
		regBase="${WORK_DIR}"/"${movBase}"_to_"${trgBase}"

		doRegister;
	else
		# Exchange mov and trg
		tmp=${mov}
		mov=${trg}
		trg=${tmp}

		# Obtain name of registration file
		movBase=`basename ${mov}`
		movBase=${movBase%%.*}
		trgBase=`basename ${trg}`
		trgBase=${trgBase%%.*}
		regBase="${WORK_DIR}"/"${movBase}"_to_"${trgBase}"

		doRegister;
	fi	

	# Combine the transforms
	if awk "BEGIN {exit !(${vol_trg} > ${vol_mov})}" # if trg is larger than mov
	then

		echo
		echo Combining warp from mov to NMT and warp from NMT to talairach.
		echo

		# Movable is smaller than NMT
        if [[ "$mode" = "nonlinear" ]] ; then
			cp ${regBase}_linear.mat ${OUTPUT_DIR}/${finalTransformLin}.mat
			convertwarp --rel --ref=${trg} --premat=${regBase}_nonlinear0GenericAffine.mat --warp1=${regBase}_nonlinear1Warp.nii.gz --out=${OUTPUT_DIR}/${finalTransformNonLin}.nii.gz

            # Apply transform to movable volume
			applywarp --rel -i ${mov} -r ${trg} -w ${OUTPUT_DIR}/${finalTransformNonLin}.nii.gz -o ${WORK_DIR}/final.nii.gz
		else
			# Also combine the linear tranform
			cp ${regBase}_linear.mat ${OUTPUT_DIR}/${finalTransformLin}.mat


			# Apply transform to movable volume
			flirt -in ${mov} -ref ${trg} -applyxfm -init ${OUTPUT_DIR}/${finalTransformLin}.mat -out ${WORK_DIR}/final.nii.gz -interp spline
        fi

	else

		echo
		echo Combining inverse warp from NMT to mov and warp from NMT to talairach
		echo
		
		# swap back
		tmp=${mov}
		mov=${trg}
		trg=${tmp}

		# Movable is larger than NMT
		# Invert the second transform here since we were going from NMT to movable
        if [[ "$mode" = "nonlinear" ]] ; then
			convert_xfm -omat ${regBase}_nonlinear0InverseAffine.mat -inverse ${regBase}_nonlinear0GenericAffine.mat
			cp ${regBase}_nonlinear0InverseAffine.mat ${OUTPUT_DIR}/${finalTransformLin}.mat
			convertwarp --rel --ref=${trg} --warp1=${regBase}_nonlinear1InverseWarp.nii.gz --postmat=${regBase}_nonlinear0InverseAffine.mat --out=${OUTPUT_DIR}/${finalTransformNonLin}.nii.gz

			# Apply transform to movable volume
			applywarp --rel -i ${mov} -r ${trg} -w ${OUTPUT_DIR}/${finalTransformNonLin}.nii.gz -o ${WORK_DIR}/final.nii.gz

			
		else
			# Also combine the linear tranform
			convert_xfm -omat ${OUTPUT_DIR}/${finalTransformLin}.mat -inverse ${regBase}_linear.mat

			# Apply transform to movable volume
			flirt -in ${mov} -ref ${trg} -applyxfm -init ${OUTPUT_DIR}/${finalTransformLin}.mat -out ${WORK_DIR}/final.nii.gz -interp spline
        fi
	fi
}

finalTransformLin=linear
finalTransformNonLin=nonlinear

# Setup directory
mkdir -p $WORK_DIR
mkdir -p $OUTPUT_DIR

# Do the registrations
register_one_step;

# Restore result
if [[ "${WORK_DIR}" != "${OUTPUT_DIR}" ]]
then
	mv -f ${WORK_DIR}/final.nii.gz $OUTPUT_DIR
fi

# Clean up work dir
if [[ "${clean_up}" -eq 1 ]]
then
	echo Removed:
	rm -rv ${WORK_DIR}
fi

echo
echo register done.
echo

exit 0