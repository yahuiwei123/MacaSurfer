#!/bin/bash
set -e
set -x

###### This script processes Original space surface generation

# Help message
usage() {
echo "
Usage: $0 --hemi <hemisphere> --python_inter <python_inter> --utils_path <utils_path> --caret7_dir <caret7_dir> --resample_dir <resample_dir> --freesurfer_dir <freesurfer_dir> --low_res_meshes <low_res_meshes> --prefix <subj_ses_prefix>

Required arguments:
--hemi                     Hemisphere
--python_inter             Python interpreter path
--utils_path               Utilities scripts path
--caret7_dir               Connectome Workbench directory path
--resample_dir             Resample directory path
--freesurfer_dir           FreeSurfer directory path
--low_res_meshes           Low resolution meshes
--prefix                   Subject/session prefix (e.g. sub-032144_ses-004)
"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --freesurfer_dir)
      FreeSurferDIR="$2"
      shift 2
      ;;
    --resample_dir)
      ResampleDIR="$2"
      shift 2
      ;;
    --hemi)
      hemisphere="$2"
      shift 2
      ;;
    --low_res_meshes)
      LowResMeshes="$2"
      shift 2
      ;;
    --python_inter)
      PYTHON_INTER="$2"
      shift 2
      ;;
    --utils_path)
      UTILS_PATH="$2"
      shift 2
      ;;
    --caret7_dir)
      CARET7DIR="$2"
      shift 2
      ;;
    --prefix)
      Prefix="$2"
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
if [[ -z "$hemisphere" || -z "$PYTHON_INTER" || -z "$UTILS_PATH" || -z "$CARET7DIR" || -z "$ResampleDIR" || -z "$FreeSurferDIR" || -z "$LowResMeshes" || -z "$Prefix" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Log message function
log_Msg() {
    echo "$(date): $*"
}

# Main execution
log_Msg "Starting Original surface generating"

OriginalFolder=${ResampleDIR}/Original
mkdir -p ${OriginalFolder}

# Create necessary directories
LowResMeshes=${LowResMeshes//@/ }
log_Msg "LowResMeshes: ${LowResMeshes}"

for LowResMesh in ${LowResMeshes} ; do
    if [ ! -e ${OriginalFolder}/fsaverage_LR${LowResMesh}k ] ; then
        mkdir ${OriginalFolder}/fsaverage_LR${LowResMesh}k
    fi
done

if [[ -e ${ResampleDIR}/Original/wb.spec ]]; then
    rm ${ResampleDIR}/Original/wb.spec
fi

# Determine registration sphere based on registration method
if [ $hemisphere = "lh" ] ; then
  Hemisphere="L"
  Structure="CORTEX_LEFT"
elif [ $hemisphere = "rh" ] ; then
  Hemisphere="R"
  Structure="CORTEX_RIGHT"
fi

AtlasSpaceFolder=${ResampleDIR}/Atlas
if [ ${RegName} = "MSMSulc" ] ; then
    RegSphere="${AtlasSpaceFolder}/Native/${Hemisphere}.sphere.MSMSulc.surf.gii"
else
    RegSphere="${AtlasSpaceFolder}/Native/${Hemisphere}.sphere.reg.reg_LR.surf.gii"
fi

# create midthickness by averaging white and pial surfaces and use it to make inflated surfacess
${CARET7DIR}/wb_command -surface-average ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii -surf ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-white.surf.gii -surf ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-pial.surf.gii
${CARET7DIR}/wb_command -set-structure ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${Structure} -surface-type ANATOMICAL -surface-secondary-type MIDTHICKNESS
# ${CARET7DIR}/wb_command -add-to-spec-file ${ResampleDIR}/Original/wb.spec $Structure ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii

# get number of vertices from native file
NativeVerts=$(${CARET7DIR}/wb_command -file-information ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii | grep 'Number of Vertices:' | cut -f2 -d: | tr -d '[:space:]')

# HCP fsaverage_LR32k used -iterations-scale 0.75. Compute new param value for native mesh density
InflateExtraScale=1
NativeInflationScale=$(echo "scale=4; $InflateExtraScale * 0.75 * $NativeVerts / 32492" | bc -l)

${CARET7DIR}/wb_command -surface-generate-inflated ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-inflated.surf.gii ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-veryinflated.surf.gii -iterations-scale $NativeInflationScale
# ${CARET7DIR}/wb_command -add-to-spec-file ${ResampleDIR}/Original/wb.spec $Structure ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-inflated.surf.gii
# ${CARET7DIR}/wb_command -add-to-spec-file ${ResampleDIR}/Original/wb.spec $Structure ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-veryinflated.surf.gii

# convert original and registered spherical surfaces and add them to the nonlinear spec file
for surf_type in sphere.reg sphere ; do
  mris_convert ${FreeSurferDIR}/surf/${hemisphere}.${surf_type} ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}.surf.gii
  ${CARET7DIR}/wb_command -set-structure ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}.surf.gii ${Structure} -surface-type SPHERICAL
done
# ${CARET7DIR}/wb_command -add-to-spec-file ${ResampleDIR}/Original/wb.spec $Structure ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-sphere.surf.gii

# add more files to the spec file and convert other FreeSurfer surface data to metric/GIFTI including sulc, curv, and thickness.
for Map in sulc@sulc@Sulc thickness@thickness@Thickness curv@curvature@Curvature ; do
    fsname=$(echo $Map | cut -d "@" -f 1)
    wbname=$(echo $Map | cut -d "@" -f 2)
    mapname=$(echo $Map | cut -d "@" -f 3)
    metric_file=${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-${wbname}.shape.gii
    tmp_metric_file=${ResampleDIR}/Original/Native/${hemisphere}.${Prefix}_hemi-${Hemisphere}_desc-${wbname}.shape.gii
    mris_convert -c ${FreeSurferDIR}/surf/${hemisphere}.${fsname} ${FreeSurferDIR}/surf/${hemisphere}.white ${tmp_metric_file}
    if [[ ! -f "${tmp_metric_file}" ]]; then
        echo "Error: mris_convert did not create expected metric file: ${tmp_metric_file}" >&2
        exit 1
    fi
    mv ${tmp_metric_file} ${metric_file}
    ${CARET7DIR}/wb_command -set-structure ${metric_file} ${Structure}
    ${CARET7DIR}/wb_command -metric-math "var * -1" ${metric_file} -var var ${metric_file}
    ${CARET7DIR}/wb_command -set-map-names ${metric_file} -map 1 "$hemisphere"_"$mapname"
    ${CARET7DIR}/wb_command -metric-palette ${metric_file} MODE_AUTO_SCALE_PERCENTAGE -pos-percent 2 98 -palette-name Gray_Interp -disp-pos true -disp-neg true -disp-zero true
done
# thickness specific operations
${CARET7DIR}/wb_command -metric-math "abs(thickness) * 0.4" ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-thickness.shape.gii -var thickness ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-thickness.shape.gii
${CARET7DIR}/wb_command -metric-palette ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-thickness.shape.gii MODE_AUTO_SCALE_PERCENTAGE -pos-percent 4 96 -interpolate true -palette-name videen_style -disp-pos true -disp-neg false -disp-zero false
${CARET7DIR}/wb_command -metric-math "thickness > 0" ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii -var thickness ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-thickness.shape.gii
${CARET7DIR}/wb_command -metric-fill-holes ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii
${CARET7DIR}/wb_command -metric-remove-islands ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii
${CARET7DIR}/wb_command -set-map-names ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii -map 1 "$hemisphere"_ROI
${CARET7DIR}/wb_command -metric-dilate ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-thickness.shape.gii ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii 10 ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-thickness.shape.gii -nearest
${CARET7DIR}/wb_command -metric-dilate ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-curvature.shape.gii ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii 10 ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-curvature.shape.gii -nearest

# label operations
for Map in aparc aparc.a2009s ; do #Remove BA because it doesn't convert properly
    if [[ -e ${FreeSurferDIR}/label/${hemisphere}.${Map}.annot ]] ; then
      mris_convert --annot ${FreeSurferDIR}/label/${hemisphere}.${Map}.annot ${FreeSurferDIR}/surf/${hemisphere}.white ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-${Map}.label.gii
      ${CARET7DIR}/wb_command -set-structure ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-${Map}.label.gii $Structure
      ${CARET7DIR}/wb_command -set-map-names ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-${Map}.label.gii -map 1 "$hemisphere"_${Map}
      ${CARET7DIR}/wb_command -gifti-label-add-prefix ${ResampleDIR}/Original/Native/${Prefix}_hemi-${Hemisphere}_desc-${Map}.label.gii "${Hemisphere}_" ${ResampleDIR}/Original/Native/${Prefix}_hemi-${hemisphere}_desc-${Map}.label.gii
    fi
done


log_Msg "Starting Original space surface and metric resampling"

# Process low resolution meshes for Original space

# Create downsampled fs_LR spec file in Original space for both hemispheres
for LowResMesh in ${LowResMeshes} ; do
    log_Msg "Processing Original space low resolution mesh: ${LowResMesh}k"

    for Surface in white midthickness pial ; do
        ${CARET7DIR}/wb_command -surface-resample ${OriginalFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-${Surface}.surf.gii ${RegSphere} ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii BARYCENTRIC ${OriginalFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-${Surface}_res-${LowResMesh}k.surf.gii
        # ${CARET7DIR}/wb_command -add-to-spec-file ${OriginalFolder}/fsaverage_LR${LowResMesh}k/${LowResMesh}k_fs_LR.wb.spec $Structure ${OriginalFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-${Surface}_res-${LowResMesh}k.surf.gii
    done

    # HCP fsaverage_LR32k used -iterations-scale 0.75. Recalculate in case using a different mesh
    LowResInflationScale=$(echo "scale=4; $InflateExtraScale * 0.75 * $LowResMesh / 32" | bc -l)

    ${CARET7DIR}/wb_command -surface-generate-inflated ${OriginalFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${LowResMesh}k.surf.gii ${OriginalFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-inflated_res-${LowResMesh}k.surf.gii ${OriginalFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-veryinflated_res-${LowResMesh}k.surf.gii -iterations-scale "$LowResInflationScale"
    # ${CARET7DIR}/wb_command -add-to-spec-file ${OriginalFolder}/fsaverage_LR${LowResMesh}k/${LowResMesh}k_fs_LR.wb.spec $Structure ${OriginalFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-inflated_res-${LowResMesh}k.surf.gii
    # ${CARET7DIR}/wb_command -add-to-spec-file ${OriginalFolder}/fsaverage_LR${LowResMesh}k/${LowResMesh}k_fs_LR.wb.spec $Structure ${OriginalFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-veryinflated_res-${LowResMesh}k.surf.gii

    # Resample metrics to Original space low resolution
    for Map in sulc thickness curvature ; do
        if [ -e ${AtlasSpaceFolder}/Native/${Hemisphere}.${Map}.shape.gii ] ; then
            ${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Hemisphere}.${Map}.shape.gii ${RegSphere} ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii ADAP_BARY_AREA ${OriginalFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-${Map}_res-${LowResMesh}k.shape.gii -area-surfs ${OriginalFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${OriginalFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${LowResMesh}k.surf.gii -current-roi ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii
            ${CARET7DIR}/wb_command -metric-mask ${OriginalFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-${Map}_res-${LowResMesh}k.shape.gii ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/hemi-${Hemisphere}_desc-atlasroi_res-${LowResMesh}k.shape.gii ${OriginalFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-${Map}_res-${LowResMesh}k.shape.gii
        fi
    done

    # Resample labels to Original space low resolution
    for Map in aparc aparc.a2009s ; do
        if [[ -e "${FreeSurferDIR}"/label/${hemisphere}.${Map}.annot ]] ; then
            ${CARET7DIR}/wb_command -label-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-${Map}.label.gii ${RegSphere} ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii BARYCENTRIC ${OriginalFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-${Map}_res-${LowResMesh}k.label.gii -largest
        fi
    done
done

log_Msg "Original space surface and metric resampling completed successfully"