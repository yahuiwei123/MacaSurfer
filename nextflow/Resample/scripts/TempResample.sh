#!/bin/bash
set -e
set -x

###### This script processes Atlas space surface generation

# Help message
usage() {
echo "
Usage: $0  --preprocess_dir <preprocess_dir> --freesurfer_dir <freesurfer_dir> --resample_dir <resample_dir> --surface_atlas_dir <surface_atlas_dir> --hemi <hemisphere> --low_res_meshes <low_res_meshes> --regname <register_name> --python_inter <python_inter> --utils_path <utils_path> --caret7_dir <caret7_dir> --prefix <subj_ses_prefix>

Required arguments:
--preprocess_dir           Preprocess directory path
--freesurfer_dir           FreeSurfer directory path
--resample_dir             Resample directory path
--surface_atlas_dir        Surface atlas directory
--hemi                     Hemisphere to process
--low_res_meshes           Low resolution meshes
--regname                  Surface registration method
--python_inter             Python interpreter path
--utils_path               Utilities scripts path
--caret7_dir               Connectome Workbench directory path
--prefix                   Subject/session prefix (e.g. sub-032144_ses-004)
"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --preprocess_dir)
      PreProcessDIR="$2"
      shift 2
      ;;
    --freesurfer_dir)
      FreeSurferDIR="$2"
      shift 2
      ;;
    --resample_dir)
      ResampleDIR="$2"
      shift 2
      ;;
    --surface_atlas_dir)
      SurfaceAtlasDIR="$2"
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
    --regname)
      RegName="$2"
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
    --high_res_mesh)
      HighResMesh="$2"
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
if [[ -z "$PreProcessDIR" || -z ${FreeSurferDIR} || -z "$ResampleDIR" || -z "$SurfaceAtlasDIR" || -z "$hemisphere" || -z "$LowResMeshes" || -z "$RegName" || -z "$PYTHON_INTER" || -z "$UTILS_PATH" || -z "$CARET7DIR" || -z "$Prefix" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Log message function
log_Msg() {
    echo "$(date): $*"
}

# Main execution
log_Msg "Starting Atlas space processing"

InverseAtlasTransform=${PreProcessDIR}/MEBRAIN/xfms/from-MEBRAIN_to-T1w_mode-image_xfm.nii.gz
AtlasTransform=${PreProcessDIR}/MEBRAIN/xfms/from-T1w_to-MEBRAIN_mode-image_xfm.nii.gz
OriginalSpaceFolder=${ResampleDIR}/Original
AtlasSpaceFolder=${ResampleDIR}/Atlas

LowResMeshes=${LowResMeshes//@/ }
log_Msg "LowResMeshes: ${LowResMeshes}"

for LowResMesh in ${LowResMeshes} ; do
    if [ ! -e ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k ] ; then
        mkdir -p ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k
    fi
done

mkdir -p ${ResampleDIR}

######
###### [step3] Create Atlas space folder
######
mkdir -p ${ResampleDIR}/Atlas
mkdir -p ${ResampleDIR}/Atlas/Volume
mkdir -p ${ResampleDIR}/Atlas/Native

if mkdir ${ResampleDIR}/Atlas/Volume/.brain_copy_lock 2>/dev/null; then
    cp ${PreProcessDIR}/MEBRAIN/${Prefix}_space-MEBRAIN_desc-restorebrain_T1w.nii.gz ${ResampleDIR}/Atlas/Volume/${Prefix}_space-MEBRAIN_desc-brain_T1w.nii.gz
    rmdir ${ResampleDIR}/Atlas/Volume/.brain_copy_lock
fi
# ${CARET7DIR}/wb_command -add-to-spec-file ${ResampleDIR}/Atlas/wb.spec INVALID ${ResampleDIR}/Atlas/Volume/${Prefix}_space-MEBRAIN_desc-brain_T1w.nii.gz

InflateExtraScale=1

if [ $hemisphere = "lh" ] ; then
  Hemisphere="L"
  Structure="CORTEX_LEFT"
elif [ $hemisphere = "rh" ] ; then
  Hemisphere="R"
  Structure="CORTEX_RIGHT"
fi

# Create BIDS-named symlinks to FreeSurfer-named metric files in Atlas/Native
# (MSMReg.sh creates files with FreeSurfer naming, but downstream expects BIDS naming)
cd ${ResampleDIR}/Atlas/Native
for metric in thickness sulc curvature ArealDistortion_FS EdgeDistortion_FS StrainJ_FS StrainR_FS ArealDistortion_MSMSulc EdgeDistortion_MSMSulc StrainJ_MSMSulc StrainR_MSMSulc ; do
    if [ -e "${Hemisphere}.${metric}.shape.gii" ] ; then
        bids_name="$metric"
        # Map FreeSurfer curvature -> BIDS curv (also create curvature variant for low-res loop)
        if [ "$metric" = "curvature" ] ; then
            ln -sf "${Hemisphere}.${metric}.shape.gii" "${Prefix}_hemi-${Hemisphere}_desc-curvature.shape.gii"
            bids_name="curv"
        fi
        # Drop underscore before FS/MSMSulc for distortion metrics (e.g. ArealDistortion_FS -> ArealDistortionFS)
        bids_name=$(echo "$bids_name" | sed 's/_FS/FS/' | sed 's/_MSMSulc/MSMSulc/')
        ln -sf "${Hemisphere}.${metric}.shape.gii" "${Prefix}_hemi-${Hemisphere}_desc-${bids_name}.shape.gii"
    fi
done
for sphere in sphere.MSMSulc sphere.reg.reg_LR ; do
    if [ -e "${Hemisphere}.${sphere}.surf.gii" ] ; then
        # Map sphere.MSMSulc -> sphereMSMSulc, sphere.reg.reg_LR -> spherereg_reg-LR
        bids_name=$(echo "$sphere" | sed 's/\.//g' | sed 's/reg_reg/reg.reg/')
        bids_name=$(echo "$bids_name" | sed 's/MSMSulc/MSMSulc/')
        # Fix: sphereMSMSulc kept as-is, spherereg.reg_LR needs to become spherereg_reg-LR
        if [ "$sphere" = "sphere.reg.reg_LR" ] ; then
            bids_name="spherereg_reg-LR"
        elif [ "$sphere" = "sphere.MSMSulc" ] ; then
            bids_name="sphereMSMSulc"
        fi
        ln -sf "${Hemisphere}.${sphere}.surf.gii" "${Prefix}_hemi-${Hemisphere}_desc-${bids_name}.surf.gii"
    fi
done
cd - > /dev/null

# Create BIDS-named symlinks to FreeSurfer-named atlas files in Atlas/
cd ${AtlasSpaceFolder}
if [ -n "${HighResMesh}" ] ; then
    for f in "${Hemisphere}.sphere.${HighResMesh}k_fs_LR.surf.gii" "${Hemisphere}.atlasroi.${HighResMesh}k_fs_LR.shape.gii" ; do
        if [ -e "$f" ] ; then
            base=$(basename "$f")
            case "$base" in
                ${Hemisphere}.sphere.*)
                    bids_name="${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${HighResMesh}k.surf.gii"
                    ;;
                ${Hemisphere}.atlasroi.*)
                    bids_name="${Prefix}_hemi-${Hemisphere}_desc-atlasroi_res-${HighResMesh}k.shape.gii"
                    ;;
            esac
            ln -sf "$f" "${bids_name}"
        fi
    done
fi
cd - > /dev/null


Types="ANATOMICAL@GRAY_WHITE ANATOMICAL@PIAL"
i=1
for surf_type in white pial ; do
    Type=$(echo "$Types" | cut -d " " -f $i)
    Secondary=$(echo "$Type" | cut -d "@" -f 2)
    Type=$(echo "$Type" | cut -d "@" -f 1)
    if [ ! $Secondary = $Type ] ; then
      Secondary=$(echo " -surface-secondary-type ""$Secondary")
    else
      Secondary=""
    fi

    # Transform Original surface to Atlas space
    ${CARET7DIR}/wb_command -surface-apply-warpfield ${OriginalSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}.surf.gii ${InverseAtlasTransform} ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}.surf.gii -fnirt ${AtlasTransform}
    ${CARET7DIR}/wb_command -set-structure ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}.surf.gii ${Structure} -surface-type $Type$Secondary
    # ${CARET7DIR}/wb_command -add-to-spec-file ${ResampleDIR}/Atlas/wb.spec $Structure ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}.surf.gii

    # surface to volume
    ${CARET7DIR}/wb_command -create-signed-distance-volume ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}.surf.gii ${ResampleDIR}/Atlas/Volume/${Prefix}_space-MEBRAIN_desc-brain_T1w.nii.gz ${ResampleDIR}/Atlas/Volume/${Prefix}_hemi-${Hemisphere}_desc-${surf_type}_dseg.nii.gz
done

# create midthickness by averaging white and pial surfaces and use it to make inflated surfacess
${CARET7DIR}/wb_command -surface-average ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii -surf ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-white.surf.gii -surf ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-pial.surf.gii
${CARET7DIR}/wb_command -set-structure ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${Structure} -surface-type ANATOMICAL -surface-secondary-type MIDTHICKNESS
# ${CARET7DIR}/wb_command -add-to-spec-file ${ResampleDIR}/Atlas/wb.spec $Structure ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii

# create ROI through thickness > 0
${CARET7DIR}/wb_command -metric-math "thickness > 0" ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii -var thickness ${ResampleDIR}/Atlas/Native/${Hemisphere}.thickness.shape.gii
${CARET7DIR}/wb_command -metric-fill-holes ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii
${CARET7DIR}/wb_command -metric-remove-islands ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii
${CARET7DIR}/wb_command -set-map-names ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii -map 1 "$hemisphere"_ROI
${CARET7DIR}/wb_command -metric-dilate ${ResampleDIR}/Atlas/Native/${Hemisphere}.thickness.shape.gii ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii 10 ${ResampleDIR}/Atlas/Native/${Hemisphere}.thickness.shape.gii -nearest
${CARET7DIR}/wb_command -metric-dilate ${ResampleDIR}/Atlas/Native/${Hemisphere}.curvature.shape.gii ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii 10 ${ResampleDIR}/Atlas/Native/${Hemisphere}.curvature.shape.gii -nearest

# get number of vertices from native file
NativeVerts=$(${CARET7DIR}/wb_command -file-information ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii | grep 'Number of Vertices:' | cut -f2 -d: | tr -d '[:space:]')

# HCP fsaverage_LR32k used -iterations-scale 0.75. Compute new param value for native mesh density
NativeInflationScale=$(echo "scale=4; $InflateExtraScale * 0.75 * $NativeVerts / 32492" | bc -l)

${CARET7DIR}/wb_command -surface-generate-inflated ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-inflated.surf.gii ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-veryinflated.surf.gii -iterations-scale $NativeInflationScale
# ${CARET7DIR}/wb_command -add-to-spec-file ${ResampleDIR}/Atlas/wb.spec $Structure ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-inflated.surf.gii
# ${CARET7DIR}/wb_command -add-to-spec-file ${ResampleDIR}/Atlas/wb.spec $Structure ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-veryinflated.surf.gii

# label operations
for Map in aparc aparc.a2009s ; do #Remove BA because it doesn't convert properly
    if [[ -e ${FreeSurferDIR}/label/${hemisphere}.${Map}.annot ]] ; then
      mris_convert --annot ${FreeSurferDIR}/label/${hemisphere}.${Map}.annot ${FreeSurferDIR}/surf/${hemisphere}.white ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-${Map}.label.gii
      ${CARET7DIR}/wb_command -set-structure ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-${Map}.label.gii $Structure
      ${CARET7DIR}/wb_command -set-map-names ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-${Map}.label.gii -map 1 "$hemisphere"_"$Map"
      ${CARET7DIR}/wb_command -gifti-label-add-prefix ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${Hemisphere}_desc-${Map}.label.gii "${Hemisphere}_" ${ResampleDIR}/Atlas/Native/${Prefix}_hemi-${hemisphere}_desc-${Map}.label.gii
    fi
done

# Set hemisphere-specific variables
if [ $Hemisphere = "L" ] ; then
    hemisphere="lh"
    Structure="CORTEX_LEFT"
elif [ $Hemisphere = "R" ] ; then
    hemisphere="rh"
    Structure="CORTEX_RIGHT"
fi

log_Msg "Processing ${Hemisphere} hemisphere for Atlas space"

# Determine registration sphere based on registration method
if [ ${RegName} = "MSMSulc" ] ; then
    RegSphere="${AtlasSpaceFolder}/Native/${Hemisphere}.sphere.MSMSulc.surf.gii"
else
    RegSphere="${AtlasSpaceFolder}/Native/${Hemisphere}.sphere.reg.reg_LR.surf.gii"
fi

# Ensure no zeros in atlas medial wall ROI
${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/${Hemisphere}.atlasroi.${HighResMesh}k_fs_LR.shape.gii ${AtlasSpaceFolder}/${Hemisphere}.sphere.${HighResMesh}k_fs_LR.surf.gii ${RegSphere} BARYCENTRIC ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-atlasroi.shape.gii -largest
${CARET7DIR}/wb_command -metric-math "(atlas + individual) > 0" ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii -var atlas ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-atlasroi.shape.gii -var individual ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii
${CARET7DIR}/wb_command -metric-mask ${AtlasSpaceFolder}/Native/${Hemisphere}.thickness.shape.gii ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii ${AtlasSpaceFolder}/Native/${Hemisphere}.thickness.shape.gii
${CARET7DIR}/wb_command -metric-mask ${AtlasSpaceFolder}/Native/${Hemisphere}.curvature.shape.gii ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii ${AtlasSpaceFolder}/Native/${Hemisphere}.curvature.shape.gii

# Populate Highres fs_LR spec file for Atlas space
for Surface in white midthickness pial ; do
    ${CARET7DIR}/wb_command -surface-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-${Surface}.surf.gii ${RegSphere} ${AtlasSpaceFolder}/${Hemisphere}.sphere.${HighResMesh}k_fs_LR.surf.gii BARYCENTRIC ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-${Surface}_res-${HighResMesh}k.surf.gii
    # ${CARET7DIR}/wb_command -add-to-spec-file ${AtlasSpaceFolder}/${HighResMesh}k_fs_LR.wb.spec $Structure ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-${Surface}_res-${HighResMesh}k.surf.gii
done

# HCP fsaverage_LR32k used -iterations-scale 0.75. Compute new param value for high res mesh density
HighResInflationScale=$(echo "scale=4; $InflateExtraScale * 0.75 * $HighResMesh / 32" | bc -l)

${CARET7DIR}/wb_command -surface-generate-inflated ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${HighResMesh}k.surf.gii ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-inflated_res-${HighResMesh}k.surf.gii ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-veryinflated_res-${HighResMesh}k.surf.gii -iterations-scale $HighResInflationScale
# ${CARET7DIR}/wb_command -add-to-spec-file ${AtlasSpaceFolder}/${HighResMesh}k_fs_LR.wb.spec $Structure ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-inflated_res-${HighResMesh}k.surf.gii
# ${CARET7DIR}/wb_command -add-to-spec-file ${AtlasSpaceFolder}/${HighResMesh}k_fs_LR.wb.spec $Structure ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-veryinflated_res-${HighResMesh}k.surf.gii

# Resample metrics to Atlas space high resolution
for Map in thickness curvature ; do
    ${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Hemisphere}.${Map}.shape.gii ${RegSphere} ${AtlasSpaceFolder}/${Hemisphere}.sphere.${HighResMesh}k_fs_LR.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-${Map}_res-${HighResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${HighResMesh}k.surf.gii -current-roi ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii
    ${CARET7DIR}/wb_command -metric-mask ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-${Map}_res-${HighResMesh}k.shape.gii ${AtlasSpaceFolder}/${Hemisphere}.atlasroi.${HighResMesh}k_fs_LR.shape.gii ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-${Map}_res-${HighResMesh}k.shape.gii
done

# Resample distortion metrics to Atlas space
${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-ArealDistortionFS.shape.gii ${RegSphere} ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${HighResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-ArealDistortionFS_res-${HighResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${HighResMesh}k.surf.gii
${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-EdgeDistortionFS.shape.gii ${RegSphere} ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${HighResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-EdgeDistortionFS_res-${HighResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${HighResMesh}k.surf.gii
${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-StrainJFS.shape.gii ${RegSphere} ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${HighResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-StrainJFS_res-${HighResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${HighResMesh}k.surf.gii
${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-StrainRFS.shape.gii ${RegSphere} ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${HighResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-StrainRFS_res-${HighResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${HighResMesh}k.surf.gii

if [ ${RegName} = "MSMSulc" ] ; then
    ${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-ArealDistortionMSMSulc.shape.gii ${RegSphere} ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${HighResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-ArealDistortionMSMSulc_res-${HighResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${HighResMesh}k.surf.gii
    ${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-EdgeDistortionMSMSulc.shape.gii ${RegSphere} ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${HighResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-EdgeDistortionMSMSulc_res-${HighResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${HighResMesh}k.surf.gii
    ${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-StrainJMSMSulc.shape.gii ${RegSphere} ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${HighResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-StrainJMSMSulc_res-${HighResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${HighResMesh}k.surf.gii
    ${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-StrainRMSMSulc.shape.gii ${RegSphere} ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${HighResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-StrainRMSMSulc_res-${HighResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${HighResMesh}k.surf.gii
fi

${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-sulc.shape.gii ${RegSphere} ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${HighResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-sulc_res-${HighResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${HighResMesh}k.surf.gii

# Resample labels to Atlas space
for Map in aparc aparc.a2009s ; do
    if [[ -e ${FreeSurferDIR}/label/${hemisphere}."$Map".annot ]] ; then
        ${CARET7DIR}/wb_command -label-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-${Map}.label.gii ${RegSphere} ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${HighResMesh}k.surf.gii BARYCENTRIC ${AtlasSpaceFolder}/${Prefix}_hemi-${Hemisphere}_desc-${Map}_res-${HighResMesh}k.label.gii -largest
    fi
done

# Process low resolution meshes for Atlas space
for LowResMesh in ${LowResMeshes} ; do
    log_Msg "Processing Atlas space low resolution mesh: ${LowResMesh}k for ${Hemisphere} hemisphere"

    # Copy Atlas Files for low resolution
    cp ${SurfaceAtlasDIR}/${Hemisphere}.sphere.${LowResMesh}k_fs_LR.surf.gii ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii
		# ${CARET7DIR}/wb_command -add-to-spec-file ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${LowResMesh}k_fs_LR.wb.spec $Structure ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii
		cp ${SurfaceAtlasDIR}/${Hemisphere}.atlasroi.${LowResMesh}k_fs_LR.shape.gii ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/hemi-${Hemisphere}_desc-atlasroi_res-${LowResMesh}k.shape.gii
		if [ -e ${SurfaceAtlasDIR}/colin.cerebral.${Hemisphere}.flat.${LowResMesh}k_fs_LR.surf.gii ] ; then
			cp ${SurfaceAtlasDIR}/colin.cerebral.${Hemisphere}.flat.${LowResMesh}k_fs_LR.surf.gii ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/hemi-${Hemisphere}_desc-flat_res-${LowResMesh}k.surf.gii
			# ${CARET7DIR}/wb_command -add-to-spec-file ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${LowResMesh}k_fs_LR.wb.spec $Structure ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-flat_res-${LowResMesh}k.surf.gii
		fi

	    # Create BIDS-named symlinks in low-res directory
	    cd ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k
	    ln -sf hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii ${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii
	    ln -sf hemi-${Hemisphere}_desc-atlasroi_res-${LowResMesh}k.shape.gii ${Prefix}_hemi-${Hemisphere}_desc-atlasroi_res-${LowResMesh}k.shape.gii
	    cd - > /dev/null

    # Create downsampled fs_LR spec files for Atlas space
    for Surface in white midthickness pial ; do
        ${CARET7DIR}/wb_command -surface-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-${Surface}.surf.gii ${RegSphere} ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii BARYCENTRIC ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-${Surface}_res-${LowResMesh}k.surf.gii
        # ${CARET7DIR}/wb_command -add-to-spec-file ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${LowResMesh}k_fs_LR.wb.spec $Structure ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-${Surface}_res-${LowResMesh}k.surf.gii
    done

    # HCP fsaverage_LR32k used -iterations-scale 0.75. Recalculate in case using a different mesh
    LowResInflationScale=$(echo "scale=4; $InflateExtraScale * 0.75 * $LowResMesh / 32" | bc -l)

    ${CARET7DIR}/wb_command -surface-generate-inflated ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${LowResMesh}k.surf.gii ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-inflated_res-${LowResMesh}k.surf.gii ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-veryinflated_res-${LowResMesh}k.surf.gii -iterations-scale "$LowResInflationScale"
    # ${CARET7DIR}/wb_command -add-to-spec-file ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${LowResMesh}k_fs_LR.wb.spec $Structure ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-inflated_res-${LowResMesh}k.surf.gii
    # ${CARET7DIR}/wb_command -add-to-spec-file ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${LowResMesh}k_fs_LR.wb.spec $Structure ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-veryinflated_res-${LowResMesh}k.surf.gii

    # Resample metrics to Atlas space low resolution
    for Map in sulc thickness curvature ; do
        ${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Hemisphere}.${Map}.shape.gii ${RegSphere} ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-${Map}_res-${LowResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${LowResMesh}k.surf.gii -current-roi ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-roi.shape.gii
        ${CARET7DIR}/wb_command -metric-mask ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-${Map}_res-${LowResMesh}k.shape.gii ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-atlasroi_res-${LowResMesh}k.shape.gii ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-${Map}_res-${LowResMesh}k.shape.gii
    done

    # Resample distortion metrics to Atlas space low resolution
    ${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-ArealDistortionFS.shape.gii ${RegSphere} ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-ArealDistortionFS_res-${LowResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${LowResMesh}k.surf.gii
    ${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-EdgeDistortionFS.shape.gii ${RegSphere} ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-EdgeDistortionFS_res-${LowResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${LowResMesh}k.surf.gii
    ${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-StrainJFS.shape.gii ${RegSphere} ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-StrainJFS_res-${LowResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${LowResMesh}k.surf.gii
    ${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-StrainRFS.shape.gii ${RegSphere} ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-StrainRFS_res-${LowResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${LowResMesh}k.surf.gii

    if [ ${RegName} = "MSMSulc" ] ; then
        ${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-ArealDistortionMSMSulc.shape.gii ${RegSphere} ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-ArealDistortionMSMSulc_res-${LowResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${LowResMesh}k.surf.gii
        ${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-EdgeDistortionMSMSulc.shape.gii ${RegSphere} ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-EdgeDistortionMSMSulc_res-${LowResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${LowResMesh}k.surf.gii
        ${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-StrainJMSMSulc.shape.gii ${RegSphere} ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-StrainJMSMSulc_res-${LowResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${LowResMesh}k.surf.gii
        ${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-StrainRMSMSulc.shape.gii ${RegSphere} ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-StrainRMSMSulc_res-${LowResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${LowResMesh}k.surf.gii
    fi

    ${CARET7DIR}/wb_command -metric-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-sulc.shape.gii ${RegSphere} ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii ADAP_BARY_AREA ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-sulc_res-${LowResMesh}k.shape.gii -area-surfs ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-midthickness.surf.gii ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-midthickness_res-${LowResMesh}k.surf.gii

    # Resample labels to Atlas space low resolution
    for Map in aparc aparc.a2009s ; do
        if [[ -e ${FreeSurferDIR}/label/${hemisphere}."$Map".annot ]] ; then
            ${CARET7DIR}/wb_command -label-resample ${AtlasSpaceFolder}/Native/${Prefix}_hemi-${Hemisphere}_desc-${Map}.label.gii ${RegSphere} ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-sphere_res-${LowResMesh}k.surf.gii BARYCENTRIC ${AtlasSpaceFolder}/fsaverage_LR${LowResMesh}k/${Prefix}_hemi-${Hemisphere}_desc-${Map}_res-${LowResMesh}k.label.gii -largest
        fi
    done
done

log_Msg "Completed ${Hemisphere} hemisphere for Atlas space"


log_Msg "Atlas space surface and metric resampling completed successfully"