#!/bin/bash
set -e
set -x

###### This script performs spherical surface registration

# Help message
usage() {
echo "
Usage: $0 --study_folder <study_folder> --subject <subject> --atlas_space_folder <atlas_space_folder> --native_folder <native_folder> --freesurfer_folder <freesurfer_folder> --surface_atlas_dir <surface_atlas_dir> --high_res_mesh <high_res_mesh> --reg_name <reg_name> --caret7_dir <caret7_dir> --msm_bindir <msm_bindir> --msm_configdir <msm_configdir> --hemi <hemisphere> 

Required arguments:
--subject                 Subject ID
--atlas_space_folder      Atlas space folder path
--native_folder           Native folder name
--freesurfer_folder       FreeSurfer folder path
--surface_atlas_dir       Surface atlas directory path
--high_res_mesh           High resolution mesh (e.g., 164)
--reg_name                Registration name (FS or MSMSulc)
--caret7_dir              Connectome Workbench directory path
--msm_bindir              MSM binary directory path
--msm_configdir           MSM configuration directory path
--hemi                    hemisphere
"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --subject_id)
      Subject="$2"
      shift 2
      ;;
    --atlas_space_folder)
      AtlasSpaceFolder="$2"
      shift 2
      ;;
    --native_folder)
      NativeFolder="$2"
      shift 2
      ;;
    --freesurfer_folder)
      FreeSurferFolder="$2"
      shift 2
      ;;
    --surface_atlas_dir)
      SurfaceAtlasDIR="$2"
      shift 2
      ;;
    --high_res_mesh)
      HighResMesh="$2"
      shift 2
      ;;
    --reg_name)
      RegName="$2"
      shift 2
      ;;
    --caret7_dir)
      CARET7DIR="$2"
      shift 2
      ;;
    --msm_bindir)
      MSMBINDIR="$2"
      shift 2
      ;;
    --msm_configdir)
      MSMCONFIGDIR="$2"
      shift 2
      ;;
    --hemi)
      hemisphere="$2"
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
if [[ -z "$Subject" || -z "$AtlasSpaceFolder" || -z "$NativeFolder" || -z "$FreeSurferFolder" || -z "$SurfaceAtlasDIR" || -z "$HighResMesh" || -z "$RegName" || -z "$CARET7DIR" || -z "$MSMBINDIR" || -z "$MSMCONFIGDIR" || -z "$hemisphere" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Log message function
log_Msg() {
    echo "$(date): $*"
}

# Main execution
log_Msg "Starting spherical surface registration"

# Create necessary directories
if [ ! -e ${AtlasSpaceFolder}/ROIs ] ; then
    mkdir -p ${AtlasSpaceFolder}/ROIs
fi
if [ ! -e ${AtlasSpaceFolder}/${NativeFolder} ] ; then
    mkdir -p ${AtlasSpaceFolder}/${NativeFolder}
fi
if [ ! -e ${AtlasSpaceFolder}/fsaverage ] ; then
	mkdir -p ${AtlasSpaceFolder}/fsaverage
fi

# Set hemisphere-specific variables
if [ $hemisphere = "lh" ] ; then
    Hemisphere="L"
    Structure="CORTEX_LEFT"
elif [ $hemisphere = "rh" ] ; then
    Hemisphere="R"
    Structure="CORTEX_RIGHT"
fi

log_Msg "Processing ${Hemisphere} hemisphere"

# Copy and convert FreeSurfer Sphere
for surf_type in sphere.reg sphere ; do
    mris_convert ${FreeSurferFolder}/${hemisphere}.${surf_type} ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.${surf_type}.surf.gii
    ${CARET7DIR}/wb_command -set-structure ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.${surf_type}.surf.gii ${Structure} -surface-type SPHERICAL
done

# Copy metrics for registration
for Map in sulc@sulc@Sulc thickness@thickness@Thickness curv@curvature@Curvature ; do
    fsname=$(echo $Map | cut -d "@" -f 1)
    wbname=$(echo $Map | cut -d "@" -f 2)
    mapname=$(echo $Map | cut -d "@" -f 3)
    metric_file=${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.${wbname}.shape.gii
    tmp_metric_file=${AtlasSpaceFolder}/${NativeFolder}/${hemisphere}.${Hemisphere}.${wbname}.shape.gii
    mris_convert -c ${FreeSurferFolder}/${hemisphere}.${fsname} ${FreeSurferFolder}/${hemisphere}.white ${tmp_metric_file}
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
${CARET7DIR}/wb_command -metric-math "abs(thickness) * 0.4" ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.thickness.shape.gii -var thickness ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.thickness.shape.gii
${CARET7DIR}/wb_command -metric-palette ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.thickness.shape.gii MODE_AUTO_SCALE_PERCENTAGE -pos-percent 4 96 -interpolate true -palette-name videen_style -disp-pos true -disp-neg false -disp-zero false


# Copy Atlas Files
cp ${SurfaceAtlasDIR}/fs_${Hemisphere}/fsaverage.${Hemisphere}.sphere.${HighResMesh}k_fs_${Hemisphere}.surf.gii ${AtlasSpaceFolder}/fsaverage/${Hemisphere}.sphere.${HighResMesh}k_fs_${Hemisphere}.surf.gii
cp ${SurfaceAtlasDIR}/fs_${Hemisphere}/fs_${Hemisphere}-to-fs_LR_fsaverage.${Hemisphere}_LR.spherical_std.${HighResMesh}k_fs_${Hemisphere}.surf.gii ${AtlasSpaceFolder}/fsaverage/${Hemisphere}.def_sphere.${HighResMesh}k_fs_${Hemisphere}.surf.gii
cp ${SurfaceAtlasDIR}/fsaverage.${Hemisphere}_LR.spherical_std.${HighResMesh}k_fs_LR.surf.gii ${AtlasSpaceFolder}/${Hemisphere}.sphere.${HighResMesh}k_fs_LR.surf.gii
${CARET7DIR}/wb_command -add-to-spec-file ${AtlasSpaceFolder}/${HighResMesh}k_fs_LR.wb.spec $Structure ${AtlasSpaceFolder}/${Hemisphere}.sphere.${HighResMesh}k_fs_LR.surf.gii
cp ${SurfaceAtlasDIR}/${Hemisphere}.atlasroi.${HighResMesh}k_fs_LR.shape.gii ${AtlasSpaceFolder}/${Hemisphere}.atlasroi.${HighResMesh}k_fs_LR.shape.gii
cp ${SurfaceAtlasDIR}/${Hemisphere}.refsulc.${HighResMesh}k_fs_LR.shape.gii ${AtlasSpaceFolder}/${Hemisphere}.refsulc.${HighResMesh}k_fs_LR.shape.gii

# Concatenate FS registration to FS --> FS_LR registration
${CARET7DIR}/wb_command -surface-sphere-project-unproject ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.reg.surf.gii ${AtlasSpaceFolder}/fsaverage/${Hemisphere}.sphere.${HighResMesh}k_fs_${Hemisphere}.surf.gii ${AtlasSpaceFolder}/fsaverage/${Hemisphere}.def_sphere.${HighResMesh}k_fs_${Hemisphere}.surf.gii ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.reg.reg_LR.surf.gii

# Make FreeSurfer Registration Areal Distortion Maps
${CARET7DIR}/wb_command -surface-vertex-areas ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.surf.gii ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.shape.gii
${CARET7DIR}/wb_command -surface-vertex-areas ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.reg.reg_LR.surf.gii ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.reg.reg_LR.shape.gii
${CARET7DIR}/wb_command -metric-math "ln(spherereg / sphere) / ln(2)" ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.ArealDistortion_FS.shape.gii -var sphere ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.shape.gii -var spherereg ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.reg.reg_LR.shape.gii
rm ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.shape.gii ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.reg.reg_LR.shape.gii
${CARET7DIR}/wb_command -set-map-names ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.ArealDistortion_FS.shape.gii -map 1 "${Subject}_${Hemisphere}_Areal_Distortion_FS"
${CARET7DIR}/wb_command -metric-palette ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.ArealDistortion_FS.shape.gii MODE_AUTO_SCALE -palette-name ROY-BIG-BL -thresholding THRESHOLD_TYPE_NORMAL THRESHOLD_TEST_SHOW_OUTSIDE -1 1

${CARET7DIR}/wb_command -surface-distortion ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.surf.gii ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.reg.reg_LR.surf.gii ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.EdgeDistortion_FS.shape.gii -edge-method

${CARET7DIR}/wb_command -surface-distortion ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.surf.gii ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.reg.reg_LR.surf.gii ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.Strain_FS.shape.gii -local-affine-method
${CARET7DIR}/wb_command -metric-merge ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.StrainJ_FS.shape.gii -metric ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.Strain_FS.shape.gii -column 1
${CARET7DIR}/wb_command -metric-merge ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.StrainR_FS.shape.gii -metric ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.Strain_FS.shape.gii -column 2
${CARET7DIR}/wb_command -metric-math "ln(var) / ln (2)" ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.StrainJ_FS.shape.gii -var var ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.StrainJ_FS.shape.gii
${CARET7DIR}/wb_command -metric-math "ln(var) / ln (2)" ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.StrainR_FS.shape.gii -var var ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.StrainR_FS.shape.gii
rm ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.Strain_FS.shape.gii

# If desired, run MSMSulc folding-based registration to FS_LR initialized with FS affine
if [ ${RegName} = "MSMSulc" ] ; then
    log_Msg "Running MSMSulc registration for ${Hemisphere} hemisphere"
    
    # Calculate Affine Transform and Apply
    if [ ! -e ${AtlasSpaceFolder}/${NativeFolder}/MSMSulc ] ; then
        mkdir ${AtlasSpaceFolder}/${NativeFolder}/MSMSulc
    fi
    ${CARET7DIR}/wb_command -surface-affine-regression ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.surf.gii ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.reg.reg_LR.surf.gii ${AtlasSpaceFolder}/${NativeFolder}/MSMSulc/${Hemisphere}.mat
    ${CARET7DIR}/wb_command -surface-apply-affine ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.surf.gii ${AtlasSpaceFolder}/${NativeFolder}/MSMSulc/${Hemisphere}.mat ${AtlasSpaceFolder}/${NativeFolder}/MSMSulc/${Hemisphere}.sphere_rot.surf.gii
    wb_command -surface-information ${AtlasSpaceFolder}/${NativeFolder}/MSMSulc/${Hemisphere}.sphere_rot.surf.gii
    ${CARET7DIR}/wb_command -surface-modify-sphere ${AtlasSpaceFolder}/${NativeFolder}/MSMSulc/${Hemisphere}.sphere_rot.surf.gii 100 ${AtlasSpaceFolder}/${NativeFolder}/MSMSulc/${Hemisphere}.sphere_rot.surf.gii
    cp ${AtlasSpaceFolder}/${NativeFolder}/MSMSulc/${Hemisphere}.sphere_rot.surf.gii ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.rot.surf.gii
    
    DIR=$(pwd)
    cd ${AtlasSpaceFolder}/${NativeFolder}/MSMSulc
    
    # Register using FreeSurfer Sulc Folding Map Using MSM Algorithm
    MSM_LD_LIBRARY_PATH="/usr/lib64"
    for libdir in /home/weiyahui/software/miniconda3/lib /home/weiyahui/software/miniconda3/envs/fslmaths-env/lib; do
        if [[ -e "${libdir}/libopenblas.so.0" ]]; then
            MSM_LD_LIBRARY_PATH="${libdir}:${MSM_LD_LIBRARY_PATH}"
        fi
    done
    if [ -x "${MSMBINDIR}" ] && [ ! -d "${MSMBINDIR}" ]; then
        log_Msg "Using MSM executable ${MSMBINDIR}"
        LD_LIBRARY_PATH=${MSM_LD_LIBRARY_PATH} ${MSMBINDIR} \
            --conf=${MSMCONFIGDIR}/MSMSulcStrainFinalconf \
            --inmesh=${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.rot.surf.gii \
            --refmesh=${AtlasSpaceFolder}/${Hemisphere}.sphere.${HighResMesh}k_fs_LR.surf.gii \
            --indata=${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sulc.shape.gii \
            --refdata=${AtlasSpaceFolder}/${Hemisphere}.refsulc.${HighResMesh}k_fs_LR.shape.gii \
            --out=${AtlasSpaceFolder}/${NativeFolder}/MSMSulc/${Hemisphere}. \
            --verbose
    elif [ -x "${MSMBINDIR}/msm_centos_v3" ]; then
        log_Msg "Using msm_centos_v3 for MSM registration"
        LD_LIBRARY_PATH=${MSM_LD_LIBRARY_PATH} ${MSMBINDIR}/msm_centos_v3 \
            --conf=${MSMCONFIGDIR}/MSMSulcStrainFinalconf \
            --inmesh=${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.rot.surf.gii \
            --refmesh=${AtlasSpaceFolder}/${Hemisphere}.sphere.${HighResMesh}k_fs_LR.surf.gii \
            --indata=${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sulc.shape.gii \
            --refdata=${AtlasSpaceFolder}/${Hemisphere}.refsulc.${HighResMesh}k_fs_LR.shape.gii \
            --out=${AtlasSpaceFolder}/${NativeFolder}/MSMSulc/${Hemisphere}. \
            --verbose
    elif [ -x "${FSLDIR}/bin/msm" ]; then
        log_Msg "Using FSL msm for MSM registration"
        LD_LIBRARY_PATH=${MSM_LD_LIBRARY_PATH} ${FSLDIR}/bin/msm \
            --conf=${MSMCONFIGDIR}/MSMSulcStrainFinalconf \
            --inmesh=${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.rot.surf.gii \
            --refmesh=${AtlasSpaceFolder}/${Hemisphere}.sphere.${HighResMesh}k_fs_LR.surf.gii \
            --indata=${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sulc.shape.gii \
            --refdata=${AtlasSpaceFolder}/${Hemisphere}.refsulc.${HighResMesh}k_fs_LR.shape.gii \
            --out=${AtlasSpaceFolder}/${NativeFolder}/MSMSulc/${Hemisphere}. \
            --verbose
    else
        log_Msg "Error: No MSM executable found"
        exit 1
    fi
    
    cp ${MSMCONFIGDIR}/MSMSulcStrainFinalconf ${AtlasSpaceFolder}/${NativeFolder}/MSMSulc/${Hemisphere}.logdir/conf
    cd $DIR
    
    cp ${AtlasSpaceFolder}/${NativeFolder}/MSMSulc/${Hemisphere}.sphere.reg.surf.gii ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.MSMSulc.surf.gii
    ${CARET7DIR}/wb_command -set-structure ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.MSMSulc.surf.gii ${Structure}

    # Make MSMSulc Registration Areal Distortion Maps
    ${CARET7DIR}/wb_command -surface-vertex-areas ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.surf.gii ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.shape.gii
    ${CARET7DIR}/wb_command -surface-vertex-areas ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.MSMSulc.surf.gii ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.MSMSulc.shape.gii
    ${CARET7DIR}/wb_command -metric-math "ln(spherereg / sphere) / ln(2)" ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.ArealDistortion_MSMSulc.shape.gii -var sphere ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.shape.gii -var spherereg ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.MSMSulc.shape.gii
    rm ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.shape.gii ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.MSMSulc.shape.gii
    ${CARET7DIR}/wb_command -set-map-names ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.ArealDistortion_MSMSulc.shape.gii -map 1 "${Subject}_${Hemisphere}_Areal_Distortion_MSMSulc"
    ${CARET7DIR}/wb_command -metric-palette ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.ArealDistortion_MSMSulc.shape.gii MODE_AUTO_SCALE -palette-name ROY-BIG-BL -thresholding THRESHOLD_TYPE_NORMAL THRESHOLD_TEST_SHOW_OUTSIDE -1 1

    ${CARET7DIR}/wb_command -surface-distortion ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.surf.gii ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.MSMSulc.surf.gii ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.EdgeDistortion_MSMSulc.shape.gii -edge-method

    ${CARET7DIR}/wb_command -surface-distortion ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.surf.gii ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.MSMSulc.surf.gii ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.Strain_MSMSulc.shape.gii -local-affine-method
    ${CARET7DIR}/wb_command -metric-merge ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.StrainJ_MSMSulc.shape.gii -metric ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.Strain_MSMSulc.shape.gii -column 1
    ${CARET7DIR}/wb_command -metric-merge ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.StrainR_MSMSulc.shape.gii -metric ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.Strain_MSMSulc.shape.gii -column 2
    ${CARET7DIR}/wb_command -metric-math "ln(var) / ln (2)" ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.StrainJ_MSMSulc.shape.gii -var var ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.StrainJ_MSMSulc.shape.gii
    ${CARET7DIR}/wb_command -metric-math "ln(var) / ln (2)" ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.StrainR_MSMSulc.shape.gii -var var ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.StrainR_MSMSulc.shape.gii
    rm ${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.Strain_MSMSulc.shape.gii

    RegSphere="${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.MSMSulc.surf.gii"
else
    RegSphere="${AtlasSpaceFolder}/${NativeFolder}/${Hemisphere}.sphere.reg.reg_LR.surf.gii"
fi

log_Msg "Completed ${Hemisphere} hemisphere registration"
