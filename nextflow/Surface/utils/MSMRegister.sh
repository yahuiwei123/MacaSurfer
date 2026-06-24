#!/bin/bash
set -e
set -x

WORK_DIR=/workspace/FreeSurfer/workspace/surf/msm
MOV_DIR=/workspace/FreeSurfer/
FIX_DIR=/MacaSurfer/global/templates/MEBRAIN/FreeSurfer

for Hemisphere in L R ; do
    if [ $Hemisphere = "L" ] ; then
      hemisphere="lh"
      Structure="CORTEX_LEFT"
    elif [ $Hemisphere = "R" ] ; then
      hemisphere="rh"
      Structure="CORTEX_RIGHT"
    fi


    mris_convert ${MOV_DIR}/workspace/surf/${hemisphere}.sphere ${WORK_DIR}/${Hemisphere}.mov.sphere.surf.gii
    mris_convert ${FIX_DIR}/workspace/surf/${hemisphere}.sphere ${WORK_DIR}/${Hemisphere}.fix.sphere.surf.gii


    ${CARET7DIR}/wb_command -surface-modify-sphere \
        ${WORK_DIR}/${Hemisphere}.mov.sphere.surf.gii \
        100 \
        ${WORK_DIR}/${Hemisphere}.mov.sphere.surf.gii

    ${CARET7DIR}/wb_command -surface-modify-sphere \
        ${WORK_DIR}/${Hemisphere}.fix.sphere.surf.gii \
        100 \
        ${WORK_DIR}/${Hemisphere}.fix.sphere.surf.gii

    
    mris_convert -c ${MOV_DIR}/workspace/surf/${hemisphere}.sulc ${MOV_DIR}/workspace/surf/${hemisphere}.white ${WORK_DIR}/${Hemisphere}.mov.sulc.shape.gii
    mris_convert -c ${FIX_DIR}/workspace/surf/${hemisphere}.sulc ${FIX_DIR}/workspace/surf/${hemisphere}.white ${WORK_DIR}/${Hemisphere}.fix.sulc.shape.gii



    cp ${MOV_DIR}/workspace/mri/middle/lh_medial_wall_binary.shape.gii ${WORK_DIR}/${Hemisphere}.MWmask.shape.gii
    ${CARET7DIR}/wb_command -metric-math "abs(var < 0.5)" ${WORK_DIR}/${Hemisphere}.MWmask.shape.gii -var var ${WORK_DIR}/${Hemisphere}.MWmask.shape.gii
	
    LD_LIBRARY=/usr/lib64 ${MSMBINDIR}/msm_centos_v3 \
        --conf=${MSMCONFIGDIR}/MSMSulcStrainFinalconf \
        --inmesh=${WORK_DIR}/${Hemisphere}.mov.sphere.surf.gii \
        --refmesh=${WORK_DIR}/${Hemisphere}.fix.sphere.surf.gii \
        --indata=${WORK_DIR}/${Hemisphere}.mov.sulc.shape.gii \
        --refdata=${WORK_DIR}/${Hemisphere}.fix.sulc.shape.gii \
        --inweight=${WORK_DIR}/${Hemisphere}.MWmask.shape.gii \
        --refweight=${FIX_DIR}/../${Hemisphere}.MWmask.shape.gii \
        --out=${WORK_DIR}/${Hemisphere}. \
        --verbose
        
    cp ${WORK_DIR}/${Hemisphere}.sphere.reg.surf.gii ${WORK_DIR}/${Hemisphere}.sphere.reg.back.surf.gii
    ${CARET7DIR}/wb_command -surface-modify-sphere \
        ${WORK_DIR}/${Hemisphere}.sphere.reg.surf.gii \
        1 \
        ${WORK_DIR}/${Hemisphere}.sphere.reg.surf.gii

    mris_convert ${WORK_DIR}/${Hemisphere}.sphere.reg.surf.gii ${MOV_DIR}/workspace/surf/${hemisphere}.sphere.reg
    mv ${WORK_DIR}/${Hemisphere}.sphere.reg.back.surf.gii ${WORK_DIR}/${Hemisphere}.sphere.reg.surf.gii

    # only register
    wb_command -label-resample \
        ${FIX_DIR}/../${Hemisphere}.aparc.label.gii \
        ${WORK_DIR}/${Hemisphere}.fix.sphere.surf.gii \
        ${WORK_DIR}/${Hemisphere}.sphere.reg.surf.gii \
        BARYCENTRIC \
        ${WORK_DIR}/${Hemisphere}.MBNA.only_reg.BARYCENTRIC.label.gii \
        -largest
    
    # freesurfer gcs
    mris_ca_label -sdir ${MOV_DIR} workspace lh ${MOV_DIR}/workspace/surf/${hemisphere}.sphere.reg ${FIX_DIR}/../lh.MBNA.gcs ${MOV_DIR}/workspace/label/${hemisphere}.aparc.annot
    mris_convert --annot ${MOV_DIR}/workspace/label/${hemisphere}.aparc.annot ${MOV_DIR}/workspace/surf/${hemisphere}.white ${WORK_DIR}/${Hemisphere}.MBNA.with_gcs.BARYCENTRIC.label.gii
done