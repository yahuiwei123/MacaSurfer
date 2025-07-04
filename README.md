# MacaSurfer Pipeline

A comprehensive pipeline for fast and accurate cortical surface reconstruction with macaque-specific parcellation

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://github.com/yahuiwei123/MacaSurfer/blob/main/doc/figures/overall.svg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig.1 (a) Overall workflow The workflow is organized into four stages. Green box: data preparation stage. Beige box: segmentation and enhancement stage. Blue box: surface initialization stage. Pink box: surface refinement stage. (b) Cortical and subcortical atlas Cortical parcellation on the MEBRAIN template surface obtained from MBNA atlas via MSM-based spherical registration; Atlas mapped from the D99 atlas on the NMT template to the MEBRAIN template volume.</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://github.com/yahuiwei123/MacaSurfer/blob/main/doc/figures/tgb.svg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig.3 (a) Cortical surface reconstructions from ten randomly selected subjects across all centers. Green contours indicate the white matter surface boundaries, and yellow contours indicate the pial surface boundaries. (b) High-quality cortical surface reconstruction was achieved even in subjects with brain tumors.</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://github.com/yahuiwei123/MacaSurfer/blob/main/doc/figures/multisites.svg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig.1 (a) Overall workflow The workflow is organized into four stages. Green box: data preparation stage. Beige box: segmentation and enhancement stage. Blue box: surface initialization stage. Pink box: surface refinement stage. (b) Cortical and subcortical atlas Cortical parcellation on the MEBRAIN template surface obtained from MBNA atlas via MSM-based spherical registration; Atlas mapped from the D99 atlas on the NMT template to the MEBRAIN template volume.</div>
</center>


## Installation

+ download the singularity image

```shell
singularity pull --arch amd64 library://weiyahui123/weiyahui123/macasurfer:v1.0
```

+ other necessary softwares
  + FreeSurfer >= 7.3.2

## Usage

+ Before running, you also need to create a folder named **RawData** inside each subject’s directory following the file structure described in **Results**. Place the T1 and T2 images in NIFTI format into this folder. Note that the T1 and T2 images must include ‘T1’ or ‘T2’ in their filenames to allow the program to identify their modality. It is also acceptable to have only T1 modality images.

+ One-click run

  ```shell
  /home/yhwei/software/singularity/bin/singularity exec --nv \
  	-B /path/to/your/freesurfer-7.3.2:/soft/freesurfer \
  	-B /path/to/your/fsl-6.0.5:/soft/fsl \
  	-B /path/to/your/subject_directory/:/workspace \
  	/path/to/your/macasurfer.sif \
      sh /MacaSurfer/Examples/recon-all.sh
  ```

+ Step-by-step run

  + Only prepare part

    ```shell
    /home/yhwei/software/singularity/bin/singularity exec --nv \
    	-B /path/to/your/freesurfer-7.3.2:/soft/freesurfer \
    	-B /path/to/your/fsl-6.0.5:/soft/fsl \
    	-B /path/to/your/subject_directory/:/workspace \
    	/path/to/your/macasurfer.sif \
        sh /MacaSurfer/Examples/prepare.sh
    ```

  + Only tissue segmentation and tissue-guided bias field correction part

    ```shell
    /home/yhwei/software/singularity/bin/singularity exec --nv \
    	-B /path/to/your/freesurfer-7.3.2:/soft/freesurfer \
    	-B /path/to/your/fsl-6.0.5:/soft/fsl \
    	-B /path/to/your/subject_directory/:/workspace \
    	/path/to/your/macasurfer.sif \
        sh /MacaSurfer/Examples/enhance.sh
    ```

  + Only surface reconstruction part

    ```shell
    /home/yhwei/software/singularity/bin/singularity exec --nv \
    	-B /path/to/your/freesurfer-7.3.2:/soft/freesurfer \
    	-B /path/to/your/fsl-6.0.5:/soft/fsl \
    	-B /path/to/your/subject_directory/:/workspace \
    	/path/to/your/macasurfer.sif \
        sh /MacaSurfer/Examples/surface.sh
    ```

  + Only post process part

    ```shell
    /home/yhwei/software/singularity/bin/singularity exec --nv \
    	-B /path/to/your/freesurfer-7.3.2:/soft/freesurfer \
    	-B /path/to/your/fsl-6.0.5:/soft/fsl \
    	-B /path/to/your/subject_directory/:/workspace \
    	/path/to/your/macasurfer.sif \
        sh /MacaSurfer/Examples/postprocess.sh
    ```

+ User-defined running scripts

  + Below is the complete content of the recon-all.sh script. Users can change command in this file and rebind it to the singularity image. Detailed tutorial will coming soon ...

  ```shell
  #!/bin/bash
  set -e
  
  HCPPIPEDIR=/MacaSurfer
  EnvironmentScript=${HCPPIPEDIR}/SetUpHCPPipelineNHP.sh
  . ${EnvironmentScript}
  
  # Input Variables
  StudyFolder=/
  Subject=workspace
  
  echo $StudyFolder
  echo $Subject
  
  if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
      echo "GPU detected. Running with GPU..."
      device="gpu"
  else
      echo "No GPU detected. Running on CPU..."
      device="cpu"
  fi
  
  #####
  ##### [step1]: Prepare stage
  #####
  sh ${HCPPIPEDIR}/Prepare/Prepare.sh "$StudyFolder"/"$Subject" 1
  sh ${HCPPIPEDIR}/Prepare/Prepare.sh "$StudyFolder"/"$Subject" 2
  sh ${HCPPIPEDIR}/Prepare/Prepare.sh "$StudyFolder"/"$Subject" 3
  
  ####
  #### [step2]: Enhance stage
  ####
  
  SPECIES="Macaque"
  segment_model=${MODEL_PATH}
  skull_strip=Yes # Yes
  enhance_contrast=No # Yes
  bfc=n4 # sqrt
  
  # auto detect T2
  if find "$StudyFolder"/"$Subject"/RawData/ -type f -name "*T2*.nii.gz" | grep -q .; then
      t2='--t2'
  else
      t2=''
  fi
  
  if [[ $t2 = '' ]]; then
      fake_t2="--fake_t2"
  fi
  
  if [[ $skull_strip = "Yes" ]]; then
      skull_strip="--skull_strip"
  fi
  
  if [[ $enhance_contrast = "Yes" ]]; then
      enhance_contrast="--enhance_contrast"
  fi
  
  OrigPath=${StudyFolder}/${Subject}/Prepare
  SubjectPath=${StudyFolder}/${Subject}
  
  sh ${HCPPIPEDIR}/Enhance/PreProcessNHP.sh \
      --in_dir=${OrigPath} \
      --out_dir=${SubjectPath} \
      --model=${segment_model} \
      --bfc=${bfc} \
      --step=1 \
      --device=$device \
      $t2 \
      --select_valid \
      $skull_strip \
      $enhance_contrast \
      --correct_orient \
      $fake_t2 \
      
  
  sh ${HCPPIPEDIR}/Enhance/PreProcessNHP.sh \
      --in_dir=${OrigPath} \
      --out_dir=${SubjectPath} \
      --model=${segment_model} \
      --bfc=${bfc} \
      --step=2 \
      --device=$device \
      $t2 \
      --select_valid \
      $skull_strip \
      $enhance_contrast \
      --correct_orient \
      $fake_t2 \
  
  sh ${HCPPIPEDIR}/Enhance/PreProcessNHP.sh \
      --in_dir=${OrigPath} \
      --out_dir=${SubjectPath} \
      --model=${segment_model} \
      --bfc=${bfc} \
      --step=3 \
      --device=$device \
      $t2 \
      --select_valid \
      $skull_strip \
      $enhance_contrast \
      --correct_orient \
      $fake_t2 \
  
  sh ${HCPPIPEDIR}/Enhance/PreProcessNHP.sh \
      --in_dir=${OrigPath} \
      --out_dir=${SubjectPath} \
      --model=${segment_model} \
      --bfc=${bfc} \
      --step=4 \
      --device=$device \
      $t2 \
      --select_valid \
      $skull_strip \
      $enhance_contrast \
      --correct_orient \
      $fake_t2 \
  
  
  
  ######
  ###### [step3]: FreeSurfer stage
  ######
  
  SPECIES="Macaque"
  PreprocessDIR="${StudyFolder}/${Subject}/Enhance"
  SubjectDIR="${StudyFolder}/${Subject}/FreeSurfer"
  SubjectID="${Subject}"
  T1wImage="${StudyFolder}/${Subject}/Enhance/T1w/T1w_acpc_iso.nii.gz" #T1w FreeSurfer Input (Full Resolution)
  T1wImageBrain="${StudyFolder}/${Subject}/Enhance/T1w/T1w_acpc_iso.nii.gz" #T1w FreeSurfer Input (Full Resolution)
  T2wImage="${StudyFolder}/${Subject}/Enhance/T1w/T2w_acpc_iso.nii.gz" #T2w FreeSurfer Input (Full Resolution)
  FSLinearTransform="${HCPPIPEDIR_Templates}/fs_xfms/eye.xfm" #Identity
  T2wFlag="${T2wFlag:=T2w}" # T2w, FLAIR or NONE. Default is T2w
  FullyDeep=0
  
  mkdir -p $SubjectDIR
  
  RunMode=0
  sh ${HCPPIPEDIR}/FreeSurfer/FreeSurferPipelineNHP.sh \
      --preprocessDIR=$PreprocessDIR \
      --subject="$Subject" \
      --subjectDIR="$SubjectDIR" \
      --t1="$T1wImage" \
      --t1brain="$T1wImageBrain" \
      --t2="$T2wImage" \
      --fslinear="$FSLinearTransform" \
      --gcadir="$GCAdir" \
      --rescaletrans="$RescaleVolumeTransform" \
      --asegedit="$AsegEdit" \
      --controlpoints="$ControlPoints" \
      --wmedit="$WmEdit" \
      --t2wflag="$T2wFlag" \
      --species="$SPECIES" \
      --fullydeep="$FullyDeep" \
      --runmode="$RunMode" \
      --seed="$Seed" \
      --printcom="$PRINTCOM" \
      --firstpass="$FirstPass"
  
  
  ######
  ###### [step4]: Postprocess
  ######
  
  if [ -e "${StudyFolder}/${Subject}/Results/" ]; then
      rm -rf "${StudyFolder}/${Subject}/Results/"
  fi
  
  sh ${HCPPIPEDIR}/PostProcess/PostProcessNHP.sh \
      --path=${StudyFolder} \
      --subject=${Subject}
  ```

  

## Results

+ Directory structure

  ```shell
  /path/to/your/subject/
  ├── Enhance
  │   ├── MEBRAIN
  │   │   ├── mebrain_T1w_04mm_brain_LIA.nii.gz
  │   │   ├── mebrain_T1w_04mm_LIA.nii.gz
  │   │   ├── mebrain_T1w_1mm_brain_LIA.nii.gz
  │   │   ├── mebrain_T2w_04mm_brain_LIA.nii.gz
  │   │   ├── orig2temp
  │   │   │   ├── check.nii.gz
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_bfc_affine0GenericAffine.mat
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_bfc_affine.nii.gz
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_bfc_init0GenericAffine.mat
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_bfc_init.nii.gz
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_bfc_linear.mat
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_bfc_linear.mat.fsl
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_bfc_linear.mat.ras
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_bfc.nii.gz
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_bfc_nonlinear0GenericAffine.mat
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_bfc_nonlinear0GenericAffine.mat.ras
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_bfc_nonlinear0InverseAffine.mat
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_bfc_nonlinear1InverseWarp.nii.gz
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_bfc_nonlinear1Warp.nii.gz
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_bfc_nonlinearInit.nii.gz
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_bfc_nonlinearRes.nii.gz
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_bfc_rigid0GenericAffine.mat
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_bfc_rigid.nii.gz
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_conform_affine0GenericAffine.mat
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_conform_affine.nii.gz
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_conform_init0GenericAffine.mat
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_conform_init.nii.gz
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_conform_linear.mat
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_conform_linear.mat.fsl
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_conform_linear.mat.ras
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_conform.nii.gz
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_conform_nonlinear0GenericAffine.mat
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_conform_nonlinear0GenericAffine.mat.ras
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_conform_nonlinear0InverseAffine.mat
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_conform_nonlinear1InverseWarp.nii.gz
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_conform_nonlinear1Warp.nii.gz
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_conform_nonlinearInit.nii.gz
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_conform_nonlinearRes.nii.gz
  │   │   │   ├── mebrain_T1w_04mm_brain_LIA_to_T1w_conform_rigid0GenericAffine.mat
  │   │   │   └── mebrain_T1w_04mm_brain_LIA_to_T1w_conform_rigid.nii.gz
  │   │   ├── T1w.nii.gz
  │   │   ├── T1w_restore_brain.nii.gz
  │   │   ├── T1w_restore.nii.gz
  │   │   └── xfms
  │   │       ├── final.nii.gz
  │   │       ├── log.txt
  │   │       ├── NonlinearRegJacobians.nii.gz
  │   │       ├── orig2standardLinear.mat
  │   │       ├── orig2standard.nii.gz
  │   │       ├── qa.txt
  │   │       └── standard2orig.nii.gz
  │   ├── nBEST
  │   │   ├── brain_cerebellum_brainstem_mask
  │   │   │   └── T1w_conform.nii.gz
  │   │   ├── brain_cerebrum
  │   │   │   └── T1w_conform.nii.gz
  │   │   ├── brain_cerebrum_mask
  │   │   │   └── T1w_conform.nii.gz
  │   │   ├── brain_img
  │   │   │   └── T1w_conform.nii.gz
  │   │   ├── brain_mask
  │   │   │   └── T1w_conform.nii.gz
  │   │   └── brain_tissue
  │   │       ├── plans.pkl
  │   │       └── T1w_conform.nii.gz
  │   ├── T1w
  │   │   ├── aseg_wm.nii.gz
  │   │   ├── BiasField_acpc_dc.nii.gz
  │   │   ├── gm_acpc_compliment_iso.nii.gz
  │   │   ├── gm_acpc_compliment.nii.gz
  │   │   ├── gm_compliment.nii.gz
  │   │   ├── T1w_acpc_aseg_iso.nii.gz
  │   │   ├── T1w_acpc_aseg.nii.gz
  │   │   ├── T1w_acpc_brainmask_iso.nii.gz
  │   │   ├── T1w_acpc_brainmask.nii.gz
  │   │   ├── T1w_acpc_cerebrum_iso.nii.gz
  │   │   ├── T1w_acpc_cerebrum.nii.gz
  │   │   ├── T1w_acpc_head.nii.gz
  │   │   ├── T1w_acpc_iso_1mm.nii.gz
  │   │   ├── T1w_acpc_iso.nii.gz
  │   │   ├── T1w_acpc_nbest_iso.nii.gz
  │   │   ├── T1w_acpc_nbest.nii.gz
  │   │   ├── T1w_acpc.nii.gz
  │   │   ├── T1w_acpc_pial_iso.nii.gz
  │   │   ├── T1w_acpc_pial.nii.gz
  │   │   ├── T1w_acpc_resample.nii.gz
  │   │   ├── T1w_acpc_vessel_iso.nii.gz
  │   │   ├── T1w_acpc_vessel.nii.gz
  │   │   ├── T1w_acpc_white_iso.nii.gz
  │   │   ├── T1w_acpc_white.nii.gz
  │   │   ├── T1w_aseg_04mm.nii.gz
  │   │   ├── T1w_aseg.nii.gz
  │   │   ├── T1w_bfc_04mm.nii.gz
  │   │   ├── T1w_bfc_1mm.nii.gz
  │   │   ├── T1w_bfc.nii.gz
  │   │   ├── T1w_bfc_pial_04mm.nii.gz
  │   │   ├── T1w_bfc_pial_bias_04mm.nii.gz
  │   │   ├── T1w_bfc_pial_bias.nii.gz
  │   │   ├── T1w_bfc_pial.nii.gz
  │   │   ├── T1w_bfc_vessel.nii.gz
  │   │   ├── T1w_bfc_white_04mm.nii.gz
  │   │   ├── T1w_bfc_white_bias_04mm.nii.gz
  │   │   ├── T1w_bfc_white_bias.nii.gz
  │   │   ├── T1w_bfc_white.nii.gz
  │   │   ├── T1w_brainmask.nii.gz
  │   │   ├── T1w_brain.nii.gz
  │   │   ├── T1w_cerebellum_brainstem_04mm.nii.gz
  │   │   ├── T1w_cerebellum_brainstem.nii.gz
  │   │   ├── T1w_cerebrum.nii.gz
  │   │   ├── T1w_conform_04mm.nii.gz
  │   │   ├── T1w_conform_1mm.nii.gz
  │   │   ├── T1w_conform_back.nii.gz
  │   │   ├── T1w_conform_brainmask.nii.gz
  │   │   ├── T1w_conform_head.nii.gz
  │   │   ├── T1w_conform.nii.gz
  │   │   ├── T1w_csf_mask.nii.gz
  │   │   ├── T1w_head.nii.gz
  │   │   ├── T1w_n4_weight.nii.gz
  │   │   ├── T1w_nbest.nii.gz
  │   │   ├── T1w_tissue_04mm.nii.gz
  │   │   ├── T1w_valid.nii.gz
  │   │   ├── T1w_vessel_mask.nii.gz
  │   │   ├── T2w_acpc_iso_1mm.nii.gz
  │   │   ├── T2w_acpc_iso.nii.gz
  │   │   ├── T2w_acpc.nii.gz
  │   │   ├── T2w_acpc_resample.nii.gz
  │   │   ├── T2w_acpc_tgb_iso.nii.gz
  │   │   ├── T2w_acpc_tgb.nii.gz
  │   │   ├── T2w_acpc_vessel_iso.nii.gz
  │   │   ├── T2w_acpc_vessel.nii.gz
  │   │   ├── T2w_bfc.nii.gz
  │   │   ├── T2w_bfc_tgb.nii.gz
  │   │   ├── T2w_bfc_vessel.nii.gz
  │   │   ├── wm_acpc_compliment1_iso.nii.gz
  │   │   ├── wm_acpc_compliment1.nii.gz
  │   │   ├── wm_acpc_compliment2_iso.nii.gz
  │   │   ├── wm_acpc_compliment2.nii.gz
  │   │   ├── wm_compliment1.nii.gz
  │   │   ├── wm_compliment2.nii.gz
  │   │   └── xfms
  │   │       ├── acpc_04mm.dat
  │   │       ├── acpc_04mm.dat~
  │   │       ├── acpc.dat
  │   │       ├── acpc.dat~
  │   │       ├── acpc.mat
  │   │       ├── identity.mat
  │   │       ├── scaling.mat
  │   │       ├── T1w2T2w.mat
  │   │       └── T2w2T1w.mat
  │   └── T2w
  │       └── xfms
  │           └── T2w2T1w.mat
  ├── FreeSurfer
  │   └── workspace
  │       ├── label
  │       │   ├── aparc.annot.ctab
  │       │   ├── lh.aparc.annot
  │       │   ├── lh.cortex.deformed.label
  │       │   ├── lh.cortex.label
  │       │   ├── lh.cortex.prehires.label
  │       │   ├── lh.nofix.cortex.label
  │       │   ├── rh.aparc.annot
  │       │   ├── rh.cortex.deformed.label
  │       │   ├── rh.cortex.label
  │       │   ├── rh.cortex.prehires.label
  │       │   └── rh.nofix.cortex.label
  │       ├── mri
  │       │   ├── amygdala_mask.mgz
  │       │   ├── amygdala_mask.nii.gz
  │       │   ├── aparc+aseg.mgz
  │       │   ├── aseg.auto.mgz
  │       │   ├── aseg.auto.nii.gz
  │       │   ├── aseg+claustrum.mgz
  │       │   ├── aseg+claustrum.nii.gz
  │       │   ├── aseg+claustrum.orig.nii.gz
  │       │   ├── aseg.mgz
  │       │   ├── aseg.nii.gz
  │       │   ├── aseg.orig.mgz
  │       │   ├── aseg.orig.nii.gz
  │       │   ├── aseg.presurf.hypos.mgz
  │       │   ├── aseg.presurf.mgz
  │       │   ├── brain.finalsurfs.mgz
  │       │   ├── brain.finalsurfs.nii.gz
  │       │   ├── brain.finalsurfs.orig.mgz
  │       │   ├── brainmask.mgz
  │       │   ├── brain.mgz
  │       │   ├── claustrum2putamen.lh.nii.gz
  │       │   ├── claustrum2putamen.rh.nii.gz
  │       │   ├── csf_mask_ero.nii.gz
  │       │   ├── csf_mask.nii.gz
  │       │   ├── csf_mask.orig.nii.gz
  │       │   ├── dilribbon_inv.nii.gz
  │       │   ├── filled.mgz
  │       │   ├── filled.nii.gz
  │       │   ├── filled.orig.mgz
  │       │   ├── filled.orig.nii.gz
  │       │   ├── gm_mask.nii.gz
  │       │   ├── gm.orig.nii.gz
  │       │   ├── hippocampus_mask.mgz
  │       │   ├── hippocampus_mask.nii.gz
  │       │   ├── lh.ribbon.mgz
  │       │   ├── lh.ribbon.nii.gz
  │       │   ├── middle
  │       │   │   ├── Left-Hemi-Dil.nii.gz
  │       │   │   ├── Left-Hemi.nii.gz
  │       │   │   ├── Middle-Wall-Eliminate.nii.gz
  │       │   │   ├── Middle-Wall-Mask.nii.gz
  │       │   │   ├── Middle-Wall-Mask-Smooth.nii.gz
  │       │   │   ├── Right-Complement.nii.gz
  │       │   │   ├── Right-Hemi-Dil.nii.gz
  │       │   │   └── Right-Hemi.nii.gz
  │       │   ├── nbest.cerebrum.mgz
  │       │   ├── nbest.cerebrum.nii.gz
  │       │   ├── nbest.cerebrum.orig.nii.gz
  │       │   ├── nbest.tissue.mgz
  │       │   ├── nbest.tissue.nii.gz
  │       │   ├── norm.mgz
  │       │   ├── norm.nii.gz
  │       │   ├── norm.nouse.nii.gz
  │       │   ├── nu.mgz
  │       │   ├── orig
  │       │   │   └── 001.mgz
  │       │   ├── orig.mgz
  │       │   ├── orig.nii.gz
  │       │   ├── rawavg.mgz
  │       │   ├── rawavg.nii.gz
  │       │   ├── rh.ribbon.mgz
  │       │   ├── rh.ribbon.nii.gz
  │       │   ├── ribbon_inv.nii.gz
  │       │   ├── ribbon.mgz
  │       │   ├── ribbon.nii.gz
  │       │   ├── ribbon_s5.nii.gz
  │       │   ├── seg
  │       │   │   ├── above_wm.nii.gz
  │       │   │   ├── accumbens.nii.gz
  │       │   │   ├── aseg.nii.gz
  │       │   │   ├── aseg_wo_gwm.nii.gz
  │       │   │   ├── below_wm.nii.gz
  │       │   │   ├── gm_nbest.nii.gz
  │       │   │   ├── left_aseg.nii.gz
  │       │   │   ├── left_gm_nbest.nii.gz
  │       │   │   ├── left_wm_nbest.nii.gz
  │       │   │   ├── pallidum.nii.gz
  │       │   │   ├── putamen.nii.gz
  │       │   │   ├── right_aseg.nii.gz
  │       │   │   ├── right_gm_nbest.nii.gz
  │       │   │   ├── right_wm_nbest.nii.gz
  │       │   │   ├── substancia.nii.gz
  │       │   │   ├── thalamus.nii.gz
  │       │   │   └── wm_nbest.nii.gz
  │       │   ├── surface.defects.mgz
  │       │   ├── T1w_hires.greynorm.mgz
  │       │   ├── T1w_hires.greynorm.nii.gz
  │       │   ├── T1w_hires.greynorm_ribbon.nii.gz
  │       │   ├── T1w_hires.nii.gz
  │       │   ├── T1w_hires.norm.mgz
  │       │   ├── T1w_hires.norm.nii.gz
  │       │   ├── T1w_hires.norm.one.mgz
  │       │   ├── T1w_hires.norm.one.nii.gz
  │       │   ├── T1w_hires.norm.two_grey_myelin.nii.gz
  │       │   ├── T1w_hires.norm.two.mgz
  │       │   ├── T1w_hires.norm.two.nii.gz
  │       │   ├── T1w_hires.norm.two_ribbon_myelin.nii.gz
  │       │   ├── T1w_hires.norm.two_ribbon.nii.gz
  │       │   ├── T1w_hires_pial.nii.gz
  │       │   ├── T1w_hires_white.mgz
  │       │   ├── T1w_hires_white.nii.gz
  │       │   ├── T1wMulT2w_hires.nii.gz
  │       │   ├── T2w_hires.nii.gz
  │       │   ├── T2w_hires.norm.mgz
  │       │   ├── T2w_hires.norm.nii.gz
  │       │   ├── T2w_hires_pial.nii.gz
  │       │   ├── transforms
  │       │   │   ├── bak
  │       │   │   ├── eye.dat
  │       │   │   ├── T2wtoT1w.dat
  │       │   │   ├── T2wtoT1w.dat~
  │       │   │   ├── T2wtoT1w.dat.mincost
  │       │   │   ├── T2wtoT1w.dat.param
  │       │   │   ├── T2wtoT1w.dat.sum
  │       │   │   ├── T2wtoT1w.log
  │       │   │   ├── T2wtoT1w.mat
  │       │   │   └── talairach.xfm
  │       │   ├── ventricleIDC.nii.gz
  │       │   ├── white.nii.gz
  │       │   ├── wm.asegedit.mgz
  │       │   ├── wm.asegedit.nii.gz
  │       │   ├── wm.compliment1.nii.gz
  │       │   ├── wm.compliment2.orig.nii.gz
  │       │   ├── wm_mask.nii.gz
  │       │   ├── wm.mgz
  │       │   ├── wm.nii.gz
  │       │   ├── wm.orig.mgz
  │       │   ├── wm.orig.nii.gz
  │       │   └── wmparc.mgz
  │       ├── scripts
  │       │   ├── build-stamp.txt
  │       │   ├── defect2seg.log
  │       │   ├── lastcall.build-stamp.txt
  │       │   ├── patchdir.txt
  │       │   ├── recon-all.cmd
  │       │   ├── recon-all.done
  │       │   ├── recon-all.env
  │       │   ├── recon-all.env.bak
  │       │   ├── recon-all.local-copy
  │       │   ├── recon-all.log
  │       │   ├── recon-all-status.log
  │       │   ├── recon-config.yaml
  │       │   └── unknown-args.txt
  │       ├── stats
  │       │   ├── brainvol.stats
  │       │   ├── lh.aparc.pial.stats
  │       │   ├── lh.aparc.stats
  │       │   ├── rh.aparc.pial.stats
  │       │   └── rh.aparc.stats
  │       ├── surf
  │       │   ├── autodet.gw.stats.lh.dat
  │       │   ├── autodet.gw.stats.rh.dat
  │       │   ├── lh.area
  │       │   ├── lh.area.deformed
  │       │   ├── lh.area.mid
  │       │   ├── lh.area.pial
  │       │   ├── lh.area.pial.deformed
  │       │   ├── lh.area.pial.T2
  │       │   ├── lh.area.pial.T2.two
  │       │   ├── lh.area.prehires
  │       │   ├── lh.curv
  │       │   ├── lh.curv.deformed
  │       │   ├── lh.curv.pial
  │       │   ├── lh.curv.pial.deformed
  │       │   ├── lh.curv.pial.T2
  │       │   ├── lh.curv.pial.T2.two
  │       │   ├── lh.curv.prehires
  │       │   ├── lh.defect_borders
  │       │   ├── lh.defect_chull
  │       │   ├── lh.defect_labels
  │       │   ├── lh.defects.pointset
  │       │   ├── lh.inflated
  │       │   ├── lh.inflated.10
  │       │   ├── lh.inflated.H
  │       │   ├── lh.inflated.K
  │       │   ├── lh.inflated.nofix
  │       │   ├── lh.inflated.nofix.10
  │       │   ├── lh.jacobian_white
  │       │   ├── lh.orig
  │       │   ├── lh.orig.nofix
  │       │   ├── lh.orig.premesh
  │       │   ├── lh.pial
  │       │   ├── lh.pial.deformed
  │       │   ├── lh.pial.deformed.out
  │       │   ├── lh.pial.nii.gz
  │       │   ├── lh.pial.one
  │       │   ├── lh.pial.preT2
  │       │   ├── lh.pial.preT2.two
  │       │   ├── lh.pial.surf.gii
  │       │   ├── lh.pial.T2
  │       │   ├── lh.pial.T2.two
  │       │   ├── lh.qsphere.nofix
  │       │   ├── lh.smoothwm
  │       │   ├── lh.smoothwm.nofix
  │       │   ├── lh.sphere
  │       │   ├── lh.sphere.reg
  │       │   ├── lh.sulc
  │       │   ├── lh.thickness
  │       │   ├── lh.thickness.deformed
  │       │   ├── lh.thickness.preT2
  │       │   ├── lh.thickness.T2
  │       │   ├── lh.thickness.T2.two
  │       │   ├── lh.volume
  │       │   ├── lh.white
  │       │   ├── lh.white.deformed
  │       │   ├── lh.white.deformed.out
  │       │   ├── lh.white.nii.gz
  │       │   ├── lh.white.preaparc
  │       │   ├── lh.white.prehires
  │       │   ├── lh.white.surf.gii
  │       │   ├── rh.area
  │       │   ├── rh.area.deformed
  │       │   ├── rh.area.mid
  │       │   ├── rh.area.pial
  │       │   ├── rh.area.pial.deformed
  │       │   ├── rh.area.pial.T2
  │       │   ├── rh.area.pial.T2.two
  │       │   ├── rh.area.prehires
  │       │   ├── rh.curv
  │       │   ├── rh.curv.deformed
  │       │   ├── rh.curv.pial
  │       │   ├── rh.curv.pial.deformed
  │       │   ├── rh.curv.pial.T2
  │       │   ├── rh.curv.pial.T2.two
  │       │   ├── rh.curv.prehires
  │       │   ├── rh.defect_borders
  │       │   ├── rh.defect_chull
  │       │   ├── rh.defect_labels
  │       │   ├── rh.defects.pointset
  │       │   ├── rh.inflated
  │       │   ├── rh.inflated.10
  │       │   ├── rh.inflated.H
  │       │   ├── rh.inflated.K
  │       │   ├── rh.inflated.nofix
  │       │   ├── rh.inflated.nofix.10
  │       │   ├── rh.jacobian_white
  │       │   ├── rh.orig
  │       │   ├── rh.orig.nofix
  │       │   ├── rh.orig.premesh
  │       │   ├── rh.pial
  │       │   ├── rh.pial.deformed
  │       │   ├── rh.pial.deformed.out
  │       │   ├── rh.pial.nii.gz
  │       │   ├── rh.pial.one
  │       │   ├── rh.pial.preT2
  │       │   ├── rh.pial.preT2.two
  │       │   ├── rh.pial.surf.gii
  │       │   ├── rh.pial.T2
  │       │   ├── rh.pial.T2.two
  │       │   ├── rh.qsphere.nofix
  │       │   ├── rh.smoothwm
  │       │   ├── rh.smoothwm.nofix
  │       │   ├── rh.sphere
  │       │   ├── rh.sphere.reg
  │       │   ├── rh.sulc
  │       │   ├── rh.thickness
  │       │   ├── rh.thickness.deformed
  │       │   ├── rh.thickness.preT2
  │       │   ├── rh.thickness.T2
  │       │   ├── rh.thickness.T2.two
  │       │   ├── rh.volume
  │       │   ├── rh.white
  │       │   ├── rh.white.deformed
  │       │   ├── rh.white.deformed.out
  │       │   ├── rh.white.nii.gz
  │       │   ├── rh.white.preaparc
  │       │   ├── rh.white.prehires
  │       │   └── rh.white.surf.gii
  │       ├── tmp
  │       ├── touch
  │       │   ├── cortical_ribbon.touch
  │       │   ├── lh.aparcstats.touch
  │       │   ├── lh.aparc.touch
  │       │   ├── lh.inflate2.touch
  │       │   ├── lh.jacobian_white.touch
  │       │   ├── lh.qsphere.touch
  │       │   ├── lh.smoothwm1.touch
  │       │   ├── lh.smoothwm2.touch
  │       │   ├── lh.sphmorph.touch
  │       │   ├── lh.tessellate.touch
  │       │   ├── lh.topofix.touch
  │       │   ├── rh.aparcstats.touch
  │       │   ├── rh.aparc.touch
  │       │   ├── rh.inflate2.touch
  │       │   ├── rh.jacobian_white.touch
  │       │   ├── rh.qsphere.touch
  │       │   ├── rh.smoothwm1.touch
  │       │   ├── rh.smoothwm2.touch
  │       │   ├── rh.sphmorph.touch
  │       │   ├── rh.tessellate.touch
  │       │   └── rh.topofix.touch
  │       └── trash
  ├── Prepare
  │   ├── avg_brainmask.jpg
  │   ├── data_config.yaml
  │   ├── data_figure.jpg
  │   ├── t1_brainmask.jpg
  │   ├── T1w
  │   │   └── runs
  │   │       └── sub-032222_ses-001_run_1_T1w
  │   │           ├── brainmask.nii.gz
  │   │           ├── brainmask_reorient.nii.gz
  │   │           ├── brain.nii.gz
  │   │           ├── brain_reorient.nii.gz
  │   │           ├── head.nii.gz
  │   │           ├── head_reorient.nii.gz
  │   │           ├── nBEST
  │   │           └── reorient
  │   │               ├── mov_06mm.nii.gz
  │   │               ├── reo.mat
  │   │               ├── reo.nii.gz
  │   │               └── t1_template_06mm.nii.gz
  │   ├── T1w_brainmask.nii.gz
  │   ├── T1w_brain.nii.gz
  │   ├── T1w_head.nii.gz
  │   └── T2w
  │       └── runs
  ├── RawData
  │   └── sub-032222_ses-001_run_1_T1w.nii.gz
  └── Results
    ├── ACPC
    │   ├── fsaverage_LR10k
    │   │   ├── 10k_fs_LR.wb.spec
    │   │   ├── L.inflated.10k_fs_LR.surf.gii
    │   │   ├── L.midthickness.10k_fs_LR.surf.gii
    │   │   ├── L.midthickness_va.10k_fs_LR.shape.gii
    │   │   ├── L.pial.10k_fs_LR.surf.gii
    │   │   ├── L.very_inflated.10k_fs_LR.surf.gii
    │   │   ├── L.white.10k_fs_LR.surf.gii
    │   │   ├── midthickness_va.10k_fs_LR.dscalar.nii
    │   │   ├── midthickness_va_norm.10k_fs_LR.dscalar.nii
    │   │   ├── R.inflated.10k_fs_LR.surf.gii
    │   │   ├── R.midthickness.10k_fs_LR.surf.gii
    │   │   ├── R.midthickness_va.10k_fs_LR.shape.gii
    │   │   ├── R.pial.10k_fs_LR.surf.gii
    │   │   ├── R.very_inflated.10k_fs_LR.surf.gii
    │   │   └── R.white.10k_fs_LR.surf.gii
    │   ├── fsaverage_LR32k
    │   │   ├── 32k_fs_LR.wb.spec
    │   │   ├── L.inflated.32k_fs_LR.surf.gii
    │   │   ├── L.midthickness.32k_fs_LR.surf.gii
    │   │   ├── L.midthickness_va.32k_fs_LR.shape.gii
    │   │   ├── L.pial.32k_fs_LR.surf.gii
    │   │   ├── L.very_inflated.32k_fs_LR.surf.gii
    │   │   ├── L.white.32k_fs_LR.surf.gii
    │   │   ├── midthickness_va.32k_fs_LR.dscalar.nii
    │   │   ├── midthickness_va_norm.32k_fs_LR.dscalar.nii
    │   │   ├── R.inflated.32k_fs_LR.surf.gii
    │   │   ├── R.midthickness.32k_fs_LR.surf.gii
    │   │   ├── R.midthickness_va.32k_fs_LR.shape.gii
    │   │   ├── R.pial.32k_fs_LR.surf.gii
    │   │   ├── R.very_inflated.32k_fs_LR.surf.gii
    │   │   └── R.white.32k_fs_LR.surf.gii
    │   ├── mri
    │   │   ├── L.pial.nii.gz
    │   │   ├── L.ribbon.nii.gz
    │   │   ├── L.white.nii.gz
    │   │   ├── ribbon.nii.gz
    │   │   ├── R.pial.nii.gz
    │   │   ├── R.ribbon.nii.gz
    │   │   ├── R.white.nii.gz
    │   │   ├── T1w_acpc_brain.nii.gz
    │   │   ├── T1w_acpc_resample.nii.gz
    │   │   └── T2w_acpc_brain.nii.gz
    │   ├── Native
    │   │   ├── L.aparc.label.gii
    │   │   ├── L.area.pial.shape.gii
    │   │   ├── L.area.white.shape.gii
    │   │   ├── L.curvature.shape.gii
    │   │   ├── lh.aparc.label.gii
    │   │   ├── L.inflated.surf.gii
    │   │   ├── L_level1_metrics.xlsx
    │   │   ├── L_level2_metrics.xlsx
    │   │   ├── L.midthickness.surf.gii
    │   │   ├── L.pial.surf.gii
    │   │   ├── L.roi.shape.gii
    │   │   ├── L.sphere.reg.surf.gii
    │   │   ├── L.sphere.surf.gii
    │   │   ├── L.sulc.shape.gii
    │   │   ├── L.thickness.shape.gii
    │   │   ├── L_total_metrics.xlsx
    │   │   ├── L.very_inflated.surf.gii
    │   │   ├── L.white.surf.gii
    │   │   ├── native.wb.spec
    │   │   ├── R.aparc.label.gii
    │   │   ├── R.area.pial.shape.gii
    │   │   ├── R.area.white.shape.gii
    │   │   ├── R.curvature.shape.gii
    │   │   ├── rh.aparc.label.gii
    │   │   ├── R.inflated.surf.gii
    │   │   ├── R_level1_metrics.xlsx
    │   │   ├── R_level2_metrics.xlsx
    │   │   ├── R.midthickness.surf.gii
    │   │   ├── R.pial.surf.gii
    │   │   ├── R.roi.shape.gii
    │   │   ├── R.sphere.reg.surf.gii
    │   │   ├── R.sphere.surf.gii
    │   │   ├── R.sulc.shape.gii
    │   │   ├── R.thickness.shape.gii
    │   │   ├── R_total_metrics.xlsx
    │   │   ├── R.very_inflated.surf.gii
    │   │   └── R.white.surf.gii
    │   └── wb.spec
    ├── Atlas
    │   ├── 164k_fs_LR.wb.spec
    │   ├── ArealDistortion_FS.164k_fs_LR.dscalar.nii
    │   ├── ArealDistortion_MSMSulc.164k_fs_LR.dscalar.nii
    │   ├── curvature.164k_fs_LR.dscalar.nii
    │   ├── EdgeDistortion_FS.164k_fs_LR.dscalar.nii
    │   ├── EdgeDistortion_MSMSulc.164k_fs_LR.dscalar.nii
    │   ├── fsaverage
    │   │   ├── L.def_sphere.164k_fs_L.surf.gii
    │   │   ├── L.sphere.164k_fs_L.surf.gii
    │   │   ├── R.def_sphere.164k_fs_R.surf.gii
    │   │   └── R.sphere.164k_fs_R.surf.gii
    │   ├── fsaverage_LR10k
    │   │   ├── 10k_fs_LR.wb.spec
    │   │   ├── ArealDistortion_FS.10k_fs_LR.dscalar.nii
    │   │   ├── ArealDistortion_MSMSulc.10k_fs_LR.dscalar.nii
    │   │   ├── curvature.10k_fs_LR.dscalar.nii
    │   │   ├── EdgeDistortion_FS.10k_fs_LR.dscalar.nii
    │   │   ├── EdgeDistortion_MSMSulc.10k_fs_LR.dscalar.nii
    │   │   ├── L.ArealDistortion_FS.10k_fs_LR.shape.gii
    │   │   ├── L.ArealDistortion_MSMSulc.10k_fs_LR.shape.gii
    │   │   ├── L.atlasroi.10k_fs_LR.shape.gii
    │   │   ├── L.curvature.10k_fs_LR.shape.gii
    │   │   ├── L.EdgeDistortion_FS.10k_fs_LR.shape.gii
    │   │   ├── L.EdgeDistortion_MSMSulc.10k_fs_LR.shape.gii
    │   │   ├── L.flat.10k_fs_LR.surf.gii
    │   │   ├── L.inflated.10k_fs_LR.surf.gii
    │   │   ├── L.midthickness.10k_fs_LR.surf.gii
    │   │   ├── L.pial.10k_fs_LR.surf.gii
    │   │   ├── L.sphere.10k_fs_LR.surf.gii
    │   │   ├── L.StrainJ_FS.10k_fs_LR.shape.gii
    │   │   ├── L.StrainJ_MSMSulc.10k_fs_LR.shape.gii
    │   │   ├── L.StrainR_FS.10k_fs_LR.shape.gii
    │   │   ├── L.StrainR_MSMSulc.10k_fs_LR.shape.gii
    │   │   ├── L.sulc.10k_fs_LR.shape.gii
    │   │   ├── L.thickness.10k_fs_LR.shape.gii
    │   │   ├── L.very_inflated.10k_fs_LR.surf.gii
    │   │   ├── L.white.10k_fs_LR.surf.gii
    │   │   ├── R.ArealDistortion_FS.10k_fs_LR.shape.gii
    │   │   ├── R.ArealDistortion_MSMSulc.10k_fs_LR.shape.gii
    │   │   ├── R.atlasroi.10k_fs_LR.shape.gii
    │   │   ├── R.curvature.10k_fs_LR.shape.gii
    │   │   ├── R.EdgeDistortion_FS.10k_fs_LR.shape.gii
    │   │   ├── R.EdgeDistortion_MSMSulc.10k_fs_LR.shape.gii
    │   │   ├── R.flat.10k_fs_LR.surf.gii
    │   │   ├── R.inflated.10k_fs_LR.surf.gii
    │   │   ├── R.midthickness.10k_fs_LR.surf.gii
    │   │   ├── R.pial.10k_fs_LR.surf.gii
    │   │   ├── R.sphere.10k_fs_LR.surf.gii
    │   │   ├── R.StrainJ_FS.10k_fs_LR.shape.gii
    │   │   ├── R.StrainJ_MSMSulc.10k_fs_LR.shape.gii
    │   │   ├── R.StrainR_FS.10k_fs_LR.shape.gii
    │   │   ├── R.StrainR_MSMSulc.10k_fs_LR.shape.gii
    │   │   ├── R.sulc.10k_fs_LR.shape.gii
    │   │   ├── R.thickness.10k_fs_LR.shape.gii
    │   │   ├── R.very_inflated.10k_fs_LR.surf.gii
    │   │   ├── R.white.10k_fs_LR.surf.gii
    │   │   ├── StrainJ_FS.10k_fs_LR.dscalar.nii
    │   │   ├── StrainJ_MSMSulc.10k_fs_LR.dscalar.nii
    │   │   ├── StrainR_FS.10k_fs_LR.dscalar.nii
    │   │   ├── StrainR_MSMSulc.10k_fs_LR.dscalar.nii
    │   │   ├── sulc.10k_fs_LR.dscalar.nii
    │   │   └── thickness.10k_fs_LR.dscalar.nii
    │   ├── fsaverage_LR32k
    │   │   ├── 32k_fs_LR.wb.spec
    │   │   ├── ArealDistortion_FS.32k_fs_LR.dscalar.nii
    │   │   ├── ArealDistortion_MSMSulc.32k_fs_LR.dscalar.nii
    │   │   ├── curvature.32k_fs_LR.dscalar.nii
    │   │   ├── EdgeDistortion_FS.32k_fs_LR.dscalar.nii
    │   │   ├── EdgeDistortion_MSMSulc.32k_fs_LR.dscalar.nii
    │   │   ├── L.ArealDistortion_FS.32k_fs_LR.shape.gii
    │   │   ├── L.ArealDistortion_MSMSulc.32k_fs_LR.shape.gii
    │   │   ├── L.atlasroi.32k_fs_LR.shape.gii
    │   │   ├── L.curvature.32k_fs_LR.shape.gii
    │   │   ├── L.EdgeDistortion_FS.32k_fs_LR.shape.gii
    │   │   ├── L.EdgeDistortion_MSMSulc.32k_fs_LR.shape.gii
    │   │   ├── L.flat.32k_fs_LR.surf.gii
    │   │   ├── L.inflated.32k_fs_LR.surf.gii
    │   │   ├── L.midthickness.32k_fs_LR.surf.gii
    │   │   ├── L.pial.32k_fs_LR.surf.gii
    │   │   ├── L.sphere.32k_fs_LR.surf.gii
    │   │   ├── L.StrainJ_FS.32k_fs_LR.shape.gii
    │   │   ├── L.StrainJ_MSMSulc.32k_fs_LR.shape.gii
    │   │   ├── L.StrainR_FS.32k_fs_LR.shape.gii
    │   │   ├── L.StrainR_MSMSulc.32k_fs_LR.shape.gii
    │   │   ├── L.sulc.32k_fs_LR.shape.gii
    │   │   ├── L.thickness.32k_fs_LR.shape.gii
    │   │   ├── L.very_inflated.32k_fs_LR.surf.gii
    │   │   ├── L.white.32k_fs_LR.surf.gii
    │   │   ├── R.ArealDistortion_FS.32k_fs_LR.shape.gii
    │   │   ├── R.ArealDistortion_MSMSulc.32k_fs_LR.shape.gii
    │   │   ├── R.atlasroi.32k_fs_LR.shape.gii
    │   │   ├── R.curvature.32k_fs_LR.shape.gii
    │   │   ├── R.EdgeDistortion_FS.32k_fs_LR.shape.gii
    │   │   ├── R.EdgeDistortion_MSMSulc.32k_fs_LR.shape.gii
    │   │   ├── R.flat.32k_fs_LR.surf.gii
    │   │   ├── R.inflated.32k_fs_LR.surf.gii
    │   │   ├── R.midthickness.32k_fs_LR.surf.gii
    │   │   ├── R.pial.32k_fs_LR.surf.gii
    │   │   ├── R.sphere.32k_fs_LR.surf.gii
    │   │   ├── R.StrainJ_FS.32k_fs_LR.shape.gii
    │   │   ├── R.StrainJ_MSMSulc.32k_fs_LR.shape.gii
    │   │   ├── R.StrainR_FS.32k_fs_LR.shape.gii
    │   │   ├── R.StrainR_MSMSulc.32k_fs_LR.shape.gii
    │   │   ├── R.sulc.32k_fs_LR.shape.gii
    │   │   ├── R.thickness.32k_fs_LR.shape.gii
    │   │   ├── R.very_inflated.32k_fs_LR.surf.gii
    │   │   ├── R.white.32k_fs_LR.surf.gii
    │   │   ├── StrainJ_FS.32k_fs_LR.dscalar.nii
    │   │   ├── StrainJ_MSMSulc.32k_fs_LR.dscalar.nii
    │   │   ├── StrainR_FS.32k_fs_LR.dscalar.nii
    │   │   ├── StrainR_MSMSulc.32k_fs_LR.dscalar.nii
    │   │   ├── sulc.32k_fs_LR.dscalar.nii
    │   │   └── thickness.32k_fs_LR.dscalar.nii
    │   ├── L.ArealDistortion_FS.164k_fs_LR.shape.gii
    │   ├── L.ArealDistortion_MSMSulc.164k_fs_LR.shape.gii
    │   ├── L.atlasroi.164k_fs_LR.shape.gii
    │   ├── L.curvature.164k_fs_LR.shape.gii
    │   ├── L.EdgeDistortion_FS.164k_fs_LR.shape.gii
    │   ├── L.EdgeDistortion_MSMSulc.164k_fs_LR.shape.gii
    │   ├── L.flat.164k_fs_LR.surf.gii
    │   ├── L.inflated.164k_fs_LR.surf.gii
    │   ├── L.midthickness.164k_fs_LR.surf.gii
    │   ├── L.pial.164k_fs_LR.surf.gii
    │   ├── L.refsulc.164k_fs_LR.shape.gii
    │   ├── L.sphere.164k_fs_LR.surf.gii
    │   ├── L.StrainJ_FS.164k_fs_LR.shape.gii
    │   ├── L.StrainJ_MSMSulc.164k_fs_LR.shape.gii
    │   ├── L.StrainR_FS.164k_fs_LR.shape.gii
    │   ├── L.StrainR_MSMSulc.164k_fs_LR.shape.gii
    │   ├── L.sulc.164k_fs_LR.shape.gii
    │   ├── L.thickness.164k_fs_LR.shape.gii
    │   ├── L.very_inflated.164k_fs_LR.surf.gii
    │   ├── L.white.164k_fs_LR.surf.gii
    │   ├── mri
    │   │   ├── L.pial.nii.gz
    │   │   ├── L.white.nii.gz
    │   │   ├── R.pial.nii.gz
    │   │   ├── R.white.nii.gz
    │   │   └── T1w_brain.nii.gz
    │   ├── Native
    │   │   ├── aparc.dlabel.nii
    │   │   ├── ArealDistortion_FS.dscalar.nii
    │   │   ├── ArealDistortion_MSMSulc.dscalar.nii
    │   │   ├── curvature.dscalar.nii
    │   │   ├── EdgeDistortion_FS.dscalar.nii
    │   │   ├── EdgeDistortion_MSMSulc.dscalar.nii
    │   │   ├── L.aparc.label.gii
    │   │   ├── L.ArealDistortion_FS.shape.gii
    │   │   ├── L.ArealDistortion_MSMSulc.shape.gii
    │   │   ├── L.atlasroi.shape.gii
    │   │   ├── L.curvature.shape.gii
    │   │   ├── L.EdgeDistortion_FS.shape.gii
    │   │   ├── L.EdgeDistortion_MSMSulc.shape.gii
    │   │   ├── lh.aparc.label.gii
    │   │   ├── L.inflated.surf.gii
    │   │   ├── L.midthickness.surf.gii
    │   │   ├── L.pial.surf.gii
    │   │   ├── L.roi.shape.gii
    │   │   ├── L.sphere.MSMSulc.surf.gii
    │   │   ├── L.sphere.reg.reg_LR.surf.gii
    │   │   ├── L.sphere.reg.surf.gii
    │   │   ├── L.sphere.rot.surf.gii
    │   │   ├── L.sphere.surf.gii
    │   │   ├── L.StrainJ_FS.shape.gii
    │   │   ├── L.StrainJ_MSMSulc.shape.gii
    │   │   ├── L.StrainR_FS.shape.gii
    │   │   ├── L.StrainR_MSMSulc.shape.gii
    │   │   ├── L.sulc.shape.gii
    │   │   ├── L.thickness.shape.gii
    │   │   ├── L.very_inflated.surf.gii
    │   │   ├── L.white.surf.gii
    │   │   ├── MSMSulc
    │   │   │   ├── L.logdir
    │   │   │   │   ├── conf
    │   │   │   │   └── MSM.log
    │   │   │   ├── L.mat
    │   │   │   ├── L.sphere.LR.reg.surf.gii
    │   │   │   ├── L.sphere.reg.surf.gii
    │   │   │   ├── L.sphere_rot.surf.gii
    │   │   │   ├── L.transformed_and_reprojected.func.gii
    │   │   │   ├── R.logdir
    │   │   │   │   ├── conf
    │   │   │   │   └── MSM.log
    │   │   │   ├── R.mat
    │   │   │   ├── R.sphere.LR.reg.surf.gii
    │   │   │   ├── R.sphere.reg.surf.gii
    │   │   │   ├── R.sphere_rot.surf.gii
    │   │   │   └── R.transformed_and_reprojected.func.gii
    │   │   ├── native.wb.spec
    │   │   ├── R.aparc.label.gii
    │   │   ├── R.ArealDistortion_FS.shape.gii
    │   │   ├── R.ArealDistortion_MSMSulc.shape.gii
    │   │   ├── R.atlasroi.shape.gii
    │   │   ├── R.curvature.shape.gii
    │   │   ├── R.EdgeDistortion_FS.shape.gii
    │   │   ├── R.EdgeDistortion_MSMSulc.shape.gii
    │   │   ├── rh.aparc.label.gii
    │   │   ├── R.inflated.surf.gii
    │   │   ├── R.midthickness.surf.gii
    │   │   ├── R.pial.surf.gii
    │   │   ├── R.roi.shape.gii
    │   │   ├── R.sphere.MSMSulc.surf.gii
    │   │   ├── R.sphere.reg.reg_LR.surf.gii
    │   │   ├── R.sphere.reg.surf.gii
    │   │   ├── R.sphere.rot.surf.gii
    │   │   ├── R.sphere.surf.gii
    │   │   ├── R.StrainJ_FS.shape.gii
    │   │   ├── R.StrainJ_MSMSulc.shape.gii
    │   │   ├── R.StrainR_FS.shape.gii
    │   │   ├── R.StrainR_MSMSulc.shape.gii
    │   │   ├── R.sulc.shape.gii
    │   │   ├── R.thickness.shape.gii
    │   │   ├── R.very_inflated.surf.gii
    │   │   ├── R.white.surf.gii
    │   │   ├── StrainJ_FS.dscalar.nii
    │   │   ├── StrainJ_MSMSulc.dscalar.nii
    │   │   ├── StrainR_FS.dscalar.nii
    │   │   ├── StrainR_MSMSulc.dscalar.nii
    │   │   ├── sulc.dscalar.nii
    │   │   └── thickness.dscalar.nii
    │   ├── R.ArealDistortion_FS.164k_fs_LR.shape.gii
    │   ├── R.ArealDistortion_MSMSulc.164k_fs_LR.shape.gii
    │   ├── R.atlasroi.164k_fs_LR.shape.gii
    │   ├── R.curvature.164k_fs_LR.shape.gii
    │   ├── R.EdgeDistortion_FS.164k_fs_LR.shape.gii
    │   ├── R.EdgeDistortion_MSMSulc.164k_fs_LR.shape.gii
    │   ├── Results
    │   ├── R.flat.164k_fs_LR.surf.gii
    │   ├── R.inflated.164k_fs_LR.surf.gii
    │   ├── R.midthickness.164k_fs_LR.surf.gii
    │   ├── ROIs
    │   ├── R.pial.164k_fs_LR.surf.gii
    │   ├── R.refsulc.164k_fs_LR.shape.gii
    │   ├── R.sphere.164k_fs_LR.surf.gii
    │   ├── R.StrainJ_FS.164k_fs_LR.shape.gii
    │   ├── R.StrainJ_MSMSulc.164k_fs_LR.shape.gii
    │   ├── R.StrainR_FS.164k_fs_LR.shape.gii
    │   ├── R.StrainR_MSMSulc.164k_fs_LR.shape.gii
    │   ├── R.sulc.164k_fs_LR.shape.gii
    │   ├── R.thickness.164k_fs_LR.shape.gii
    │   ├── R.very_inflated.164k_fs_LR.surf.gii
    │   ├── R.white.164k_fs_LR.surf.gii
    │   ├── StrainJ_FS.164k_fs_LR.dscalar.nii
    │   ├── StrainJ_MSMSulc.164k_fs_LR.dscalar.nii
    │   ├── StrainR_FS.164k_fs_LR.dscalar.nii
    │   ├── StrainR_MSMSulc.164k_fs_LR.dscalar.nii
    │   ├── sulc.164k_fs_LR.dscalar.nii
    │   ├── thickness.164k_fs_LR.dscalar.nii
    │   └── wb.spec
    ├── Original
    │   ├── mri
    │   │   ├── L.pial.nii.gz
    │   │   ├── L.ribbon.nii.gz
    │   │   ├── L.white.nii.gz
    │   │   ├── ribbon.nii.gz
    │   │   ├── R.pial.nii.gz
    │   │   ├── R.ribbon.nii.gz
    │   │   ├── R.white.nii.gz
    │   │   └── T1w_brain.nii.gz
    │   ├── Native
    │   │   ├── L.aparc.label.gii
    │   │   ├── L.curvature.shape.gii
    │   │   ├── lh.aparc.label.gii
    │   │   ├── L.inflated.surf.gii
    │   │   ├── L.midthickness.surf.gii
    │   │   ├── L.pial.surf.gii
    │   │   ├── L.roi.shape.gii
    │   │   ├── L.sphere.reg.surf.gii
    │   │   ├── L.sphere.surf.gii
    │   │   ├── L.sulc.shape.gii
    │   │   ├── L.thickness.shape.gii
    │   │   ├── L.very_inflated.surf.gii
    │   │   ├── L.white.surf.gii
    │   │   ├── R.aparc.label.gii
    │   │   ├── R.curvature.shape.gii
    │   │   ├── rh.aparc.label.gii
    │   │   ├── R.inflated.surf.gii
    │   │   ├── R.midthickness.surf.gii
    │   │   ├── R.pial.surf.gii
    │   │   ├── R.roi.shape.gii
    │   │   ├── R.sphere.reg.surf.gii
    │   │   ├── R.sphere.surf.gii
    │   │   ├── R.sulc.shape.gii
    │   │   ├── R.thickness.shape.gii
    │   │   ├── R.very_inflated.surf.gii
    │   │   └── R.white.surf.gii
    │   └── wb.spec
    └── report_acpc.svg
  ```

