# MacaSurfer
A comprehensive pipeline for fast and accurate cortical surface reconstruction with macaque-specific parcellation

## Installation

+ download the singularity image

```shell
singularity pull --arch amd64 library://weiyahui123/weiyahui123/macasurfer:v1.0
```

+ other necessary softwares
  + FreeSurfer >= 7.3.2
  + FSL >= 6.0.5

## Useage

+ Before running, you also need to create a folder named **RawData** inside each subject’s directory following the file structure described in **Results**. Place the T1 and T2 images in NIFTI format into this folder. Note that the T1 and T2 images must include ‘T1’ or ‘T2’ in their filenames to allow the program to identify their modality. It is also acceptable to have only T1 modality images.

+ One-click run

  ```shell
  /home/yhwei/software/singularity/bin/singularity exec --nv --writable \
  	-B /path/to/your/freesurfer-7.3.2:/soft/freesurfer \
  	-B /path/to/your/fsl-6.0.5:/soft/fsl \
  	-B /path/to/your/subject_directory/:/workspace \
  	/path/to/your/MacaSurfer.simg \
      sh /MacaSurfer/Examples/recon-all.sh
  ```

+ Step-by-step run

  + Only prepare part

    ```shell
    /home/yhwei/software/singularity/bin/singularity exec --nv --writable \
    	-B /path/to/your/freesurfer-7.3.2:/soft/freesurfer \
    	-B /path/to/your/fsl-6.0.5:/soft/fsl \
    	-B /path/to/your/subject_directory/:/workspace \
    	/path/to/your/MacaSurfer.simg \
        sh /MacaSurfer/Examples/prepare.sh
    ```

  + Only tissue segmentation and tissue-guided bias field correction part

    ```shell
    /home/yhwei/software/singularity/bin/singularity exec --nv --writable \
    	-B /path/to/your/freesurfer-7.3.2:/soft/freesurfer \
    	-B /path/to/your/fsl-6.0.5:/soft/fsl \
    	-B /path/to/your/subject_directory/:/workspace \
    	/path/to/your/MacaSurfer.simg \
        sh /MacaSurfer/Examples/enhance.sh
    ```

  + Only surface reconstruction part

    ```shell
    /home/yhwei/software/singularity/bin/singularity exec --nv --writable \
    	-B /path/to/your/freesurfer-7.3.2:/soft/freesurfer \
    	-B /path/to/your/fsl-6.0.5:/soft/fsl \
    	-B /path/to/your/subject_directory/:/workspace \
    	/path/to/your/MacaSurfer.simg \
        sh /MacaSurfer/Examples/surface.sh
    ```

  + Only post process part

    ```shell
    /home/yhwei/software/singularity/bin/singularity exec --nv --writable \
    	-B /path/to/your/freesurfer-7.3.2:/soft/freesurfer \
    	-B /path/to/your/fsl-6.0.5:/soft/fsl \
    	-B /path/to/your/subject_directory/:/workspace \
    	/path/to/your/MacaSurfer.simg \
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
  - subject directory
      - RawData
        - *T1*.nii.gz
        - *T2*.nii.gz
      - Enhance
        - MEBRAIN
        - nBEST
        - T1w
        - T2w
      - Prepare
        - T1w
        	- folder of every T1 run ...
        - T2w
          - folder of every T2 run ...
        - data_config.yaml
        - data_figure.jpg
        - t1_brainmask.jpg
        - t2_brainmask.jpg
        - avg_brainmask.jpg
      - FreeSurfer
        - subjectID
          - mri
          - surf
          - label
          - scripts
          - stats
          - tmp
          - touch
          - trash
      - Results
        - ACPC
          - fsaverage_LR10k
          - fsaverage_LR32k
          - Native
          - mri
        - Atlas
          - fsaverage_LR10k
          - fsaverage_LR32k
          - Native
          - mri
        - Original
          - Native
          - mri
  ```

