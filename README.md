# MacaSurfer Pipeline

MacaSurfer offers a standardized preprocessing pipeline specifically tailored for structural MRI data of non-human primates. It features a streamlined architecture with full automation and exhibits strong robustness against variations in image quality and the presence of anatomical abnormalities. The pipeline establishes a systematic workflow from data curation to cortical surface modeling, addressing key challenges in multimodal segmentation and surface reconstruction for macaque brains through an integration of classical image processing techniques and modern deep learning algorithms. MacaSurfer supports a unified atlas framework, multi-contrast and multi-sequence fusion, as well as fine-grained tissue segmentation and cortical modeling.

+ Overview

<center>
  <img style="border-radius: 0.3125em;
              box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);
              display: block;
              margin-bottom: 2px;" 
       src="https://github.com/yahuiwei123/MacaSurfer/blob/main/doc/figures/overall.svg" />


  <div style="color:orange; border-bottom: 1px solid #d9d9d9;
              display: inline-block;
              color: #999;
              padding: 2px;
              margin-top: 0px;">Fig.1 (a) Overall workflow The workflow is organized into four stages. Green box: data preparation stage. Beige box: segmentation and enhancement stage. Blue box: surface initialization stage. Pink box: surface refinement stage. (b) Cortical and subcortical atlas Cortical parcellation on the MEBRAIN template surface obtained from MBNA atlas via MSM-based spherical registration; Atlas mapped from the D99 atlas on the NMT template to the MEBRAIN template volume.</div>

</center>

+ Generalizability

<center>
  <img style="border-radius: 0.3125em;
              box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);
              display: block;
              margin-bottom: 2px;" 
       src="https://github.com/yahuiwei123/MacaSurfer/blob/main/doc/figures/multisites.jpg" />


  <div style="color:orange; border-bottom: 1px solid #d9d9d9;
              display: inline-block;
              color: #999;
              padding: 2px;
              margin-top: 0px;">Fig2. (a) Cortical surface reconstructions from ten randomly selected subjects across all centers. Green contours indicate the white matter surface boundaries, and yellow contours indicate the pial surface boundaries. (b) High-quality cortical surface reconstruction was achieved even in subjects with brain tumors.</div>

</center>

+ Accuracy

<center>
  <img style="border-radius: 0.3125em;
              box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);
              display: block;
              margin-bottom: 2px;" 
       src="https://github.com/yahuiwei123/MacaSurfer/blob/main/doc/figures/tgb.svg" />


  <div style="color:orange; border-bottom: 1px solid #d9d9d9;
              display: inline-block;
              color: #999;
              padding: 2px;
              margin-top: 0px;">Fig3. Comparison of Bias Field Correction Methods (a) Heatmaps visualizing the estimated bias field after N4 correction (top) and the proposed tissue-guided correction (bottom). The proposed method more effectively reduces intra-tissue bias, particularly in regions with thin white matter structures such as the frontal and occipital lobes. (b) White matter surface reconstructions after applying each correction method. Compared to N4, our method yields more anatomically accurate surfaces, especially in the occipital region, where white matter boundaries are better preserved.</div>

</center>


## Installation

+ download the singularity image

```shell
singularity pull --arch amd64 library://weiyahui123/weiyahui123/macasurfer:v3.0
```


## Usage

+ Before running, you also need to create a folder named **RawData** inside each subject’s directory following the file structure described in **Results**. Place the T1 and T2 images in NIFTI format into this folder. Note that the T1 and T2 images must include ‘T1’ or ‘T2’ in their filenames to allow the program to identify their modality. It is also acceptable to have only T1 modality images.

+ One-click run

  ```shell
  /usr/bin/singularity exec --nv --writable \
    -B ${subj_dir}/${subj_id}:/workspace \
    ~/software/macasurfer_v3.0 \
    sh /MacaSurfer/MacaSurfer.sh --bfc_method tgbfc --fix_white true --deep_white false --vessel_detect false
  ```

+ Step-by-step run

  + Only before check part

    ```shell
    /usr/bin/singularity exec --nv --writable \
    -B ${subj_dir}/${subj_id}:/workspace \
    ~/software/macasurfer_v3.0 \
    sh /MacaSurfer/MacaSurfer.sh --bfc_method tgbfc --fix_white true --deep_white false --vessel_detect false --before_check true
    ```

  + After check part

    ```shell
    /usr/bin/singularity exec --nv --writable \
    -B ${subj_dir}/${subj_id}:/workspace \after
    ~/software/macasurfer_v3.0 \
    sh /MacaSurfer/MacaSurfer.sh --bfc_method tgbfc --fix_white true --deep_white false --vessel_detect false --after_check true
    ```

## Results

+ Directory structure
