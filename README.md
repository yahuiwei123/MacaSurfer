# MacaSurfer Pipeline

MacaSurfer offers a standardized preprocessing pipeline specifically tailored for structural MRI data of non-human primates. It features a streamlined architecture with full automation and exhibits strong robustness against variations in image quality and the presence of anatomical abnormalities. The pipeline establishes a systematic workflow from data curation to cortical surface modeling, addressing key challenges in multimodal segmentation and surface reconstruction for macaque brains through an integration of classical image processing techniques and modern deep learning algorithms. MacaSurfer supports a unified atlas framework, multi-contrast and multi-sequence fusion, as well as fine-grained tissue segmentation and cortical modeling.

+ Overview

(a) Overall workflow The workflow is organized into four stages. Green box: data preparation stage. Beige box: segmentation and enhancement stage. Blue box: surface initialization stage. Pink box: surface refinement stage. 

(b) Cortical and subcortical atlas Cortical parcellation on the MEBRAIN template surface obtained from MBNA atlas via MSM-based spherical registration; Atlas mapped from the D99 atlas on the NMT template to the MEBRAIN template volume.

<center>
  <img style="border-radius: 0.3125em;
              box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);
              display: block;
              margin-bottom: 2px;" 
       src="https://github.com/yahuiwei123/MacaSurfer/blob/main/figures/overall.svg" />
</center>
+ Generalizability on Multi-sites

We evaluated the robustness of our pipeline across macaque datasets from 24 acquisition sites. The pipeline consistently produced high-quality reconstructions in multi-site adult macaques and also successfully reconstructed the cortex in the 3-month-old UNC macaque dataset.

<center>
  <img style="border-radius: 0.3125em;
              box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);
              display: block;
              margin-bottom: 2px;" 
       src="https://github.com/yahuiwei123/MacaSurfer/blob/main/figures/multisites.svg" />
</center>

+ Tissue-guided Bias Field Correction

The deep learning–based macaque tissue segmentation algorithm nBEST (https://github.com/TaoZhong11/nBEST) provides robust cerebrospinal fluid, gray matter and white matter labels. We leverage these high-quality tissue maps to guide bias-field correction, enabling the removal of higher-frequency components that are typically difficult to estimate. This correction substantially improves intensity homogeneity and, in turn, facilitates more accurate reconstruction of the white-matter surface.
<center>
  <img style="border-radius: 0.3125em;
              box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);
              display: block;
              margin-bottom: 2px;" 
       src="https://github.com/yahuiwei123/MacaSurfer/blob/main/figures/tgb.svg" />
</center>

+ Cortical Region Extraction

FreeSurfer often exhibits inaccuracies in identifying the medial wall, and even small errors in this region can substantially disrupt spherical registration, sometimes leading to global shifts of the cortical surface in spherical space. To address this limitation, we developed a robust medial-wall extraction algorithm that reliably delineates the cortical region in individual subjects, providing a more stable foundation for downstream surface registration and cortical parcellation. We manually inspected the cortical-region extraction results for all subjects in the PRIME-DE dataset and confirmed that the algorithm produced anatomically correct cortical masks for every subject, with no obvious failures.

<center>
  <img style="border-radius: 0.3125em;
              box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);
              display: block;
              margin-bottom: 2px;" 
       src="https://github.com/yahuiwei123/MacaSurfer/blob/main/figures/medial_wall.svg" />
</center>

+ Surface-aware Volume Registration

Voxel-level group analyses—particularly in deep learning applications—require precise alignment of each subject’s cortical ribbon to a common template. Traditional volume-only registration methods, however, struggle to achieve accurate correspondence in the ribbon, leading to suboptimal voxelwise alignment of both cortical regions and BOLD signals. To overcome these limitations, we extend the FireANTs  framework (https://github.com/rohitrango/FireANTs) with a Surface-Aware symmetric diffeomorphic registration approach that enables substantially more precise voxelwise cortical alignment across subjects.


<center>
  <img style="border-radius: 0.3125em;
              box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);
              display: block;
              margin-bottom: 2px;" 
       src="https://github.com/yahuiwei123/MacaSurfer/blob/main/figures/surface_aware.svg" />
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

## Citation

+ Coming soon...

## Contacts

For questions/bugs/feedback, please contact: 

Yahui Wei, Ph.D., weiyahui2023@ia.ac.cn

*Institute of Automation, Chinese Academy of Sciences*
