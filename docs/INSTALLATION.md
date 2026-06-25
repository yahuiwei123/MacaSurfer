# MacaSurfer v3.0 — Installation & User Guide

MacaSurfer is a ground-up MRI processing pipeline purpose-built for macaque monkeys, supporting structural and functional (BOLD) MRI preprocessing, surface reconstruction, MSM registration, and normative modeling.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Decision: Container vs Source](#quick-decision-container-vs-source)
3. [Option A: Container Image (macasurfer.sif)](#option-a-container-image-macasurfersif)
   - [Prerequisites](#a1-prerequisites)
   - [Getting the Image](#a2-getting-the-image)
   - [Running the Pipeline](#a3-running-the-pipeline)
   - [Running Individual Stages](#a4-running-individual-stages)
   - [GPU Support](#a5-gpu-support)
4. [Option B: Source Installation](#option-b-source-installation)
   - [System Requirements](#b1-system-requirements)
   - [Step 1: Install System Libraries](#b2-step-1-install-system-libraries)
   - [Step 2: Install Java & Nextflow](#b3-step-2-install-java--nextflow)
   - [Step 3: Install FreeSurfer](#b4-step-3-install-freesurfer)
   - [Step 4: Install FSL](#b5-step-4-install-fsl)
   - [Step 5: Install ANTs](#b6-step-5-install-ants)
   - [Step 6: Install Connectome Workbench](#b7-step-6-install-connectome-workbench)
   - [Step 7: Install MSM](#b8-step-7-install-msm)
   - [Step 8: Install MATLAB Runtime](#b9-step-8-install-matlab-runtime-optional)
   - [Step 9: Set up Python Environment (macapipe)](#b10-step-9-set-up-python-environment-macapipe)
   - [Step 10: Clone MacaSurfer](#b11-step-10-clone-macasurfer)
   - [Step 11: Configure Paths](#b12-step-11-configure-paths)
   - [Step 12: Verify Installation](#b13-step-12-verify-installation)
5. [Input Data: BIDS Format](#input-data-bids-format)
6. [Running the Pipeline](#running-the-pipeline)
7. [Output Structure](#output-structure)
8. [Parameter Reference](#parameter-reference)
9. [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Overview

![MacaSurfer Pipeline](../figures/pipeline.png)

For a visual overview of the complete project — from data collection through model training to deployment — see the [project workflow diagram](../figures/workflow.png).

MacaSurfer supports two species: **Macaque Mulatta** & **Macaque Fascicularis** (MEBRAIN templates).

---

## Quick Decision: Container vs Source

| | Container (macasurfer.sif) | Source Install |
|---|---|---|
| **Setup time** | ~5 minutes | ~2-4 hours |
| **Disk space** | ~15 GB (image) | ~50+ GB (all software) |
| **Dependencies** | Singularity/Apptainer only | Java, Nextflow, FreeSurfer, FSL, ANTs, Workbench, MSM, MATLAB, Python |
| **Portability** | Single .sif file, runs anywhere | Tied to specific machine |
| **Customization** | Limited (can overlay config) | Full control |
| **GPU support** | Yes (`--nv` flag) | Native |
| **Recommended for** | Quick start, HPC clusters, reproducibility | Development, frequent customization |

**If you just want to process data**, use the container.  
**If you need to modify the pipeline or develop new features**, install from source.

---

## Option A: Container Image (macasurfer.sif)

### A.1 Prerequisites

- **Singularity / Apptainer** ≥ 3.5
  ```bash
  # Check version
  singularity --version
  # or
  apptainer --version
  ```
- **GPU support** (optional): NVIDIA drivers + `nvidia-container-cli`

### A.2 Getting the Image

```bash
# Pull from container registry
singularity pull macasurfer.sif docker://your-registry/macasurfer:3.0

# Or if provided as a file
# macasurfer.sif (single file, ~15 GB)
```

The image bundles:
- FreeSurfer 7.3.2, FSL, ANTs 2.6.5, Connectome Workbench
- MSM (Ubuntu binaries), NiftyReg
- Python 3.10 environment (`macapipe`) with all required packages
- Java 17, Nextflow
- All MacaSurfer scripts, templates (MEBRAIN), models, atlases

### A.3 Running the Pipeline

#### Basic run (structural + BOLD)

```bash
singularity run --nv \
  --bind /path/to/data:/data \
  --bind /path/to/output:/output \
  macasurfer.sif \
  --bids_dir /data/rawdata \
  --participant_label sub-001 \
  --out_dir /output
```

#### Directory bindings explained

| Container path | Purpose | Bind example |
|---|---|---|
| `/data` | BIDS input | `--bind /your/bids:/data` |
| `/output` | Pipeline output | `--bind /your/output:/output` |
| `/work` | Nextflow work directory (tmp) | `--bind /your/work:/work` |
| `/freesurfer_license` | FreeSurfer license | `--bind /your/license.txt:/freesurfer_license` |

#### Recommended: Use a wrapper script

```bash
#!/bin/bash
# run_macasurfer.sh — customize these paths
BIDS_DIR="/data/PRIME-DE/site-mcgill/rawdata"
OUT_DIR="/data/output"
WORK_DIR="/data/work"
SUBJ="sub-032206"

singularity run --nv \
  --bind "${BIDS_DIR}:/data/rawdata" \
  --bind "${OUT_DIR}:/output" \
  --bind "${WORK_DIR}:/work" \
  /path/to/macasurfer.sif \
  --bids_dir /data/rawdata \
  --participant_label "${SUBJ}" \
  --out_dir /output \
  --work_dir /work
```

#### Run modes

```bash
# BOLD-only (reuse existing structural output)
singularity run --nv \
  --bind /data:/data --bind /output:/output \
  macasurfer.sif \
  --bids_dir /data/rawdata \
  --participant_label sub-001 \
  --out_dir /output \
  --bold_only

# Anatomical-only (skip BOLD)
singularity run --nv \
  --bind /data:/data --bind /output:/output \
  macasurfer.sif \
  --bids_dir /data/rawdata \
  --participant_label sub-001 \
  --out_dir /output \
  --anat_only

# Resume a failed run
singularity run --nv \
  --bind /data:/data --bind /output:/output \
  macasurfer.sif \
  --bids_dir /data/rawdata \
  --participant_label sub-001 \
  --out_dir /output \
  --resume
```

#### Customizing parameters in the container

```bash
# Override with a custom config file
singularity run --nv \
  --bind /data:/data --bind /output:/output \
  --bind ./my_config.config:/config/my_config.config \
  macasurfer.sif \
  --bids_dir /data/rawdata \
  --participant_label sub-001 \
  --out_dir /output \
  --config_file /config/my_config.config

# Or pass parameters directly (see PARAMETERS.md for all options)
singularity run --nv \
  --bind /data:/data --bind /output:/output \
  macasurfer.sif \
  --bids_dir /data/rawdata \
  --participant_label sub-001 \
  --out_dir /output \
  --bfc_method n4 \
  --seg_tool nbest \
  --bold_bandpass 0.008-0.08
```

### A.4 Running Individual Stages

The container includes an `Examples/` directory with scripts for running individual stages:

```bash
# Shell into the container
singularity shell --bind /data:/data --bind /output:/output macasurfer.sif

# Inside the container:
source /MacaSurfer/SetUpHCPPipelineNHP.sh

# Run a specific stage (e.g., surface annotation only)
sh /MacaSurfer/Examples/register_with_surf.sh
```

### A.5 GPU Support

The `--nv` flag passes NVIDIA GPUs into the container. GPU-accelerated steps include:

- **Template registration** (ANTS, if `--device gpu`)
- **Brain extraction** (macaUNet, PyTorch)
- **Tissue segmentation** (macaBrainNet, PyTorch)
- **Surface registration** (FireANTs, PyTorch)

```bash
# Restrict to specific GPUs
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0,1
singularity run --nv \
  --bind /data:/data --bind /output:/output \
  macasurfer.sif \
  --bids_dir /data/rawdata \
  --participant_label sub-001 \
  --out_dir /output
```

---

## Option B: Source Installation

### B.1 System Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 20.04+ / CentOS 7+ | Ubuntu 22.04 LTS |
| CPU cores | 8 | 64+ |
| RAM | 32 GB | 128 GB |
| GPU | NVIDIA 8 GB VRAM | NVIDIA 24+ GB VRAM |
| Disk | 100 GB | 500 GB+ (for BOLD temp files) |
| Software space | ~50 GB | ~80 GB (with MATLAB) |

### B.2 Step 1: Install System Libraries

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake \
  libgomp1 libx11-6 libxmu6 libxext6 libxt6 libgl1-mesa-glx \
  libjpeg62 libtiff5 libpng16-16 \
  gfortran libopenblas-dev liblapack-dev \
  tcsh wget unzip git curl \
  libglu1-mesa libxi6 libxrender1 libxrandr2 libxcursor1 \
  libxinerama1 libxxf86vm1

# CentOS / RHEL
sudo yum install -y epel-release
sudo yum groupinstall -y "Development Tools"
sudo yum install -y \
  cmake libgomp libX11 libXmu libXext libXt mesa-libGL \
  libjpeg-turbo libtiff libpng \
  gcc-gfortran openblas-devel lapack-devel \
  tcsh wget unzip git curl \
  libglu mesa-libXi libXrender libXrandr libXcursor \
  libXinerama libXxf86vm
```

### B.3 Step 2: Install Java & Nextflow

```bash
# Java 17
wget https://download.java.net/java/GA/jdk17.0.2/dfd4a8d0985749f896bed70d7138bb7f/8/GPL/openjdk-17.0.2_linux-x64_bin.tar.gz
tar -xzf openjdk-17.0.2_linux-x64_bin.tar.gz -C /usr/local/
export JAVA_HOME=/usr/local/jdk-17.0.2
export PATH=$JAVA_HOME/bin:$PATH

# Nextflow
cd /usr/local/
curl -s https://get.nextflow.io | bash
# Alternatively: wget -qO- https://get.nextflow.io | bash
chmod +x nextflow
export PATH=/usr/local:$PATH

# Verify
nextflow -version
# Expected: nextflow version 23.x.x or later
```

### B.4 Step 3: Install FreeSurfer

```bash
# Download FreeSurfer 7.3.2
wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.3.2/freesurfer-linux-centos7_x86_64-7.3.2.tar.gz
tar -xzf freesurfer-linux-centos7_x86_64-7.3.2.tar.gz -C /usr/local/
export FREESURFER_HOME=/usr/local/freesurfer-7.3.2
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Register (free): https://surfer.nmr.mgh.harvard.edu/registration.html
# Place license.txt at $FREESURFER_HOME/license.txt

# Verify
recon-all --version
```

**Note:** The pipeline uses FreeSurfer tools (mri_convert, mris_*, etc.) but does NOT run `recon-all`. The FreeSurfer version is used for surface tools and file format conversions.

### B.5 Step 4: Install FSL

```bash
# Download FSL 6.0.6+
wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py
python fslinstaller.py -d /usr/local/fsl

export FSLDIR=/usr/local/fsl
source $FSLDIR/etc/fslconf/fsl.sh
export PATH=$FSLDIR/bin:$PATH

# Verify
flirt -version
fslmaths --version
```

### B.6 Step 5: Install ANTs

```bash
# Option 1: Pre-built binaries (recommended)
wget https://github.com/ANTsX/ANTs/releases/download/v2.5.4/ants-2.5.4-ubuntu22.04-X64-gcc.zip
unzip ants-2.5.4-ubuntu22.04-X64-gcc.zip -d /usr/local/
mv /usr/local/ants-2.5.4 /usr/local/ants
export ANTSPATH=/usr/local/ants/bin
export PATH=$ANTSPATH:$PATH

# Option 2: Build from source
git clone https://github.com/ANTsX/ANTs.git
cd ANTs
git checkout v2.5.4
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/ants
make -j$(nproc)
make install

# Verify
ANTS --version
DenoiseImage --help
```

### B.7 Step 6: Install Connectome Workbench

```bash
wget https://www.humanconnectome.org/storage/app/media/workbench/workbench-linux64-v1.5.0.zip
unzip workbench-linux64-v1.5.0.zip -d /usr/local/
export CARET7DIR=/usr/local/workbench/bin_linux64
export PATH=$CARET7DIR:$PATH

# Verify
wb_command -version
```

### B.8 Step 7: Install MSM

```bash
# MSM (Multimodal Surface Matching) — Ubuntu binary
# Place the MSM binary at:
export MSMBINDIR=/usr/local/msm
# The binary should be: $MSMBINDIR/msm

# MacaSurfer includes MSM configuration files in its MSMConfig/ directory.
# No separate MSM config installation needed.
```

### B.9 Step 8: Install MATLAB Runtime (optional)

MATLAB Runtime is only needed for MSMAll group-level analysis. For per-subject processing, skip this step.

```bash
# Download MCR R2019b from MathWorks
# https://www.mathworks.com/products/compiler/mcr/index.html
# Install to /usr/local/MATLAB/MATLAB_Compiler_Runtime
export MATLAB_COMPILER_RUNTIME=/usr/local/MATLAB/MATLAB_Compiler_Runtime
```

### B.10 Step 9: Set up Python Environment (macapipe)

```bash
# Create conda environment
conda create -n macapipe python=3.10 -y
conda activate macapipe

# Install from MacaSurfer's requirements.txt
cd /path/to/MacaSurfer
pip install -r requirements.txt

# Verify key imports
python -c "
import numpy
import scipy
import nibabel
import SimpleITK
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
import monai
import nibabel
print('All core packages OK')
"
```

**requirements.txt** includes: numpy, scipy, nibabel, SimpleITK, torch, monai, pytorch3d, timm, scikit-image, opencv-python, Pillow, scikit-learn, pandas, tqdm, pcntoolkit, nipype, niworkflows, nitransforms, sdcflows, pybids, h5py, surfplot, batchgenerators, numba, pyyaml, requests, omegaconf, hydra-core, nvidia-ml-py3, ray, einops, trimesh, pyrender, tifffile.

### B.11 Step 10: Clone MacaSurfer

```bash
git clone https://github.com/your-org/MacaSurfer.git /usr/local/MacaSurfer
cd /usr/local/MacaSurfer

# Directory structure after cloning:
# MacaSurfer/
# ├── MacaSurfer.sh          # Main entry point
# ├── SetUpHCPPipelineNHP.sh # Environment setup
# ├── requirements.txt       # Python dependencies
# ├── README.md
# ├── PARAMETERS.md          # Full parameter reference
# ├── nextflow/
# │   ├── macasurfer.nf      # Main pipeline
# │   ├── Prepare/           # Prepare stage scripts
# │   ├── Enhance/           # Enhance stage scripts
# │   ├── Surface/           # Surface stage scripts
# │   ├── Resample/          # Resample stage scripts
# │   ├── BOLD/              # BOLD stage scripts
# │   ├── macasurfer.common.config
# │   └── macasurfer.local.config
# ├── global/
# │   ├── templates/MEBRAIN/ # MEBRAIN macaque templates
# │   ├── config/            # LUTs, FNIRT configs
# │   ├── scripts/           # Global utility scripts
# │   └── normative_model/   # Normative model (BLR)
# ├── shared/
# │   ├── brainextractor/    # macaUNet brain extractor
# │   ├── tissueextractor/   # nnU-Net tissue segmenter
# │   ├── surf_fireants/     # FireANTs surface registration
# │   ├── volume_register/   # Volume registration helpers
# │   ├── utils/             # QC + BFC + utility scripts
# │   └── reports/           # Report generation
# ├── MSMConfig/             # MSM registration configs
# └── docs/
#     └── INSTALLATION.md    # This document
```

### B.12 Step 11: Configure Paths

Edit `nextflow/macasurfer.common.config` to match your installation paths:

```groovy
params {
    // Software paths — update these to your installation
    freesurfer_home  = "/usr/local/freesurfer-7.3.2"
    fsl_dir          = "/usr/local/fsl"
    niftyreg_install = "/usr/local/niftyreg"
    antspath         = "/usr/local/ants/bin"
    python_env       = "/opt/conda/envs/macapipe"
    python_inter     = "/opt/conda/envs/macapipe/bin/python"
    hcppipedir       = "/usr/local/MacaSurfer"
    caret7dir        = "/usr/local/workbench/bin_linux64"
    msmbindir        = "/usr/local/msm"
    fixdir           = "/usr/local/fix"

    // Model paths — if macaBrainNet installed separately
    maca_brainnet_dir = "/usr/local/macaBrainNet_v2"
}
```

Alternatively, create a local override file:

```bash
cat > my_config.config << 'EOF'
params {
    freesurfer_home  = "/usr/local/freesurfer-7.3.2"
    fsl_dir          = "/usr/local/fsl"
    antspath         = "/usr/local/ants/bin"
    python_inter     = "/opt/conda/envs/macapipe/bin/python"
}
EOF

# Use it:
./MacaSurfer.sh --config_file ./my_config.config \
  --bids_dir /data/rawdata --participant_label sub-001 --out_dir /output
```

**Important:** You must also update `SetUpHCPPipelineNHP.sh` if your software paths differ from the defaults. This file sets environment variables used by the shell scripts called within Nextflow processes.

### B.13 Step 12: Verify Installation

```bash
# Source the environment
source /usr/local/MacaSurfer/SetUpHCPPipelineNHP.sh

# Verify each tool is in PATH
which fslmaths         # FSL
which antsRegistration  # ANTs
which mri_convert      # FreeSurfer
which wb_command       # Workbench
which python           # Should be macapipe env

# Verify Python environment
python -c "import nibabel; import torch; import monai; print('All OK')"

# Run the pipeline help
./MacaSurfer.sh --help

# Test on a small dataset
./MacaSurfer.sh \
  --bids_dir /path/to/test/rawdata \
  --participant_label sub-test \
  --out_dir /tmp/test_output
```

---

## Input Data: BIDS Format

MacaSurfer expects BIDS-formatted input. The minimum required structure:

```
rawdata/
└── sub-<label>/
    └── ses-<label>/          # Optional for single-session
        ├── anat/
        │   ├── sub-*_T1w.nii.gz          # Required: T1w anatomical
        │   └── sub-*_T2w.nii.gz          # Optional: T2w anatomical
        └── func/
            ├── sub-*_task-rest_bold.nii.gz   # BOLD runs
            ├── sub-*_task-rest_bold.json     # BOLD metadata
            ├── sub-*_task-rest_sbref.nii.gz  # Single-band reference (optional)
            └── sub-*_dir-AP_epi.nii.gz       # Fieldmap (optional)
```

**Modalities supported:**
- **T1w**: Required for structural pipeline
- **T2w**: Optional. Used for pial surface refinement and sqrt BFC
- **bold**: Required for BOLD pipeline. Task name filterable via `--bold_task_type`

For detailed BIDS specification: https://bids.neuroimaging.io/

---

## Running the Pipeline

### Source install usage

```bash
cd /usr/local/MacaSurfer

# Minimal run
./MacaSurfer.sh \
  --bids_dir /data/rawdata \
  --participant_label sub-001 \
  --out_dir /data/output

# With all options
./MacaSurfer.sh \
  --bids_dir /data/rawdata \
  --participant_label sub-001,sub-002 \
  --session_id ses-001 \
  --out_dir /data/output \
  --work_dir /data/work \
  --resume \
  --debug \
  --bfc_method gauss \
  --seg_tool macabrainnet \
  --fix_white true \
  --bold_task_type rest \
  --bold_skip_frame 4 \
  --bold_bandpass 0.008-0.08 \
  --bold_cifti true
```

### Container usage

```bash
singularity run --nv \
  --bind /data:/data \
  --bind /output:/output \
  macasurfer.sif \
  --bids_dir /data/rawdata \
  --participant_label sub-001 \
  --out_dir /output
```

### Common run modes

| Scenario | Flags |
|---|---|
| Full processing (structural + BOLD) | *(default, no extra flags)* |
| Structural only | `--anat_only` |
| BOLD only (pre-existing structural) | `--bold_only` |
| Resume failed run | `--resume` |
| Debug mode | `--debug` |
| Single stage only | `--process_stage biasfield` |
| Custom BFC method | `--bfc_method n4` |
| Legacy segmentation | `--seg_tool nbest` |
| CIFTI output | `--bold_cifti true` |

See **[PARAMETERS.md](PARAMETERS.md)** for the complete parameter reference.

---

## Output Structure

```
output/
└── sub-<label>/
    └── ses-<label>/              # Or directly under sub-<label>/ if single session
        ├── Prepare/
        │   ├── T1w/              # Skull-stripped, aligned, averaged images
        │   └── QC/               # Skullstrip QC images
        ├── Enhance/
        │   ├── T1w/
        │   │   ├── *desc-brain_T1w.nii.gz      # Brain-extracted T1w
        │   │   ├── *desc-brain_T2w.nii.gz      # Brain-extracted T2w
        │   │   ├── *desc-bfc_T1w.nii.gz        # Bias-field corrected T1w
        │   │   ├── *desc-bfc_T2w.nii.gz        # Bias-field corrected T2w
        │   │   ├── *desc-dseg_dseg.nii.gz      # Tissue segmentation
        │   │   ├── *space-acpc_res-04mm_*.nii.gz # ACPC-aligned isotropic
        │   │   └── xfms/                        # Transform matrices
        │   └── QC/                              # Enhancement QC images
        ├── Surface/
        │   ├── workspace/
        │   │   ├── surf/          # White, pial, sphere surfaces
        │   │   ├── label/         # Cortical labels
        │   │   └── stats/         # Surface statistics
        │   └── QC/                # Surface QC
        ├── Resample/
        │   ├── Atlas/             # Atlas-space resampled data
        │   ├── ACPC/              # ACPC-space resampled data
        │   ├── Native/            # Native-space resampled data
        │   └── Annot/             # Cortical annotation + normative stats
        ├── BOLD/
        │   ├── preprocess/        # Preprocessed BOLD runs
        │   ├── confounds/         # Confound regressor TSV files
        │   ├── normalize/         # Template-normalized BOLD
        │   ├── surface/           # Surface-projected BOLD (if CIFTI)
        │   └── QC/                # BOLD QC (motion, tSNR, carpet plots)
        └── run_info/              # Run metadata, YAML configs
```

---

## Parameter Reference

All pipeline parameters are documented in **[PARAMETERS.md](PARAMETERS.md)**, organized by processing stage:

- **Pipeline Control** — `bids_dir`, `participant_label`, `process_stage`, `anat_only`, `bold_only`, etc.
- **GPU / Resources** — `gpus`, `per_gpu`, `device`
- **BOLD Processing** — `bold_task_type`, `bold_bandpass`, `bold_sdc`, `bold_cifti`, etc.
- **Enhancement** — `seg_tool`, `bfc_method`, `fix_white`, `vessel_detect`, `t2_refine_pial`, etc.
- **Registration / Surface** — `reg_name`, `high_res_mesh`, etc.
- **Species-Specific** — Templates, mesh resolution, smoothing parameters
- **Software Paths** — All external tool locations
- **Atlas & Templates** — MEBRAIN, MNI, standard mesh atlases

---

## Troubleshooting & FAQ

### General

**Q: Pipeline fails with "command not found: nextflow"**  
A: Ensure Nextflow is in PATH: `export PATH=/usr/local:$PATH`. For source install, run `which nextflow`. For container, Nextflow is bundled inside.

**Q: "No space left on device" during BOLD processing**  
A: BOLD preprocessing generates large temporary files. Set `--work_dir` to a location with sufficient space (500 GB+ recommended for multi-run datasets), and clean it after successful completion: `rm -rf $work_dir/nextflow/`.

**Q: Pipeline runs but GPU is not used**  
A: Check: (1) `--device gpu` is set (default), (2) `nvidia-smi` works, (3) container is launched with `--nv`. Verify with `--debug` to see detected GPUs.

### Container-specific

**Q: "singularity: command not found"**  
A: Install Singularity/Apptainer: https://apptainer.org/docs/admin/main/installation.html

**Q: Cannot write to bind-mounted directories**  
A: Ensure the host directories exist and have write permissions for the user running singularity. Singularity runs as the host user by default.

**Q: FreeSurfer license not found**  
A: The container may include a bundled license. If not, bind your license: `--bind /path/to/license.txt:/usr/local/freesurfer-7.3.2/license.txt`

### Source install-specific

**Q: "ImportError: libgomp.so.1: cannot open shared object file"**  
A: Install missing system library: `sudo apt-get install libgomp1` (Ubuntu) or `yum install libgomp` (CentOS).

**Q: ANTs registration fails with "Killed"**  
A: Reduce thread count: set `--omp_num_threads 16` in config or CLI. Registration can consume >32 GB RAM with high thread counts.

**Q: "mri_convert: error while loading shared libraries"**  
A: FreeSurfer requires specific glibc. On newer Ubuntu (22.04+), install compatibility libs: `sudo apt-get install libtiff5 libjpeg62`.

**Q: nnU-Net or macaBrainNet models not found**  
A: Check `maca_brainnet_dir` and `nbest_model_path` in config. These are separate downloads; macaBrainNet_v2 includes trained model weights (~2 GB).

### Data issues

**Q: "No BOLD runs found for subject sub-XXX"**  
A: Check: (1) BOLD files follow BIDS naming (`sub-*_task-*_bold.nii.gz`), (2) JSON sidecars exist, (3) `--bold_task_type` filter matches the task name.

**Q: T2w processing skipped despite T2w being present**  
A: T2w must be in the same session directory as T1w and follow BIDS naming (`sub-*_T2w.nii.gz`).

---

## Further Help

- **Parameter reference**: [PARAMETERS.md](PARAMETERS.md)
- **Issue tracker**: https://github.com/yahuiwei123/MacaSurfer/issues
