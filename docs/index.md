# MacaSurfer v3.0

**NHP MRI Processing Pipeline** — structural and functional MRI preprocessing, cortical surface reconstruction, multimodal surface matching (MSM) registration, and normative modeling for non-human primates.

A ground-up MRI processing pipeline purpose-built for **Macaque** (Macaca Mulatta & Macaca Fascicularis), covering structural and functional (BOLD) preprocessing, cortical surface reconstruction, MSM registration, and normative modeling.

---

## Pipeline Overview

![MacaSurfer Pipeline](../figures/pipeline.png)

MacaSurfer processes BIDS-formatted MRI data through six Nextflow workflows:

| # | Workflow | Description |
|---|----------|-------------|
| 1 | **info** | Parse BIDS structure, initialize per-session output directories |
| 2 | **prepare** | Skull stripping (deep learning), brainmask fixing, orientation correction, within-subject run alignment & averaging |
| 3 | **enhance** | Conforming, T2w→T1w registration, template registration (MEBRAIN), tissue segmentation (deep learning), bias-field correction, vessel detection, ACPC alignment, white matter fix |
| 4 | **surface** | Tessellation, white surface reconstruction, pial surface refinement, FreeSurfer spherical registration, MSM sulcal registration |
| 5 | **resample** | Ribbon generation, resampling to atlas/ACPC/original spaces, cortical annotation, normative statistics (BLR) |
| 6 | **bold_wf** | BOLD run discovery, anatomical reference preparation, fieldmap estimation, motion correction, distortion correction, confound regression, template normalization, volume-to-surface projection, comprehensive QC |

## Project Development

![Project Workflow](../figures/workflow.png)

The complete development cycle — from multi-center data collection, manual label curation, pipeline assembly, to deep learning model training and validation.

## Model Validation

### Generalization Across Centers and Ages

![Generalization](../figures/general.png)

Validated across multiple primate imaging centers and a wide age range, demonstrating consistent performance independent of scanner vendor, field strength, site protocol, or subject age.

### Robustness to Image Noise

![Noise Robustness](../figures/robustness.png)

Systematic noise perturbation experiments confirm that MacaSurfer maintains high segmentation accuracy and surface reconstruction quality even under severely degraded acquisition conditions.

### Test-Retest Reliability

![Test-Retest Reliability](../figures/test-retest.png)

Cortical morphometric measures (thickness, curvature, sulcal depth, volume) derived from repeated within-session scans demonstrate excellent intraclass correlation (ICC > 0.9), validating the pipeline for longitudinal and multi-session experimental designs.

## Quick Start

=== "Container (recommended)"

    ```bash
    singularity run --nv \
      --bind /path/to/data:/data \
      --bind /path/to/output:/output \
      macasurfer.sif \
      --bids_dir /data/rawdata \
      --participant_label sub-001 \
      --out_dir /output
    ```

=== "Source Install"

    ```bash
    sh MacaSurfer.sh \
      --bids_dir /path/to/bids/rawdata \
      --participant_label sub-001 \
      --out_dir /path/to/output
    ```

## Documentation

| Guide | Description |
|---|---|
| **[Installation Guide](INSTALLATION.md)** | Container (macasurfer.sif) usage & full source installation with step-by-step instructions |
| **[Parameter Reference](PARAMETERS.md)** | Complete parameters grouped by processing stage with CLI flags, defaults, and usage examples |

## Related Projects

MacaSurfer integrates several standalone tools, each available as an independent repository:

| Tool | Repository | Description |
|------|-----------|-------------|
| **AutoOrientation** | [yahuiwei123/AutoOrientation](https://github.com/yahuiwei123/AutoOrientation) | Automatic anatomical orientation correction for raw NIfTI volumes |
| **MacaBrainNet** | [yahuiwei123/MacaBrainNet](https://github.com/yahuiwei123/MacaBrainNet) | Ensemble 3D U-Net for whole-brain tissue segmentation (GM, WM, CSF) |
| **Macaque Normative Modeling** | [yahuiwei123/MacaqueNormativeModeling](https://github.com/yahuiwei123/MacaqueNormativeModeling) | Bayesian linear regression normative models for cortical morphometry |
| **Surface-Aware Registration** | [yahuiwei123/SurfaceAwareRegistration](https://github.com/yahuiwei123/SurfaceAwareRegistration) | Cortical surface geometry-constrained volume registration |
| **Tissue-Guided Bias Field Correction** | [yahuiwei123/TissueGuidedBiasFieldCorrection](https://github.com/yahuiwei123/TissueGuidedBiasFieldCorrection) | Tissue-label-informed high- and low-frequency bias field correction |

## Citation

If you use MacaSurfer in your research, please cite:

> Wei, Y. et al. MacaSurfer: unified surface-volume mapping of the macaque brain across the lifespan. 2026.06.14.732101 Preprint at https://doi.org/10.64898/2026.06.14.732101 (2026).
