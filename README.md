# MacaSurfer v3.0

NHP MRI Processing Pipeline based on the Human Connectome Project (HCP) pipelines.

---

## Pipeline Overview

![MacaSurfer Pipeline](figures/pipeline.tif)

MacaSurfer processes structural and functional MRI data from BIDS format through six Nextflow workflows:

| # | Workflow | Description |
|---|----------|-------------|
| 1 | **info** | Parse BIDS structure, initialize session directories |
| 2 | **prepare** | Skull stripping, brainmask fixing, alignment, averaging |
| 3 | **enhance** | Conforming, template registration, tissue segmentation, bias correction |
| 4 | **surface** | Tessellation, white/pial surface reconstruction, spherical registration |
| 5 | **resample** | Resampling to atlas space, annotation, normative statistics |
| 6 | **bold_wf** | BOLD preprocessing, confound regression, normalization, surface projection |

## Project Workflow

![Workflow](figures/workflow.tif)

The complete development process behind MacaSurfer — from multi-center data collection, pipeline assembly, manual label curation, to deep learning model training.

## Model Validation

### Generalization across centers and ages

![Generalization](figures/general.tif)

MacaSurfer's segmentation and surface reconstruction were validated across multiple primate imaging centers and a wide age range, demonstrating consistent performance independent of scanner, site, or subject age.

### Robustness to image noise

![Noise Robustness](figures/robustness.tif)

Artificial noise was systematically added to test images to evaluate the pipeline's stability under degraded acquisition conditions. MacaSurfer maintains high segmentation and surface reconstruction quality even under substantial noise perturbations.

### Test-retest reliability

![Test-Retest](figures/test-retest.tif)

Cortical morphometric measures (thickness, curvature, volume) derived from repeated scans of the same session show high intraclass correlation, confirming the pipeline's reproducibility and suitability for longitudinal and multi-session studies.

## Quick Start

### Container (recommended)

```bash
singularity run --nv \
  --bind /path/to/data:/data \
  --bind /path/to/output:/output \
  macasurfer.sif \
  --bids_dir /data/rawdata \
  --participant_label sub-001 \
  --out_dir /output
```

### Source install

```bash
sh MacaSurfer.sh \
  --bids_dir /path/to/bids/rawdata \
  --participant_label sub-001 \
  --out_dir /path/to/output
```

## Documentation

| Document | Description |
|---|---|
| **[docs/INSTALLATION.md](docs/INSTALLATION.md)** | Full installation guide — container usage, source install step-by-step, troubleshooting |
| **[PARAMETERS.md](PARAMETERS.md)** | Complete parameter reference grouped by processing stage, with usage examples |
| **[requirements.txt](requirements.txt)** | Python package list for the `macapipe` conda environment |

## Requirements

- **Container**: Singularity/Apptainer ≥ 3.5 (no other dependencies)
- **Source install**: Nextflow, FreeSurfer 7.3.2, FSL, ANTs 2.5+, Connectome Workbench, MSM, Python 3.10 (`macapipe` env), MATLAB Runtime (optional)

See **[docs/INSTALLATION.md](docs/INSTALLATION.md)** for detailed setup instructions.
