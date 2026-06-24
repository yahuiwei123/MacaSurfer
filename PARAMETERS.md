# MacaSurfer Parameter Reference

> **Single source of truth**: `nextflow/macasurfer.common.config`  
> All defaults are defined there. CLI flags (via `MacaSurfer.sh`) override them.

## Pipeline Architecture

MacaSurfer is a Nextflow-based MRI processing pipeline for non-human primates (NHP), adapted from the Human Connectome Project (HCP) pipelines. It processes structural and functional MRI from BIDS input through 5 main workflows:

| # | Workflow | Description |
|---|----------|-------------|
| 1 | **info** | Parse BIDS YAML configs, initialize per-session output directories |
| 2 | **prepare** | Skull stripping, brainmask fixing, orientation correction, run alignment & averaging |
| 3 | **enhance** | Conforming, modality registration, template registration, tissue segmentation, bias-field correction, vessel detection, ACPC alignment, white matter fix |
| 4 | **surface** | Tessellation, white/pial surface reconstruction, spherical & MSM sulcal registration |
| 5 | **resample** | Ribbon generation, resampling to atlas/ACPC/original space, cortical annotation, normative stats |
| 6 | **bold_wf** | BOLD discovery, anatomical prep, fieldmap estimation, preprocessing, confounds, normalization, volume-to-surface projection, QC |

---

## Quick Reference

| Parameter | Default | Stage | Description |
|-----------|---------|-------|-------------|
| `bids_dir` | *(required)* | All | BIDS input directory |
| `participant_label` | *(required)* | All | Subject ID(s) |
| `out_dir` | *(required)* | All | Output root directory |
| `session_id` | `""` | All | Session ID filter |
| `process_stage` | `"all"` | All | Stage selector |
| `before_check` | `"false"` | info | Generate config only |
| `after_check` | `"false"` | All | Skip config init |
| `anat_only` | `"false"` | All | Structural only |
| `bold_only` | `"false"` | All | BOLD only |
| `debug` | `"false"` | All | Print params dump |
| `gpus` | `null` | All | GPU indices |
| `per_gpu` | `null` | All | GPU concurrency |
| `device` | `"gpu"` | enhance | GPU/CPU for registration |
| `bold_task_type` | `""` | bold_wf | BOLD task filter |
| `bold_skip_frame` | `0` | bold_wf | Skip initial frames |
| `bold_bandpass` | `"0.01-0.08"` | bold_wf | Bandpass filter |
| `bold_sdc` | `"true"` | bold_wf | Distortion correction |
| `bold_reg_method` | `"flirt"` | bold_wf | Registration method |
| `bold_confounds` | `"true"` | bold_wf | Confound regressors |
| `bold_cifti` | `"false"` | bold_wf | CIFTI output |
| `bold_volume_space` | `"MEBRAIN"` | bold_wf | Normalization space |
| `seg_tool` | `"macabrainnet"` | enhance | Segmentation tool |
| `bfc_method` | `"gauss"` | enhance | Bias correction |
| `fix_white` | `"false"` | enhance | WM fix |
| `vessel_detect` | `"false"` | enhance | Vessel detection |
| `deep_white` | `"false"` | surface | Deep WM surface |
| `t2_refine_pial` | `"true"` | surface | T2w pial refinement |
| `qc_grid_rows` | `6` | prepare | QC mosaic rows |
| `qc_grid_cols` | `6` | prepare | QC mosaic columns |
| `denoise_rician_rad` | `2` | prepare | DenoiseImage radius |
| `tessellation_cores` | `8` | surface | Tessellation cores |
| `reg_name` | `"MSMSulc"` | surface/resample | Registration name |
| `high_res_mesh` | `"164"` | surface/resample | Mesh resolution |
| `species` | `"Macaque"` | All | Species selector |
| `brain_size` | `"60"` | surface | Brain size (mm) |
| `low_res_mesh` | `"32@10"` | resample | Low-res mesh(es) |

---

## 1. Pipeline Control

Parameters that control the overall execution flow.

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `bids_dir` | path | `""` | `--bids_dir` | Root of the BIDS dataset. Must contain `sub-*/ses-*/` folders. |
| `participant_label` | string | `""` | `--participant_label` | Subject label(s) to process, e.g. `sub-032206` or `sub-001,sub-002`. |
| `session_id` | string | `""` | `--session_id` | Session ID(s) to process, e.g. `ses-001`. Empty = all sessions. |
| `out_dir` | path | `""` | `--out_dir` | Output root. Sub-directories per subject/session are created within. |
| `process_stage` | string | `"all"` | `--process_stage` | Run only a specific process stage. Values: `all`, `prepare`, `enhance`, `surface`, `resample`, `bold`, `biasfield`, `detect_vessel`, `fake_t2`, `acpc_isotropy`, `fix_wm`, `tessel`, `white`, `pial`. |
| `before_check` | bool | `"false"` | `--before_check` | If `"true"`, only generate BIDS→YAML configuration (QC directory setup). |
| `after_check` | bool | `"false"` | `--after_check` | If `"true"`, skip config init and start directly from an existing QC directory. |
| `anat_only` | bool | `"false"` | `--anat_only` | Skip the BOLD fMRI pipeline; run structural processing only. |
| `bold_only` | bool | `"false"` | `--bold_only` | Skip the structural pipeline; run BOLD preprocessing only. Requires structural output from a previous run. |
| `debug` | bool | `"false"` | `--debug` | Print the full params map before pipeline execution. |

### Process stage values in detail

| Value | Workflow | What runs |
|-------|----------|-----------|
| `all` | All | Full pipeline (default) |
| `prepare` | prepare | Skullstrip, brainmask fix, orientation, alignment, average |
| `enhance` | enhance | Full enhancement pipeline |
| `surface` | surface | Full surface pipeline |
| `resample` | resample | Full resample pipeline |
| `bold` | bold_wf | Full BOLD pipeline |
| `biasfield` | enhance (subset) | Bias field correction only |
| `detect_vessel` | enhance (subset) | Vessel detection only |
| `fake_t2` | enhance (subset) | Fake T2w generation only |
| `acpc_isotropy` | enhance (subset) | ACPC alignment + isotropy only |
| `fix_wm` | enhance (subset) | White matter fix only |
| `tessel` | surface (subset) | Tessellation only |
| `white` | surface (subset) | White surface only |
| `pial` | surface (subset) | Pial surface only |

---

## 2. GPU / Resources

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `gpus` | string | `null` | `--gpus` | Comma-separated GPU indices, e.g. `"0,1"`. `null` = auto-detect all GPUs. |
| `per_gpu` | int | `null` | `--per_gpu` | Number of concurrent processes per GPU. `null` = 1. Total maxForks = `per_gpu * GPU_count`. |
| `device` | string | `"gpu"` | `--device` | Compute device for template registration (ANTS). `"gpu"` or `"cpu"`. |

### GPU auto-detection logic

The pipeline runs `nvidia-smi` to discover available GPUs. If `params.gpus` is null, all visible GPUs are used. The `per_gpu` multiplier controls how many process instances can share each GPU:

```
maxForks = (params.per_gpu ?: 1) * GPU_count
```

---

## 3. BOLD Processing

Parameters for the fMRI preprocessing workflow (`bold_wf`).

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `bold_task_type` | string | `""` | `--bold_task_type` | BOLD task name to filter, e.g. `"rest"`. Empty = process all tasks found. |
| `bold_skip_frame` | int | `0` | `--bold_skip_frame` | Number of initial volumes to discard from each BOLD run. |
| `bold_bandpass` | string | `"0.01-0.08"` | `--bold_bandpass` | Bandpass filter range in Hz, format: `low-high`. |
| `bold_sdc` | bool | `"true"` | `--bold_sdc` | Enable susceptibility distortion correction (requires fieldmaps). |
| `bold_reg_method` | string | `"flirt"` | `--bold_reg_method` | Registration method for BOLD-to-T1w alignment. |
| `bold_confounds` | bool | `"true"` | `--bold_confounds` | Compute confound regressor time-series (motion, CSF, WM, etc.). |
| `bold_cifti` | bool | `"false"` | `--bold_cifti` | Generate CIFTI-2 dense timeseries outputs in addition to volume outputs. |
| `bold_volume_space` | string | `"MEBRAIN"` | `--bold_volume_space` | Template space for normalized volume output (e.g., `"MEBRAIN"`). |

### BOLD preprocessing pipeline steps

1. **bold_get_func** — Discover BOLD runs from BIDS structure
2. **bold_anat_prepare** — Extract reference anatomy (brain, mask, WM/GM/CSF probability maps, WM segmentation, FSNative→T1w transform)
3. **bold_fieldmap_estimate** — Estimate fieldmaps for SDC (if `bold_sdc=true`)
4. **bold_preprocess** — Motion correction, distortion correction, registration to T1w
5. **bold_confounds** — Compute confound time-series (if `bold_confounds=true`)
6. **bold_normalize** — Warp to template space
7. **bold_vol2surf** — Project to surface (if `bold_cifti=true`)
8. **QC** — Motion, registration, tSNR, carpet plot, surface, normalization

---

## 4. Enhancement

Parameters for anatomical image preprocessing (`enhance` workflow).

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `seg_tool` | string | `"macabrainnet"` | `--seg_tool` | Tissue segmentation tool. `"macabrainnet"` (ensemble deep-learning model, recommended) or `"nbest"` (legacy nnU-Net). |
| `bfc_method` | string | `"gauss"` | `--bfc_method` | Bias-field correction. Options: `"gauss"` (Gaussian mixture, recommended), `"rbf"` (RBF scatter), `"n4"` (N4ITK), `"sqrt"` (sqrt(T1w×T2w)), `"none"`. `"tgbfc"` accepted as alias for `"gauss"`. |
| `fix_white` | bool | `"false"` | `--fix_white` | Enable white matter topology fix after segmentation. Corrects holes and handles in the WM surface. |
| `vessel_detect` | bool | `"false"` | `--vessel_detect` | Enable vessel/artery detection. Useful for vascular studies. |
| `deep_white` | bool | `"false"` | `--deep_white` | Reconstruct deep white matter surface (subcortical WM boundary). |
| `t2_refine_pial` | bool | `"true"` | `--t2_refine_pial` | Use T2w contrast to refine the pial surface boundary. Only applies when T2w is available. |
| `qc_grid_rows` | int | `6` | `--qc_grid_cols` | Number of rows in QC mosaic images (skullstrip, etc.). |
| `qc_grid_cols` | int | `6` | `--qc_grid_cols` | Number of columns in QC mosaic images. |
| `denoise_rician_rad` | int | `2` | `--denoise_rician_rad` | Radial window size for `DenoiseImage` (ANTS). Applied to averaged brain/head images when fewer than 4-5 inputs exist. |
| `tessellation_cores` | int | `8` | `--tessellation_cores` | Number of CPU cores for surface tessellation (`TessellationPre.sh`). |

### Tissue segmentation tools

- **macabrainnet** (default): Ensemble of 3D U-Net models trained specifically on macaque brains. Provides tissue probability maps and segmentation labels.
- **nbest** (legacy v2.0): nnU-Net based tissue extractor. Kept for backward compatibility.

### Bias-field correction methods

- **gauss** (default): Gaussian mixture model BFC using tissue priors. Supports both single-label and multilabel (tissue19) modes. Recommended for macaque data. `tgbfc` is accepted as an alias.
- **rbf**: RBF scatter-based BFC. Uses radial basis functions to model the bias field.
- **n4**: Standard N4ITK bias correction. Good general-purpose option.
- **sqrt**: sqrt(T1w × T2w) method. Requires T2w. Improves GM/WM boundary contrast.
- **none**: Skip bias correction entirely.

---

## 5. Registration / Surface

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `reg_name` | string | `"MSMSulc"` | `--reg_name` | Surface registration name. Controls the MSM registration configuration used for spherical alignment. |
| `high_res_mesh` | string | `"164"` | `--high_res_mesh` | High-resolution mesh vertex count (in thousands). 164 = 164k vertices per hemisphere. Must match the surface atlas resolution. |
| `low_res_mesh` | string | `"32@10"` | *(config only)* | Low-resolution mesh(es) for downsampled outputs. Format: `"N@K"` where N=k vertices, K=number of levels. |
| `omp_num_threads` | int | `128` | *(config only)* | OpenMP thread count for ANTs and other multi-threaded tools. |
| `itk_global_default_number_of_threads` | int | `128` | *(config only)* | ITK global thread limit. |

---

## 6. Species-Specific Parameters

Template selection and anatomical constants depend on `species`. Other parameters are config defaults.

| Species | `brain_size` | `high_res_mesh` | `final_fmri_resolution` | Template |
|---------|-------------|-----------------|------------------------|----------|
| **Macaque** (default) | 60 | 164 | 1.25 | MEBRAIN 0.4mm |
| **Human** | 150 | 164 | 2.0 | MNI152 |
| **Marmoset** | 40 | 164 | 0.5 | RIKEN Marmoset |

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `species` | string | `"Macaque"` | Species: `"Macaque"`, `"Human"`, or `"Marmoset"`. Controls template paths in `SetUpHCPPipelineNHP.sh`. |
| `brain_size` | string | `"60"` | Brain size in mm. Used by FreeSurfer surface reconstruction. |
| `high_res_mesh` | string | `"164"` | High-resolution mesh vertex count (k). |
| `low_res_mesh` | string | `"32@10"` | Low-resolution mesh(es). |
| `final_fmri_resolution` | string | `"1.25"` | Final fMRI volume resolution in mm. |
| `smoothing_fwhm` | string | `"1.25"` | Volume smoothing FWHM in mm. |
| `grayordinates_resolution` | string | `"1.25"` | Grayordinates (CIFTI) resolution in mm. |
| `myelin_mapping_fwhm` | string | `"2"` | Myelin mapping smoothing FWHM in mm. |
| `surface_smoothing_fwhm` | string | `"2"` | Surface smoothing FWHM in mm. |
| `correction_sigma` | string | `"5"` | Topology correction sigma. |

---

## 7. Software Paths

Paths to external software dependencies. All paths below are the default installation locations and can be overridden.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `freesurfer_home` | `/home/weiyahui/software/freesurfer-7.3.2` | FreeSurfer installation |
| `fsl_dir` | `/home/weiyahui/software/fsl` | FSL installation |
| `niftyreg_install` | `/home/weiyahui/software/niftyreg` | NiftyReg installation |
| `antspath` | `/home/weiyahui/software/ants-2.6.5/bin` | ANTs binaries |
| `python_env` | `.../miniconda3/envs/macapipe` | Python conda environment |
| `python_inter` | `.../miniconda3/envs/macapipe/bin/python` | Python interpreter |
| `hcppipedir` | `.../macasurfer_v3.0/MacaSurfer` | Pipeline root directory |
| `caret7dir` | `/home/weiyahui/software/workbench/bin_linux64` | Connectome Workbench |
| `msmbindir` | `/home/weiyahui/software/msm/msm_ubuntu_v3` | MSM binaries |
| `fixdir` | `/home/weiyahui/software/fix` | FIX (FMRIB's ICA-based Xnoiseifier) |
| `fs_license_file` | `""` | FreeSurfer license file. Auto-detected if empty. |

---

## 8. Script Locations

Derived from `hcppipedir`. Normally not changed.

| Parameter | Default |
|-----------|---------|
| `prepare_script_dir` | `${hcppipedir}/nextflow/Prepare` |
| `enhance_script_dir` | `${hcppipedir}/nextflow/Enhance` |
| `surface_script_dir` | `${hcppipedir}/nextflow/Surface` |
| `resample_script_dir` | `${hcppipedir}/nextflow/Resample` |

---

## 9. Shared Data / Models

Deep learning models and shared utility scripts.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `shared_dir` | `${hcppipedir}/shared` | Shared scripts root |
| `maca_brainnet_dir` | `/home/weiyahui/projects/monkey/macaBrainNet_v2` | BrainNet ensemble models |
| `maca_unet_path` | `${shared_dir}/brainextractor/macaUNet` | 3D U-Net brain extractor |
| `nbest_model_path` | `${shared_dir}/tissueextractor` | nnU-Net tissue segmenter |
| `utils_path` | `${shared_dir}/utils` | QC and utility scripts |
| `surf_reg_dir` | `${shared_dir}/surf_fireants` | FireANTs surface registration |
| `msmconfigdir` | `${hcppipedir}/MSMConfig` | MSM registration configs |
| `norm_model_path` | `${hcppipedir}/global/normative_model/blr` | Bayesian linear regression normative models |

---

## 10. Atlas & Template Files

MEBRAIN template files for macaque (default species).

| Parameter | Description |
|-----------|-------------|
| `t1w_template` | T1w template (0.4mm, LIA orientation) |
| `t1w_template_brain` | T1w brain-only template |
| `t1w_template_atlas` | Tissue atlas (GCA format) |
| `t1w_template_2mm` | T1w 1mm downsampled template |
| `template_mask` | Template brain mask (0.4mm) |
| `template_2mm_mask` | Template brain mask (1mm) |
| `gca_dir` | GCA atlas directory |
| `wm_compliment1` | WM compliment 1 (subcortical WM) |
| `wm_compliment2` | WM compliment 2 (cerebellar WM) |
| `gm_compliment` | GM compliment |
| `fake_talairch_transform` | Fake Talairach transform file |
| `fnirt_config` | FNIRT config (T1→MNI NHP) |
| `freesurfer_labels` | FreeSurfer label LUT |
| `surface_atlas_dir` | Surface atlas (standard mesh) |
| `grayordinates_space_dir` | Grayordinates space files |
| `reference_myelin_maps` | Reference myelin map (dscalar) |
| `atlases_dir` | Cortical atlas directory (fsaverage 32k) |
| `cortical_atlases` | Space-separated atlas names |

---

## How to Override Parameters

### Method 1: CLI flags (recommended for per-run changes)

```bash
sh MacaSurfer.sh \
  --bids_dir /data/rawdata \
  --participant_label sub-001 \
  --out_dir /data/output \
  --bfc_method n4 \
  --seg_tool nbest \
  --bold_skip_frame 5
```

### Method 2: Custom config file

Create a file (e.g., `my_config.config`) and pass it with `--config_file`:

```groovy
params {
    bfc_method = "sqrt"
    seg_tool = "nbest"
    bold_bandpass = "0.008-0.1"
    omp_num_threads = 16
}
```

```bash
sh MacaSurfer.sh --config_file ./my_config.config ...
```

### Method 3: Environment variables

Software paths and thread counts can be set via environment before launching:

```bash
export OMP_NUM_THREADS=64
sh MacaSurfer.sh ...
```

### Precedence

CLI flags > custom config file > `macasurfer.common.config` defaults.

---

## Usage Examples

### Minimal structural + BOLD run

```bash
sh MacaSurfer.sh \
  --bids_dir /data/PRIME-DE/site-mcgill/rawdata \
  --participant_label sub-032206 \
  --out_dir /data/output
```

### BOLD-only (re-using structural output)

```bash
sh MacaSurfer.sh \
  --bids_dir /data/PRIME-DE/site-mcgill/rawdata \
  --participant_label sub-032206 \
  --out_dir /data/output \
  --bold_only
```

### Structural only with custom settings

```bash
sh MacaSurfer.sh \
  --bids_dir /data/rawdata \
  --participant_label sub-001 \
  --out_dir /data/output \
  --anat_only \
  --seg_tool macabrainnet \
  --bfc_method gauss \
  --fix_white true \
  --t2_refine_pial false
```

### Resume a failed run with debug

```bash
sh MacaSurfer.sh \
  --bids_dir /data/rawdata \
  --participant_label sub-001 \
  --out_dir /data/output \
  --resume --debug
```

### Run only bias-field correction

```bash
sh MacaSurfer.sh \
  --bids_dir /data/rawdata \
  --participant_label sub-001 \
  --out_dir /data/output \
  --process_stage biasfield
```

### Multiple subjects with GPU control

```bash
sh MacaSurfer.sh \
  --bids_dir /data/rawdata \
  --participant_label sub-001,sub-002,sub-003 \
  --out_dir /data/output \
  --gpus 0,1 \
  --per_gpu 2 \
  --device gpu
```

### Custom BOLD preprocessing

```bash
sh MacaSurfer.sh \
  --bids_dir /data/rawdata \
  --participant_label sub-001 \
  --out_dir /data/output \
  --bold_task_type rest \
  --bold_skip_frame 4 \
  --bold_bandpass 0.008-0.08 \
  --bold_sdc false \
  --bold_cifti true \
  --bold_volume_space MEBRAIN
```
