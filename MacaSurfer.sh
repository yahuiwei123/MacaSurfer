#!/bin/bash
# =============================================================================
# MacaSurfer — NHP MRI Pipeline (Nextflow wrapper)
# =============================================================================
# Usage:  See --help or run without arguments.
#
# Parameter defaults are defined in nextflow/macasurfer.common.config
# (single source of truth). CLI flags here only pass user overrides.
# =============================================================================
if [ -z "${BASH_VERSION:-}" ]; then
  exec /bin/bash "$0" "$@"
fi
set -e

args=("$@")
echo "INFO: args: ${args[*]:-}"

# =============================================================================
# Version
# =============================================================================
VERSION="3.0"

# =============================================================================
# Help text — grouped by pipeline stage
# =============================================================================
help_text() {
  cat <<EOF
MacaSurfer v${VERSION} — NHP MRI Processing Pipeline

USAGE:
  MacaSurfer.sh --bids_dir <path> --participant_label <ID> [OPTIONS]

╔══════════════════════════════════════════════════════════════════════════════╗
║  REQUIRED                                                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  --bids_dir <path>              BIDS-format input directory                 ║
║  --participant_label <ID>       Subject ID (comma-separated for multiple)   ║
║  --out_dir <path>               Output directory                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  DIRECTORIES / RUNTIME                                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  --macasurfer_home <path>       Pipeline install root (default: auto)       ║
║  --config_file <path>           Custom Nextflow config file                 ║
║  --work_dir <path>              Nextflow work directory                     ║
║  --resume                       Resume previous Nextflow run                ║
║  --debug                        Print full parameter dump                   ║
║  --session_id <ID>              Session ID(s), comma-separated              ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  PIPELINE CONTROL                                                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  --anat_only                    Structural-only (skip BOLD)                 ║
║  --bold_only                    BOLD-only (skip structural)                 ║
║  --process_stage <stage>        Run specific stage(s) — see below           ║
║  --before_check                 Generate BIDS→YAML config only              ║
║  --after_check                  Skip config init, start from QC dir         ║
║                                                                             ║
║  process_stage values:                                                      ║
║    all (default), prepare, enhance, surface, resample, bold                 ║
║    biasfield, detect_vessel, fake_t2, acpc_isotropy                         ║
║    fix_wm, tessel, white, pial                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  GPU / RESOURCES                                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  --gpus <indices>               Comma-separated GPU indices (default: auto) ║
║  --per_gpu <N>                  Concurrency per GPU (default: 1)            ║
║  --device <type>                "gpu" or "cpu" for template registration   ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  ENHANCEMENT (anatomical preprocessing)                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  --seg_tool <tool>              Tissue segmentation: macabrainnet | nbest   ║
║  --bfc_method <method>          Bias-field correction: gauss|rbf|n4|sqrt|none║
║  --fix_white <bool>             White matter topology fix (true/false)      ║
║  --vessel_detect <bool>         Vessel/artery detection (true/false)        ║
║  --deep_white <bool>            Deep white surface reconstruction           ║
║  --t2_refine_pial <bool>        Use T2w to refine pial surface              ║
║  --qc_grid_rows <N>             QC mosaic rows (default: 6)                 ║
║  --qc_grid_cols <N>             QC mosaic columns (default: 6)              ║
║  --denoise_rician_rad <N>       DenoiseImage radial window (default: 2)     ║
║  --tessellation_cores <N>       CPU cores for tessellation (default: 8)     ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  REGISTRATION / SURFACE                                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  --reg_name <name>              Surface registration name (default: MSMSulc)║
║  --high_res_mesh <N>            High-resolution mesh (default: 164)         ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  BOLD PROCESSING (fMRI)                                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  --bold_task_type <name>        BOLD task name filter                       ║
║  --bold_skip_frame <N>          Frames to discard from start (default: 0)   ║
║  --bold_bandpass <low>-<high>   Bandpass filter in Hz (default: 0.01-0.08)  ║
║  --bold_sdc <bool>              Susceptibility distortion correction        ║
║  --bold_reg_method <method>     BOLD→T1w registration (default: flirt)      ║
║  --bold_confounds <bool>        Compute confound regressors                 ║
║  --bold_cifti <bool>            Generate CIFTI dense timeseries             ║
║  --bold_volume_space <space>    Normalization template (default: MEBRAIN)   ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  SPECIES                                                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  --species <name>               Macaque (default) | Human | Marmoset        ║
╚══════════════════════════════════════════════════════════════════════════════╝

EXAMPLES:
  # Minimal structural+BOLD run
  MacaSurfer.sh \\
    --bids_dir /data/PRIME-DE/site-mcgill/rawdata \\
    --participant_label sub-032206 \\
    --out_dir /data/output

  # BOLD-only on pre-processed structurals
  MacaSurfer.sh \\
    --bids_dir /data/PRIME-DE/site-mcgill/rawdata \\
    --participant_label sub-032206 \\
    --out_dir /data/output \\
    --bold_only

  # Structural only, with custom segmentation
  MacaSurfer.sh \\
    --bids_dir /data/rawdata \\
    --participant_label sub-001 \\
    --out_dir /data/output \\
    --anat_only \\
    --seg_tool nbest \\
    --bfc_method n4

  # Resume a failed run with debug output
  MacaSurfer.sh \\
    --bids_dir /data/rawdata \\
    --participant_label sub-001 \\
    --out_dir /data/output \\
    --resume --debug

For full parameter documentation, see PARAMETERS.md
EOF
}

# Show help if no args or --help/-h
if [[ $# -eq 0 ]]; then
  help_text
  exit 0
fi
if [[ "$1" == "-h" || "$1" == "--help" || "$1" == "-help" ]]; then
  help_text
  exit 0
fi
if [[ "$1" == "--version" || "$1" == "-version" ]]; then
  echo "MacaSurfer v${VERSION}"
  exit 0
fi

# =============================================================================
# Default paths
# =============================================================================
macasurfer_home="/home/weiyahui/projects/monkey/macasurfer_v3.0/MacaSurfer"
pipeline_file="${macasurfer_home}/nextflow/macasurfer.nf"
default_config="${macasurfer_home}/nextflow/macasurfer.common.config"
config_file=""
output_dir=""
work_dir=""
participant_label=""
bids_dir=""
resume=""
debug=""

# Associative array: only params explicitly provided by user are stored here.
# After parsing, they are appended to nxf_params — the config file handles defaults.
declare -A user_overrides
nxf_params=()

# =============================================================================
# Parse CLI arguments — grouped by stage
# =============================================================================
while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    # -------------------------------------------------------------------------
    # Directories / runtime
    # -------------------------------------------------------------------------
    --macasurfer_home)
      macasurfer_home="$2"
      pipeline_file="${macasurfer_home}/nextflow/macasurfer.nf"
      default_config="${macasurfer_home}/nextflow/macasurfer.common.config"
      shift 2 ;;
    --config_file)   config_file="$2";       shift 2 ;;
    --participant_label) participant_label="$2"; shift 2 ;;
    --bids_dir)      bids_dir="$2";          shift 2 ;;
    --out_dir)       output_dir="$2";         shift 2 ;;
    --work_dir)      work_dir="$2";           shift 2 ;;
    --session_id)    user_overrides["session_id"]="$2"; shift 2 ;;
    --resume)        resume="1";             shift 1 ;;
    --debug)         debug="1";              shift 1 ;;

    # -------------------------------------------------------------------------
    # Pipeline control
    # -------------------------------------------------------------------------
    --anat_only)       user_overrides["anat_only"]="true";       shift 1 ;;
    --bold_only)       user_overrides["bold_only"]="true";       shift 1 ;;
    --process_stage)   user_overrides["process_stage"]="$2";     shift 2 ;;
    --before_check)    user_overrides["before_check"]="true";    shift 1 ;;
    --after_check)     user_overrides["after_check"]="true";     shift 1 ;;

    # -------------------------------------------------------------------------
    # GPU / resources
    # -------------------------------------------------------------------------
    --gpus)     user_overrides["gpus"]="$2";     shift 2 ;;
    --per_gpu)  user_overrides["per_gpu"]="$2";  shift 2 ;;
    --device)   user_overrides["device"]="$2";   shift 2 ;;

    # -------------------------------------------------------------------------
    # Enhancement
    # -------------------------------------------------------------------------
    --seg_tool)         user_overrides["seg_tool"]="$2";       shift 2 ;;
    --bfc_method)       user_overrides["bfc_method"]="$2";     shift 2 ;;
    --fix_white)        user_overrides["fix_white"]="$2";      shift 2 ;;
    --vessel_detect)    user_overrides["vessel_detect"]="$2";  shift 2 ;;
    --deep_white)       user_overrides["deep_white"]="$2";     shift 2 ;;
    --t2_refine_pial)   user_overrides["t2_refine_pial"]="$2"; shift 2 ;;
    --qc_grid_rows)     user_overrides["qc_grid_rows"]="$2";   shift 2 ;;
    --qc_grid_cols)     user_overrides["qc_grid_cols"]="$2";   shift 2 ;;
    --denoise_rician_rad) user_overrides["denoise_rician_rad"]="$2"; shift 2 ;;
    --tessellation_cores) user_overrides["tessellation_cores"]="$2"; shift 2 ;;

    # -------------------------------------------------------------------------
    # Registration / surface
    # -------------------------------------------------------------------------
    --reg_name)       user_overrides["reg_name"]="$2";       shift 2 ;;
    --high_res_mesh)  user_overrides["high_res_mesh"]="$2";  shift 2 ;;

    # -------------------------------------------------------------------------
    # BOLD processing
    # -------------------------------------------------------------------------
    --bold_task_type)     user_overrides["bold_task_type"]="$2";     shift 2 ;;
    --bold_skip_frame)    user_overrides["bold_skip_frame"]="$2";    shift 2 ;;
    --bold_bandpass)      user_overrides["bold_bandpass"]="$2";      shift 2 ;;
    --bold_sdc)           user_overrides["bold_sdc"]="$2";           shift 2 ;;
    --bold_reg_method)    user_overrides["bold_reg_method"]="$2";    shift 2 ;;
    --bold_confounds)     user_overrides["bold_confounds"]="$2";     shift 2 ;;
    --bold_cifti)         user_overrides["bold_cifti"]="$2";         shift 2 ;;
    --bold_volume_space)  user_overrides["bold_volume_space"]="$2";  shift 2 ;;

    # -------------------------------------------------------------------------
    # Species
    # -------------------------------------------------------------------------
    --species)  user_overrides["species"]="$2";  shift 2 ;;

    # -------------------------------------------------------------------------
    # Wildcard: pass any unrecognized --flag [value] directly to Nextflow
    # -------------------------------------------------------------------------
    *)
      nxf_params+=("$key"); shift 1
      if [[ $# -gt 0 && "$1" != -* ]]; then
        nxf_params+=("$1"); shift 1
      fi ;;
  esac
done

# =============================================================================
# Validation
# =============================================================================
if [[ -z "${participant_label:-}" ]]; then
  echo "ERROR: --participant_label is required" >&2
  echo "Run with --help for usage." >&2
  exit 1
fi
if [[ -z "${bids_dir:-}" ]]; then
  echo "ERROR: --bids_dir is required" >&2
  echo "Run with --help for usage." >&2
  exit 1
fi
if [[ ! -d "${bids_dir}" ]]; then
  echo "ERROR: --bids_dir does not exist: ${bids_dir}" >&2
  exit 1
fi
if [[ -z "${output_dir:-}" ]]; then
  echo "WARN: --out_dir not set, using ./MacaSurfer_output" >&2
  output_dir="./MacaSurfer_output"
fi
mkdir -p "${output_dir}"

# =============================================================================
# Config file
# =============================================================================
if [[ -z "${config_file}" ]]; then
  config_file="${default_config}"
fi
if [[ ! -f "${pipeline_file}" ]]; then
  echo "ERROR: pipeline file not found: ${pipeline_file}" >&2
  exit 1
fi
if [[ ! -f "${config_file}" ]]; then
  echo "ERROR: config file not found: ${config_file}" >&2
  exit 1
fi

# =============================================================================
# Work directory
# =============================================================================
if [[ -z "${work_dir}" ]]; then
  work_dir="${output_dir}/WorkDir"
fi
mkdir -p "${work_dir}"

# =============================================================================
# Pass user overrides to Nextflow (config handles defaults)
# =============================================================================
for key in "${!user_overrides[@]}"; do
  nxf_params+=("--${key}" "${user_overrides[$key]}")
done

# =============================================================================
# Debug output
# =============================================================================
if [[ -n "${debug}" ]]; then
  echo "DEBUG: macasurfer_home   = ${macasurfer_home}"
  echo "DEBUG: pipeline_file     = ${pipeline_file}"
  echo "DEBUG: config_file       = ${config_file}"
  echo "DEBUG: bids_dir          = ${bids_dir}"
  echo "DEBUG: participant_label = ${participant_label}"
  echo "DEBUG: out_dir           = ${output_dir}"
  echo "DEBUG: work_dir          = ${work_dir}"
  echo "DEBUG: user_overrides    ="
  for k in "${!user_overrides[@]}"; do
    echo "DEBUG:   --${k} = ${user_overrides[$k]}"
  done
  echo "DEBUG: extra nxf_params  = ${nxf_params[*]}"
fi

# =============================================================================
# Source environment
# =============================================================================
source "${macasurfer_home}/SetUpHCPPipelineNHP.sh"

# =============================================================================
# Build and execute Nextflow command
# =============================================================================
cmd=(/home/weiyahui/software/nextflow run "${pipeline_file}"
     -c "${config_file}"
     -w "${work_dir}/nextflow/${participant_label}"
     --bids_dir "${bids_dir}"
     --participant_label "${participant_label}"
     --out_dir "${output_dir}")

if [[ -n "${resume}" ]]; then
  cmd+=("-resume")
fi
cmd+=("${nxf_params[@]}")

if [[ -n "${debug}" ]]; then
  echo "DEBUG: final command:"
  printf '  %q' "${cmd[@]}"
  echo
fi

"${cmd[@]}"
