#!/bin/bash
set -euo pipefail

# =========================================================================
# BOLD Get Functional Files
# Discovers BOLD fMRI runs in a BIDS directory, writes per-run text files
# following DeepPrep's pattern for Nextflow channel consumption.
# =========================================================================

usage() {
  echo "
Usage: $0 --bids_dir <dir> --out_dir <dir> [--participant_label <sub>] [--session_id <ses>] [--bold_task_type <task>] [--bold_only true|false]

Required:
  --bids_dir           BIDS dataset root directory
  --out_dir            Output directory (BOLD files written under out_dir/BOLD/)

Optional:
  --participant_label  Subject ID(s) to process (space-separated)
  --session_id         Session ID(s) to process (space-separated)
  --bold_task_type     Task label(s) to filter (space-separated, e.g. 'rest')
  --bold_only          If 'true', skip T1w availability check (default: false)
"
}

bids_dir=""
out_dir=""
participant_label=""
session_id=""
bold_task_type=""
bold_only="false"

while [[ $# -gt 0 ]]; do
  case $1 in
    --bids_dir)           bids_dir="$2"; shift 2 ;;
    --out_dir)            out_dir="$2"; shift 2 ;;
    --participant_label)  participant_label="$2"; shift 2 ;;
    --session_id)         session_id="$2"; shift 2 ;;
    --bold_task_type)     bold_task_type="$2"; shift 2 ;;
    --bold_only)          bold_only="$2"; shift 2 ;;
    --help)               usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [ -z "${bids_dir}" ] || [ -z "${out_dir}" ]; then
  echo "ERROR: --bids_dir and --out_dir are required"
  usage; exit 1
fi

bold_preprocess_path="${out_dir}/BOLD"
mkdir -p "${bold_preprocess_path}"

# Also create output in CWD for Nextflow to discover
mkdir -p "BOLD"

# Find all BOLD NIfTI files
echo "[INFO] Searching for BOLD files in ${bids_dir}..."

bold_files=$(find "${bids_dir}" -maxdepth 5 -name '*_bold.nii.gz' -type f | sort)

if [ -z "${bold_files}" ]; then
  echo "[ERROR] No BOLD files found in ${bids_dir}"
  exit 1
fi

# Filter by subject
if [ -n "${participant_label}" ]; then
  filtered=""
  for sub in ${participant_label}; do
    sub_match=$(echo "${bold_files}" | grep "/${sub}/" || true)
    filtered="${filtered}${sub_match}"$'\n'
  done
  bold_files=$(echo "${filtered}" | grep '_bold.nii.gz' | sort)
fi

# Filter by session
if [ -n "${session_id}" ]; then
  filtered=""
  for ses in ${session_id}; do
    ses_match=$(echo "${bold_files}" | grep "/${ses}/" || true)
    filtered="${filtered}${ses_match}"$'\n'
  done
  bold_files=$(echo "${filtered}" | grep '_bold.nii.gz' | sort)
fi

# Filter by task type
if [ -n "${bold_task_type}" ]; then
  filtered=""
  for task in ${bold_task_type}; do
    task_match=$(echo "${bold_files}" | grep "task-${task}" || true)
    filtered="${filtered}${task_match}"$'\n'
  done
  bold_files=$(echo "${filtered}" | grep '_bold.nii.gz' | sort)
fi

if [ -z "${bold_files}" ]; then
  echo "[ERROR] No BOLD files match the specified filters"
  echo "  participant: ${participant_label:-any}"
  echo "  session:     ${session_id:-any}"
  echo "  task:        ${bold_task_type:-any}"
  exit 1
fi

# Write per-run files (DeepPrep style: 2-line file with sub-id + bold path)
count=0
for bold_file in ${bold_files}; do
  # Extract BIDS entities from path
  bold_basename=$(basename "${bold_file}" .nii.gz)
  sub=$(echo "${bold_basename}" | grep -oP 'sub-[^_]+' | head -1)
  ses=$(echo "${bold_basename}" | grep -oP 'ses-[^_]+' || echo "")

  if [ -z "${sub}" ]; then
    echo "[WARN] Cannot parse subject from: ${bold_file}"
    continue
  fi

  # Write subject-level output file (out_dir for downstream use)
  sub_out_dir="${bold_preprocess_path}/${sub}"
  mkdir -p "${sub_out_dir}"

  # Per-run job file named by bold_id (suffix before _bold)
  job_file="${sub_out_dir}/${bold_basename}"

  # DeepPrep format: line 0 = sub-XXX, line 1 = full path to BOLD
  echo "${sub}" > "${job_file}"
  echo "${bold_file}" >> "${job_file}"

  # Also write to CWD for Nextflow process output discovery
  mkdir -p "BOLD/${sub}"
  echo "${sub}" > "BOLD/${sub}/${bold_basename}"
  echo "${bold_file}" >> "BOLD/${sub}/${bold_basename}"

  echo "[INFO] ${bold_basename} -> ${job_file}"
  count=$((count + 1))
done

echo "[DONE] Discovered ${count} BOLD runs across $(echo "${bold_files}" | grep -oP 'sub-[^/]+' | sort -u | wc -l) subjects"
