#!/bin/bash
set -euo pipefail

# =========================================================================
# BOLD Normalize to Template
# Applies existing FNIRT/ANTS nonlinear warp to resample preprocessed BOLD
# from T1w native space to MEBRAIN template space.
# =========================================================================

usage() {
  echo "
Usage: $0 --bold_preprocess_dir <dir> --subj <sub> --ses <ses> --bold_id <id> --warp_file <file> --t1w_ref <file> [--template_res <mm>] [--template_space <name>]

Required:
  --bold_preprocess_dir  BOLD preprocessing output directory
  --subj                 Subject ID
  --ses                  Session ID
  --bold_id              BOLD run identifier
  --warp_file            Nonlinear warp from T1w to template (from-T1w_to-MEBRAIN_mode-image_xfm.nii.gz)
  --t1w_ref              T1w reference in native space

Optional:
  --template_res         Template resolution in mm (default: 0.4)
  --template_space       Template space name (default: MEBRAIN)
"
}

bold_preprocess_dir=""
subj=""
ses=""
bold_id=""
warp_file=""
t1w_ref=""
template_res="0.4"
template_space="MEBRAIN"

while [[ $# -gt 0 ]]; do
  case $1 in
    --bold_preprocess_dir) bold_preprocess_dir="$2"; shift 2 ;;
    --subj)                subj="$2"; shift 2 ;;
    --ses)                 ses="$2"; shift 2 ;;
    --bold_id)             bold_id="$2"; shift 2 ;;
    --warp_file)           warp_file="$2"; shift 2 ;;
    --t1w_ref)             t1w_ref="$2"; shift 2 ;;
    --template_res)        template_res="$2"; shift 2 ;;
    --template_space)      template_space="$2"; shift 2 ;;
    --help)                usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

for var in bold_preprocess_dir subj ses bold_id warp_file t1w_ref; do
  if [ -z "${!var}" ]; then echo "ERROR: --${var} is required"; usage; exit 1; fi
done

if [ ! -f "${warp_file}" ]; then
  echo "ERROR: Warp file not found: ${warp_file}"
  exit 1
fi

func_dir="${bold_preprocess_dir}/func"
prefix="${subj}_${ses}"

echo "============================================"
echo "BOLD Normalization: ${bold_id}"
echo "  Template:  ${template_space}"
echo "  Warp:      ${warp_file}"
echo "============================================"

# --- Step 1: Locate T1w-space preprocessed BOLD ---
bold_t1w="${func_dir}/${bold_id}_space-T1w_desc-preproc_bold.nii.gz"
if [ ! -f "${bold_t1w}" ]; then
  echo "ERROR: T1w-space BOLD not found: ${bold_t1w}"
  exit 1
fi

# --- Step 2: Apply warp to each volume ---
echo "[STEP 2] Applying nonlinear warp to template space..."
bold_template="${func_dir}/${bold_id}_space-${template_space}_desc-preproc_bold.nii.gz"

applywarp --ref="${t1w_ref}" \
  --in="${bold_t1w}" \
  --warp="${warp_file}" \
  --out="${bold_template}" \
  --interp=spline

echo "[OK]  Template-space BOLD: ${bold_template}"

# --- Step 3: Generate template-space reference images ---
echo "[STEP 3] Generating template-space BOLD reference..."

# Create 4D mean over time (for QC)
boldref_template="${func_dir}/${bold_id}_space-${template_space}_boldref.nii.gz"
fslmaths "${bold_template}" -Tmean "${boldref_template}"
echo "[OK]  Template-space BOLDref: ${boldref_template}"

echo ""
echo "[DONE] BOLD normalization complete"
echo "  Template: ${bold_template}"
echo "  BOLDref:  ${boldref_template}"
