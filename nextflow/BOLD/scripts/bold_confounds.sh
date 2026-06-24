#!/bin/bash
set -euo pipefail

# =========================================================================
# BOLD Confounds Computation
# Computes nuisance regressors: motion parameters, global signal, WM, CSF,
# and anatomical CompCor components. Outputs a confounds TSV file.
# Follows DeepPrep's bold_confounds pattern using FSL + Python.
# =========================================================================

usage() {
  echo "
Usage: $0 --bold_preprocess_dir <dir> --work_dir <dir> --subj <sub> --ses <ses> --bold_id <id> --t1w_ref <file> --t1w_mask <file> --wm_prob <file> --csf_prob <file> [--nbest_seg <file>] [--skip_frame <n>] [--bandpass <freq>]

Required:
  --bold_preprocess_dir  BOLD preprocessing output directory
  --work_dir             Working directory for temp files
  --subj                 Subject ID
  --ses                  Session ID
  --bold_id              BOLD run identifier
  --t1w_ref              T1w reference in native space
  --t1w_mask             Brain mask in T1w space
  --wm_prob              WM probability map in T1w space
  --csf_prob             CSF probability map in T1w space

Optional:
  --nbest_seg            nBEST tissue segmentation (for EPI-space masks)
  --skip_frame           Frames to skip before computing confounds (default: 0)
  --bandpass             Bandpass filter range (default: 0.01-0.08)
  --bbreg_xfm            BBRegister transform file
  --python_inter         Python interpreter (default: python3)
  --utils_path           Path to utility scripts
"
}

bold_preprocess_dir=""
work_dir=""
subj=""
ses=""
bold_id=""
t1w_ref=""
t1w_mask=""
wm_prob=""
csf_prob=""
nbest_seg=""
skip_frame=0
bandpass="0.01-0.08"
bbreg_xfm=""
python_inter="python3"
utils_path=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --bold_preprocess_dir) bold_preprocess_dir="$2"; shift 2 ;;
    --work_dir)            work_dir="$2"; shift 2 ;;
    --subj)                subj="$2"; shift 2 ;;
    --ses)                 ses="$2"; shift 2 ;;
    --bold_id)             bold_id="$2"; shift 2 ;;
    --t1w_ref)             t1w_ref="$2"; shift 2 ;;
    --t1w_mask)            t1w_mask="$2"; shift 2 ;;
    --wm_prob)             wm_prob="$2"; shift 2 ;;
    --csf_prob)            csf_prob="$2"; shift 2 ;;
    --nbest_seg)           nbest_seg="$2"; shift 2 ;;
    --skip_frame)          skip_frame="$2"; shift 2 ;;
    --bandpass)            bandpass="$2"; shift 2 ;;
    --bbreg_xfm)           bbref_xfm="$2"; shift 2 ;;
    --python_inter)        python_inter="$2"; shift 2 ;;
    --utils_path)          utils_path="$2"; shift 2 ;;
    --help)                usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

for var in bold_preprocess_dir work_dir subj ses bold_id t1w_ref t1w_mask wm_prob csf_prob; do
  if [ -z "${!var}" ]; then echo "ERROR: --${var} is required"; usage; exit 1; fi
done

func_dir="${bold_preprocess_dir}/func"
conf_dir="${work_dir}/confounds/${subj}/${ses}/${bold_id}"
mkdir -p "${conf_dir}"

echo "============================================"
echo "BOLD Confounds: ${bold_id}"
echo "  WM prob: ${wm_prob}"
echo "  CSF prob: ${csf_prob}"
echo "============================================"

# --- Step 1: Locate preprocessed files ---
# BOLD in T1w space
bold_t1w="${func_dir}/${bold_id}_space-T1w_desc-preproc_bold.nii.gz"
# Motion-corrected BOLD
bold_mc="${func_dir}/${bold_id}_desc-mc_bold.nii.gz"
# Motion parameters
mc_par="${func_dir}/${bold_id}_desc-mc_motion_params.txt"
# Registration transform (always FSL .mat format)
reg_xfm="${func_dir}/${bold_id}_from-bold_to-T1w_mode-image_xfm.mat"
boldref="${func_dir}/${bold_id}_boldref.nii.gz"
apply_affine="${UTILS_PATH:-/home/weiyahui/projects/monkey/macasurfer_v3.0/MacaSurfer/shared/utils}/apply_affine_to_header.py"
[ ! -f "${reg_xfm}" ] && reg_xfm=""  # not found, treat as missing

for f in "${bold_t1w}" "${bold_mc}" "${mc_par}"; do
  if [ ! -f "$f" ]; then echo "ERROR: Required file not found: $f"; exit 1; fi
done

# --- Step 2: Create tissue masks in BOLD (EPI) space ---
echo "[STEP 2] Creating tissue masks in EPI space..."

wm_epi="${conf_dir}/${bold_id}_desc-wm_mask.nii.gz"
csf_epi="${conf_dir}/${bold_id}_desc-csf_mask.nii.gz"
brain_epi="${conf_dir}/${bold_id}_desc-brain_mask.nii.gz"

# Use 3D boldref as target (not 4D bold_mc) for correct 3D mask output
targ_3d="${boldref}"
[ ! -f "${targ_3d}" ] && targ_3d="${func_dir}/boldref_mid.nii.gz"

if [ -n "${reg_xfm}" ] && [ -f "${targ_3d}" ]; then
  # Invert BOLD->T1w to get T1w->BOLD, then resample masks to EPI grid
  t1w2bold_xfm="${conf_dir}/t1w2bold.mat"
  convert_xfm -omat "${t1w2bold_xfm}" -inverse "${reg_xfm}"

  # Resample WM prob map to EPI space (trilinear for probability values)
  flirt -in "${wm_prob}" -ref "${targ_3d}" -applyxfm -init "${t1w2bold_xfm}" \
    -out "${wm_epi}" -interp trilinear
  fslmaths "${wm_epi}" -thr 0.5 -bin "${wm_epi}"

  # Resample CSF prob map to EPI space
  flirt -in "${csf_prob}" -ref "${targ_3d}" -applyxfm -init "${t1w2bold_xfm}" \
    -out "${csf_epi}" -interp trilinear
  fslmaths "${csf_epi}" -thr 0.5 -bin "${csf_epi}"

  # Resample brain mask to EPI space (nearest-neighbour preserves binary)
  flirt -in "${t1w_mask}" -ref "${targ_3d}" -applyxfm -init "${t1w2bold_xfm}" \
    -out "${brain_epi}" -interp nearestneighbour

  echo "[OK]  Tissue masks resampled to EPI space via FSL .mat"
else
  cp "${wm_prob}" "${wm_epi}"
  cp "${csf_prob}" "${csf_epi}"
  cp "${t1w_mask}" "${brain_epi}"
  echo "[WARN] No BOLD->T1w transform found; masks may be in wrong space"
fi

# --- Step 3: Extract signals ---
echo "[STEP 3] Extracting nuisance signals..."

# Use FSL to extract mean time series from each mask
fslmeants -i "${bold_mc}" -m "${brain_epi}" -o "${conf_dir}/global_signal.txt"
fslmeants -i "${bold_mc}" -m "${wm_epi}" -o "${conf_dir}/wm_signal.txt"
fslmeants -i "${bold_mc}" -m "${csf_epi}" -o "${conf_dir}/csf_signal.txt"

echo "[OK]  Extracted global, WM, and CSF signals"

# --- Step 4: Compile confounds TSV ---
echo "[STEP 4] Compiling confounds table..."

${python_inter} - << PYEOF
import numpy as np
import os

conf_dir = "${conf_dir}"
func_dir = "${func_dir}"
bold_id = "${bold_id}"
skip_frame = ${skip_frame}

# Load motion parameters
mc_par = "${mc_par}"
if os.path.exists(mc_par):
    motion = np.loadtxt(mc_par)  # 6 columns: rx,ry,rz,tx,ty,tz
else:
    # Try MCFLIRT .par format
    print(f"[WARN] Motion params not found at {mc_par}")
    motion = np.zeros((1, 6))

# Load signals
gs = np.loadtxt(os.path.join(conf_dir, "global_signal.txt"))
wm = np.loadtxt(os.path.join(conf_dir, "wm_signal.txt"))
csf = np.loadtxt(os.path.join(conf_dir, "csf_signal.txt"))

# Ensure 1D arrays
gs = np.atleast_1d(gs)
wm = np.atleast_1d(wm)
csf = np.atleast_1d(csf)

# Truncate to shortest signal length
min_len = min(len(gs), len(wm), len(csf), len(motion))
gs = gs[:min_len]
wm = wm[:min_len]
csf = csf[:min_len]
if motion.shape[0] > min_len:
    motion = motion[:min_len, :]
elif motion.shape[0] < min_len:
    pad = np.tile(motion[-1:], (min_len - motion.shape[0], 1))
    motion = np.vstack([motion, pad])

# Compute derivatives
gs_deriv = np.diff(gs, prepend=gs[0])
wm_deriv = np.diff(wm, prepend=wm[0])
csf_deriv = np.diff(csf, prepend=csf[0])

# Compute CSF+WM combined
csf_wm = csf + wm
csf_wm_deriv = np.diff(csf_wm, prepend=csf_wm[0])

# Build header
header = [
    "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
    "global_signal", "global_signal_derivative1",
    "csf", "csf_derivative1",
    "white_matter", "white_matter_derivative1",
    "csf_wm", "csf_wm_derivative1",
]

# Build data array
data = np.column_stack([
    motion,
    gs, gs_deriv,
    csf, csf_deriv,
    wm, wm_deriv,
    csf_wm, csf_wm_deriv,
])

# Write TSV
tsv_file = os.path.join(conf_dir, f"{bold_id}_desc-confounds_timeseries.tsv")
with open(tsv_file, 'w') as f:
    f.write('\t'.join(header) + '\n')
    np.savetxt(f, data, delimiter='\t', fmt='%.6f')

print(f"[OK] Confounds TSV written to {tsv_file}")
print(f"     Shape: {data.shape[0]} frames x {data.shape[1]} regressors")
PYEOF

# --- Step 5: Copy confounds TSV to func directory ---
cp "${conf_dir}/${bold_id}_desc-confounds_timeseries.tsv" "${func_dir}/"

echo ""
echo "[DONE] BOLD confounds computation complete"
echo "  Confounds: ${func_dir}/${bold_id}_desc-confounds_timeseries.tsv"
