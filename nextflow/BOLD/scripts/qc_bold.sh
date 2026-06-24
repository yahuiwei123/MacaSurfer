#!/bin/bash
set -euo pipefail

# =========================================================================
# BOLD QC
# Quality control for BOLD preprocessing: TSNR computation, carpet plot,
# and registration overlay check.
# =========================================================================

usage() {
  echo "
Usage: $0 --bold_preprocess_dir <dir> --qc_dir <dir> --subj <sub> --ses <ses> --bold_id <id> [--t1w_ref <file>]

Required:
  --bold_preprocess_dir  BOLD preprocessing output directory
  --qc_dir               QC output root directory
  --subj                 Subject ID
  --ses                  Session ID
  --bold_id              BOLD run identifier

Optional:
  --t1w_ref              T1w reference for registration overlay
  --python_inter         Python interpreter (default: python3)
  --utils_path           Path to utility scripts (for QC plotting)
"
}

bold_preprocess_dir=""
qc_dir=""
subj=""
ses=""
bold_id=""
t1w_ref=""
python_inter="python3"
utils_path=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --bold_preprocess_dir) bold_preprocess_dir="$2"; shift 2 ;;
    --qc_dir)              qc_dir="$2"; shift 2 ;;
    --subj)                subj="$2"; shift 2 ;;
    --ses)                 ses="$2"; shift 2 ;;
    --bold_id)             bold_id="$2"; shift 2 ;;
    --t1w_ref)             t1w_ref="$2"; shift 2 ;;
    --python_inter)        python_inter="$2"; shift 2 ;;
    --utils_path)          utils_path="$2"; shift 2 ;;
    --help)                usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

for var in bold_preprocess_dir qc_dir subj ses bold_id; do
  if [ -z "${!var}" ]; then echo "ERROR: --${var} is required"; usage; exit 1; fi
done

func_dir="${bold_preprocess_dir}/func"
session_qc_dir="${qc_dir}/${subj}/${ses}"
mkdir -p "${session_qc_dir}"

echo "============================================"
echo "BOLD QC: ${bold_id}"
echo "  QC dir: ${session_qc_dir}"
echo "============================================"

# --- 1. Compute TSNR ---
echo "[QC 1] Computing TSNR..."
bold_preproc="${func_dir}/${bold_id}_desc-preproc_bold.nii.gz"
bold_t1w="${func_dir}/${bold_id}_space-T1w_desc-preproc_bold.nii.gz"

if [ -f "${bold_preproc}" ]; then
  tsnr_img="${session_qc_dir}/${bold_id}_tsnr.nii.gz"
  fslmaths "${bold_preproc}" -Tmean "${func_dir}/${bold_id}_mean_tmp.nii.gz"
  fslmaths "${bold_preproc}" -Tstd "${func_dir}/${bold_id}_std_tmp.nii.gz"
  fslmaths "${func_dir}/${bold_id}_mean_tmp.nii.gz" -div "${func_dir}/${bold_id}_std_tmp.nii.gz" "${tsnr_img}"

  # Generate TSNR PNG using slicer
  tsnr_png="${session_qc_dir}/qc_bold_tsnr_${bold_id}.png"
  slicer "${tsnr_img}" -i 0 100 -a "${tsnr_png}" 2>/dev/null || {
    # Fallback: use python matplotlib
    ${python_inter} -c "
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

img = nib.load('${tsnr_img}')
data = img.get_fdata()
mid_z = data.shape[2] // 2
plt.figure(figsize=(8, 6))
plt.imshow(np.rot90(data[:, :, mid_z]), cmap='hot', vmin=0, vmax=100)
plt.colorbar(label='TSNR')
plt.title('${bold_id} TSNR (z=${mid_z})')
plt.tight_layout()
plt.savefig('${tsnr_png}', dpi=100)
plt.close()
" 2>/dev/null || echo "[WARN] Could not generate TSNR plot"
  }
  rm -f "${func_dir}/${bold_id}_mean_tmp.nii.gz" "${func_dir}/${bold_id}_std_tmp.nii.gz"
  echo "[OK]  TSNR: ${tsnr_png}"

  # Compute mean TSNR
  mean_tsnr=$(fslstats "${tsnr_img}" -M 2>/dev/null || echo "N/A")
  echo "       Mean TSNR: ${mean_tsnr}"
else
  echo "[WARN] Preprocessed BOLD not found for TSNR: ${bold_preproc}"
fi

# --- 2. Carpet plot ---
echo "[QC 2] Generating carpet plot..."
bold_mask="${func_dir}/${bold_id}_desc-brain_mask.nii.gz"
carpet_png="${session_qc_dir}/qc_bold_carpet_${bold_id}.png"

if [ -f "${bold_preproc}" ] && [ -f "${bold_mask}" ]; then
  ${python_inter} -c "
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load data
bold = nib.load('${bold_preproc}').get_fdata()
mask = nib.load('${bold_mask}').get_fdata() > 0

# Extract brain voxels
n_t = bold.shape[3]
brain_voxels = bold[mask, :]

# Subsample for speed
max_vox = 2000
if brain_voxels.shape[0] > max_vox:
  idx = np.linspace(0, brain_voxels.shape[0]-1, max_vox, dtype=int)
  brain_voxels = brain_voxels[idx, :]

# Normalize each voxel
brain_voxels = brain_voxels - np.mean(brain_voxels, axis=1, keepdims=True)
brain_voxels = brain_voxels / (np.std(brain_voxels, axis=1, keepdims=True) + 1e-8)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [4, 1]})

# Carpet
axes[0].imshow(brain_voxels, aspect='auto', cmap='gray', vmin=-3, vmax=3)
axes[0].set_ylabel('Brain voxels')
axes[0].set_title('${bold_id} Carpet Plot')

# Global signal
gs = np.mean(brain_voxels, axis=0)
axes[1].plot(gs, 'b', linewidth=0.5)
axes[1].set_xlabel('Time (volumes)')
axes[1].set_ylabel('GS')

plt.tight_layout()
plt.savefig('${carpet_png}', dpi=100)
plt.close()
print('[OK] Carpet plot saved')
" 2>/dev/null || echo "[WARN] Could not generate carpet plot"
  echo "[OK]  Carpet: ${carpet_png}"
else
  echo "[WARN] Missing inputs for carpet plot"
fi

# --- 3. Registration overlay ---
echo "[QC 3] Checking registration overlay..."
reg_png="${session_qc_dir}/qc_bold_registration_${bold_id}.png"

if [ -f "${bold_t1w}" ] && [ -n "${t1w_ref}" ] && [ -f "${t1w_ref}" ]; then
  # Generate overlay of BOLD edges on T1w
  boldref_t1w="${func_dir}/${bold_id}_space-T1w_boldref_tmp.nii.gz"
  fslmaths "${bold_t1w}" -Tmean "${boldref_t1w}"

  ${python_inter} -c "
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage

bold = nib.load('${boldref_t1w}').get_fdata()
t1w = nib.load('${t1w_ref}').get_fdata()

# Get middle slice
z = min(bold.shape[2], t1w.shape[2]) // 2
bold_slice = bold[:, :, z].squeeze()
t1w_slice = t1w[:, :min(t1w.shape[2], z)].squeeze()

# Edge detection on BOLD
bold_edges = np.abs(ndimage.sobel(bold_slice))

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(np.rot90(t1w_slice), cmap='gray')
ax.imshow(np.rot90(bold_edges), cmap='Reds', alpha=0.5)
ax.set_title('${bold_id} BOLD→T1w Overlay')
ax.axis('off')
plt.tight_layout()
plt.savefig('${reg_png}', dpi=100)
plt.close()
" 2>/dev/null || echo "[WARN] Could not generate registration overlay"
  rm -f "${boldref_t1w}"
  echo "[OK]  Registration: ${reg_png}"
else
  echo "[WARN] Missing inputs for registration overlay"
fi

rm -f "${func_dir}/"*_tmp.nii.gz 2>/dev/null || true

echo ""
echo "[DONE] BOLD QC complete for ${bold_id}"
echo "  QC outputs in: ${session_qc_dir}/"
