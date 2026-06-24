#!/bin/bash
set -euo pipefail

# =========================================================================
# BOLD Anatomical Prepare
# Bridge step: converts MacaSurfer structural outputs to BOLD-ready format.
# Follows DeepPrep's bold_anat_prepare.py pattern using FSL tools.
# =========================================================================

usage() {
  echo "
Usage: $0 --bold_path <dir> --subj <sub> [--ses <ses>] --t1w_conform <file> --brainmask <file> --nbest_seg <file> [--t2w_conform <file>]

Required:
  --bold_path      BOLD output root (e.g., out_dir/sub-XXX/ses-XXX/BOLD)
  --subj           Subject ID (e.g., sub-032213)
  --t1w_conform    T1w conformed image in native space
  --brainmask      Brain mask (binary)
  --nbest_seg      nBEST 5-class tissue segmentation (1=CSF,2=GM,3=WM,4=CB,5=BS)

Optional:
  --ses            Session ID (e.g., ses-001)
  --t2w_conform    T2w conformed image (for improved tissue segmentation)
"
}

bold_path=""
subj=""
ses=""
t1w_conform=""
brainmask=""
nbest_seg=""
t2w_conform=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --bold_path)    bold_path="$2"; shift 2 ;;
    --subj)         subj="$2"; shift 2 ;;
    --ses)          ses="$2"; shift 2 ;;
    --t1w_conform)  t1w_conform="$2"; shift 2 ;;
    --brainmask)    brainmask="$2"; shift 2 ;;
    --nbest_seg)    nbest_seg="$2"; shift 2 ;;
    --t2w_conform)  t2w_conform="$2"; shift 2 ;;
    --help)         usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# Validation
for var in bold_path subj t1w_conform brainmask nbest_seg; do
  if [ -z "${!var}" ]; then
    echo "ERROR: --${var} is required"; usage; exit 1
  fi
done

for f in "${t1w_conform}" "${brainmask}" "${nbest_seg}"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: File not found: $f"; exit 1
  fi
done

anat_dir="${bold_path}/anat"
mkdir -p "${anat_dir}"

if [ -n "${ses}" ]; then
  prefix="${subj}_${ses}"
else
  prefix="${subj}"
fi

echo "============================================"
echo "BOLD Anatomical Prepare"
echo "  Subject: ${subj}/${ses}"
echo "  T1w:     ${t1w_conform}"
echo "  Mask:    ${brainmask}"
echo "  nBEST:   ${nbest_seg}"
echo "============================================"

# --- 1. Copy/Link T1w reference and brain mask ---
t1w_out="${anat_dir}/${prefix}_desc-preproc_T1w.nii.gz"
mask_out="${anat_dir}/${prefix}_desc-brain_mask.nii.gz"

cp "${t1w_conform}" "${t1w_out}"
cp "${brainmask}" "${mask_out}"
echo "[OK] Copied T1w and brain mask"

# --- 2. Generate tissue probability maps using nBEST segmentation ---
# nBEST labels: 1=CSF, 2=GM, 3=WM, 4=Cerebellum, 5=Brainstem
wm_prob="${anat_dir}/${prefix}_label-WM_probseg.nii.gz"
gm_prob="${anat_dir}/${prefix}_label-GM_probseg.nii.gz"
csf_prob="${anat_dir}/${prefix}_label-CSF_probseg.nii.gz"
wm_dseg="${anat_dir}/${prefix}_label-WM_dseg.nii.gz"

# Create binary tissue masks from nBEST
fslmaths "${nbest_seg}" -thr 3 -uthr 3 -bin "${wm_prob}"
fslmaths "${nbest_seg}" -thr 2 -uthr 2 -bin "${gm_prob}"
fslmaths "${nbest_seg}" -thr 1 -uthr 1 -bin "${csf_prob}"

# Include cerebellum (label 4) and brainstem (label 5) in GM mask too
fslmaths "${nbest_seg}" -thr 4 -uthr 4 -bin -add "${gm_prob}" "${gm_prob}"
fslmaths "${nbest_seg}" -thr 5 -uthr 5 -bin -add "${gm_prob}" "${gm_prob}"

# Ensure tissue masks are in same geometry as T1w
fslcpgeom "${t1w_conform}" "${wm_prob}"
fslcpgeom "${t1w_conform}" "${gm_prob}"
fslcpgeom "${t1w_conform}" "${csf_prob}"

# Create discrete WM segmentation (for fMRIPrep-style input)
cp "${wm_prob}" "${wm_dseg}"
fslmaths "${wm_dseg}" -mul 2 "${wm_dseg}"  # scale to match fMRIPrep convention (2=WM)

echo "[OK] Generated tissue probability maps from nBEST"

# --- 3. Create fsnative-to-T1w identity transform ---
fsnative2t1w_xfm="${anat_dir}/${prefix}_from-fsnative_to-T1w_mode-image_xfm.txt"
cat > "${fsnative2t1w_xfm}" << 'XMFTXT'
#Insight Transform File V1.0
#Transform 0
Transform: AffineTransform_float_3_3
Parameters: 1 0 0 0 1 0 0 0 1 0 0 0
FixedParameters: 0 0 0
XMFTXT
echo "[OK] Created identity fsnative2T1w transform"

# --- Optional: Generate improved tissue maps using T2w if available ---
if [ -n "${t2w_conform}" ] && [ -f "${t2w_conform}" ]; then
  t2w_in_t1w="${anat_dir}/${prefix}_desc-preproc_T2w.nii.gz"
  cp "${t2w_conform}" "${t2w_in_t1w}"
  echo "[OK] Copied T2w reference"
fi

echo "[DONE] BOLD anatomical preparation complete"
echo "  Outputs in: ${anat_dir}/"
ls -la "${anat_dir}/"
