#!/bin/bash
set -euo pipefail

# =========================================================================
# BOLD Pre-Processing
# Core fMRI preprocessing: frame skipping, reorientation, motion correction,
# and BOLD-to-T1w coregistration via FreeSurfer BBRegister.
# Replaces fMRIPrep's init_bold_wf for macaque data.
# =========================================================================

usage() {
  echo "
Usage: $0 --bold_file <file> --bold_preprocess_dir <dir> --subj <sub> --ses <ses> --t1w_ref <file> --t1w_mask <file> --t1w_dseg <file> --subjects_dir <dir> --bids_dir <dir> [--fs_license <file>] [--skip_frame <n>] [--sdc <true|false>] [--viz <true|false>]

Required:
  --bold_file             Original BOLD NIfTI file from BIDS
  --bold_preprocess_dir   Output root (bold output directory)
  --subj                  Subject ID (e.g., sub-032213)
  --ses                   Session ID (e.g., ses-001)
  --t1w_ref               T1w reference NIfTI
  --t1w_mask              Brain mask in T1w space
  --t1w_dseg              WM discrete segmentation
  --subjects_dir          FreeSurfer SUBJECTS_DIR
  --bids_dir              Root BIDS directory (for fieldmap metadata)

Optional:
  --fs_license            FreeSurfer license file path
  --skip_frame            Number of initial volumes to skip (default: 0)
  --sdc                   Susceptibility distortion correction (default: false)
  --viz                   Enable debug visualization PNGs (default: true)
  --reg_method            BOLD-T1w registration: flirt (default) or bbregister
"
}

bold_file=""
bold_preprocess_dir=""
subj=""
ses=""
t1w_ref=""
t1w_mask=""
t1w_dseg=""
subjects_dir=""
bids_dir=""
fs_license=""
skip_frame=0
sdc="false"
viz="true"
reg_method="flirt"

while [[ $# -gt 0 ]]; do
  case $1 in
    --bold_file)              bold_file="$2"; shift 2 ;;
    --bold_preprocess_dir)    bold_preprocess_dir="$2"; shift 2 ;;
    --subj)                   subj="$2"; shift 2 ;;
    --ses)                    ses="$2"; shift 2 ;;
    --t1w_ref)                t1w_ref="$2"; shift 2 ;;
    --t1w_mask)               t1w_mask="$2"; shift 2 ;;
    --t1w_dseg)               t1w_dseg="$2"; shift 2 ;;
    --subjects_dir)           subjects_dir="$2"; shift 2 ;;
    --bids_dir)               bids_dir="$2"; shift 2 ;;
    --fs_license)             fs_license="$2"; shift 2 ;;
    --skip_frame)             skip_frame="$2"; shift 2 ;;
    --sdc)                    sdc="$2"; shift 2 ;;
    --viz)                    viz="$2"; shift 2 ;;
    --reg_method)             reg_method="$2"; shift 2 ;;
    --help)                   usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

for var in bold_file bold_preprocess_dir subj ses t1w_ref t1w_mask t1w_dseg subjects_dir bids_dir; do
  if [ -z "${!var}" ]; then echo "ERROR: --${var} is required"; usage; exit 1; fi
done

# Set FS license
if [ -n "${fs_license}" ] && [ -f "${fs_license}" ]; then
  export FS_LICENSE="${fs_license}"
fi

# Derive BOLD ID from filename
bold_basename=$(basename "${bold_file}" _bold.nii.gz)
bold_id="${bold_basename}"

func_dir="${bold_preprocess_dir}/func"
mkdir -p "${func_dir}"

prefix="${subj}_${ses}"

echo "============================================"
echo "BOLD Preprocessing: ${bold_id}"
echo "  T1w ref:     ${t1w_ref}"
echo "  SubjectsDir: ${subjects_dir}"
echo "  Skip frames: ${skip_frame}"
echo "  SDC:         ${sdc}"
echo "  Viz:         ${viz}"
echo "============================================"

# --- Debug visualization helper ---
viz_debug() {
    if [ "${viz}" != "true" ]; then
        return 0
    fi
    viz_dir="${func_dir}/viz"
    mkdir -p "${viz_dir}"
    python_inter="${PYTHON_INTER:-python3}"
    bold_scripts_dir="${BOLD_SCRIPT_DIR:-$(dirname "$0")}"
    viz_py="${bold_scripts_dir}/bold_viz.py"
    if [ ! -f "${viz_py}" ]; then
        echo "[VIZ] WARNING: bold_viz.py not found at ${viz_py}"
        return 1
    fi
    "$python_inter" "$viz_py" "$@"
}

# --- Step 1: Skip initial frames and reorient to RAS ---
bold_skip="${func_dir}/${bold_id}_desc-skip_bold.nii.gz"
if [ "${skip_frame}" -gt 0 ]; then
  echo "[STEP 1] Skipping ${skip_frame} initial frames..."
  nvols=$(fslnvols "${bold_file}")
  new_count=$((nvols - skip_frame))
  fslroi "${bold_file}" "${bold_skip}" ${skip_frame} ${new_count}
  echo "[OK]  Retained ${new_count}/${nvols} volumes"
else
  cp "${bold_file}" "${bold_skip}"
  echo "[STEP 1] No frames skipped"
fi

bold_reorient="${func_dir}/${bold_id}_space-reorient_bold.nii.gz"
orient=$(fslorient -getorient "${bold_skip}" 2>/dev/null || echo "")
if [ "${orient}" = "RAS" ]; then
  cp "${bold_skip}" "${bold_reorient}"
  echo "[STEP 1b] Already RAS orientation"
else
  echo "[STEP 1b] Reorienting from ${orient:-unknown} to RAS..."
  fslreorient2std "${bold_skip}" "${bold_reorient}"
  echo "[OK]  Reoriented to RAS"
fi

# [VIZ] After skip+reorient
viz_debug --mode multi-slice --input "${bold_reorient}" \
    --output "${func_dir}/viz/${bold_id}_step1_reorient.png" \
    --title "Step 1: After skip+reorient ${bold_id}" \
    --n_slices 4 || true

# --- Step 2a: MCFLIRT motion correction ---
echo "[STEP 2a] Motion correction (MCFLIRT)..."
bold_mc="${func_dir}/${bold_id}_desc-mc_bold.nii.gz"
bold_mc_par="${func_dir}/${bold_id}_desc-mc_motion_params.txt"
bold_mc_rel="${func_dir}/${bold_id}_desc-mc_rel.rms"
bold_mc_abs="${func_dir}/${bold_id}_desc-mc_abs.rms"

nvols_mid=$(( $(fslnvols "${bold_reorient}") / 2 ))
fslroi "${bold_reorient}" "${func_dir}/boldref_mid.nii.gz" ${nvols_mid} 1

mcflirt -in "${bold_reorient}" \
  -refvol ${nvols_mid} \
  -out "${bold_mc}" \
  -mats -plots \
  -rmsrel -rmsabs

if [ -f "${bold_mc}.par" ]; then
  mv "${bold_mc}.par" "${bold_mc_par}"
fi
mc_base=$(dirname "${bold_mc}")
mc_name=$(basename "${bold_mc}")
rms_rel=$(find "${mc_base}" -name "*_rel.rms" -newer "${bold_reorient}" 2>/dev/null | head -1 || echo "")
rms_abs=$(find "${mc_base}" -name "*_abs.rms" -newer "${bold_reorient}" 2>/dev/null | head -1 || echo "")
[ -n "${rms_rel}" ] && mv "${rms_rel}" "${bold_mc_rel}" 2>/dev/null || true
[ -n "${rms_abs}" ] && mv "${rms_abs}" "${bold_mc_abs}" 2>/dev/null || true

echo "[OK]  Motion correction complete"

# [VIZ] After MCFLIRT
viz_debug --mode compare --input "${bold_reorient}" "${bold_mc}" \
    --output "${func_dir}/viz/${bold_id}_step2a_mcflirt.png" \
    --title "Step 2a: MCFLIRT motion correction ${bold_id}" \
    --vol_t ${nvols_mid} || true
viz_debug --mode motion \
    --input "${bold_mc_rel}" "${bold_mc_abs}" \
    --output "${func_dir}/viz/${bold_id}_step2a_motion.png" \
    --title "Step 2a: Motion parameters ${bold_id}" || true

# --- Step 2b: Apply SDC on top of MCFLIRT output ---
bold_preproc="${func_dir}/${bold_id}_desc-preproc_bold.nii.gz"
sdc_applied="false"

if [ "${sdc}" = "true" ]; then
  fmap_dir="${bold_preprocess_dir}/fmap"
  fmap_index="${fmap_dir}/fieldmap_index.json"

  if [ -f "${fmap_index}" ]; then
    fmap_count=$(python3 -c "import json; d=json.load(open('${fmap_index}')); print(len(d))" 2>/dev/null || echo "0")

    if [ "${fmap_count}" -gt 0 ]; then
      echo "[STEP 2b] Applying SDC via sdcflows..."

      python_inter="${PYTHON_INTER:-python3}"
      bold_scripts_dir="${BOLD_SCRIPT_DIR:-$(dirname "$0")}"

      boldref_mc="${func_dir}/${bold_id}_boldref_mc.nii.gz"
      fslmaths "${bold_mc}" -Tmean "${boldref_mc}"

      "${python_inter}" "${bold_scripts_dir}/bold_sdc_apply.py" \
        --bids_dir "${bids_dir}" \
        --bold_file "${bold_reorient}" \
        --boldref_file "${boldref_mc}" \
        --fmap_dir "${fmap_dir}" \
        --nvols $(fslnvols "${bold_reorient}") \
        --bold_mc_file "${bold_mc}" \
        --sdc_file "${bold_preproc}" \
        --subject_id "${subj}" \
        --bold_id "${bold_id}" \
        --omp_nthreads 2

      if [ $? -eq 0 ] && [ -f "${bold_preproc}" ]; then
        sdc_applied="true"
        echo "[OK]  SDC+HMC correction complete"

        # [VIZ] Compare MCFLIRT vs SDC
        viz_debug --mode compare --input "${bold_mc}" "${bold_preproc}" \
            --output "${func_dir}/viz/${bold_id}_step2b_sdc_compare.png" \
            --title "Step 2b: MCFLIRT vs SDC ${bold_id}" \
            --vol_t ${nvols_mid} || true
      else
        echo "[WARN] SDC application failed, falling back to MCFLIRT-only"
      fi

      rm -f "${boldref_mc}" 2>/dev/null || true
    else
      echo "[WARN] SDC requested but no fieldmaps found for ${subj}${ses:+/}${ses}. Using MCFLIRT only."
    fi
  else
    echo "[WARN] SDC requested but fieldmap index not found: ${fmap_index}"
    echo "       Please run bold_fieldmap_estimate first. Using MCFLIRT only."
  fi
fi

if [ "${sdc_applied}" = "false" ]; then
  cp "${bold_mc}" "${bold_preproc}"
  echo "[STEP 2b] SDC skipped, using MCFLIRT-only as preproc"
fi

# --- Step 2c: Generate BOLD reference ---
echo "[STEP 2c] Generating BOLD reference from preprocessed data..."
boldref="${func_dir}/${bold_id}_boldref.nii.gz"
fslmaths "${bold_preproc}" -Tmean "${boldref}"
echo "[OK]  BOLD reference: ${boldref}"

viz_debug --mode single --input "${boldref}" \
    --output "${func_dir}/viz/${bold_id}_step2c_boldref.png" \
    --title "Step 2c: BOLD reference ${bold_id}" || true

# --- Step 3: BOLD-to-T1w coregistration ---
echo "[STEP 3] BOLD-to-T1w coregistration (method: ${reg_method})..."
export SUBJECTS_DIR="${subjects_dir}"

reg_xfm="${func_dir}/${bold_id}_from-bold_to-T1w_mode-image_xfm.mat"
boldref_t1w="${func_dir}/viz/${bold_id}_boldref_to_T1w.nii.gz"
apply_affine="${UTILS_PATH:-$(dirname "$0")/../../../../shared/utils}/apply_affine_to_header.py"

if [ "${reg_method}" = "bbregister" ]; then
    # --- BBRegister path ---
    # Build a temporary FreeSurfer subject directory with correct-space
    # surfaces from Resample/Original (workspace has fake headers for FS compatibility)
    echo "[STEP 3a] Setting up temporary FS subject for BBRegister..."
    resample_dir="$(dirname "${subjects_dir}")/Resample"
    fake_fs_subj="${func_dir}/.bbregister_fs_subj"
    rm -rf "${fake_fs_subj}"
    mkdir -p "${fake_fs_subj}/mri" "${fake_fs_subj}/surf"

    # Copy T1w volume (in original/correct space) as T1.mgz
    orig_t1w="${resample_dir}/Original/Volume/${prefix}_space-orig_desc-brain_T1w.nii.gz"
    if [ ! -f "${orig_t1w}" ]; then
        echo "[ERROR] Original T1w not found: ${orig_t1w}"
        exit 1
    fi
    mri_convert "${orig_t1w}" "${fake_fs_subj}/mri/T1.mgz"
    # BBRegister needs several standard FS files; link/copy T1 as needed
    for f in orig.mgz brainmask.mgz norm.mgz nu.mgz; do
        ln -sf T1.mgz "${fake_fs_subj}/mri/${f}" 2>/dev/null || \
            cp "${fake_fs_subj}/mri/T1.mgz" "${fake_fs_subj}/mri/${f}"
    done
    echo "[OK]  T1.mgz + links from ${orig_t1w}"

    # Convert GIFTI surfaces (scanner RAS) → FreeSurfer native format (tkRAS)
    gifti2fs="${UTILS_PATH}/gifti_surf2fs_surf.py"
    for hemi in L R; do
        if [ "$hemi" = "L" ]; then fs_hemi="lh"; else fs_hemi="rh"; fi
        for stype in white pial; do
            gifti_surf="${resample_dir}/Original/Native/${prefix}_hemi-${hemi}_desc-${stype}.surf.gii"
            fs_surf="${fake_fs_subj}/surf/${fs_hemi}.${stype}"
            if [ -f "${gifti_surf}" ]; then
                ${python_inter} "${gifti2fs}" \
                    --gifti_surf "${gifti_surf}" \
                    --volume "${orig_t1w}" \
                    --output "${fs_surf}"
            else
                echo "[WARN] Missing surface: ${gifti_surf}"
            fi
        done
        # Convert thickness shape.gii → FS .thickness via nibabel
        gifti_thick="${resample_dir}/Original/Native/${prefix}_hemi-${hemi}_desc-thickness.shape.gii"
        fs_thick="${fake_fs_subj}/surf/${fs_hemi}.thickness"
        if [ -f "${gifti_thick}" ]; then
            ${python_inter} -c "
import nibabel as nb
gii = nb.load('${gifti_thick}')
nb.freesurfer.write_morph_data('${fs_thick}', gii.darrays[0].data)
print(f'  thickness: {len(gii.darrays[0].data)} verts')
"
        fi
    done
    echo "[OK]  Surfaces converted to FS format"

    # Run BBRegister on the fake subject
    echo "[STEP 3b] Running BBRegister..."
    bbref_dat="${func_dir}/${bold_id}_from-bold_to-T1w_mode-image_xfm.dat"

    t2w_flag=""
    t2w_ref="${bold_preprocess_dir}/anat/${prefix}_desc-preproc_T2w.nii.gz"
    [ -f "${t2w_ref}" ] && t2w_flag="--t2"

    # BBRegister looks in $SUBJECTS_DIR for the subject, so point it there
    SUBJECTS_DIR="${func_dir}" bbregister --s ".bbregister_fs_subj" \
      --mov "${boldref}" --bold \
      --reg "${bbref_dat}" --init-fsl ${t2w_flag}

    rm -f "${bbref_dat}.mincost" "${bbref_dat}.param" "${bbref_dat}.sum" 2>/dev/null || true
    echo "[OK]  BBRegister .dat: ${bbref_dat}"

    # Convert .dat to FSL .mat via lta_convert
    echo "[STEP 3c] Converting BBRegister .dat to FSL .mat..."
    lta_convert --inreg "${bbref_dat}" --outfsl "${reg_xfm}" \
      --src "${boldref}" --trg "${t1w_ref}"
    echo "[OK]  FSL .mat: ${reg_xfm}"

    # Clean up temporary FS subject
    rm -rf "${fake_fs_subj}"
else
    # --- FLIRT path (default) ---
    echo "[STEP 3a] Running FLIRT..."
    flirt -in "${boldref}" -ref "${t1w_ref}" -dof 6 -omat "${reg_xfm}"
    echo "[OK]  FLIRT .mat: ${reg_xfm}"
fi

# Apply to boldref header for viz
# ${python_inter} "${apply_affine}" \
#   --input "${boldref}" --matrix "${reg_xfm}" --reference "${t1w_ref}" \
#   --output "${boldref_t1w}"
flirt -in "${boldref}" -ref "${t1w_ref}" -applyxfm -init "${reg_xfm}" \
  -out "${boldref_t1w}"

# [VIZ] Verify registration
viz_debug --mode align_check --input "${boldref_t1w}" \
    --reference "${t1w_ref}" \
    --output "${func_dir}/viz/${bold_id}_step3_reg_check.png" \
    --title "Step 3: BOLD\u2192T1w (${reg_method}) ${bold_id}" || true

# --- Step 4: Resample BOLD to T1w space ---
echo "[STEP 4] Resampling BOLD to T1w space (${reg_method})..."
bold_t1w="${func_dir}/${bold_id}_space-T1w_desc-preproc_bold.nii.gz"

# ${python_inter} "${apply_affine}" \
#   --input "${bold_preproc}" --matrix "${reg_xfm}" --reference "${t1w_ref}" \
#   --output "${bold_t1w}"
flirt -in "${bold_preproc}" -ref "${t1w_ref}" -applyxfm -init "${reg_xfm}" \
  -out "${bold_t1w}" -interp nearestneighbour

echo "[OK]  BOLD in T1w space: ${bold_t1w}"

viz_debug --mode align_check --input "${bold_t1w}" \
    --reference "${t1w_ref}" \
    --output "${func_dir}/viz/${bold_id}_step4_t1w_align.png" \
    --title "Step 4: BOLD\u2192T1w (${reg_method}) ${bold_id}" || true

# --- Step 5: Create brain mask in EPI space ---
echo "[STEP 5] Creating brain mask in EPI space (${reg_method})..."
bold_mask="${func_dir}/${bold_id}_desc-brain_mask.nii.gz"

# Invert BOLD\u2192T1w to get T1w\u2192BOLD
t1w2bold_xfm="${func_dir}/${bold_id}_from-T1w_to-bold_mode-image_xfm.mat"
convert_xfm -omat "${t1w2bold_xfm}" -inverse "${reg_xfm}"

flirt -in "${t1w_mask}" -ref "${boldref}" -applyxfm -init "${t1w2bold_xfm}" \
  -out "${bold_mask}" -interp nearestneighbour

fslmaths "${bold_mask}" -thr 0.5 -bin "${bold_mask}"
echo "[OK]  EPI brain mask: ${bold_mask}"

viz_debug --mode single --input "${bold_preproc}" \
    --output "${func_dir}/viz/${bold_id}_step5_brainmask.png" \
    --title "Step 5: Brain mask overlaid ${bold_id}" \
    --overlay "${bold_mask}" || true

echo ""
echo "[DONE] BOLD preprocessing complete for ${bold_id}"
echo "  Viz outputs in: ${func_dir}/viz/"
ls -la "${func_dir}/viz/"*.png 2>/dev/null || echo "  (no viz generated)"
echo ""
echo "  NIfTI outputs in: ${func_dir}/"
ls -la "${func_dir}/"*bold*.nii.gz "${func_dir}/"*motion* "${func_dir}/"*dat 2>/dev/null || true
