#!/bin/bash
#JSUB -m "c-0-1 || c-0-2 || c-0-3 || c-0-4 || c-0-5 || c-0-6 || c-0-7 || c-0-8 || c-0-9 || c-0-10 || c-0-11 || c-0-12 || c-0-13 || c-0-14 || c-0-15 || f-2-0 || f-2-1"
#JSUB -n 2
#JSUB -o error_%J.log
#JSUB -o output_%J.log
#JSUB -J stats
set -e
set -x

###### This script performs ROI statistics and normative model prediction

# Help message
usage() {
echo "
Usage: $0 --subjects_dir <subjects_dir> --subject <subject_name> --session <session_name> --out_dir <out_dir> --meta_csv <meta_csv> --python_inter <python_inter> --utils_path <utils_path> --norm_model_dir <norm_model_dir> --atlases_dir <atlases_dir> [--atlases <atlases>]

Required arguments:
--subjects_dir             Subjects directory path (parent folder containing subject folders)
--subject                  Subject name (folder name of the subject)
--session                  Session name
--out_dir                  Output directory for statistics and predictions
--meta_csv                 Meta CSV file containing subject_id, age, sex, site, breed, weight (kg)
--python_inter             Python interpreter path
--utils_path               Utilities scripts path (shared/utils)
--norm_model_dir           Normative model directory path
--atlases_dir              Atlases directory

Optional arguments:
--atlases                  Space-separated list of atlases (default: 'MBNA124 Modalities M129 M132')
--help                     Show this help message
"
}

# Default values
ATLASES="MBNA124 Modalities M129 M132"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --subjects_dir)
      SUBJECTS_DIR="$2"
      shift 2
      ;;
    --subject)
      SUBJECT="$2"
      shift 2
      ;;
    --session)
      SESSION="$2"
      shift 2
      ;;
    --out_dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --meta_csv)
      META_CSV="$2"
      shift 2
      ;;
    --python_inter)
      PYTHON_INTER="$2"
      shift 2
      ;;
    --utils_path)
      UTILS_PATH="$2"
      shift 2
      ;;
    --norm_model_dir)
      NORM_MODEL_DIR="$2"
      shift 2
      ;;
    --atlases_dir)
      ATLAS_DIR="$2"
      shift 2
      ;;
    --atlases)
      ATLASES="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

# Validate required arguments
if [[ -z "$SUBJECTS_DIR" || -z "$SUBJECT" || -z "$SESSION" || -z "$OUT_DIR" || -z "$META_CSV" || -z "$PYTHON_INTER" || -z "$UTILS_PATH" || -z "$NORM_MODEL_DIR" || -z "$ATLAS_DIR" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Log message function
log_Msg() {
    echo "$(date): $*"
}

# Check whether meta CSV has the columns required for normative model prediction
# Returns "true" if age/sex/site are present with at least one non-empty value each
check_normative_ready() {
    local meta_csv="$1"
    local python_inter="$2"

    if [[ ! -f "$meta_csv" ]]; then
        echo "false"
        return
    fi

    ${python_inter} -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('${meta_csv}')
    required = ['age', 'sex', 'site']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f'Normative check: meta_csv missing columns: {missing}')
        print('false')
        sys.exit(0)

    # Check each required column has at least one non-empty, non-NaN value
    for col in required:
        valid_count = df[col].notna().sum()
        if valid_count == 0:
            print(f'Normative check: column \"{col}\" is entirely empty in meta_csv')
            print('false')
            sys.exit(0)
        # Also check no empty strings
        valid_str = (df[col].notna() & (df[col].astype(str).str.strip() != '')).sum()
        if valid_str == 0:
            print(f'Normative check: column \"{col}\" has only empty string values in meta_csv')
            print('false')
            sys.exit(0)

    print('Normative check: meta_csv has valid age/sex/site data, normative prediction enabled')
    print('true')
except Exception as e:
    print(f'Normative check error: {e}')
    print('false')
"
}

# Determine if normative prediction can run
RUN_NORMATIVE=$(check_normative_ready "${META_CSV}" "${PYTHON_INTER}")
log_Msg "Normative prediction enabled: ${RUN_NORMATIVE}"

# Main execution
log_Msg "Starting ROI statistics and normative model prediction"
log_Msg "Subject: ${SUBJECT}"
log_Msg "Subject: ${SESSION}"
log_Msg "Subjects directory: ${SUBJECTS_DIR}"
log_Msg "Output directory: ${OUT_DIR}"
log_Msg "Atlases: ${ATLASES}"

# Create output directory
mkdir -p ${OUT_DIR}

# Loop through atlases
for atlas in ${ATLASES}; do
    log_Msg "Processing atlas: ${atlas}"

    # Find atlas label files for both hemispheres

    # Process each hemisphere
    for hemi in "L" "R"; do
        log_Msg "Processing hemisphere: ${hemi}"

        # Find atlas label file
        ATLAS_FILE="${ATLAS_DIR}/${hemi}.${atlas}.label.gii"

        if [[ ! -f "$ATLAS_FILE" ]]; then
            # Try alternative naming patterns
            ATLAS_FILE="${ATLAS_DIR}/${hemi}.map_${atlas}.label.gii"
        fi

        if [[ ! -f "$ATLAS_FILE" ]]; then
            log_Msg "Warning: Atlas file not found for ${atlas}/${hemi}, skipping..."
            continue
        fi

        log_Msg "Using atlas file: ${ATLAS_FILE}"

        # Run ROI statistics
        log_Msg "Running ROI statistics for ${atlas}/${hemi}..."
        ${PYTHON_INTER} ${UTILS_PATH}/roi_stats.py \
            --atlas ${ATLAS_FILE} \
            --subjects_dir ${SUBJECTS_DIR} \
            --subject ${SUBJECT} \
            --session ${SESSION} \
            --out_dir ${OUT_DIR} \
            --meta_csv ${META_CSV} \
            --export_area

        # Run normative model prediction for each metric (only if meta CSV has required columns)
        if [[ "${RUN_NORMATIVE}" == "true" ]]; then
        for metric in "cortvol" "curvature" "sulc" "thickness" "area"; do
            log_Msg "Running normative prediction for ${atlas}/${hemi}/${metric}..."

            # Input CSV generated by roi_stats.py: out_dir/cort/{atlas}/{hemi}/{metric}.csv
            INPUT_CSV="${OUT_DIR}/cort/${atlas}/${hemi}/${metric}.csv"

            if [[ ! -f "$INPUT_CSV" ]]; then
                log_Msg "Warning: Input CSV not found: ${INPUT_CSV}, skipping prediction..."
                continue
            fi

            ${PYTHON_INTER} ${UTILS_PATH}/norm_predict.py \
                --atlas ${atlas} \
                --hemi ${hemi} \
                --metric ${metric} \
                --input_csv ${INPUT_CSV} \
                --norm_model_dir ${NORM_MODEL_DIR} \
                --out_dir ${OUT_DIR}/cort/${atlas}/${hemi}
        done

        # Subcortical volume prediction (separate path convention)
        for metric in "subvol"; do
            INPUT_CSV="${OUT_DIR}/subcort/aseg/${hemi}/volume.csv"
            if [[ ! -f "$INPUT_CSV" ]]; then
                log_Msg "Warning: Subcortical CSV not found: ${INPUT_CSV}, skipping prediction..."
            else
                log_Msg "Running normative prediction for subcortical ${hemi}..."
                ${PYTHON_INTER} ${UTILS_PATH}/norm_predict.py \
                    --atlas ${atlas} \
                    --hemi ${hemi} \
                    --metric ${metric} \
                    --input_csv ${INPUT_CSV} \
                    --norm_model_dir ${NORM_MODEL_DIR} \
                    --out_dir ${OUT_DIR}/subcort/aseg/${hemi}
            fi
        done
        else
            log_Msg "Skipping normative prediction: meta CSV lacks valid age/sex/site data"
        fi
    done
done

log_Msg "ROI statistics and normative model prediction completed"