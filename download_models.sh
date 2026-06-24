#!/bin/bash
# =============================================================================
# MacaSurfer — Download pretrained model weights and large template files
# =============================================================================
# These files are too large (>100 MB) for GitHub and must be downloaded
# separately from cloud storage (KS3 / Kingsoft Cloud).
#
# Usage:
#   bash download_models.sh [--dest /path/to/MacaSurfer]
# =============================================================================
set -e

DEST="$(cd "$(dirname "$0")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

echo "=== MacaSurfer Model Download ==="
echo "Destination: ${DEST}"
echo ""

# ── Brain extractor (macaUNet) ──────────────────────────────────────────────
BRAIN_MODEL_DIR="${DEST}/shared/brainextractor/macaUNet/models/3_times_conv_on_downsample/fold_all"
mkdir -p "${BRAIN_MODEL_DIR}"

echo "[1/3] Downloading macaUNet brain extractor models..."
# NOTE: Replace these URLs with actual download links from your KS3 bucket or
# other cloud storage. Example using ks3util:
#
#   ks3util cp ks3://your-bucket/models/macaunet/best_metric_model.pth "${BRAIN_MODEL_DIR}/"
#   ks3util cp ks3://your-bucket/models/macaunet/curr_epoch_model.pth "${BRAIN_MODEL_DIR}/"
#
# Or with wget/curl:
#   wget -O "${BRAIN_MODEL_DIR}/best_metric_model.pth" "https://your-cdn.example.com/models/macaunet/best_metric_model.pth"
#   wget -O "${BRAIN_MODEL_DIR}/curr_epoch_model.pth" "https://your-cdn.example.com/models/macaunet/curr_epoch_model.pth"

echo "  → Skipped (no download URLs configured — edit this script to add them)"

# ── Tissue segmenter (nnU-Net / nBEST) ──────────────────────────────────────
TISSUE_MODEL_DIR="${DEST}/shared/tissueextractor/nnUNet_trained_models/nnUNet/3d_fullres/Task509_tissue_infant/nnUNetTrainerV2_DA3_BN_UNeXt_axial_attn__nnUNetPlans_pretrained_IDENTIFIER/fold_1"
mkdir -p "${TISSUE_MODEL_DIR}"

echo "[2/3] Downloading tissue segmentation models..."
# NOTE: Replace these URLs with actual download links.
#
#   ks3util cp ks3://your-bucket/models/tissueseg/model_ts.model    "${TISSUE_MODEL_DIR}/"
#   ks3util cp ks3://your-bucket/models/tissueseg/model_be.model    "${TISSUE_MODEL_DIR}/"
#   ks3util cp ks3://your-bucket/models/tissueseg/model_rmcere.model "${TISSUE_MODEL_DIR}/"

echo "  → Skipped (no download URLs configured — edit this script to add them)"

# ── FreeSurfer GCA atlas ────────────────────────────────────────────────────
GCA_DIR="${DEST}/global/templates/MEBRAIN"
mkdir -p "${GCA_DIR}"

echo "[3/3] Downloading FreeSurfer GCA atlas..."
# NOTE: Replace this URL with the actual download link.
#
#   ks3util cp ks3://your-bucket/templates/RB_all_2020-01-02.gca "${GCA_DIR}/"

echo "  → Skipped (no download URLs configured — edit this script to add them)"

echo ""
echo "=== Done ==="
echo "If download URLs are configured, models will be at:"
echo "  ${BRAIN_MODEL_DIR}/"
echo "  ${TISSUE_MODEL_DIR}/"
echo "  ${GCA_DIR}/"
