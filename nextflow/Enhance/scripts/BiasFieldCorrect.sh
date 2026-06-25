#!/bin/bash
set -e
set -x

###### This script performs bias field correction using various methods
###### It supports sqrt(T1w*T1w), N4, and TGBFC methods

# Help message
usage() {
echo "
Usage: $0 --t1w_conform <t1w_conform> --t2w_conform <t2w_conform> --t1w_nbest <t1w_nbest> --t1w_cerebellum_brainstem <t1w_cerebellum_brainstem> --enhance_dir <enhance_dir> --contain_t2 <contain_t2> --bfc_method <bfc_method> --python_inter <python_inter> --utils_path <utils_path> --script_dir <script_dir>

Required arguments:
--t1w_conform               T1w conformed image
--t2w_conform               T2w conformed image
--t1w_nbest                 T1w nBEST segmentation
--t1w_cerebellum_brainstem  T1w cerebellum brainstem mask
--enhance_dir               Enhance directory
--contain_t2                Contains T2 flag (True/False)
--bfc_method                BFC method (tgbfc/sqrt/n4/gauss/rbf/none)
--python_inter              Python interpreter path
--utils_path                Utils scripts path
--script_dir                Enhancement scripts directory
"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --t1w_conform)
      t1w_conform="$2"
      shift 2
      ;;
    --t2w_conform)
      t2w_conform="$2"
      shift 2
      ;;
    --t1w_nbest)
      t1w_nbest="$2"
      shift 2
      ;;
    --t1w_cerebellum_brainstem)
      t1w_cerebellum_brainstem="$2"
      shift 2
      ;;
    --enhance_dir)
      enhance_dir="$2"
      shift 2
      ;;
    --contain_t2)
      contain_t2="$2"
      shift 2
      ;;
    --bfc_method)
      bfc_method="$2"
      shift 2
      ;;
    --python_inter)
      python_inter="$2"
      shift 2
      ;;
    --utils_path)
      utils_path="$2"
      shift 2
      ;;
    --script_dir)
      script_dir="$2"
      shift 2
      ;;
    --t1w_tissue19)
      t1w_tissue19="$2"
      shift 2
      ;;
    --use_tissue19)
      use_tissue19="$2"
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
if [[ -z "$t1w_conform" || -z "$t2w_conform" || -z "$t1w_nbest" || -z "$t1w_cerebellum_brainstem" || -z "$enhance_dir" || -z "$contain_t2" || -z "$bfc_method" || -z "$python_inter" || -z "$utils_path" || -z "$script_dir" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Validate bfc_method against currently implemented methods
case "$bfc_method" in
    tgbfc)
        bfc_method="gauss"
        ;;
    sqrt|n4|gauss|rbf|none)
        ;;
    *)
        echo "Error: Unsupported --bfc_method '${bfc_method}'." >&2
        echo "Supported methods: tgbfc, sqrt, n4, gauss, rbf, none." >&2
        exit 1
        ;;
esac

# Set paths
# Derive BIDS subject/session prefix from .../<sub>/<ses>/Enhance
subj_bids=$(basename "$(dirname "$(dirname "$enhance_dir")")")
ses_bids=$(basename "$(dirname "$enhance_dir")")
if [[ "${subj_bids}" != sub-* ]]; then
    subj_bids="sub-${subj_bids}"
fi
if [[ "${ses_bids}" != ses-* ]]; then
    ses_bids="ses-${ses_bids}"
fi
prefix="${subj_bids}_${ses_bids}"

t1w_path="${enhance_dir}/T1w"
t1w_xfm_path="${t1w_path}/xfms"
mkdir -p "${t1w_xfm_path}"
if [ ! -f "${t1w_xfm_path}/identity.mat" ]; then
  echo "1 0 0 0"  > "${t1w_xfm_path}/identity.mat"
  echo "0 1 0 0" >> "${t1w_xfm_path}/identity.mat"
  echo "0 0 1 0" >> "${t1w_xfm_path}/identity.mat"
  echo "0 0 0 1" >> "${t1w_xfm_path}/identity.mat"
  echo "[OK] Created identity.mat"
fi
t1w_final_corrected="${t1w_path}/${prefix}_desc-bfc_T1w.nii.gz"
t2w_final_corrected="${t1w_path}/${prefix}_desc-bfc_T2w.nii.gz"
t1w_white="${t1w_path}/${prefix}_desc-whitebfc_T1w.nii.gz"
t1w_pial="${t1w_path}/${prefix}_desc-pialbfc_T1w.nii.gz"
t2w_pial="${t1w_path}/${prefix}_desc-pialbfc_T2w.nii.gz"

echo "Starting bias field correction using method: $bfc_method"
cd ${t1w_path}

if [[ ${contain_t2} == "True" ]]; then
    echo "Processing with T2w image..."
    
    case $bfc_method in
        "sqrt")
            echo "Using sqrt(T1w*T1w) method..."
            mkdir -p BiasFieldCorrection_sqrtT1wXT1w
            sh ${script_dir}/utils/BiasFieldCorrection_sqrtT1wXT1w.sh \
                --workingdir=BiasFieldCorrection_sqrtT1wXT1w \
                --T1im=${t1w_conform} \
                --T1brain=${t1w_conform} \
                --T2im=${t2w_conform} \
                --obias=BiasField_acpc_dc \
                --oT1im=${t1w_final_corrected} \
                --oT1brain=${t1w_final_corrected} \
                --oT2im=${t2w_final_corrected} \
                --oT2brain=${t2w_final_corrected} \
                --bfsigma=5
            ;;
            
        "n4")
            echo "Using N4 bias field correction method..."
            # Generate WM probability map for N4
            fslmaths ${t1w_nbest} -thr 3 -uthr 3 -bin ${prefix}_desc-n4weight_mask.nii.gz
            fslmaths ${t1w_cerebellum_brainstem} -bin -add ${prefix}_desc-n4weight_mask.nii.gz -bin ${prefix}_desc-n4weight_mask.nii.gz
            fslcpgeom ${t1w_conform} ${prefix}_desc-n4weight_mask.nii.gz
            
            N4BiasFieldCorrection -d 3 -i ${t1w_conform} -o ["${t1w_final_corrected}","${prefix}_desc-bfcbias_T1w.nii.gz"] -w ${prefix}_desc-n4weight_mask.nii.gz -s 1
            N4BiasFieldCorrection -d 3 -i ${t2w_conform} -o ${t2w_final_corrected} -w ${prefix}_desc-n4weight_mask.nii.gz -s 1
            ;;
            
        "gauss")
            echo "Using gaussian mixture method at native resolution..."

            if [[ "${use_tissue19}" == "true" ]]; then
                # Use merged complete aseg directly (native resolution, multilabel BFC)
                ${python_inter} ${utils_path}/conform.py --input ${t1w_tissue19} --output ${prefix}_desc-bfc_tissue19.nii.gz --reorient LIA

                # Process T1w white matter bias field (native resolution, multilabel)
                ${python_inter} ${utils_path}/bfc_gauss_mixture_multilabel.py \
                    --input_msk ${prefix}_desc-conform_mask.nii.gz \
                    --input_img ${t1w_conform} \
                    --input_lab ${prefix}_desc-bfc_tissue19.nii.gz \
                    --output_img ${t1w_white} \
                    --output_bias ${prefix}_desc-bfcbias_T1w.nii.gz \
                    --bias_max_iter 20 --max_kernel 7.0 --min_kernel 3.0 \
                    --refine_max_iter 1 --label_soft_kernel 0.25

                # Normalize T1 intensity
                ${python_inter} ${utils_path}/conform.py \
                    --input ${t1w_white} \
                    --output ${t1w_white} \
                    --norm 255 --gamma 1.0 --modal T1
                cp ${t1w_white} ${t1w_pial}

                # Process T2w bias field (native resolution, multilabel)
                fslmaths ${t2w_conform} -mas ${prefix}_desc-conform_mask.nii.gz ${t2w_conform}
                ${python_inter} ${utils_path}/bfc_gauss_mixture_multilabel.py \
                    --input_msk ${prefix}_desc-conform_mask.nii.gz \
                    --input_img ${t2w_conform} \
                    --input_lab ${prefix}_desc-bfc_tissue19.nii.gz \
                    --output_img ${t2w_pial} \
                    --output_bias ${prefix}_desc-bfcbias_T2w.nii.gz \
                    --bias_max_iter 20 --max_kernel 9.0 --min_kernel 5.0 \
                    --refine_max_iter 1 --label_soft_kernel 0.05

                # Normalize T2 intensity
                ${python_inter} ${utils_path}/conform.py \
                    --input ${t2w_pial} \
                    --output ${t2w_pial} \
                    --norm 255 --gamma 1.0 --modal T2
            else
                # Use nbest (native resolution) + aseg enhancement
                ${python_inter} ${utils_path}/conform.py --input ${t1w_nbest} --output ${prefix}_desc-bfc_dseg.nii.gz --reorient LIA
                fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 10 -uthr 10 -bin -ero -mul 3 -max ${prefix}_desc-bfc_dseg.nii.gz ${prefix}_desc-bfc_dseg.nii.gz
                fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 49 -uthr 49 -bin -ero -mul 3 -max ${prefix}_desc-bfc_dseg.nii.gz ${prefix}_desc-bfc_dseg.nii.gz
                fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 13 -uthr 13 -bin -ero -mul 3 -max ${prefix}_desc-bfc_dseg.nii.gz ${prefix}_desc-bfc_dseg.nii.gz
                fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 52 -uthr 52 -bin -ero -mul 3 -max ${prefix}_desc-bfc_dseg.nii.gz ${prefix}_desc-bfc_dseg.nii.gz
                BFC_LAB="${prefix}_desc-bfc_dseg.nii.gz"

                # Process T1w white matter bias field (native resolution)
                ${python_inter} ${utils_path}/bfc_gauss_mixture.py \
                    --input_msk ${prefix}_desc-conform_mask.nii.gz \
                    --input_img ${t1w_conform} \
                    --input_lab ${BFC_LAB} \
                    --output_img ${t1w_white} \
                    --output_bias ${prefix}_desc-bfcbias_T1w.nii.gz \
                    --bias_max_iter 15 --max_kernel 7.0 --min_kernel 3.0 \
                    --refine_max_iter 1 --label_soft_kernel 0.25

                # Normalize T1 intensity
                ${python_inter} ${utils_path}/conform.py \
                    --input ${t1w_white} \
                    --output ${t1w_white} \
                    --norm 255 --gamma 1.0 --modal T1
                cp ${t1w_white} ${t1w_pial}

                # Process T2w bias field (native resolution)
                fslmaths ${t2w_conform} -mas ${prefix}_desc-conform_mask.nii.gz ${t2w_conform}
                cp ${BFC_LAB} ${prefix}_desc-bfc_dseg_T2w.nii.gz
                ${python_inter} ${utils_path}/bfc_rbf_scatter.py \
                    --input_img ${t2w_conform} \
                    --input_lab ${prefix}_desc-bfc_dseg_T2w.nii.gz \
                    --input_msk ${prefix}_desc-conform_mask.nii.gz \
                    --output_img ${t2w_pial} \
                    --output_bias ${prefix}_desc-bfcbias_T2w.nii.gz \
                    --use_label_value 2 \
                    --no_soft_labels

                # Normalize T2 intensity
                ${python_inter} ${utils_path}/conform.py \
                    --input ${t2w_pial} \
                    --output ${t2w_pial} \
                    --norm 255 --gamma 1.0 --modal T2
            fi

            cp ${prefix}_desc-initcorrected_T1w.nii.gz ${t1w_final_corrected}
            cp ${prefix}_desc-initcorrected_T2w.nii.gz ${t2w_final_corrected}
            ;;
        "rbf")
            echo "Using rbf scatter method..."
            ${python_inter} ${utils_path}/conform.py --input ${t1w_nbest} --output ${t1w_nbest} --reorient LIA
            ${python_inter} ${utils_path}/conform.py --input ${t1w_cerebellum_brainstem} --output ${t1w_cerebellum_brainstem} --reorient LIA

            if [[ "${use_tissue19}" == "true" ]]; then
                # Use multilabel BFC with merged tissue segmentation
                ${python_inter} ${utils_path}/conform.py --input ${t1w_tissue19} --output ${prefix}_desc-bfc_tissue19.nii.gz --reorient LIA

                # Process T1w white matter bias field (multilabel)
                ${python_inter} ${utils_path}/bfc_gauss_mixture_multilabel.py \
                    --input_msk ${prefix}_desc-conform_mask.nii.gz \
                    --input_img ${t1w_conform} \
                    --input_lab ${prefix}_desc-bfc_tissue19.nii.gz \
                    --output_img ${t1w_white} \
                    --output_bias ${prefix}_desc-bfcbias_T1w.nii.gz \
                    --bias_max_iter 20 --max_kernel 7.0 --min_kernel 3.0 \
                    --refine_max_iter 1 --label_soft_kernel 0.25

                # Normalize T1 intensity
                ${python_inter} ${utils_path}/conform.py \
                    --input ${t1w_white} \
                    --output ${t1w_white} \
                    --norm 255 --gamma 1.0 --modal T1
                cp ${t1w_white} ${t1w_pial}

                # Process T2w bias field (multilabel)
                fslmaths ${t2w_conform} -mas ${prefix}_desc-conform_mask.nii.gz ${t2w_conform}
                ${python_inter} ${utils_path}/bfc_gauss_mixture_multilabel.py \
                    --input_msk ${prefix}_desc-conform_mask.nii.gz \
                    --input_img ${t2w_conform} \
                    --input_lab ${prefix}_desc-bfc_tissue19.nii.gz \
                    --output_img ${t2w_pial} \
                    --output_bias ${prefix}_desc-bfcbias_T2w.nii.gz \
                    --bias_max_iter 20 --max_kernel 9.0 --min_kernel 5.0 \
                    --refine_max_iter 1 --label_soft_kernel 0.05
            else
                # Enhanced tissue segmentation
                fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 10 -uthr 10 -bin -ero -mul 3 -max ${t1w_nbest} ${prefix}_desc-bfc_dseg.nii.gz
                fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 49 -uthr 49 -bin -ero -mul 3 -max ${prefix}_desc-bfc_dseg.nii.gz ${prefix}_desc-bfc_dseg.nii.gz
                fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 13 -uthr 13 -bin -ero -mul 3 -max ${prefix}_desc-bfc_dseg.nii.gz ${prefix}_desc-bfc_dseg.nii.gz
                fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 52 -uthr 52 -bin -ero -mul 3 -max ${prefix}_desc-bfc_dseg.nii.gz ${prefix}_desc-bfc_dseg.nii.gz

                # Process T1w white matter bias field
                ${python_inter} ${utils_path}/bfc_rbf_scatter.py \
                    --input_img ${t1w_conform} \
                    --input_lab ${prefix}_desc-bfc_dseg.nii.gz \
                    --input_msk ${prefix}_desc-bfc_dseg.nii.gz \
                    --output_img ${t1w_white} \
                    --output_bias ${prefix}_desc-bfcbias_T1w.nii.gz \
                    --no_soft_labels --debug

                # Normalize T1 intensity
                ${python_inter} ${utils_path}/conform.py \
                    --input ${t1w_white} \
                    --output ${t1w_white} \
                    --norm 255 --gamma 1.0 --modal T1
                cp ${t1w_white} ${t1w_pial}

                # Process T2w bias field
                fslmaths ${t2w_conform} -mas ${prefix}_desc-conform_mask.nii.gz ${t2w_conform}
                cp ${prefix}_desc-bfc_dseg.nii.gz ${prefix}_desc-bfc_dseg_T2w.nii.gz
                ${python_inter} ${utils_path}/bfc_rbf_scatter.py \
                    --input_img ${t2w_conform} \
                    --input_lab ${prefix}_desc-bfc_dseg_T2w.nii.gz \
                    --input_msk ${prefix}_desc-bfc_dseg_T2w.nii.gz \
                    --output_img ${t2w_pial} \
                    --output_bias ${prefix}_desc-bfcbias_T2w.nii.gz \
                    --use_label_value 2 \
                    --no_soft_labels
            fi

            # Normalize T2 intensity
            ${python_inter} ${utils_path}/conform.py \
                --input ${t2w_pial} \
                --output ${t2w_pial} \
                --norm 255 --gamma 1.0 --modal T2
            
            cp ${prefix}_desc-initcorrected_T1w.nii.gz ${t1w_final_corrected}
            cp ${prefix}_desc-initcorrected_T2w.nii.gz ${t2w_final_corrected}
            ;;
        *)
            echo "No bias field correction applied"
            cp ${t1w_conform} ${t1w_final_corrected}
            cp ${t2w_conform} ${t2w_final_corrected}
            fslmaths ${t1w_final_corrected} -mul 0 ${prefix}_desc-bfcbias_T1w.nii.gz
            ;;
    esac
else
    echo "Processing without T2w image..."
    if [ "$bfc_method" == "sqrt" ] || [ "$bfc_method" == "n4" ]; then
        # Generate wm probability map to enhance preformance of N4
        fslmaths ${t1w_nbest} -thr 3 -uthr 3 -bin ${prefix}_desc-n4weight_mask.nii.gz
        fslmaths ${prefix}_label-cerebellum-brainstem_dseg.nii.gz -bin -add ${prefix}_desc-n4weight_mask.nii.gz -bin ${prefix}_desc-n4weight_mask.nii.gz
        fslcpgeom ${t1w_conform} ${prefix}_desc-n4weight_mask.nii.gz
        N4BiasFieldCorrection -d 3 -i ${t1w_conform} -o [${t1w_final_corrected},"${prefix}_desc-bfcbias_T1w.nii.gz"] -w ${prefix}_desc-n4weight_mask.nii.gz -s 1
    fi
    if [ "$bfc_method" == "gauss" ]; then
        echo "Using gaussian mixture method at native resolution..."

        if [[ "${use_tissue19}" == "true" ]]; then
            # Multilabel BFC with merged tissue segmentation at native resolution
            echo "Using multilabel BFC with tissue segmentation..."
            ${python_inter} ${utils_path}/conform.py --input ${t1w_tissue19} --output ${prefix}_desc-bfc_tissue19.nii.gz --reorient LIA

            ${python_inter} ${utils_path}/bfc_gauss_mixture_multilabel.py \
                --input_msk ${prefix}_desc-conform_mask.nii.gz \
                --input_img ${t1w_conform} \
                --input_lab ${prefix}_desc-bfc_tissue19.nii.gz \
                --output_img ${t1w_white} \
                --output_bias ${prefix}_desc-bfcbias_T1w.nii.gz \
                --bias_max_iter 20 --max_kernel 7.0 --min_kernel 3.0 \
                --refine_max_iter 1 --label_soft_kernel 0.25
            ${python_inter} ${utils_path}/conform.py \
                --input ${t1w_white} \
                --output ${t1w_white} \
                --norm 255 --gamma 1.0 --modal T1
            cp ${t1w_white} ${t1w_pial}
        else
            # Use nbest (native resolution) + aseg enhancement
            ${python_inter} ${utils_path}/conform.py --input ${t1w_nbest} --output ${prefix}_desc-bfc_dseg.nii.gz --reorient LIA
            fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 10 -uthr 10 -bin -ero -mul 3 -max ${prefix}_desc-bfc_dseg.nii.gz ${prefix}_desc-bfc_dseg.nii.gz
            fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 49 -uthr 49 -bin -ero -mul 3 -max ${prefix}_desc-bfc_dseg.nii.gz ${prefix}_desc-bfc_dseg.nii.gz
            fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 13 -uthr 13 -bin -ero -mul 3 -max ${prefix}_desc-bfc_dseg.nii.gz ${prefix}_desc-bfc_dseg.nii.gz
            fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 52 -uthr 52 -bin -ero -mul 3 -max ${prefix}_desc-bfc_dseg.nii.gz ${prefix}_desc-bfc_dseg.nii.gz
            BFC_LAB="${prefix}_desc-bfc_dseg.nii.gz"

            # Process T1w white matter bias field (native resolution)
            ${python_inter} ${utils_path}/bfc_gauss_mixture.py \
                --input_msk ${prefix}_desc-conform_mask.nii.gz \
                --input_img ${t1w_conform} \
                --input_lab ${BFC_LAB} \
                --output_img ${t1w_white} \
                --output_bias ${prefix}_desc-bfcbias_T1w.nii.gz \
                --bias_max_iter 15 --max_kernel 7.0 --min_kernel 3.0 \
                --refine_max_iter 1 --label_soft_kernel 0.25

            # Normalize T1 intensity
            ${python_inter} ${utils_path}/conform.py \
                --input ${t1w_white} \
                --output ${t1w_white} \
                --norm 255 --gamma 1.0 --modal T1
            cp ${t1w_white} ${t1w_pial}
        fi
        cp ${prefix}_desc-initcorrected_T1w.nii.gz ${t1w_final_corrected}
    elif [ "$bfc_method" == "rbf" ]; then
        echo "Using rbf scatter method..."
        ${python_inter} ${utils_path}/conform.py --input ${t1w_nbest} --output ${t1w_nbest} --reorient LIA
        ${python_inter} ${utils_path}/conform.py --input ${t1w_cerebellum_brainstem} --output ${t1w_cerebellum_brainstem} --reorient LIA

        if [[ "${use_tissue19}" == "true" ]]; then
            # Use multilabel BFC with merged tissue segmentation
            ${python_inter} ${utils_path}/conform.py --input ${t1w_tissue19} --output ${prefix}_desc-bfc_tissue19.nii.gz --reorient LIA

            ${python_inter} ${utils_path}/bfc_gauss_mixture_multilabel.py \
                --input_msk ${prefix}_desc-conform_mask.nii.gz \
                --input_img ${t1w_conform} \
                --input_lab ${prefix}_desc-bfc_tissue19.nii.gz \
                --output_img ${t1w_white} \
                --output_bias ${prefix}_desc-bfcbias_T1w.nii.gz \
                --bias_max_iter 20 --max_kernel 7.0 --min_kernel 3.0 \
                --refine_max_iter 1 --label_soft_kernel 0.25
            ${python_inter} ${utils_path}/conform.py \
                --input ${t1w_white} \
                --output ${t1w_white} \
                --norm 255 --gamma 1.0 --modal T1
            cp ${t1w_white} ${t1w_pial}
        else
            # Enhanced tissue segmentation
            fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 10 -uthr 10 -bin -ero -mul 3 -max ${t1w_nbest} ${prefix}_desc-bfc_dseg.nii.gz
            fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 49 -uthr 49 -bin -ero -mul 3 -max ${prefix}_desc-bfc_dseg.nii.gz ${prefix}_desc-bfc_dseg.nii.gz
            fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 13 -uthr 13 -bin -ero -mul 3 -max ${prefix}_desc-bfc_dseg.nii.gz ${prefix}_desc-bfc_dseg.nii.gz
            fslmaths ${prefix}_desc-aseg_dseg.nii.gz -thr 52 -uthr 52 -bin -ero -mul 3 -max ${prefix}_desc-bfc_dseg.nii.gz ${prefix}_desc-bfc_dseg.nii.gz

            # Process T1w white matter bias field
            ${python_inter} ${utils_path}/bfc_rbf_scatter.py \
                --input_img ${t1w_conform} \
                --input_lab ${prefix}_desc-bfc_dseg.nii.gz \
                --input_msk ${prefix}_desc-bfc_dseg.nii.gz \
                --output_img ${t1w_white} \
                --output_bias ${prefix}_desc-bfcbias_T1w.nii.gz \
                --no_soft_labels
            ${python_inter} ${utils_path}/conform.py \
                --input ${t1w_white} \
                --output ${t1w_white} \
                --norm 255 --gamma 1.0 --modal T1
            cp ${t1w_white} ${t1w_pial}
        fi

        cp ${prefix}_desc-initcorrected_T1w.nii.gz ${t1w_final_corrected}
    else
        cp ${t1w_conform} ${t1w_final_corrected}
        fslmaths ${t1w_final_corrected} -mul 0 ${prefix}_desc-bfcbias_T1w.nii.gz
    fi
fi

# Ensure output files exist
if [[ ! -f ${t1w_white} ]]; then
    cp ${t1w_final_corrected} ${t1w_white}
    cp ${t1w_final_corrected} ${t1w_pial}
    cp ${t2w_final_corrected} ${t2w_pial}
fi

echo "Bias field correction completed successfully"