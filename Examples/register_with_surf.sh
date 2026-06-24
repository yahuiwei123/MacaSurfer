#!/bin/bash
set -x
sh /MacaSurfer/nextflow/Resample/scripts/Annot.sh --preprocess_dir /workspace/Enhance/ --resample_dir /workspace/Resample/ --python_inter /soft/macapipe/bin/python --surf_reg_dir /MacaSurfer/shared/volume_register/ --template_dir /MacaSurfer/global/templates/MEBRAIN --utils_path /MacaSurfer/shared/utils/
