#!/bin/bash

source /MacaSurfer/SetUpHCPPipelineNHP.sh
nextflow run /MacaSurfer/nextflow/macasurfer.nf -c /MacaSurfer/nextflow/macasurfer.common.config --subject_dir /workspace --after_check True --deep_white True