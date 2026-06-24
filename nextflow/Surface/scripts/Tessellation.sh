#!/bin/bash
set -e
set -x

###### This script performs tessellation and related processing for T1w and T2w images.
###### It includes steps like pre-tessellation, surface inflation, registration, and generating surfaces for both hemispheres (left and right).

# Help message
usage() {
    echo "
Usage: $0 --subject_dir <subject_dir> --subject_id <subject_id> --hemi <hemi>

Required arguments:
--subject_dir            Subject directory path
--subject_id             Subject ID
--hemi                   Hemisphere ('${hemi}' or 'rh') to process
  
Optional arguments:
--help                   Display this help message
"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --subject_dir)
      SubjectDIR="$2"
      shift 2
      ;;
    --subject_id)
      SubjectID="$2"
      shift 2
      ;;
    --hemi)
      hemi="$2"
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
if [[ -z "$SubjectDIR" || -z "$SubjectID" || -z "$hemi" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Check if hemisphere is valid
if [[ "$hemi" != "lh" && "$hemi" != "rh" ]]; then
    echo "Error: Invalid hemisphere. Please specify either 'lh' or 'rh'."
    usage
    exit 1
fi

# Log message function
log_Msg() {
    echo "$(date): $*"
}

# Print the arguments for confirmation
log_Msg "Subject directory: $SubjectDIR"
log_Msg "Subject ID: $SubjectID"
log_Msg "Hemisphere: $hemi"

mri_dir=${SubjectDIR}/${SubjectID}/mri
surf_dir=${SubjectDIR}/${SubjectID}/surf
label_dir=${SubjectDIR}/${SubjectID}/label

# Main tessellation function
tessellate() {
    local hemi=$1
    
    cd ${mri_dir}
    # Step 1: Pre-tessellate the filled.mgz and norm.mgz
    if [[ ${hemi} == 'lh' ]]; then
        mri_pretess filled.mgz 255 norm.mgz filled-pretess255.mgz
        mri_tessellate filled-pretess255.mgz 255 "${surf_dir}/${hemi}.orig.nofix"
        rm -f filled-pretess255.mgz
    else
        mri_pretess filled.mgz 127 norm.mgz filled-pretess127.mgz
        mri_tessellate filled-pretess127.mgz 127 "${surf_dir}/${hemi}.orig.nofix"
        rm -f filled-pretess127.mgz
    fi

    cd ${surf_dir}
    # Step 2: Extract main component and smooth
    mris_extract_main_component "${hemi}.orig.nofix" "${hemi}.orig.nofix"
    mris_smooth -nw "${hemi}.orig.nofix" "${hemi}.smoothwm.nofix"

    # Step 3: Inflate surfaces
    mris_inflate -n 10 -no-save-sulc "${hemi}.smoothwm.nofix" "${hemi}.inflated.nofix"
    mv "${hemi}.inflated.nofix" "${hemi}.inflated.nofix.10"
    mris_inflate -n 35 -no-save-sulc "${hemi}.smoothwm.nofix" "${hemi}.inflated.nofix"

    # Step 4: Fix topology and apply transformations
    mris_sphere -q "${hemi}.inflated.nofix" "${hemi}.qsphere.nofix"
    cp "${hemi}.orig.nofix" "${hemi}.orig"
    cp "${hemi}.inflated.nofix" "${hemi}.inflated"
    mris_fix_topology -sdir ${SubjectDIR} -mgz -sphere "qsphere.nofix" -ga ${SubjectID} "${hemi}"
    mris_euler_number "${hemi}.orig"
    mris_remove_intersection "${hemi}.orig" "${hemi}.orig"

    # Step 5: Generate surfaces
    if command -v mris_autodet_gwstats > /dev/null 2>&1; then
        mris_autodet_gwstats --o autodet.gw.stats.${hemi}.dat --i ${mri_dir}/brain.finalsurfs.mgz --wm ${mri_dir}/wm.mgz --${hemi}-surf "$SubjectDIR"/"$SubjectID"/surf/${hemi}.orig
        cp ${hemi}.orig ${hemi}.white.preaparc
        mris_make_surfaces -aseg aseg.presurf -whiteonly -noaparc -mgz -T1 brain.finalsurfs -sdir $SubjectDIR $SubjectID ${hemi} -hires
        if [[ -e ${label_dir}/${hemi}.aparc.annot ]]; then
          rm ${label_dir}/${hemi}.aparc.annot
        fi
        cortex_ctab=${label_dir}/${hemi}.cortex.ctab
        printf "1 cortex 205 62 78 0\n" > ${cortex_ctab}
        mris_label2annot --s $SubjectID --h ${hemi} --l ${label_dir}/${hemi}.cortex.label --a aparc --ctab ${cortex_ctab} --sd $SubjectDIR
    else
        recon-all -subjid $SubjectID -sd $SubjectDIR -white -hires
    fi

    # Step 6: Handle the missing ${hemi}.white
    if [[ ! -f ${surf_dir}/${hemi}.white ]]; then
        echo "${hemi}.white not found. Copying ${hemi}.white.preaparc to ${hemi}.white..."
        cp "$SubjectDIR"/"$SubjectID"/surf/${hemi}.white.preaparc "$SubjectDIR"/"$SubjectID"/surf/${hemi}.white
        echo "${hemi}.white has been copied from ${hemi}.white.preaparc."
    else
        echo "${hemi}.white already exists."
    fi
}

# Step 7: Call tessellate function for the specified hemisphere
tessellate "$hemi"

# Done
log_Msg "Tessellation processing completed for hemisphere: $hemi"
cd $DIR
