#! /usr/bin/env python3
"""
Apply susceptibility distortion correction (SDC) to MCFLIRT-motion-corrected BOLD.

Uses sdcflows ResampleSeries to apply fieldmap-based distortion correction
on top of the already motion-corrected BOLD data. The MCFLIRT transforms
are passed through as identity (since motion is already corrected) while
the fieldmap displacement is applied for SDC.

Workflow:
  1. Enhance + skullstrip BOLD reference
  2. Register fieldmap reference to BOLD reference (init_coeff2epi_wf)
  3. Reconstruct fieldmap in BOLD space (ReconstructFieldmap)
  4. Apply SDC via combined resampling (ResampleSeries)
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import nibabel as nb
import numpy as np
import bids
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.utility import KeySelect
from niworkflows.interfaces.nitransforms import ConcatenateXFMs
from niworkflows.func.util import init_enhance_and_skullstrip_bold_wf

from sdcflows.workflows.apply.registration import init_coeff2epi_wf
from bold_resampling import ResampleSeries, ReconstructFieldmap, DistortionParameters


def create_identity_hmc_xfm(output_file, nvols):
    """Create an ITK multi-frame transform file with identity transforms.

    Since MCFLIRT already motion-corrected the BOLD data, we use identity
    HMC transforms. ResampleSeries still applies the SDC fieldmap displacement.
    """
    template = (
        "#Transform {idx}\n"
        "Transform: MatrixOffsetTransformBase_double_3_3\n"
        "Parameters: 1 0 0 0 1 0 0 0 1 0 0 0\n"
        "FixedParameters: 0 0 0\n"
    )
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('#Insight Transform File V1.0\n')
        for i in range(nvols):
            f.write(template.format(idx=i))


def main():
    parser = argparse.ArgumentParser(
        description='Apply SDC (Susceptibility Distortion Correction) to BOLD data.'
    )
    parser.add_argument('--bids_dir', required=True,
                        help='Root directory of the BIDS dataset')
    parser.add_argument('--bold_file', required=True,
                        help='Original BOLD NIfTI file (reoriented, before MCFLIRT)')
    parser.add_argument('--boldref_file', required=True,
                        help='BOLD reference image (mean of motion-corrected)')
    parser.add_argument('--fmap_dir', required=True,
                        help='Directory containing fieldmap_index.json and sdcflows outputs')
    parser.add_argument('--nvols', type=int, required=True,
                        help='Total number of volumes in BOLD series')
    parser.add_argument('--bold_mc_file', required=True,
                        help='MCFLIRT motion-corrected BOLD (input for SDC)')
    parser.add_argument('--sdc_file', required=True,
                        help='Output: SDC-corrected BOLD file')
    parser.add_argument('--subject_id', required=True,
                        help='Subject ID (e.g., sub-032213)')
    parser.add_argument('--bold_id', required=True,
                        help='BOLD run identifier')
    parser.add_argument('--omp_nthreads', type=int, default=1,
                        help='Number of threads')
    args = parser.parse_args()

    fmap_dir = Path(args.fmap_dir)
    sdc_file = Path(args.sdc_file)

    # Check fieldmap index
    index_file = fmap_dir / 'fieldmap_index.json'
    if not index_file.exists():
        print(f'ERROR: Fieldmap index not found at {index_file}')
        return 1

    with open(index_file) as f:
        index = json.load(f)

    if not index:
        print('ERROR: Empty fieldmap index. Cannot apply SDC.')
        return 1

    # Get the first available estimator (typically only one per session)
    fieldmap_id = list(index.keys())[0]
    est = index[fieldmap_id]
    print(f'Using fieldmap estimator: {fieldmap_id} (method={est.get("method", "unknown")})')

    # Get fieldmap files
    fmap_ref = est.get('fmap_ref', [None])[0] if est.get('fmap_ref') else None
    fmap_coeffs = est.get('fmap_coeff', [])
    fmap_mask = est.get('fmap_mask', [None])[0] if est.get('fmap_mask') else None

    if not fmap_ref or not fmap_coeffs:
        print(f'ERROR: Missing fieldmap outputs: ref={fmap_ref}, coeffs={len(fmap_coeffs)}')
        return 1

    # If no mask file (common for PEPOLAR), generate one from fmap_ref
    if not fmap_mask:
        import subprocess
        gen_mask = Path(fmap_dir) / 'generated_fmap_mask.nii.gz'
        subprocess.run([
            'fslmaths', fmap_ref, '-bin', '-dilM', str(gen_mask)
        ], check=True, capture_output=True)
        fmap_mask = str(gen_mask)
        print(f'  Generated fmap_mask from reference: {fmap_mask}')

    print(f'  fmap_ref:   {fmap_ref}')
    print(f'  fmap_coeff: {fmap_coeffs}')
    print(f'  fmap_mask:  {fmap_mask}')

    # Ensure all referenced files exist
    if not Path(fmap_ref).exists():
        print(f'ERROR: fmap_ref file not found: {fmap_ref}')
        return 1
    for cf in fmap_coeffs:
        if not Path(cf).exists():
            print(f'ERROR: fmap_coeff file not found: {cf}')
            return 1

    # Load BIDS metadata for PE direction and readout time
    # The input bold_file may be a reoriented copy in the output directory.
    # We must find the original BIDS sidecar JSON, not look next to the reoriented file.
    metadata = {}
    # 1) Try sidecar JSON next to bold_file (works when bold_file is the original BIDS file)
    sidecar_json = Path(str(args.bold_file)).with_suffix('').with_suffix('.json')
    if not sidecar_json.exists():
        # 2) Handle reoriented files (e.g. *_space-reorient_bold.nii.gz) —
        #    find the original BIDS file via the rawdata layout
        bold_stem = sidecar_json.name.replace('.nii', '').replace('.json', '')
        # Remove known processing suffixes added by the pipeline
        for suffix in ['_space-reorient', '_desc-preproc', '_desc-mc', '_desc-skip',
                       '_desc-brain', '_desc-sdc']:
            if suffix in bold_stem:
                bold_stem = bold_stem.replace(suffix, '')
                sidecar_json = Path(str(args.bids_dir)).rglob(f'{bold_stem}.json')
                try:
                    sidecar_json = next(sidecar_json)
                except StopIteration:
                    sidecar_json = None
                break
    if sidecar_json and Path(sidecar_json).exists():
        with open(sidecar_json, 'r') as f:
            metadata = json.load(f)
    if not metadata:
        # 3) Last resort: scan rawdata for matching bold JSON by _bold suffix
        try:
            for jf in Path(args.bids_dir).rglob('*_bold.json'):
                if args.subject_id in str(jf):
                    with open(jf, 'r') as f:
                        metadata = json.load(f)
                    break
        except Exception:
            pass

    # Create temp directory for workflow
    tmp_dir = sdc_file.parent / f'.sdc_tmp_{args.bold_id}'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Create identity HMC transform file (motion already corrected by MCFLIRT)
    hmc_xfm_file = tmp_dir / 'hmc_xfm.txt'
    create_identity_hmc_xfm(str(hmc_xfm_file), args.nvols)

    # Build SDC apply workflow
    workflow = Workflow(name=f'{args.bold_id}_sdc_wf')

    # 1. Enhance + skullstrip BOLD reference
    enhance_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=args.omp_nthreads)
    enhance_wf.inputs.inputnode.in_file = args.boldref_file

    # 2. Select fieldmap outputs for this estimator
    fmap_select = pe.Node(
        KeySelect(fields=['fmap_ref', 'fmap_coeff', 'fmap_mask'],
                   key=fieldmap_id, keys=[fieldmap_id]),
        name='fmap_select', run_without_submitting=True,
    )
    fmap_select.inputs.fmap_ref = [fmap_ref]
    fmap_select.inputs.fmap_coeff = fmap_coeffs if isinstance(fmap_coeffs, list) else [fmap_coeffs]
    fmap_select.inputs.fmap_mask = [fmap_mask]

    # 3. Register fieldmap reference to BOLD reference
    fmapreg_wf = init_coeff2epi_wf(
        omp_nthreads=args.omp_nthreads,
        name='fmapreg_wf',
    )

    # 4. Convert registration to ITK text format
    itk_mat2txt = pe.Node(ConcatenateXFMs(out_fmt='itk'), name='itk_mat2txt')

    # 5. Buffer node for fieldmap registration output
    fmapreg_buffer = pe.Node(
        niu.IdentityInterface(fields=['boldref2fmap_xfm', 'boldref']),
        name='fmapreg_buffer',
    )

    # 6. Extract PE direction and readout time from BIDS metadata
    distortion_params = pe.Node(
        DistortionParameters(metadata=metadata, in_file=args.bold_file),
        name='distortion_params', run_without_submitting=True,
    )

    # 7. Reconstruct fieldmap (Hz) in BOLD reference space
    boldref_fmap = pe.Node(
        ReconstructFieldmap(inverse=[True]),
        name='boldref_fmap', mem_gb=1,
    )

    # 8. Combined resampling with HMC (identity) + SDC
    boldref_bold = pe.Node(
        ResampleSeries(jacobian=True),
        name='boldref_bold', n_procs=args.omp_nthreads, mem_gb='2GB',
    )
    # Use the MCFLIRT motion-corrected file as input (so we only apply SDC on top)
    boldref_bold.inputs.in_file = args.bold_mc_file
    boldref_bold.inputs.transforms = [str(hmc_xfm_file)]

    # Wire workflow connections (mirrors DeepPrep's bold_sdc.py architecture)
    # fmt:off
    workflow.connect([
        # BOLD reference -> fieldmap registration
        (enhance_wf, fmapreg_wf, [
            ('outputnode.bias_corrected_file', 'inputnode.target_ref'),
            ('outputnode.mask_file', 'inputnode.target_mask'),
        ]),
        # Fieldmap data -> fieldmap registration
        (fmap_select, fmapreg_wf, [
            ('fmap_ref', 'inputnode.fmap_ref'),
            ('fmap_mask', 'inputnode.fmap_mask'),
            ('fmap_coeff', 'inputnode.fmap_coeff'),
        ]),
        # Registration output -> ITK text
        (fmapreg_wf, itk_mat2txt, [('outputnode.target2fmap_xfm', 'in_xfms')]),
        (itk_mat2txt, fmapreg_buffer, [('out_xfm', 'boldref2fmap_xfm')]),
        # BOLD reference for fieldmap reconstruction
        (enhance_wf, fmapreg_buffer, [('outputnode.bias_corrected_file', 'boldref')]),
        # Fieldmap reconstruction in BOLD space
        (fmapreg_buffer, boldref_fmap, [
            ('boldref', 'target_ref_file'),
            ('boldref2fmap_xfm', 'transforms'),
        ]),
        (fmap_select, boldref_fmap, [
            ('fmap_coeff', 'in_coeffs'),
            ('fmap_ref', 'fmap_ref_file'),
        ]),
        # BOLD reference for resampling
        (fmapreg_buffer, boldref_bold, [('boldref', 'ref_file')]),
        # Distortion parameters for resampling
        (distortion_params, boldref_bold, [
            ('readout_time', 'ro_time'),
            ('pe_direction', 'pe_dir'),
        ]),
        # Fieldmap for SDC application
        (boldref_fmap, boldref_bold, [('out_file', 'fieldmap')]),
    ])
    # fmt:on

    print(f'Running SDC apply workflow...')
    workflow.base_dir = str(tmp_dir)
    try:
        workflow.run()
    except Exception as e:
        print(f'ERROR: SDC workflow failed: {e}')
        import traceback
        traceback.print_exc()
        return 1

    # Find output from ResampleSeries
    try:
        sdc_result = list(tmp_dir.rglob('boldref_bold/sub-*resampled.nii.gz'))
        if not sdc_result:
            sdc_result = list(tmp_dir.rglob('*resampled*.nii.gz'))
        if not sdc_result:
            sdc_result = list(tmp_dir.rglob('sub-*.nii.gz'))
        if sdc_result:
            result_path = sdc_result[0]
            print(f'Moving SDC result: {result_path} -> {sdc_file}')
            sdc_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(result_path), str(sdc_file))
        else:
            print('ERROR: No SDC output file found in workflow directory')
            return 1
    except Exception as e:
        print(f'ERROR: Failed to locate SDC output: {e}')
        return 1

    # Cleanup temp directory
    try:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
    except Exception:
        pass

    print(f'SDC applied successfully: {sdc_file}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
