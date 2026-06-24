#! /usr/bin/env python3
"""
Fieldmap estimation for MacaSurfer BOLD preprocessing.

Discovers fieldmaps from BIDS fmap/ directory using sdcflows,
estimates B0 field inhomogeneity maps, and saves outputs
to a persistent shared directory for reuse across BOLD runs.

Supports:
  - PEPOLAR (SE-EPI pairs with opposite phase encoding)
  - PHASEDIFF (Siemens GRE magnitude/phase difference)
  - MAPPED (pre-computed fieldmaps)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import bids


def find_outputs(output_dir, bids_id):
    """Locate sdcflows DataSink outputs for a given estimator.

    sdcflows writes outputs via DerivativesDataSink with BIDS naming.
    Files are organized as: <output_dir>/sub-<subject>/fmap/<files>
    or directly in output_dir depending on ``out_path_base`` setting.
    """
    out = {}
    sanitized = re.sub(r'[^a-zA-Z0-9]', '', bids_id)
    fmap_dir = Path(output_dir)
    if not fmap_dir.exists():
        return out

    all_niftis = sorted(fmap_dir.rglob('*.nii.gz'))
    for f in all_niftis:
        name = f.name
        if sanitized not in name and bids_id.replace('.', '') not in name and bids_id.replace('_', '') not in name:
            continue
        # Check basename (without extension) for key suffixes
        stem = f.name.replace('.nii.gz', '')
        if 'fieldmapref' in stem or 'fmap_ref' in stem:
            out.setdefault('fmap_ref', []).append(str(f))
        elif 'fieldmapcoeff' in stem or 'fieldcoef' in stem or 'coeff' in stem or 'coef' in stem:
            out.setdefault('fmap_coeff', []).append(str(f))
        elif 'fmapmask' in stem or 'fieldmap_mask' in stem:
            out.setdefault('fmap_mask', []).append(str(f))
        elif 'fieldmap' in stem:
            out.setdefault('fmap', []).append(str(f))
            # PEPOLAR reference file is named desc-epi_fieldmap (corrected EPI)
            if 'desc-epi' in stem:
                out.setdefault('fmap_ref', []).append(str(f))

    return out


def main():
    parser = argparse.ArgumentParser(
        description='Estimate B0 fieldmap from BIDS fmap directory using sdcflows.'
    )
    parser.add_argument('--bids_dir', required=True,
                        help='Root directory of the BIDS dataset')
    parser.add_argument('--bold_preprocess_dir', required=True,
                        help='BOLD preprocessing output root')
    parser.add_argument('--subject_id', required=True,
                        help='Subject ID (e.g., sub-032213)')
    parser.add_argument('--session_id', default='',
                        help='Session ID (e.g., ses-001)')
    parser.add_argument('--omp_nthreads', type=int, default=1,
                        help='Number of threads for parallel processing')
    args = parser.parse_args()

    bids_dir = Path(args.bids_dir)
    bold_preprocess_dir = Path(args.bold_preprocess_dir)
    sub = args.subject_id
    ses = args.session_id

    # Deterministic output directory (outside Nextflow work dir)
    # bold_preprocess_dir already includes sub/ses, write fmap directly under it
    fmap_out_dir = bold_preprocess_dir / 'fmap'
    fmap_out_dir.mkdir(parents=True, exist_ok=True)

    # Load BIDS layout
    layout = bids.BIDSLayout(str(bids_dir))
    subject_short = sub.replace('sub-', '')

    # Discover fieldmaps
    from sdcflows.utils.wrangler import find_estimators
    from sdcflows import fieldmaps as fm

    print(f'Searching for fieldmaps: subject={sub} session={ses or "(none)"}')
    try:
        estimators = find_estimators(
            layout=layout,
            subject=subject_short,
            fmapless=False,
        )
    except Exception as e:
        print(f'WARNING: find_estimators failed: {e}')
        estimators = []

    index = {}

    if not estimators:
        print(f'No fieldmaps found for {sub} {ses}')
        with open(fmap_out_dir / 'fieldmap_index.json', 'w') as f:
            json.dump(index, f, indent=2)
        print(f'Wrote empty index to {fmap_out_dir / "fieldmap_index.json"}')
        return

    print(f'Found {len(estimators)} estimator(s):')
    for est in estimators:
        print(f'  - {est.bids_id} ({est.method}), sources: {[s.path.name for s in est.sources]}')

    # Build and run fieldmap preprocessing workflow
    from sdcflows.workflows.base import init_fmap_preproc_wf

    fmap_wf = init_fmap_preproc_wf(
        estimators=estimators,
        omp_nthreads=args.omp_nthreads,
        output_dir=str(fmap_out_dir),
        subject=subject_short,
    )

    # Override out_path_base for all DataSinks to write directly to fmap_out_dir
    for node in fmap_wf.list_node_names():
        if node.split('.')[-1].startswith('ds_'):
            try:
                fmap_wf.get_node(node).interface.out_path_base = ''
            except Exception:
                pass

    # Wire estimator inputs
    for estimator in estimators:
        if estimator.method == fm.EstimatorType.PEPOLAR:
            suffices = [s.suffix for s in estimator.sources]
            if len(suffices) >= 2 and all(suf in ('epi', 'bold', 'sbref') for suf in suffices):
                wf_inputs = getattr(fmap_wf.inputs, f'in_{estimator.bids_id}')
                wf_inputs.in_data = [str(s.path) for s in estimator.sources]
                wf_inputs.metadata = [s.metadata for s in estimator.sources]
            else:
                print(f'WARNING: Skipping unsupported PEPOLAR estimator {estimator.bids_id}')
                continue
        elif estimator.method in (fm.EstimatorType.MAPPED, fm.EstimatorType.PHASEDIFF):
            # Connected internally by sdcflows
            pass
        else:
            print(f'WARNING: Unknown estimator type {estimator.method} for {estimator.bids_id}')

    # Run the workflow
    work_dir = fmap_out_dir / 'work'
    fmap_wf.base_dir = str(work_dir)
    print(f'Running fieldmap estimation workflow...')
    fmap_wf.run()
    print('Fieldmap estimation workflow complete.')

    # Collect outputs per estimator
    for estimator in estimators:
        bid = estimator.bids_id
        outputs = find_outputs(fmap_out_dir, bid)
        if outputs:
            index[bid] = {
                'method': str(estimator.method).rpartition('.')[-1],
                **outputs,
            }
            print(f'  Outputs for {bid}: {list(outputs.keys())}')
        else:
            print(f'  WARNING: No outputs found for {bid}')
            # Try broader search
            broad = find_outputs(fmap_out_dir, estimator.bids_id.replace('_', ''))
            if broad:
                index[bid] = {
                    'method': str(estimator.method).rpartition('.')[-1],
                    **broad,
                }
                print(f'  Fallback outputs found for {bid}')

    with open(fmap_out_dir / 'fieldmap_index.json', 'w') as f:
        json.dump(index, f, indent=2, default=str)

    print(f'\nFieldmap estimation complete. {len(index)} estimator(s) saved to {fmap_out_dir}')


if __name__ == '__main__':
    main()
