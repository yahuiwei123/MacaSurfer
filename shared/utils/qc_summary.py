#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a summary HTML page for all QC results in the output directory
"""
import os
import re
import argparse
from pathlib import Path
from collections import defaultdict

# QC type display names and descriptions
QC_TYPES = {
    'qc_skullstrip': {
        'name': 'Skull Stripping',
        'description': 'Brain extraction result, blue overlay is the brain mask'
    },
    'qc_fixed_brainmask': {
        'name': 'Brain Mask Correction',
        'description': 'Comparison of original and corrected brain masks'
    },
    'qc_corr_orient': {
        'name': 'Orientation Correction',
        'description': 'Result of image orientation standardization to RAS space'
    },
    'qc_modality_register': {
        'name': 'Multi-modality Registration',
        'description': 'Alignment quality between T1w, T2w and FLAIR images'
    },
    'qc_template_register': {
        'name': 'Template Registration',
        'description': 'Alignment quality between subject image and standard template'
    },
    'qc_tissue_segment': {
        'name': 'Tissue Segmentation',
        'description': 'Gray matter, white matter, CSF and brain structure segmentation results'
    },
    'qc_bias_field_corr': {
        'name': 'Bias Field Correction',
        'description': 'Result of intensity inhomogeneity correction'
    },
    'qc_surface': {
        'name': 'Cortical Surface Reconstruction',
        'description': 'Quality of cortical surface and parcellation results'
    },
    'qc_bold_motion': {
        'name': 'BOLD Motion Correction',
        'description': 'Head motion parameters, translations, rotations and framewise displacement'
    },
    'qc_bold_registration': {
        'name': 'BOLD-to-T1w Registration',
        'description': 'Alignment quality between BOLD and T1w reference'
    },
    'qc_bold_tsnr': {
        'name': 'BOLD TSNR',
        'description': 'Temporal signal-to-noise ratio map of preprocessed BOLD'
    },
    'qc_bold_carpet': {
        'name': 'BOLD Carpet Plot',
        'description': 'Carpet plot showing voxel-wise timeseries and global signal'
    },
    'qc_bold_normalize': {
        'name': 'BOLD Template Normalization',
        'description': 'Alignment quality between normalized BOLD and template space'
    },
    'qc_bold_surf': {
        'name': 'BOLD Surface Projection',
        'description': 'Quality check for BOLD projection onto cortical surface'
    }
}

def parse_qc_file(file_path):
    """
    Parse QC file path to get subject, session, modality and qc type
    Supports:
    - qc_dir/subject/session/qc_<type>_<modality>_<run>.png (new format)
    - qc_dir/subject/session/qc_bold_<type>_<bold_id>.png (BOLD QC format)
    - .../subject/session/Prepare/Volume/.../qc_*.png (old format)
    """
    path = Path(file_path)
    parts = path.parts

    sub_id = None
    ses_id = None
    modality = None
    qc_type = path.stem

    # Try BIDS-style prefix matching
    for i, part in enumerate(parts):
        if part.startswith('sub-') and sub_id is None:
            sub_id = part
        elif part.startswith('ses-') and ses_id is None:
            ses_id = part
        elif part in ['T1', 'T2', 'FLAIR', 'T1w', 'T2w', 'FLAIRw'] and modality is None:
            modality = part.rstrip('w')

    # Fallback: look at parent directory structure (qc_dir/subject/session/qc_*.png)
    if sub_id is None or ses_id is None:
        # QC file is in session directory: parent is session, grandparent is subject
        if len(parts) >= 3:
            if parts[-2].startswith('ses-') and parts[-3].startswith('sub-'):
                ses_id = parts[-2]
                sub_id = parts[-3]

    # BOLD QC files have pattern: qc_bold_<type>_<bold_id>.png
    # Extract the qc_type prefix (e.g., qc_bold_motion) from the filename
    if path.name.startswith('qc_bold_'):
        modality = 'BOLD'
        # Match: qc_bold_motion, qc_bold_registration, qc_bold_tsnr,
        #        qc_bold_carpet, qc_bold_normalize, qc_bold_surf
        bold_qc_match = re.match(r'(qc_bold_\w+?)_', path.stem)
        if bold_qc_match:
            qc_type = bold_qc_match.group(1)

    # If modality not found in path, try to extract from filename
    if modality is None:
        # Filename pattern: qc_skullstrip_T1_run-1.png
        match = re.search(r'(?:^|_)(T1|T2|FLAIR)(?:_|$)', path.name, re.IGNORECASE)
        if match:
            modality = match.group(1).upper()

    return {
        'subject': sub_id,
        'session': ses_id,
        'modality': modality or 'Unknown',
        'qc_type': qc_type,
        'path': str(path),
        'rel_path': str(path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path),
        'filename': path.name
    }

def generate_html(qc_data, output_path, root_dir):
    """
    Generate the HTML summary page
    """
    # Get all unique subjects, sessions, modalities and qc types
    subjects = sorted(qc_data.keys())
    sessions = set()
    modalities = set()
    qc_types = set()

    # Safe, explicit traversal with type checking
    for sub_id, sessions_dict in qc_data.items():
        for ses_id, modalities_dict in sessions_dict.items():
            sessions.add(ses_id)
            for mod, qc_types_dict in modalities_dict.items():
                modalities.add(mod)
                if isinstance(qc_types_dict, dict):
                    for qt in qc_types_dict.keys():
                        qc_types.add(qt)

    # Strict filtering: only keep string values for all filter categories
    subjects = sorted([s for s in subjects if isinstance(s, str)])
    sessions = sorted([s for s in sessions if isinstance(s, str)])
    modalities = sorted([m for m in modalities if isinstance(m, str)])

    qc_types_filtered = []
    for qt in qc_types:
        if isinstance(qt, str):
            qc_types_filtered.append(qt)
        else:
            print(f"⚠️  Skipping non-string qc_type: {qt} (type: {type(qt).__name__})")
    qc_types = sorted(qc_types_filtered)

    # Count total QC images
    total_qc = 0
    for sub in qc_data.values():
        for ses in sub.values():
            for mod in ses.values():
                if isinstance(mod, dict):
                    for qt in mod.values():
                        total_qc += len(qt)

    # Pre-render filter options (outside f-string for safety)
    subject_options = ''.join([f'<option value="{sub}">{sub}</option>' for sub in subjects])
    session_options = ''.join([f'<option value="{ses}">{ses}</option>' for ses in sessions])
    modality_options = ''.join([f'<option value="{mod}">{mod}</option>' for mod in modalities])

    qc_type_options_parts = []
    for qt in qc_types:
        qc_info = QC_TYPES.get(qt, {'name': qt.replace('qc_', '').title()})
        qc_type_options_parts.append(f'<option value="{qt}">{qc_info["name"]}</option>')
    qc_type_options = ''.join(qc_type_options_parts)

    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MacaSurfer QC Summary Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
        }}
        .qc-card {{
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .qc-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        .qc-image {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            object-position: top center;
            cursor: pointer;
            border-bottom: 1px solid #dee2e6;
        }}
        .qc-info {{
            padding: 10px;
        }}
        .qc-type-badge {{
            font-size: 0.75rem;
            margin-right: 5px;
        }}
        .modal-image {{
            max-width: 100%;
            max-height: 90vh;
        }}
        .nav-link.active {{
            font-weight: bold;
        }}
        .header-section {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem 0;
            margin-bottom: 2rem;
        }}
        .stats-card {{
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 10px;
            padding: 1rem;
        }}
        .filter-section {{
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
        }}
        .empty-state {{
            text-align: center;
            padding: 3rem;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <!-- Header -->
    <div class="header-section">
        <div class="container">
            <h1><i class="bi bi-images me-2"></i>MacaSurfer QC Summary Report</h1>
            <p class="lead">Automatic preprocessing quality control for non-human primate brain MRI</p>
            <div class="row mt-4">
                <div class="col-md-3">
                    <div class="stats-card text-center">
                        <h3>{len(subjects)}</h3>
                        <p class="mb-0">Subjects</p>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stats-card text-center">
                        <h3>{len(sessions)}</h3>
                        <p class="mb-0">Sessions</p>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stats-card text-center">
                        <h3>{len(modalities)}</h3>
                        <p class="mb-0">Modalities</p>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stats-card text-center">
                        <h3>{total_qc}</h3>
                        <p class="mb-0">QC Images</p>
                    </div>
                </div>
            </div>
            <div class="mt-3 small">
                <p class="mb-0"><i class="bi bi-folder me-1"></i> Output directory: {root_dir}</p>
                <p class="mb-0"><i class="bi bi-calendar me-1"></i> Generated on: {os.popen('date').read().strip()}</p>
            </div>
        </div>
    </div>

    <div class="container">
        <!-- Filters -->
        <div class="filter-section">
            <h5><i class="bi bi-filter me-2"></i>Filter Results</h5>
            <div class="row mt-3">
                <div class="col-md-3">
                    <label for="subject-filter" class="form-label">Subject</label>
                    <select class="form-select" id="subject-filter">
                        <option value="all">All Subjects</option>
                        {subject_options}
                    </select>
                </div>
                <div class="col-md-3">
                    <label for="session-filter" class="form-label">Session</label>
                    <select class="form-select" id="session-filter">
                        <option value="all">All Sessions</option>
                        {session_options}
                    </select>
                </div>
                <div class="col-md-3">
                    <label for="modality-filter" class="form-label">Modality</label>
                    <select class="form-select" id="modality-filter">
                        <option value="all">All Modalities</option>
                        {modality_options}
                    </select>
                </div>
                <div class="col-md-3">
                    <label for="qc-type-filter" class="form-label">QC Type</label>
                    <select class="form-select" id="qc-type-filter">
                        <option value="all">All QC Types</option>
                        {qc_type_options}
                    </select>
                </div>
            </div>
        </div>

        <!-- Tabs for different views -->
        <ul class="nav nav-tabs mb-4" id="qcTabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="by-subject-tab" data-bs-toggle="tab" data-bs-target="#by-subject" type="button" role="tab">
                    <i class="bi bi-people me-1"></i>By Subject
                </button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="by-qc-type-tab" data-bs-toggle="tab" data-bs-target="#by-qc-type" type="button" role="tab">
                    <i class="bi bi-list-check me-1"></i>By QC Type
                </button>
            </li>
        </ul>

        <div class="tab-content" id="qcTabsContent">
            <!-- By Subject View -->
            <div class="tab-pane fade show active" id="by-subject" role="tabpanel">
                {generate_by_subject_content(qc_data)}
            </div>

            <!-- By QC Type View -->
            <div class="tab-pane fade" id="by-qc-type" role="tabpanel">
                {generate_by_qc_type_content(qc_data)}
            </div>
        </div>
    </div>

    <!-- Image Preview Modal -->
    <div class="modal fade" id="imageModal" tabindex="-1" aria-labelledby="imageModalLabel" aria-hidden="true">
        <div class="modal-dialog modal-xl">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="imageModalLabel">QC Image Preview</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body text-center">
                    <img src="" id="modalImage" class="modal-image" alt="QC Image">
                </div>
                <div class="modal-footer justify-content-between">
                    <div id="modalImageInfo" class="text-start small text-muted"></div>
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Image preview functionality
        document.querySelectorAll('.qc-image').forEach(img => {{
            img.addEventListener('click', function() {{
                const src = this.getAttribute('src');
                const info = this.getAttribute('data-info');
                document.getElementById('modalImage').src = src;
                document.getElementById('modalImageInfo').textContent = info;
                new bootstrap.Modal(document.getElementById('imageModal')).show();
            }});
        }});

        // Filter functionality
        function applyFilters() {{
            const subjectFilter = document.getElementById('subject-filter').value;
            const sessionFilter = document.getElementById('session-filter').value;
            const modalityFilter = document.getElementById('modality-filter').value;
            const qcTypeFilter = document.getElementById('qc-type-filter').value;

            document.querySelectorAll('.qc-card-container').forEach(container => {{
                const subject = container.getAttribute('data-subject');
                const session = container.getAttribute('data-session');
                const modality = container.getAttribute('data-modality');
                const qcType = container.getAttribute('data-qc-type');

                let show = true;
                if (subjectFilter !== 'all' && subject !== subjectFilter) show = false;
                if (sessionFilter !== 'all' && session !== sessionFilter) show = false;
                if (modalityFilter !== 'all' && modality !== modalityFilter) show = false;
                if (qcTypeFilter !== 'all' && qcType !== qcTypeFilter) show = false;

                container.style.display = show ? 'block' : 'none';
            }});

            // Update empty state visibility
            document.querySelectorAll('.subject-section').forEach(section => {{
                const visibleCards = section.querySelectorAll('.qc-card-container[style*="display: block"]');
                const emptyState = section.querySelector('.empty-state');
                if (visibleCards.length === 0) {{
                    emptyState.style.display = 'block';
                }} else {{
                    emptyState.style.display = 'none';
                }}
            }});
        }}

        document.getElementById('subject-filter').addEventListener('change', applyFilters);
        document.getElementById('session-filter').addEventListener('change', applyFilters);
        document.getElementById('modality-filter').addEventListener('change', applyFilters);
        document.getElementById('qc-type-filter').addEventListener('change', applyFilters);

        // Collapsible sections
        document.querySelectorAll('.btn-collapse').forEach(btn => {{
            btn.addEventListener('click', function() {{
                const target = this.getAttribute('data-bs-target');
                const icon = this.querySelector('i');
                if (icon.classList.contains('bi-chevron-down')) {{
                    icon.classList.replace('bi-chevron-down', 'bi-chevron-up');
                }} else {{
                    icon.classList.replace('bi-chevron-up', 'bi-chevron-down');
                }}
            }});
        }});
    </script>
</body>
</html>
    """

    # Write HTML to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"✅ QC summary report generated successfully: {output_path}")
    print(f"📊 Total QC images included: {total_qc}")
    print(f"👥 Subjects: {len(subjects)}")
    print(f"📂 Sessions: {len(sessions)}")

def generate_by_subject_content(qc_data):
    """Generate HTML content for the By Subject view"""
    html = ""

    for sub_id, sessions in sorted(qc_data.items()):
        html += f"""
        <div class="subject-section mb-5" data-subject="{sub_id}">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h3><i class="bi bi-person me-2"></i>{sub_id}</h3>
                <button class="btn btn-sm btn-outline-primary btn-collapse" type="button"
                        data-bs-toggle="collapse" data-bs-target="#collapse-{sub_id}"
                        aria-expanded="true" aria-controls="collapse-{sub_id}">
                    <i class="bi bi-chevron-down"></i> Toggle
                </button>
            </div>
            <div class="collapse show" id="collapse-{sub_id}">
        """

        for ses_id, modalities in sorted(sessions.items()):
            html += f"""
                <div class="session-section mb-4" data-session="{ses_id}">
                    <h5 class="text-secondary mb-3"><i class="bi bi-calendar-event me-2"></i>{ses_id}</h5>
                    <div class="empty-state" style="display: none;">
                        <i class="bi bi-search fs-1 mb-3"></i>
                        <p>No QC results match your current filter criteria</p>
                    </div>
                    <div class="row row-cols-1 row-cols-md-2 row-cols-lg-3 row-cols-xl-4 g-4">
            """

            for mod, qc_types in sorted(modalities.items()):
                for qc_type, qc_files in sorted(qc_types.items()):
                    for qc_file in qc_files:
                        qc_info = QC_TYPES.get(qc_type, {
                            'name': qc_type.replace('qc_', '').title(),
                            'description': ''
                        })

                        # Calculate relative path from HTML file location
                        rel_path = os.path.relpath(qc_file['path'], os.path.dirname(args.output))
                        # Make path URL friendly
                        rel_path = rel_path.replace(os.path.sep, '/')

                        html += f"""
                        <div class="col qc-card-container"
                             data-subject="{sub_id}"
                             data-session="{ses_id}"
                             data-modality="{mod}"
                             data-qc-type="{qc_type}">
                            <div class="card qc-card h-100">
                                <img src="{rel_path}"
                                     class="qc-image"
                                     alt="{qc_info['name']}"
                                     data-info="{sub_id} | {ses_id} | {mod} | {qc_info['name']}">
                                <div class="card-body qc-info">
                                    <div class="d-flex justify-content-between align-items-start mb-2">
                                        <span class="badge bg-primary qc-type-badge">{mod}</span>
                                        <span class="badge bg-secondary qc-type-badge">{qc_info['name']}</span>
                                    </div>
                                    <h6 class="card-title mb-1">{sub_id}</h6>
                                    <p class="card-text small text-muted mb-1">{ses_id}</p>
                                    <p class="card-text small text-muted">{os.path.basename(qc_file['path'])}</p>
                                    {f'<p class="card-text small text-secondary">{qc_info["description"]}</p>' if qc_info['description'] else ''}
                                </div>
                            </div>
                        </div>
                        """

            html += """
                    </div>
                </div>
            </div>
        </div>
            """

    if not html:
        html = """
        <div class="empty-state">
            <i class="bi bi-inbox fs-1 mb-3"></i>
            <h3>No QC results found</h3>
            <p>No QC images were found in the output directory.</p>
        </div>
        """

    return html

def generate_by_qc_type_content(qc_data):
    """Generate HTML content for the By QC Type view"""
    # Group QC files by QC type
    qc_type_groups = defaultdict(list)

    for sub_id, sessions in qc_data.items():
        for ses_id, modalities in sessions.items():
            for mod, qc_types in modalities.items():
                for qc_type, qc_files in qc_types.items():
                    for qc_file in qc_files:
                        qc_type_groups[qc_type].append({
                            **qc_file,
                            'subject': sub_id,
                            'session': ses_id,
                            'modality': mod
                        })

    html = ""

    for qc_type, qc_files in sorted(qc_type_groups.items()):
        qc_info = QC_TYPES.get(qc_type, {
            'name': qc_type.replace('qc_', '').title(),
            'description': ''
        })

        html += f"""
        <div class="qc-type-section mb-5" data-qc-type="{qc_type}">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h3><i class="bi bi-check2-square me-2"></i>{qc_info['name']}</h3>
                <button class="btn btn-sm btn-outline-primary btn-collapse" type="button"
                        data-bs-toggle="collapse" data-bs-target="#collapse-{qc_type}"
                        aria-expanded="true" aria-controls="collapse-{qc_type}">
                    <i class="bi bi-chevron-down"></i> Toggle
                </button>
            </div>
            {f'<p class="text-muted mb-3">{qc_info["description"]}</p>' if qc_info['description'] else ''}
            <div class="collapse show" id="collapse-{qc_type}">
                <div class="empty-state" style="display: none;">
                    <i class="bi bi-search fs-1 mb-3"></i>
                    <p>No QC results match your current filter criteria</p>
                </div>
                <div class="row row-cols-1 row-cols-md-2 row-cols-lg-3 row-cols-xl-4 g-4">
        """

        for qc_file in sorted(qc_files, key=lambda x: (x['subject'], x['session'], x['modality'])):
            # Calculate relative path from HTML file location
            rel_path = os.path.relpath(qc_file['path'], os.path.dirname(args.output))
            # Make path URL friendly
            rel_path = rel_path.replace(os.path.sep, '/')

            html += f"""
            <div class="col qc-card-container"
                 data-subject="{qc_file['subject']}"
                 data-session="{qc_file['session']}"
                 data-modality="{qc_file['modality']}"
                 data-qc-type="{qc_type}">
                <div class="card qc-card h-100">
                    <img src="{rel_path}"
                         class="qc-image"
                         alt="{qc_info['name']}"
                         data-info="{qc_file['subject']} | {qc_file['session']} | {qc_file['modality']} | {qc_info['name']}">
                    <div class="card-body qc-info">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            <span class="badge bg-primary qc-type-badge">{qc_file['modality']}</span>
                            <span class="badge bg-secondary qc-type-badge">{qc_file['subject']}</span>
                        </div>
                        <h6 class="card-title mb-1">{qc_file['subject']}</h6>
                        <p class="card-text small text-muted mb-1">{qc_file['session']}</p>
                        <p class="card-text small text-muted">{os.path.basename(qc_file['path'])}</p>
                    </div>
                </div>
            </div>
            """

        html += """
                </div>
            </div>
        </div>
        """

    if not html:
        html = """
        <div class="empty-state">
            <i class="bi bi-inbox fs-1 mb-3"></i>
            <h3>No QC results found</h3>
            <p>No QC images were found in the output directory.</p>
        </div>
        """

    return html

def main():
    parser = argparse.ArgumentParser(description='Generate QC summary HTML report')
    parser.add_argument('--output-dir', required=True, help='Root output directory of the pipeline')
    parser.add_argument('--output', default='qc_summary.html', help='Output HTML file path')
    parser.add_argument('--subject', default=None, help='Filter by subject ID (e.g., sub-032217)')
    parser.add_argument('--session', default=None, help='Filter by session ID (e.g., ses-003)')
    parser.add_argument('--qc-dir', required=True, help='QC directory path (qc_dir/subject/session/)')

    global args
    args = parser.parse_args()

    qc_dir = os.path.abspath(args.qc_dir)

    if args.subject and args.session:
        print(f"🔍 Generating QC report for {args.subject} / {args.session}")
        # Look directly in qc_dir/subject/session/
        session_qc_dir = Path(qc_dir) / args.subject / args.session
        if not session_qc_dir.exists():
            print(f"⚠️  Session QC directory not found: {session_qc_dir}")
            generate_html({}, args.output, qc_dir)
            return

        # Find all QC png files in this session's QC directory
        qc_files = list(session_qc_dir.glob('qc_*.png'))
    else:
        print(f"🔍 Scanning for QC images in: {qc_dir}")
        # Find all QC png files in qc_dir/subject/session/ structure
        qc_files = []
        for ext in ['qc_*.png']:
            qc_files.extend(list(Path(qc_dir).rglob(ext)))

    print(f"📸 Found {len(qc_files)} QC images")

    if not qc_files:
        print("⚠️  No QC images found. Exiting.")
        generate_html({}, args.output, qc_dir)
        return

    # Parse all QC files
    # 4-level nested dict: qc_data[subject][session][modality][qc_type] = list_of_qc_files
    qc_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for qc_file in qc_files:
        try:
            parsed = parse_qc_file(qc_file)
            # Provide fallbacks and ensure ALL keys are strings
            subject = str(parsed['subject']) if parsed['subject'] else 'sub-unknown'
            session = str(parsed['session']) if parsed['session'] else 'ses-default'
            modality = str(parsed['modality']) if parsed['modality'] else 'Unknown'
            qc_type = str(parsed['qc_type']) if parsed['qc_type'] else 'qc_unknown'

            # Additional safety: ensure qc_type actually starts with 'qc_'
            if not qc_type.startswith('qc_'):
                qc_type = 'qc_' + qc_type

            qc_data[subject][session][modality][qc_type].append(parsed)
        except Exception as e:
            print(f"⚠️  Error parsing {qc_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate HTML
    generate_html(qc_data, args.output, args.output_dir)

if __name__ == "__main__":
    main()
