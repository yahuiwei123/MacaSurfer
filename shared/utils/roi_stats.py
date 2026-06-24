#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
ROI statistics extraction for a single subject/session.
Outputs CSV files in the same structure as the batch mode of gaojinquan_atlas_stats.py:
  - cortical metrics: out_dir/cort/{atlas}/{hemi}/{thickness,curvature,sulc,area,cortvol,vertex_count}.csv
  - subcortical volumes: out_dir/subcort/aseg/{L,R}/volume.csv
Each CSV contains one row per subject/session and is automatically merged with existing files.
"""

import os
import glob
import re
import sys
import argparse
from collections import defaultdict

import numpy as np
import nibabel as nib
import pandas as pd

# -------------------------
# Subcortical collapsing map (given)
# -------------------------
HEMI_COLLAPSE_MAP = {
    # Cerebrum
    41: 2,   # Right-Cerebral-White-Matter  -> Left-Cerebral-White-Matter
    42: 3,   # Right-Cerebral-Cortex        -> Left-Cerebral-Cortex

    # Ventricles
    43: 4,   # Right-Lateral-Ventricle      -> Left-Lateral-Ventricle
    44: 5,   # Right-Inf-Lat-Vent           -> Left-Inf-Lat-Vent

    # Cerebellum
    46: 7,   # Right-Cerebellum-White-Matter -> Left-Cerebellum-White-Matter
    47: 8,   # Right-Cerebellum-Cortex       -> Left-Cerebellum-Cortex

    # Deep gray nuclei
    49: 10,  # Right-Thalamus-Proper -> Left-Thalamus-Proper
    50: 11,  # Right-Caudate         -> Left-Caudate
    51: 12,  # Right-Putamen         -> Left-Putamen
    52: 13,  # Right-Pallidum        -> Left-Pallidum

    # Limbic
    53: 17,  # Right-Hippocampus -> Left-Hippocampus
    54: 18,  # Right-Amygdala    -> Left-Amygdala

    # Others commonly used
    58: 26,  # Right-Accumbens-area -> Left-Accumbens-area
    59: 27,  # Right-Substancia-Nigra -> Left-Substancia-Nigra
    60: 28,  # Right-VentralDC      -> Left-VentralDC
    63: 31,  # Right-choroid-plexus -> Left-choroid-plexus
    139: 138, # Right-Claustrum -> Left-Claustrum
    140: 140, # Cornea -> Cornea
}

# Minimal LUT for labels you explicitly listed (and their L counterparts)
ASEG_LABEL_NAME = {
    2: "Left-Cerebral-White-Matter",
    3: "Left-Cerebral-Cortex",
    4: "Left-Lateral-Ventricle",
    5: "Left-Inf-Lat-Vent",
    7: "Left-Cerebellum-White-Matter",
    8: "Left-Cerebellum-Cortex",
    10: "Left-Thalamus-Proper",
    11: "Left-Caudate",
    12: "Left-Putamen",
    13: "Left-Pallidum",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    26: "Left-Accumbens-area",
    27: "Left-Substancia-Nigra",
    28: "Left-VentralDC",
    31: "Left-choroid-plexus",
    138: "Left-Claustrum",
    140: "Cornea",

    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    43: "Right-Lateral-Ventricle",
    44: "Right-Inf-Lat-Vent",
    46: "Right-Cerebellum-White-Matter",
    47: "Right-Cerebellum-Cortex",
    49: "Right-Thalamus-Proper",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    58: "Right-Accumbens-area",
    59: "Right-Substancia-Nigra",
    60: "Right-VentralDC",
    63: "Right-choroid-plexus",
    139: "Right-Claustrum",
}

def _tet_volume(a, b, c, d):
    u = b - a
    v = c - a
    w = d - a
    return np.abs(np.einsum("ij,ij->i", np.cross(u, v), w)) / 6.0

def face_prism_volumes(white_xyz, pial_xyz, faces):
    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
    W0, W1, W2 = white_xyz[f0], white_xyz[f1], white_xyz[f2]
    P0, P1, P2 = pial_xyz[f0],  pial_xyz[f1],  pial_xyz[f2]

    v1 = _tet_volume(W0, W1, W2, P0)
    v2 = _tet_volume(P0, P1, P2, W1)
    v3 = _tet_volume(P0, P2, W2, W1)
    return v1 + v2 + v3

def cortical_parcel_volumes_prism(
    white_surf_gii: str,
    pial_surf_gii: str,
    label_ids: np.ndarray,
    label_table,
    distribute: str = "vertex_thirds",
):
    w = nib.load(white_surf_gii)
    p = nib.load(pial_surf_gii)

    white_xyz = np.asarray(w.darrays[0].data, dtype=np.float64)
    faces = np.asarray(w.darrays[1].data, dtype=np.int64)
    pial_xyz = np.asarray(p.darrays[0].data, dtype=np.float64)

    if white_xyz.shape != pial_xyz.shape:
        raise ValueError("white/pial vertex count mismatch.")
    if label_ids.shape[0] != white_xyz.shape[0]:
        raise ValueError("label_ids vertex count mismatch with surfaces.")

    face_vol = face_prism_volumes(white_xyz, pial_xyz, faces)
    total_vol = float(np.sum(face_vol))

    id2name = label_table.get_labels_as_dict()

    lab = np.asarray(label_ids, dtype=np.int64)
    roi_ids = np.unique(lab)
    roi_ids = roi_ids[roi_ids != 0]

    if roi_ids.size == 0:
        return total_vol, {}

    max_id = int(lab.max())
    lut = -np.ones(max_id + 1, dtype=np.int32)
    lut[roi_ids] = np.arange(roi_ids.size, dtype=np.int32)

    face_labs = lab[faces]
    face_idx = lut[face_labs]

    acc = np.zeros(roi_ids.size, dtype=np.float64)

    if distribute == "vertex_thirds":
        share = face_vol / 3.0
        for k in range(3):
            idxk = face_idx[:, k]
            m = idxk >= 0
            np.add.at(acc, idxk[m], share[m])
    else:
        raise ValueError(f"Unknown distribute mode: {distribute}")

    out = {}
    for i, rid in enumerate(roi_ids.tolist()):
        name = id2name.get(int(rid), f"Label{rid}")
        out[name] = float(acc[i])

    return total_vol, out

def sanitize_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def strip_lr_prefix(name: str) -> str:
    if name.startswith("Left-"):
        return name[len("Left-"):]
    if name.startswith("Right-"):
        return name[len("Right-"):]
    return name

def parse_label_names(label_array, label_table):
    label_dict = label_table.get_labels_as_dict()
    labels = []
    for idx in label_array:
        idx_int = int(idx)
        # 标签ID <= 0的都是中墙/无效区域
        if idx_int <= 0:
            labels.append("Unknown")
        else:
            lab = label_dict.get(idx_int, "Unknown")
            # 把所有中墙相关标签统一为Unknown
            MEDIAL_WALL_LABELS = {'???', 'MEDIAL.WALL'}
            if lab in MEDIAL_WALL_LABELS or lab.startswith('MedialWall_'):
                lab = "Unknown"
            labels.append(lab)
    return labels

def compute_metrics(label_list, metric_array, metric='mean'):
    label_dict = defaultdict(list)
    # 中墙标签集合，包含所有已知的中墙/无效区域标签
    MEDIAL_WALL_LABELS = {'Unknown', '???', 'MEDIAL.WALL'}
    # 构建有效顶点掩码：排除中墙标签和以MedialWall开头的标签
    label_arr = np.array(label_list)
    valid_mask = np.ones(len(label_list), dtype=bool)
    for i, lab in enumerate(label_list):
        if lab in MEDIAL_WALL_LABELS or lab.startswith('MedialWall_'):
            valid_mask[i] = False
    valid_metrics = metric_array[valid_mask]

    for label, value in zip(label_list, metric_array):
        if label in MEDIAL_WALL_LABELS or label.startswith('MedialWall_'):
            continue
        label_dict[label].append(float(value))

    if metric == 'mean':
        label_avg = {k: (sum(v) / len(v) if len(v) > 0 else 0.0) for k, v in label_dict.items()}
        # 仅使用有效皮层顶点计算全局平均值
        total_avg = float(np.mean(valid_metrics)) if len(valid_metrics) > 0 else 0.0
        return total_avg, label_avg
    elif metric == 'sum':
        label_sum = {k: float(sum(v)) for k, v in label_dict.items()}
        # 仅使用有效皮层顶点计算全局总和
        total_sum = float(np.sum(valid_metrics))
        return total_sum, label_sum
    elif metric == 'count':
        label_count = {k: int(len(v)) for k, v in label_dict.items()}
        # 仅统计有效皮层顶点的数量
        total_count = int(len(valid_metrics))
        return total_count, label_count
    else:
        raise ValueError(f"Unknown metric mode: {metric}")

def _resolve_file(subject_fsavg_dir, hemi, wb_name, bids_desc):
    """
    Try Workbench-style path first, then fall back to BIDS-style naming.
    wb_name: e.g. 'L.curvature.32k_fs_LR.shape.gii'
    bids_desc: e.g. 'curvature'
    Returns file path if found, otherwise None.
    """
    wb_path = os.path.join(subject_fsavg_dir, wb_name)
    if os.path.exists(wb_path):
        return wb_path
    # Fallback: scan for BIDS-style file: *_hemi-{LR}_desc-{desc}_res-32k.{ext}
    # Workbench names are like L.{metric}.32k_fs_LR.{type}, extract type as suffix
    ext = wb_name.split('32k_fs_LR.', 1)[-1]  # 'shape.gii' or 'surf.gii'
    bids_pattern = f"*_hemi-{hemi}_desc-{bids_desc}_res-32k.{ext}"
    matches = glob.glob(os.path.join(subject_fsavg_dir, bids_pattern))
    return matches[0] if matches else None


def get_metric_files(subject_fsavg_dir, hemi):
    metric_files = {}

    curv_file = _resolve_file(subject_fsavg_dir, hemi,
                              f'{hemi}.curvature.32k_fs_LR.shape.gii', 'curvature')
    sulc_file = _resolve_file(subject_fsavg_dir, hemi,
                              f'{hemi}.sulc.32k_fs_LR.shape.gii', 'sulc')
    thick_file = _resolve_file(subject_fsavg_dir, hemi,
                               f'{hemi}.thickness.32k_fs_LR.shape.gii', 'thickness')

    white_surf = _resolve_file(subject_fsavg_dir, hemi,
                               f'{hemi}.white.32k_fs_LR.surf.gii', 'white')
    pial_surf = _resolve_file(subject_fsavg_dir, hemi,
                              f'{hemi}.pial.32k_fs_LR.surf.gii', 'pial')
    wmarea_file = _resolve_file(subject_fsavg_dir, hemi,
                                f'{hemi}.area.white.32k_fs_LR.shape.gii', 'area.white')
    gmarea_file = _resolve_file(subject_fsavg_dir, hemi,
                                f'{hemi}.area.pial.32k_fs_LR.shape.gii', 'area.pial')

    if white_surf and not os.path.exists(wmarea_file or ''):
        if wmarea_file is None:
            wmarea_file = os.path.join(subject_fsavg_dir, f'{hemi}.area.white.32k_fs_LR.shape.gii')
        print(f"Creating white matter area file: {wmarea_file}")
        os.system(f"wb_command -surface-vertex-areas {white_surf} {wmarea_file}")

    if pial_surf and not os.path.exists(gmarea_file or ''):
        if gmarea_file is None:
            gmarea_file = os.path.join(subject_fsavg_dir, f'{hemi}.area.pial.32k_fs_LR.shape.gii')
        print(f"Creating pial area file: {gmarea_file}")
        os.system(f"wb_command -surface-vertex-areas {pial_surf} {gmarea_file}")

    if curv_file:
        metric_files['curvature'] = curv_file
    else:
        print(f"Warning: Curvature file not found for hemi {hemi} in {subject_fsavg_dir}")

    if sulc_file:
        metric_files['sulc'] = sulc_file
    else:
        print(f"Warning: Sulc file not found for hemi {hemi} in {subject_fsavg_dir}")

    if thick_file:
        metric_files['thickness'] = thick_file
    else:
        print(f"Warning: Thickness file not found for hemi {hemi} in {subject_fsavg_dir}")

    if wmarea_file and os.path.exists(wmarea_file):
        metric_files['wmarea'] = wmarea_file
    else:
        print(f"Warning: White matter area file not found for hemi {hemi} in {subject_fsavg_dir}")

    if gmarea_file and os.path.exists(gmarea_file):
        metric_files['gmarea'] = gmarea_file
    else:
        print(f"Warning: Pial area file not found for hemi {hemi} in {subject_fsavg_dir}")

    return metric_files, white_surf, pial_surf

def _find_fsavg32k_dir(subjects_dir: str, subject_name: str, session: str = None) -> str | None:
    if session:
        cand = os.path.join(subjects_dir, subject_name, session, "Resample", "Original", "fsaverage_LR32k")
        if os.path.exists(cand):
            return cand

    cand = os.path.join(subjects_dir, subject_name, "Resample", "Original", "fsaverage_LR32k")
    if os.path.exists(cand):
        return cand

    possible_dirs = []
    if session:
        possible_dirs.extend([
            os.path.join(subjects_dir, subject_name, session, "Resample", "Original", "fsaverage_LR32k"),
            os.path.join(subjects_dir, subject_name, session, "fsaverage_LR32k"),
        ])
    possible_dirs.extend([
        os.path.join(subjects_dir, "Resample", "Original", "fsaverage_LR32k"),
        os.path.join(subjects_dir, subject_name, "fsaverage_LR32k"),
        os.path.join(subjects_dir, "fsaverage_LR32k"),
        os.path.join(subjects_dir, "32k"),
    ])
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            return dir_path
    return None

def process_subject_with_atlas(subject_dir, subject_name, atlas_file, session=None):
    # -------- helpers: robustly extract coords/faces & prism volume --------
    def _extract_coords_faces(gii: nib.gifti.GiftiImage):
        coords = None
        faces = None
        for da in gii.darrays:
            arr = np.asarray(da.data)
            if arr.ndim == 2 and arr.shape[1] == 3:
                if np.issubdtype(arr.dtype, np.floating):
                    if coords is None:
                        coords = arr.astype(np.float64, copy=False)
                elif np.issubdtype(arr.dtype, np.integer):
                    if faces is None:
                        faces = arr.astype(np.int64, copy=False)
        if coords is None or faces is None:
            raise ValueError("Failed to parse coords/faces from surf.gii (expected Nx3 float coords and Fx3 int faces).")
        return coords, faces

    def _tet_vol(a, b, c, d):
        u = b - a
        v = c - a
        w = d - a
        return np.abs(np.einsum("ij,ij->i", np.cross(u, v), w)) / 6.0

    def _face_prism_volumes(white_xyz, pial_xyz, faces):
        f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
        W0, W1, W2 = white_xyz[f0], white_xyz[f1], white_xyz[f2]
        P0, P1, P2 = pial_xyz[f0],  pial_xyz[f1],  pial_xyz[f2]

        v1 = _tet_vol(W0, W1, W2, P0)
        v2 = _tet_vol(P0, P1, P2, W1)
        v3 = _tet_vol(P0, P2, W2, W1)
        return v1 + v2 + v3

    def _vertex_volumes_from_prisms(white_surf_path, pial_surf_path):
        w_gii = nib.load(white_surf_path)
        p_gii = nib.load(pial_surf_path)

        white_xyz, faces = _extract_coords_faces(w_gii)
        pial_xyz, faces_p = _extract_coords_faces(p_gii)

        if faces_p.shape != faces.shape or not np.array_equal(faces_p, faces):
            raise ValueError("white/pial faces are not identical; cannot safely compute prism volumes.")

        if white_xyz.shape != pial_xyz.shape:
            raise ValueError("white/pial vertex count mismatch.")

        face_vol = _face_prism_volumes(white_xyz, pial_xyz, faces)
        total_vol = float(np.sum(face_vol))

        n_vert = white_xyz.shape[0]
        vtx_vol = np.zeros(n_vert, dtype=np.float64)

        share = face_vol / 3.0
        np.add.at(vtx_vol, faces[:, 0], share)
        np.add.at(vtx_vol, faces[:, 1], share)
        np.add.at(vtx_vol, faces[:, 2], share)

        vtx_vol = np.nan_to_num(vtx_vol, nan=0.0, posinf=0.0, neginf=0.0)
        return vtx_vol, total_vol

    # -------- atlas name / hemi detection --------
    atlas_basename = os.path.basename(atlas_file)
    atlas_name = atlas_basename
    for suf in [".label.gii", ".func.gii", ".shape.gii", ".surf.gii", ".gii"]:
        if atlas_name.endswith(suf):
            atlas_name = atlas_name[: -len(suf)]
            break

    if atlas_basename.startswith('L.'):
        hemi = 'L'
    elif atlas_basename.startswith('R.'):
        hemi = 'R'
    else:
        atlas_lower = atlas_basename.lower()
        if 'left' in atlas_lower or '_l' in atlas_lower or '.l.' in atlas_lower:
            hemi = 'L'
        elif 'right' in atlas_lower or '_r' in atlas_lower or '.r.' in atlas_lower:
            hemi = 'R'
        else:
            print(f"Warning: Could not determine hemisphere from atlas filename: {atlas_basename}")
            print("Assuming left hemisphere (L)")
            hemi = 'L'

    print(f"Processing subject: {subject_name}")
    if session:
        print(f"Session: {session}")
    print(f"Using atlas: {atlas_basename}")
    print(f"Detected hemisphere: {hemi}")

    subject_fsavg_dir = _find_fsavg32k_dir(subject_dir, subject_name, session)
    if subject_fsavg_dir is None:
        print(f"Error: Could not find fsaverage_LR32k directory for subject {subject_name}")
        if session:
            print(f"  (session: {session})")
        return None

    print(f"Using metrics directory: {subject_fsavg_dir}")

    metric_files = get_metric_files(subject_fsavg_dir, hemi)[0]
    if not metric_files:
        print(f"Error: No metric files found for subject {subject_name}")
        return None

    try:
        label_gii = nib.load(atlas_file)
        label_array = label_gii.darrays[0].data
        label_names = parse_label_names(label_array, label_gii.labeltable)
    except Exception as e:
        print(f"Error loading atlas file: {e}")
        return None

    # 统计并打印有效/无效顶点信息
    MEDIAL_WALL_LABELS = {'Unknown', '???', 'MEDIAL.WALL'}
    total_vertices = len(label_names)
    invalid_vertices = sum(1 for lab in label_names if lab in MEDIAL_WALL_LABELS or lab.startswith('MedialWall_'))
    valid_vertices = total_vertices - invalid_vertices
    invalid_ratio = invalid_vertices / total_vertices if total_vertices > 0 else 0.0
    print(f"\n[Vertex Statistics]")
    print(f"  Total vertices: {total_vertices}")
    print(f"  Valid cortex vertices: {valid_vertices}")
    print(f"  Invalid (midwall/non-cortex) vertices: {invalid_vertices}")
    print(f"  Invalid ratio: {invalid_ratio:.2%}\n")

    results = {
        'subject': subject_name,
        'atlas': atlas_name,
        'hemisphere': hemi,
        'total': {
            'invalid_vertices': invalid_vertices,
            'total_vertices': total_vertices,
            'invalid_ratio': invalid_ratio
        },
        'labels': defaultdict(dict)
    }

    for metric_name, metric_file in metric_files.items():
        try:
            metric_data = nib.load(metric_file).darrays[0].data
            if len(metric_data) != len(label_array):
                print(f"Warning: Dimension mismatch for {metric_name}")
                print(f"  Atlas has {len(label_array)} vertices, {metric_name} has {len(metric_data)} vertices")
                continue

            if metric_name in ['wmarea', 'gmarea']:
                metric_total, metric_labels = compute_metrics(label_names, metric_data, 'sum')
            else:
                metric_total, metric_labels = compute_metrics(label_names, metric_data, 'mean')

            results['total'][metric_name] = metric_total
            for label, value in metric_labels.items():
                results['labels'][label][metric_name] = value

        except Exception as e:
            print(f"Error processing {metric_name}: {e}")

    white_surf = _resolve_file(subject_fsavg_dir, hemi,
                               f'{hemi}.white.32k_fs_LR.surf.gii', 'white')
    pial_surf  = _resolve_file(subject_fsavg_dir, hemi,
                               f'{hemi}.pial.32k_fs_LR.surf.gii', 'pial')

    if white_surf and pial_surf and os.path.exists(white_surf) and os.path.exists(pial_surf):
        try:
            vtx_vol, total_vol = _vertex_volumes_from_prisms(white_surf, pial_surf)

            if len(vtx_vol) != len(label_array):
                raise ValueError(
                    f"Vertex volume length mismatch: vtx_vol={len(vtx_vol)}, atlas={len(label_array)}"
                )

            cort_total, cort_labels = compute_metrics(label_names, vtx_vol, 'sum')
            results['total']['cortvol'] = float(cort_total)

            for label, value in cort_labels.items():
                results['labels'][label]['cortvol'] = float(value)

        except Exception as e:
            print(f"Warning: failed to compute cortical parcel volume (cortvol): {e}")
    else:
        print(f"Warning: white/pial surfaces not found; skipping cortvol.")
        print(f"  white: {white_surf}")
        print(f"  pial : {pial_surf}")

    vertex_count_total, vertex_count_labels = compute_metrics(label_names, label_array, 'count')
    results['total']['vertex_count'] = vertex_count_total
    for label, value in vertex_count_labels.items():
        results['labels'][label]['vertex_count'] = value

    return results

def _find_aseg_path(subjects_dir: str, subject_name: str, session: str = None) -> str | None:
    if session:
        cand = os.path.join(subjects_dir, subject_name, session, "Enhance", "T1w", "T1w_aseg.nii.gz")
        if os.path.exists(cand):
            return cand

    cand = os.path.join(subjects_dir, subject_name, "Enhance", "T1w", "T1w_aseg.nii.gz")
    if os.path.exists(cand):
        return cand

    pats = [
        os.path.join(subjects_dir, subject_name, "**", "T1w_aseg.nii.gz"),
        os.path.join(subjects_dir, subject_name, "**", "*aseg*.nii.gz"),
    ]
    if session:
        pats = [
            os.path.join(subjects_dir, subject_name, session, "**", "T1w_aseg.nii.gz"),
            os.path.join(subjects_dir, subject_name, session, "**", "*aseg*.nii.gz"),
        ] + pats
    for p in pats:
        hits = glob.glob(p, recursive=True)
        if hits:
            hits_sorted = sorted(hits, key=lambda x: (os.path.basename(x) != "T1w_aseg.nii.gz", len(x)))
            return hits_sorted[0]
    return None

def compute_aseg_label_volumes_mm3(aseg_path: str) -> tuple[dict[int, float], float]:
    img = nib.load(aseg_path)
    zooms = img.header.get_zooms()[:3]
    voxel_vol = float(abs(zooms[0] * zooms[1] * zooms[2]))
    data = img.get_fdata(dtype=np.float32)
    seg = np.rint(data).astype(np.int32)

    labels, counts = np.unique(seg, return_counts=True)
    vol = {}
    for lab, cnt in zip(labels.tolist(), counts.tolist()):
        lab = int(lab)
        if lab == 0:
            continue
        vol[lab] = float(cnt) * voxel_vol
    return vol, voxel_vol

def collapse_label_volumes(vol_by_id: dict[int, float], collapse_map: dict[int, int]) -> dict[int, float]:
    out = defaultdict(float)
    for lab, v in vol_by_id.items():
        tgt = collapse_map.get(lab, lab)
        out[int(tgt)] += float(v)
    return dict(out)

def aseg_volumes_to_feature_row(
    vol_raw_by_id: dict[int, float],
    collapse_map: dict[int, int],
    export_uncollapsed_lr: bool = True,
    export_collapsed_total: bool = True,
    export_all_labels_if_unknown: bool = False,
) -> dict:
    row = {}

    if export_uncollapsed_lr:
        for lab, v in vol_raw_by_id.items():
            name = ASEG_LABEL_NAME.get(lab, None)
            if name is None and not export_all_labels_if_unknown:
                continue
            if name is None:
                name = f"Label{lab}"

            if name.startswith("Left-"):
                base = strip_lr_prefix(name)
                col = f"subvol_L_{sanitize_name(base)}"
            elif name.startswith("Right-"):
                base = strip_lr_prefix(name)
                col = f"subvol_R_{sanitize_name(base)}"
            else:
                col = f"subvol_{sanitize_name(name)}"

            row[col] = float(v)

    if export_collapsed_total:
        vol_collapsed = collapse_label_volumes(vol_raw_by_id, collapse_map)
        for lab, v in vol_collapsed.items():
            name = ASEG_LABEL_NAME.get(lab, None)
            if name is None and not export_all_labels_if_unknown:
                continue
            if name is None:
                name = f"Label{lab}"

            base = strip_lr_prefix(name)
            col = f"subvol_T_{sanitize_name(base)}"
            row[col] = float(v)

    return row

def results_to_feature_row(
    results: dict,
    meta: dict,
    export_area: bool = False,
    export_vertex_count: bool = False,
    export_cortical_volume: bool = True,
) -> dict:
    subject = results["subject"]
    hemi = results["hemisphere"]
    atlas = results["atlas"]

    row = {
        "subject_id": subject,
        "age": meta.get("age", None),
        "sex": meta.get("sex", None),
        "site": meta.get("site", None),
        "breed": meta.get("breed", None),
        "weight (kg)": meta.get("weight (kg)", None),
    }

    atlas_key = sanitize_name(os.path.basename(atlas))
    if atlas_key.startswith("L_"):
        atlas_key = atlas_key.split("L_")[-1]
    if  atlas_key.startswith("R_"):
        atlas_key = atlas_key.split("R_")[-1]
    hemi_key = "L" if hemi.upper().startswith("L") else "R"

    for roi, mdict in results["labels"].items():
        roi_key = sanitize_name(roi)
        if roi_key == "": continue

        if "thickness" in mdict:
            row[f"thick_{hemi_key}_{atlas_key}_{roi_key}"] = mdict["thickness"]
        if "curvature" in mdict:
            row[f"curv_{hemi_key}_{atlas_key}_{roi_key}"] = mdict["curvature"]
        if "sulc" in mdict:
            row[f"sulc_{hemi_key}_{atlas_key}_{roi_key}"] = mdict["sulc"]

        if export_cortical_volume and "cortvol" in mdict:
            row[f"cvol_{hemi_key}_{atlas_key}_{roi_key}"] = mdict["cortvol"]

        if export_area:
            if "wmarea" in mdict:
                row[f"wmarea_{hemi_key}_{atlas_key}_{roi_key}"] = mdict["wmarea"]
            if "gmarea" in mdict:
                row[f"gmarea_{hemi_key}_{atlas_key}_{roi_key}"] = mdict["gmarea"]

        if export_vertex_count and "vertex_count" in mdict:
            row[f"vcount_{hemi_key}_{atlas_key}_{roi_key}"] = mdict["vertex_count"]

    return row

def load_meta_csv(meta_csv: str) -> dict:
    mdf = pd.read_csv(meta_csv)
    required = {"participant_id", "age", "sex", "site", "breed", "weight (kg)"}
    missing = required - set(mdf.columns)
    if missing:
        print(f"Warning: meta_csv missing columns: {missing}. Filling with empty values.")
        for col in missing:
            mdf[col] = None

    if "participant_id" not in mdf.columns:
        raise ValueError("meta_csv must contain at least 'participant_id' column")

    has_session = "session_id" in mdf.columns

    def _safe_float(val):
        """Try to convert value to float, return None on failure."""
        if pd.isna(val):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    meta = {}
    for _, r in mdf.iterrows():
        sid = str(r["participant_id"]).strip()
        session = str(r["session_id"]).strip() if (has_session and pd.notna(r["session_id"])) else None

        meta_dict = {
            "age": _safe_float(r["age"]),
            "sex": str(r["sex"]) if pd.notna(r["sex"]) else None,
            "site": str(r["site"]) if pd.notna(r["site"]) else None,
            "breed": str(r["breed"]) if pd.notna(r["breed"]) else None,
            "weight (kg)": str(r["weight (kg)"]) if pd.notna(r["weight (kg)"]) else None,
        }

        # 始终存储仅 participant_id 作为 key（当没有 session 或为了兼容回退查找）
        if not has_session or session is None:
            meta[sid] = meta_dict

        if has_session and session:
            # 使用 (participant_id, session_id) 作为 key
            key = (sid, session)
            meta[key] = meta_dict
            # 同时也用 (participant_id, None) 作为 key 存储，兼容回退查找
            if sid not in meta:
                meta[sid] = meta_dict

    return meta

def _atlas_key_from_results(results: dict) -> str:
    atlas_key = sanitize_name(str(results.get("atlas", "atlas")))
    if atlas_key.startswith("L_"):
        atlas_key = atlas_key.split("L_", 1)[-1]
    if atlas_key.startswith("R_"):
        atlas_key = atlas_key.split("R_", 1)[-1]
    return atlas_key or "atlas"

def _hemi_key_from_results(results: dict) -> str:
    hemi = str(results.get("hemisphere", "U"))
    return "L" if hemi.upper().startswith("L") else ("R" if hemi.upper().startswith("R") else hemi.upper())

def results_to_rows_by_type(
    results: dict,
    meta: dict,
    session: str = None,
    export_area: bool = False,
    export_vertex_count: bool = False,
    export_cortical_volume: bool = True,
) -> dict[str, dict]:
    subject = results["subject"]
    atlas_key = _atlas_key_from_results(results)
    hemi_key = _hemi_key_from_results(results)
    total = results.get("total", {})

    # 构造包含 session 的唯一标识符
    participant_id = subject  # 原 subject 是子文件夹名（如 sub-001），我们将在外层添加 session
    if session:
        participant_id = f"{subject}"  # 例如 sub-001_ses-20210620

    base_meta = {
        "subject_id": participant_id,
        "session_id": session,
        "age": meta.get("age", None),
        "sex": meta.get("sex", None),
        "site": meta.get("site", None),
        "breed": meta.get("breed", None),
        "weight (kg)": meta.get("weight (kg)", None),
    }

    out = {}

    # 无效区域统计信息（所有metric共享）
    invalid_vertices = total.get("invalid_vertices", None)
    total_vertices = total.get("total_vertices", None)
    invalid_ratio = total.get("invalid_ratio", None)

    out["thickness"] = dict(base_meta)
    out["thickness"]["global_mean_thickness"] = total.get("thickness", None)
    out["thickness"]["global_invalid_vertices"] = invalid_vertices
    out["thickness"]["global_total_vertices"] = total_vertices
    out["thickness"]["global_invalid_ratio"] = invalid_ratio

    out["curvature"] = dict(base_meta)
    out["curvature"]["global_mean_curvature"] = total.get("curvature", None)
    out["curvature"]["global_invalid_vertices"] = invalid_vertices
    out["curvature"]["global_total_vertices"] = total_vertices
    out["curvature"]["global_invalid_ratio"] = invalid_ratio

    out["sulc"] = dict(base_meta)
    out["sulc"]["global_mean_sulc"] = total.get("sulc", None)
    out["sulc"]["global_invalid_vertices"] = invalid_vertices
    out["sulc"]["global_total_vertices"] = total_vertices
    out["sulc"]["global_invalid_ratio"] = invalid_ratio

    if export_area:
        out["area"] = dict(base_meta)
        out["area"]["global_total_wmarea"] = total.get("wmarea", None)
        out["area"]["global_total_gmarea"] = total.get("gmarea", None)

    if export_cortical_volume:
        out["cortvol"] = dict(base_meta)
        out["cortvol"]["global_total_cortvol"] = total.get("cortvol", None)

    if export_vertex_count:
        out["vertex_count"] = dict(base_meta)
        out["vertex_count"]["global_vertex_count"] = total.get("vertex_count", None)

    for roi, mdict in results["labels"].items():
        roi_key = sanitize_name(roi)
        if not roi_key:
            continue

        if "thickness" in mdict:
            out["thickness"][f"{roi_key}"] = mdict["thickness"]

        if "curvature" in mdict:
            out["curvature"][f"{roi_key}"] = mdict["curvature"]

        if "sulc" in mdict:
            out["sulc"][f"{roi_key}"] = mdict["sulc"]

        if export_area:
            if "wmarea" in mdict:
                out["area"][f"{roi_key}_wmarea"] = mdict["wmarea"]
            if "gmarea" in mdict:
                out["area"][f"{roi_key}_gmarea"] = mdict["gmarea"]

        if export_cortical_volume and "cortvol" in mdict:
            out["cortvol"][f"{roi_key}"] = mdict["cortvol"]

        if export_vertex_count and "vertex_count" in mdict:
            out["vertex_count"][f"{roi_key}"] = mdict["vertex_count"]

    return out

def build_subcortical_row(
    vol_raw_by_id: dict[int, float],
    voxel_vol: float,
    meta: dict,
    subject_id: str,
    session: str = None,
    hemi: str = "LR",
    export_uncollapsed_lr: bool = True,
    export_collapsed_total: bool = True,
    export_all_labels_if_unknown: bool = False,
) -> dict:
    row = {
        "subject_id": subject_id,
        "session_id": session,
        "age": meta.get("age", None),
        "sex": meta.get("sex", None),
        "site": meta.get("site", None),
        "breed": meta.get("breed", None),
        "weight (kg)": meta.get("weight (kg)", None),
    }

    icv = sum(vol_raw_by_id.values())
    row["global_estimated_ICV_mm3"] = icv

    subcortical_labels = [
        10, 11, 12, 13,
        17, 18,
        26, 27, 28,
        49, 50, 51, 52,
        53, 54,
        58, 59, 60,
    ]
    subcortical_vol = sum(vol_raw_by_id.get(lab, 0) for lab in subcortical_labels)
    row["global_total_subcortical_vol_mm3"] = subcortical_vol

    sub_row = aseg_volumes_to_feature_row(
        vol_raw_by_id=vol_raw_by_id,
        collapse_map=HEMI_COLLAPSE_MAP,
        export_uncollapsed_lr=export_uncollapsed_lr,
        export_collapsed_total=export_collapsed_total,
        export_all_labels_if_unknown=export_all_labels_if_unknown,
    )
    row.update(sub_row)

    return row

def _save_merged_rows(rows: list[dict], out_csv: str, is_subcort: bool = False):
    """
    保存合并的行到CSV文件，基于 (subject_id, session_id) 去重并更新。
    若已存在相同键的行，则用新行覆盖；否则添加新行。
    """
    if not rows:
        return

    df_new = pd.DataFrame(rows)

    # 确保必要的列存在
    required_cols = ["subject_id", "session_id"]
    for col in required_cols:
        if col not in df_new.columns:
            df_new[col] = None

    front_cols = ["subject_id", "session_id", "age", "sex", "site", "breed", "weight (kg)"]
    front_cols = [c for c in front_cols if c in df_new.columns]

    # 如果文件已存在，读取并合并
    if os.path.exists(out_csv):
        print(f"  Merging with existing file: {out_csv}")
        df_existing = pd.read_csv(out_csv)

        # 确保现有文件也有键列
        for col in required_cols:
            if col not in df_existing.columns:
                df_existing[col] = None

        # 构建字典：键为 (subject_id, session_id) 元组，值为行数据（Series）
        # 注意处理 NaN 和 None：转换为字符串，空值处理为 ""
        existing_dict = {}
        for idx, row in df_existing.iterrows():
            subj = str(row.get("subject_id", "")) if pd.notna(row.get("subject_id")) else ""
            sess = str(row.get("session_id", "")) if pd.notna(row.get("session_id")) else ""
            key = (subj, sess)
            existing_dict[key] = row.to_dict()

        # 更新字典：用新行覆盖或添加
        for idx, row in df_new.iterrows():
            subj = str(row.get("subject_id", "")) if pd.notna(row.get("subject_id")) else ""
            sess = str(row.get("session_id", "")) if pd.notna(row.get("session_id")) else ""
            key = (subj, sess)
            existing_dict[key] = row.to_dict()  # 覆盖或添加

        # 从字典重建 DataFrame
        df_combined = pd.DataFrame.from_dict(existing_dict, orient='index')
        # 重置索引，将 (subject_id, session_id) 作为列
        df_combined.reset_index(drop=True, inplace=True)

        # 确保 subject_id 和 session_id 存在（字典中可能丢失了列？实际上行数据已包含）
        # 如果丢失，从键恢复？但行数据已包含，无需处理。

    else:
        df_combined = df_new.copy()

    # 重新排列列顺序
    global_cols = sorted([c for c in df_combined.columns if c.startswith("global_")])
    other_cols = sorted([c for c in df_combined.columns if c not in front_cols and c not in global_cols])
    all_cols_ordered = front_cols + global_cols + other_cols
    # 确保所有列都存在（缺失的补 NaN）
    for col in all_cols_ordered:
        if col not in df_combined.columns:
            df_combined[col] = None
    df_combined = df_combined[all_cols_ordered]

    # 写出文件
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_combined.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved/Updated: {out_csv} ({len(df_combined)} rows)")

def process_one_subject_to_csvs(
    subjects_dir: str,
    subject_name: str,
    session: str,
    atlas_file: str,
    out_dir: str,
    meta_map: dict,
    export_area: bool = False,
    export_vertex_count: bool = False,
    export_cortical_volume: bool = True,
    export_subcortical_volume: bool = True,
    subcortical_export_uncollapsed_lr: bool = True,
    subcortical_export_collapsed_total: bool = True,
    subcortical_export_all_unknown_labels: bool = False,
):
    # 皮层处理
    results = process_subject_with_atlas(
        subject_dir=subjects_dir,
        subject_name=subject_name,
        atlas_file=atlas_file,
        session=session,
    )
    if results is None:
        raise RuntimeError(f"Failed to process subject={subject_name} session={session} with atlas={atlas_file}")

    # 先尝试用 (subject_name, session) 作为 key 获取 meta
    meta = {}
    subject_name_clean = str(subject_name).strip() if subject_name else subject_name
    session_clean = str(session).strip() if session else session

    if session_clean:
        # 先尝试精确匹配
        key = (subject_name_clean, session_clean)
        meta = meta_map.get(key, {})
        # 如果没找到，尝试查找是否有类似的key（可能是空格问题）
        if not meta:
            # 遍历所有key来寻找匹配项
            for k in meta_map.keys():
                if isinstance(k, tuple) and len(k) == 2:
                    if str(k[0]).strip() == subject_name_clean and str(k[1]).strip() == session_clean:
                        meta = meta_map[k]
                        print(f"Info: Found meta using loose matching for key {k}")
                        break
    # 如果没找到，回退到只用 subject_name 作为 key
    if not meta:
        meta = meta_map.get(subject_name_clean, {})
        if not meta:
            # 尝试在所有key中查找
            for k in meta_map.keys():
                if isinstance(k, str) and str(k).strip() == subject_name_clean:
                    meta = meta_map[k]
                    print(f"Info: Found meta using loose matching for subject {k}")
                    break
    if not meta:
        if session:
            print(f"Warning: no meta found for subject={subject_name}, session={session} in meta_csv; meta fields will be empty.")
            # 打印前5个可用的key供调试
            sample_keys = list(meta_map.keys())[:5]
            print(f"  Sample keys in meta_map: {sample_keys}")
        else:
            print(f"Warning: no meta found for subject={subject_name} in meta_csv; meta fields will be empty.")

    atlas_key = _atlas_key_from_results(results)
    hemi_key = _hemi_key_from_results(results)

    rows_by_type = results_to_rows_by_type(
        results=results,
        meta=meta,
        session=session,
        export_area=export_area,
        export_vertex_count=export_vertex_count,
        export_cortical_volume=export_cortical_volume,
    )

    # 保存皮层各类型 CSV（使用 _save_merged_rows 追加合并）
    for metric_type, row in rows_by_type.items():
        out_csv = os.path.join(out_dir, "cort", atlas_key, hemi_key, f"{metric_type}.csv")
        _save_merged_rows([row], out_csv, is_subcort=False)

    # 皮下处理
    if export_subcortical_volume:
        aseg_path = _find_aseg_path(subjects_dir, subject_name, session)
        if aseg_path is None:
            print("Warning: subcortical volume requested but aseg file not found; skipping subcortical csv.")
        else:
            try:
                vol_raw_by_id, voxel_vol = compute_aseg_label_volumes_mm3(aseg_path)

                participant_id = f"{subject_name}" if session else subject_name

                # 左半球皮下（包含左侧结构和全局ICV）
                sub_row_L = build_subcortical_row(
                    vol_raw_by_id=vol_raw_by_id,
                    voxel_vol=voxel_vol,
                    meta=meta,
                    subject_id=participant_id,
                    session=session,
                    hemi="L",
                    export_uncollapsed_lr=subcortical_export_uncollapsed_lr,
                    export_collapsed_total=subcortical_export_collapsed_total,
                    export_all_labels_if_unknown=subcortical_export_all_unknown_labels,
                )
                # 过滤只保留左侧结构
                sub_row_L_filtered = {k: v for k, v in sub_row_L.items()
                                      if not k.startswith("subvol_R_")}
                out_csv_L = os.path.join(out_dir, "subcort", "aseg", "L", "volume.csv")
                _save_merged_rows([sub_row_L_filtered], out_csv_L, is_subcort=True)

                # 右半球皮下
                sub_row_R = build_subcortical_row(
                    vol_raw_by_id=vol_raw_by_id,
                    voxel_vol=voxel_vol,
                    meta=meta,
                    subject_id=participant_id,
                    session=session,
                    hemi="R",
                    export_uncollapsed_lr=subcortical_export_uncollapsed_lr,
                    export_collapsed_total=subcortical_export_collapsed_total,
                    export_all_labels_if_unknown=subcortical_export_all_unknown_labels,
                )
                sub_row_R_filtered = {k: v for k, v in sub_row_R.items()
                                      if not k.startswith("subvol_L_")}
                out_csv_R = os.path.join(out_dir, "subcort", "aseg", "R", "volume.csv")
                _save_merged_rows([sub_row_R_filtered], out_csv_R, is_subcort=True)

            except Exception as e:
                print(f"Warning: failed to compute subcortical volumes for {subject_name}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="按指标类型分别导出 CSV（单被试单session）：thickness/curvature/sulc/area/cortvol + subcortical(aseg) + global",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--atlas", type=str, required=True,
                        help="32k图谱文件路径 (.label.gii)，例如 L.map_0.label.gii")
    parser.add_argument("--subjects_dir", type=str, required=True,
                        help="被试根目录，包含各个被试的文件夹")
    parser.add_argument("--subject", type=str, required=True,
                        help="只处理该被试（文件夹名）")
    parser.add_argument("--session", type=str, required=True,
                        help="只处理该被试的一个session（文件夹名，如 ses-20210620T152859）")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="输出目录（会在其中写多个 CSV）")
    parser.add_argument("--meta_csv", type=str, required=True,
                        help="包含 participant_id,age,sex,site,breed,weight (kg) 的CSV（participant_id 为 subject 名，不含 session）")

    parser.add_argument("--export_area", action="store_true", help="also export wmarea/gmarea per ROI")
    parser.add_argument("--export_vertex_count", action="store_true", help="also export vertex_count per ROI")

    parser.add_argument("--no_cortical_volume", action="store_true",
                        help="disable cortical ROI volume (cortvol) export")
    parser.add_argument("--no_subcortical_volume", action="store_true",
                        help="disable aseg-based subcortical volume export")

    parser.add_argument("--subvol_no_lr", action="store_true",
                        help="do not export raw L/R subcortical volumes")
    parser.add_argument("--subvol_no_collapsed", action="store_true",
                        help="do not export collapsed (L+R) subcortical volumes")
    parser.add_argument("--subvol_export_all_labels", action="store_true",
                        help="export unknown aseg labels as Label<ID> too (may create many columns)")

    args = parser.parse_args()

    if not os.path.exists(args.atlas):
        print(f"Error: Atlas file not found: {args.atlas}")
        sys.exit(1)
    if not os.path.exists(args.subjects_dir):
        print(f"Error: Subjects directory not found: {args.subjects_dir}")
        sys.exit(1)
    if not os.path.exists(args.meta_csv):
        print(f"Error: meta_csv not found: {args.meta_csv}")
        sys.exit(1)

    subj_path = os.path.join(args.subjects_dir, args.subject)
    if not os.path.isdir(subj_path):
        print(f"Error: subject directory not found: {subj_path}")
        sys.exit(1)

    if args.session:
        session_path = os.path.join(subj_path, args.session)
        if not os.path.isdir(session_path):
            print(f"Error: session directory not found: {session_path}")
            sys.exit(1)

    meta_map = load_meta_csv(args.meta_csv)

    export_cortical_volume = (not args.no_cortical_volume)
    export_subcortical_volume = (not args.no_subcortical_volume)

    process_one_subject_to_csvs(
        subjects_dir=args.subjects_dir,
        subject_name=args.subject,
        session=args.session,
        atlas_file=args.atlas,
        out_dir=args.out_dir,
        meta_map=meta_map,
        export_area=args.export_area,
        export_vertex_count=args.export_vertex_count,
        export_cortical_volume=export_cortical_volume,
        export_subcortical_volume=export_subcortical_volume,
        subcortical_export_uncollapsed_lr=(not args.subvol_no_lr),
        subcortical_export_collapsed_total=(not args.subvol_no_collapsed),
        subcortical_export_all_unknown_labels=args.subvol_export_all_labels,
    )

    print("\nProcessing completed!")

if __name__ == "__main__":
    main()