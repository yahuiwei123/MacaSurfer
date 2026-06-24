#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import yaml
import argparse
import csv
import fcntl
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# BIDS helpers
# -----------------------------
def _strip_bids_ext(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".tsv.json"):
        return name[:-9]
    return str(Path(name).with_suffix("")).rstrip(".")


def get_bids_suffix(filename: str) -> Optional[str]:
    base = _strip_bids_ext(Path(filename).name)
    if "_" not in base:
        return None
    suffix = base.split("_")[-1]
    return suffix or None


def _has_token(base: str, token: str, ignore_case: bool = True) -> bool:
    flags = re.IGNORECASE if ignore_case else 0
    return re.search(rf"(^|_){re.escape(token)}(_|$)", base, flags=flags) is not None


def get_modality(filename: str) -> str:
    """
    Rules:
      1) If FLAIR token appears anywhere -> FLAIR (even if suffix is T2w)
      2) Else use BIDS suffix: T1w/T2w/FLAIR
      3) Else conservative fallback token match
    """
    base = _strip_bids_ext(Path(filename).name)

    if _has_token(base, "FLAIR", ignore_case=True):
        return "FLAIR"

    suffix = get_bids_suffix(filename)
    if suffix == "T1w":
        return "T1"
    if suffix == "T2w":
        return "T2"
    if suffix and suffix.upper() == "FLAIR":
        return "FLAIR"

    if _has_token(base, "T1w", ignore_case=False):
        return "T1"
    if _has_token(base, "T2w", ignore_case=False):
        return "T2"

    return "unknown"


_SUB_RE = re.compile(r"(^|/)(sub-[^/]+)(/|$)")
_SES_RE = re.compile(r"(^|/)(ses-[^/]+)(/|$)")


def _infer_sub_ses_from_path_and_name(path: Path) -> Tuple[str, str]:
    """
    Infer sub-XXX and ses-XXX using:
      1) directory tokens /sub-xxx/ and /ses-xxx/
      2) filename tokens _sub-xxx_ and _ses-xxx_
      3) fallback: ses-NA if missing
    """
    p = path.as_posix()

    subj = None
    ses = None

    m = _SUB_RE.search(p)
    if m:
        subj = m.group(2)

    m = _SES_RE.search(p)
    if m:
        ses = m.group(2)

    # fallback to filename
    base = _strip_bids_ext(path.name)
    if subj is None:
        m2 = re.search(r"(^|_)sub-([A-Za-z0-9]+)($|_)", base)
        if m2:
            subj = f"sub-{m2.group(2)}"
    if ses is None:
        m3 = re.search(r"(^|_)ses-([A-Za-z0-9]+)($|_)", base)
        if m3:
            ses = f"ses-{m3.group(2)}"

    if subj is None:
        subj = "sub-UNKNOWN"
    if ses is None:
        ses = "ses-NA"

    return subj, ses


def _normalize_bids_input(data_dir: str) -> List[Path]:
    """
    Accept:
      - BIDS root (contains sub-*/ or dataset_description.json)
      - subject dir (sub-xxx)
      - session dir (ses-xxx)
    Return a list of scan roots.
    """
    p = Path(data_dir).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"data_dir not found: {p}")

    # If user points to a session dir directly: scan that session dir
    if p.is_dir() and p.name.startswith("ses-"):
        return [p]

    # If user points to a subject dir directly: scan that subject dir
    if p.is_dir() and p.name.startswith("sub-"):
        return [p]

    # Otherwise treat as a container dir (BIDS root, site-*, etc.)
    # Prefer scanning sub-* dirs if present, else scan the directory itself.
    sub_dirs = sorted([x for x in p.iterdir() if x.is_dir() and x.name.startswith("sub-")])
    if sub_dirs:
        return sub_dirs

    return [p]


# -----------------------------
# Filters
# -----------------------------
def resolve_filters(
    data_dir: str,
    cli_subj: Optional[str] = None,
    cli_ses: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Determine (filter_subj, filter_ses).
    Priority: CLI args > inferred from data_dir.
    """
    p = Path(data_dir).expanduser().resolve()

    infer_subj = None
    infer_ses = None

    if p.is_dir() and p.name.startswith("ses-"):
        infer_ses = p.name
        if p.parent.is_dir() and p.parent.name.startswith("sub-"):
            infer_subj = p.parent.name
    elif p.is_dir() and p.name.startswith("sub-"):
        infer_subj = p.name

    filter_subj = cli_subj or infer_subj
    filter_ses = cli_ses or infer_ses

    if filter_subj is not None and str(filter_subj).strip() == "":
        filter_subj = None
    if filter_ses is not None and str(filter_ses).strip() == "":
        filter_ses = None

    return filter_subj, filter_ses


# -----------------------------
# Per-session YAML generation
# -----------------------------
def _empty_session_config() -> Dict[str, Any]:
    return {"T1": {}, "T2": {}, "FLAIR": {}}


def _sorted_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: d[k] for k in sorted(d.keys())}


def generate_session_yamls(
    data_dir: str,
    qc_dir: str,
    filter_subj: Optional[str] = None,
    filter_ses: Optional[str] = None,
    generate_files: bool = True
) -> Tuple[List[Path], Dict[Tuple[str, str], Dict]]:
    """
    Scan BIDS directory and generate per-session YAML under:

      qc_dir/sub-xxx/ses-yyy/orig_config.yaml

    Each YAML contains ONLY:
      { sub-xxx: { ses-yyy: { T1: {...}, T2: {...}, FLAIR: {...} } } }

    If filter_subj / filter_ses provided, only generate matched sessions.

    Returns: (list of generated YAML paths, session config dictionary)
    """
    scan_roots = _normalize_bids_input(data_dir)
    qc_root = Path(qc_dir).expanduser().resolve()
    qc_root.mkdir(parents=True, exist_ok=True)

    sess_cfg: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # per-session counters for is_refer
    t1_seen: Dict[Tuple[str, str], int] = {}
    flair_seen: Dict[Tuple[str, str], int] = {}
    t2_seen: Dict[Tuple[str, str], int] = {}

    def add_run(run_path: Path) -> None:
        subj, ses = _infer_sub_ses_from_path_and_name(run_path)

        # filtering
        if filter_subj and subj != filter_subj:
            return
        if filter_ses and ses != filter_ses:
            return

        modality = get_modality(str(run_path))
        if modality not in ("T1", "T2", "FLAIR"):
            return

        key = (subj, ses)
        if key not in sess_cfg:
            sess_cfg[key] = {subj: {ses: _empty_session_config()}}

        node = sess_cfg[key][subj][ses]
        run_name = _strip_bids_ext(run_path.name)

        if modality == "T1":
            cnt = t1_seen.get(key, 0)
            is_refer = (cnt == 0)
            t1_seen[key] = cnt + 1
            node["T1"][run_name] = {
                "orig": str(run_path),
                "is_brain": False,
                "good_mask": True,
                "is_refer": is_refer,
                "good_quality": True,
            }
        elif modality == "T2":
            cnt = t2_seen.get(key, 0)
            t2_seen[key] = cnt + 1
            node["T2"][run_name] = {
                "orig": str(run_path),
                "is_brain": False,
                "good_mask": False,
                "is_refer": False,
                "good_quality": True,
            }
        else:  # FLAIR
            cnt = flair_seen.get(key, 0)
            is_refer = (cnt == 0)
            flair_seen[key] = cnt + 1
            node["FLAIR"][run_name] = {
                "orig": str(run_path),
                "is_brain": False,
                "good_mask": False,
                "is_refer": is_refer,
                "good_quality": True,
            }

    # Only consider anat/*nii(.gz)
    for base_root in scan_roots:
        for root, _, files in os.walk(str(base_root)):
            if "anat" not in Path(root).parts:
                continue
            for fn in sorted([f for f in files if f.endswith(".nii.gz") or f.endswith(".nii")]):
                add_run(Path(root) / fn)

    if not sess_cfg:
        print("⚠️ No runs found (after filtering). Checked scan roots:")
        for r in scan_roots:
            print(f"  - {r}")
        if filter_subj or filter_ses:
            print(f"  filters: subj={filter_subj or 'ANY'} ses={filter_ses or 'ANY'}")
        return [], sess_cfg

    out_paths: List[Path] = []
    if generate_files:
        for (subj, ses), cfg in sorted(sess_cfg.items(), key=lambda x: (x[0][0], x[0][1])):
            cfg = _sorted_dict(cfg)
            cfg[subj] = _sorted_dict(cfg[subj])
            for mod in ("T1", "T2", "FLAIR"):
                cfg[subj][ses][mod] = _sorted_dict(cfg[subj][ses].get(mod, {}))

            out_dir = qc_root / subj / ses
            out_dir.mkdir(parents=True, exist_ok=True)

            out_yaml = out_dir / "orig_config.yaml"
            with open(out_yaml, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

            out_paths.append(out_yaml)

        print(f"✅ Generated {len(out_paths)} session YAML(s) under: {qc_root}")
    else:
        print(f"ℹ️ Skipping YAML generation (meta-only mode)")
    return out_paths, sess_cfg


# -----------------------------
# QC plotting (per-session YAML -> one JPG)
# -----------------------------
def calculate_image_centroid(image: np.ndarray) -> Tuple[int, int, int]:
    total = image.sum()
    if total == 0:
        return (image.shape[0] // 2, image.shape[1] // 2, image.shape[2] // 2)
    indices = np.indices(image.shape)
    centroid = [np.sum(indices[dim] * image) / total for dim in range(image.ndim)]
    return tuple(int(round(c)) for c in centroid)


def plot_one_session_from_yaml(yaml_file: str, output_fig_path: str) -> None:
    with open(yaml_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    run_list: List[Tuple[str, str]] = []
    for subj, ses_dict in config.items():
        for ses, mod_dict in (ses_dict or {}).items():
            for modality in ("T1", "T2", "FLAIR"):
                runs = (mod_dict or {}).get(modality, {}) or {}
                for run_name in sorted(runs.keys()):
                    run_info = runs[run_name]
                    label = f"{subj}/{ses}/{modality}/{run_name}"
                    run_list.append((label, run_info.get("orig", "")))

    if not run_list:
        print(f"⚠️ No runs found in {yaml_file}, skip plotting.")
        return

    cols = 3
    rows = len(run_list)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    def plot_slice(ax, image, title):
        ax.imshow(image.T, cmap="gray", origin="lower")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    for i, (label, image_path) in enumerate(run_list):
        try:
            if not image_path:
                raise ValueError("Missing 'orig' path in YAML")

            mri = nib.load(image_path)
            mri = nib.as_closest_canonical(mri)
            data = mri.get_fdata()

            if data.ndim != 3:
                raise ValueError(f"Not 3D: shape={data.shape}")

            centroid = calculate_image_centroid(data)
            axial = data[:, :, centroid[2]]
            sagittal = data[centroid[0], :, :]
            coronal = data[:, centroid[1], :]

            plot_slice(axes[i, 0], axial, f"{label}\nAxial")
            plot_slice(axes[i, 1], sagittal, f"{label}\nSagittal")
            plot_slice(axes[i, 2], coronal, f"{label}\nCoronal")

        except Exception as e:
            print(f"Error processing {label}: {e}")
            for j in range(cols):
                axes[i, j].axis("off")
                axes[i, j].set_title(f"{label}\nERROR", fontsize=9)

    plt.tight_layout()
    Path(output_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_fig_path, dpi=150)
    plt.close()
    print(f"✅ Figure saved to {output_fig_path}")


def plot_all_session_figures(session_yaml_paths: List[Path]) -> None:
    for yml in session_yaml_paths:
        out_fig = yml.parent / "orig_check.jpg"
        plot_one_session_from_yaml(str(yml), str(out_fig))


# -----------------------------
# participants.tsv -> meta.csv processing
# -----------------------------
def find_bids_root(start_path: Path) -> Optional[Path]:
    """
    Walk up from start_path until we find a directory containing
    either 'participants.tsv' or 'dataset_description.json'.
    Returns None if not found.
    """
    current = start_path.resolve()
    while current != current.parent:
        if (current / "participants.tsv").exists() or (current / "dataset_description.json").exists():
            return current
        current = current.parent
    return None


def process_participants_tsv(bids_root: Path, qc_dir: Path, subj_to_ses: Dict[str, List[str]]) -> None:
    """
    从 bids_root/participants.tsv 读取信息，为每个实际存在的会话生成一行，
    保留 participants.tsv 中已有的会话特定信息，缺失的会话用被试通用信息填充。
    与已有 meta.csv 合并，只更新当前处理的被试，保留其他被试不变。
    强制包含必需的元数据列（site, weight (kg), age, sex, breed），
    若 site 列缺失或值为空，则用 bids_root 的文件夹名填充。

    注意：此函数使用文件锁确保多进程并发安全。
    """
    participants_tsv = bids_root / "participants.tsv"
    fieldnames: List[str] = []
    rows: List[Dict[str, str]] = []
    has_session_col = False
    session_col: Optional[str] = None
    participant_general: Dict[str, Dict[str, str]] = {}
    per_session_info: Dict[Tuple[str, str], Dict[str, str]] = {}
    use_tsv = participants_tsv.exists()

    if use_tsv:
        # 读取 participants.tsv
        try:
            with open(participants_tsv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                fieldnames = reader.fieldnames or []
                rows = list(reader)
        except Exception as e:
            print(f"⚠️ Failed to read participants.tsv: {e}")
            return

        if not rows:
            print("⚠️ participants.tsv is empty, nothing to write.")
            return

        # 查找 session_id 列（不区分大小写）
        for col in fieldnames:
            if col.lower() == "session_id":
                session_col = col
                break
        has_session_col = session_col is not None

        # 构建原始信息字典
        for row in rows:
            pid = row.get("participant_id", "").strip()
            if not pid:
                continue
            if pid not in participant_general:
                participant_general[pid] = row.copy()

            if has_session_col:
                ses_val = row.get(session_col, "").strip()
                if ses_val:
                    per_session_info[(pid, ses_val)] = row.copy()
    else:
        print(f"⚠️ participants.tsv not found in {bids_root}, "
              f"generating meta.csv from directory structure for {len(subj_to_ses)} subjects.")
        fieldnames = ["participant_id", "session_id"]

    # 构建需要写入的新行：仅针对当前处理的被试
    new_rows_by_key = {}

    for pid, sessions in subj_to_ses.items():
        if not use_tsv:
            # 无 participants.tsv，从目录结构自动生成最小行
            for ses in sessions:
                new_rows_by_key[(pid, ses)] = {
                    "participant_id": pid,
                    "session_id": ses,
                }
            continue

        general_row = participant_general.get(pid)
        if general_row is None:
            print(f"⚠️ Participant {pid} not found in participants.tsv, skipping.")
            continue

        for ses in sessions:
            key = (pid, ses)
            if has_session_col and key in per_session_info:
                row = per_session_info[key].copy()
                if session_col in row:
                    row[session_col] = ses
                else:
                    row[session_col] = ses
            else:
                row = general_row.copy()
                if has_session_col:
                    row[session_col] = ses
                else:
                    row["session_id"] = ses
            new_rows_by_key[key] = row

    # 确定输出列名：包含 participants.tsv 的所有列，并确保有 session_id
    out_fieldnames = fieldnames.copy()
    if not has_session_col:
        if "session_id" not in out_fieldnames:
            out_fieldnames.append("session_id")
    # 强制包含必需的元数据列（即使原文件没有）
    required_meta_cols = ["site", "weight (kg)", "age", "sex", "breed"]
    for col in required_meta_cols:
        if col not in out_fieldnames:
            out_fieldnames.append(col)

    # 获取默认的 site 值（从 bids_root 目录名）
    default_site = bids_root.name  # 例如 "site-001" 或 "bids"

    # ========== 带文件锁的读取-修改-写入操作 ==========
    meta_csv = Path(qc_dir) / "meta.csv"
    lock_file = meta_csv.with_suffix(".csv.lock")
    meta_csv.parent.mkdir(parents=True, exist_ok=True)

    max_retries = 1000
    retry_delay = 0.2  # 200ms

    lock_fd = None
    try:
        # 打开锁文件（使用 'w+' 模式确保文件存在）
        lock_fd = open(lock_file, 'w+')

        # 尝试获取排他锁，带重试
        acquired = False
        for attempt in range(max_retries):
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except (IOError, OSError):
                # 锁被其他进程持有，等待后重试
                time.sleep(retry_delay)

        if not acquired:
            print(f"⚠️ Failed to acquire lock for {meta_csv} after {max_retries} attempts, skipping meta.csv update.")
            return

        # ========== 持有锁期间执行读取-修改-写入 ==========
        existing_rows_by_key = {}
        if meta_csv.exists():
            try:
                with open(meta_csv, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f, delimiter=",")
                    if reader.fieldnames:
                        for row in reader:
                            pid = row.get("participant_id", "").strip()
                            ses_val = row.get("session_id", "").strip()
                            if pid:
                                existing_rows_by_key[(pid, ses_val)] = row
            except Exception as e:
                print(f"⚠️ Failed to read existing meta.csv, will start fresh: {e}")

        # 合并：保留其他所有行，只覆盖当前要更新的特定session
        final_rows_by_key = {}
        for (pid, ses_val), row in existing_rows_by_key.items():
            # 只有当这个session不是当前要更新的，才保留
            if (pid, ses_val) not in new_rows_by_key:
                final_rows_by_key[(pid, ses_val)] = row
        final_rows_by_key.update(new_rows_by_key)

        # 写入最终的 meta.csv（先写临时文件再原子重命名）
        temp_csv = meta_csv.with_suffix(".csv.tmp")
        try:
            with open(temp_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=out_fieldnames, delimiter=",")
                writer.writeheader()
                for key in sorted(final_rows_by_key.keys()):
                    row = final_rows_by_key[key]
                    # 确保行包含所有输出列，缺失的补空字符串
                    for col in out_fieldnames:
                        if col not in row:
                            row[col] = ""
                    # 处理 site 列：如果为空或缺失，填充默认值
                    site_val = row.get("site", "").strip()
                    if not site_val:
                        row["site"] = default_site
                    writer.writerow(row)

            # 原子重命名（在 Unix 上是原子操作）
            os.replace(temp_csv, meta_csv)
            print(f"✅ meta.csv updated at {meta_csv} with {len(final_rows_by_key)} rows.")
        except Exception as e:
            print(f"⚠️ Failed to write meta.csv: {e}")
            # 清理临时文件
            if temp_csv.exists():
                try:
                    temp_csv.unlink()
                except:
                    pass

    finally:
        # 释放锁
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except:
                pass
            # 尝试删除锁文件（不强制，因为可能有其他进程刚打开它）
            try:
                if lock_file.exists():
                    lock_file.unlink(missing_ok=True)
            except:
                pass


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-session YAML + QC JPG under qc_dir/sub-*/ses-* (supports filtering by subject/session)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="BIDS root OR a sub-xxx directory OR a ses-xxx directory",
    )
    parser.add_argument("--qc_dir", type=str, required=True, help="QC root directory")

    parser.add_argument("--subj", type=str, default=None, help="Only process this subject, e.g. sub-001")
    parser.add_argument("--ses", type=str, default=None, help="Only process this session, e.g. ses-20210620T152859")
    parser.add_argument("--meta-only", action="store_true", help="Only update meta.csv, skip generating YAML configs and QC figures")

    args = parser.parse_args()

    filter_subj, filter_ses = resolve_filters(args.data_dir, args.subj, args.ses)

    session_yamls, sess_cfg = generate_session_yamls(
        args.data_dir,
        args.qc_dir,
        filter_subj=filter_subj,
        filter_ses=filter_ses,
        generate_files=not args.meta_only
    )

    if not args.meta_only:
        plot_all_session_figures(session_yamls)
    else:
        print(f"ℹ️ Skipping QC figure generation (meta-only mode)")

    # Build subject -> list of sessions mapping from sess_cfg
    subj_to_ses: Dict[str, List[str]] = {}
    for (subj, ses) in sess_cfg.keys():
        subj_to_ses.setdefault(subj, []).append(ses)

    # Determine BIDS root for participants.tsv
    start_path = Path(args.data_dir).expanduser().resolve()

    bids_root = find_bids_root(start_path)
    if bids_root is None:
        print(f"⚠️ Could not determine BIDS root from {start_path}, skipping meta.csv generation.")
    else:
        process_participants_tsv(bids_root, args.qc_dir, subj_to_ses)


if __name__ == "__main__":
    main()