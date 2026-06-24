#!/usr/bin/env python
import argparse
import math
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import nibabel as nib


def world_to_voxel(inv_affine: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    Convert a 3D point in world/RAS coordinates into voxel indices.

    Parameters
    ----------
    inv_affine : np.ndarray
        Inverse of the NIfTI affine matrix (4x4).
    xyz : np.ndarray
        World coordinates, shape (3,).

    Returns
    -------
    np.ndarray
        Voxel indices (i, j, k) as integers.
    """
    xyz_h = np.concatenate([xyz, [1.0]])  # homogeneous coords [x, y, z, 1]
    ijk = inv_affine.dot(xyz_h)[:3]
    return np.round(ijk).astype(int)


def collect_votes_for_hemisphere(
    white_path: str,
    pial_path: str,
    label_path: str,
    ribbon_data: np.ndarray,
    ribbon_value: int,
    inv_affine: np.ndarray,
    n_steps: int = 10,
    num_threads: int = 1,
    hemi_name: str = "lh",
) -> dict:
    """
    Collect per-voxel label votes for one hemisphere.

    For each vertex:
      - Take the segment from white to pial.
      - Sample n_steps points along this segment.
      - If the sampled voxel lies in the current hemisphere's ribbon
        (ribbon_data == ribbon_value), cast a vote for that voxel's label.

    Parameters
    ----------
    white_path : str
        Path to white surface (.surf.gii) for this hemisphere.
    pial_path : str
        Path to pial surface (.surf.gii) for this hemisphere.
    label_path : str
        Path to hemi.aparc.label.gii (per-vertex labels).
    ribbon_data : np.ndarray
        3D ribbon volume; hemisphere is encoded by ribbon_value.
    ribbon_value : int
        Value in ribbon_data that denotes this hemisphere
        (e.g., 3 for lh, 42 for rh).
    inv_affine : np.ndarray
        Inverse affine of the ribbon volume (4x4).
    n_steps : int
        Number of samples along the white→pial direction.
    num_threads : int
        Number of threads to use for vertex processing.
    hemi_name : str
        Name for logging ("lh" or "rh").

    Returns
    -------
    dict[int, Counter]
        Mapping from flattened voxel index to Counter(label -> count).
    """
    if white_path is None or pial_path is None or label_path is None:
        print(f"[{hemi_name}] Skipped: white / pial / label path missing.")
        return {}

    print(f"[{hemi_name}] Loading white surface: {white_path}")
    white_gii = nib.load(white_path)
    white_verts = white_gii.darrays[0].data.astype(np.float64)  # (N, 3)

    print(f"[{hemi_name}] Loading pial surface: {pial_path}")
    pial_gii = nib.load(pial_path)
    pial_verts = pial_gii.darrays[0].data.astype(np.float64)    # (N, 3)

    if white_verts.shape != pial_verts.shape:
        raise ValueError(
            f"[{hemi_name}] white and pial vertex counts do not match: "
            f"{white_verts.shape} vs {pial_verts.shape}"
        )

    print(f"[{hemi_name}] Loading label GIFTI: {label_path}")
    label_gii = nib.load(label_path)
    labels = np.asarray(label_gii.darrays[0].data).squeeze().astype(np.int32)

    if labels.shape[0] != white_verts.shape[0]:
        raise ValueError(
            f"[{hemi_name}] label vertex count {labels.shape[0]} does not match "
            f"surface vertex count {white_verts.shape[0]}"
        )

    nx, ny, nz = ribbon_data.shape
    n_verts = white_verts.shape[0]
    print(
        f"[{hemi_name}] #vertices={n_verts}, n_steps={n_steps}, "
        f"num_threads={num_threads}"
    )

    # Shared sampling positions along the cortical thickness
    sample_ts = np.linspace(0.0, 1.0, n_steps)

    def process_chunk(v_start: int, v_end: int) -> dict:
        """
        Process a subset of vertices [v_start, v_end) and collect votes.

        Returns
        -------
        dict[int, Counter]
            Local votes for this chunk.
        """
        local_votes = defaultdict(Counter)
        for v in range(v_start, v_end):
            label = labels[v]
            if label == 0:
                # Usually label 0 means "background" – skip it.
                continue

            w = white_verts[v]
            p = pial_verts[v]
            direction = p - w

            for t in sample_ts:
                xyz = w + t * direction
                i, j, k = world_to_voxel(inv_affine, xyz)

                if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                    # Only vote inside this hemisphere's ribbon voxels
                    if ribbon_data[i, j, k] == ribbon_value:
                        idx_flat = (i * ny + j) * nz + k
                        local_votes[idx_flat][label] += 1

        return local_votes

    # If the data is small or only 1 thread is requested, do it serially
    if num_threads <= 1 or n_verts < 1000:
        print(f"[{hemi_name}] Using a single thread for voting ...")
        return process_chunk(0, n_verts)

    # Multi-threaded processing
    print(f"[{hemi_name}] Using multi-threaded voting ...")
    chunk_size = math.ceil(n_verts / num_threads)
    votes = defaultdict(Counter)
    futures = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for start in range(0, n_verts, chunk_size):
            end = min(start + chunk_size, n_verts)
            futures.append(executor.submit(process_chunk, start, end))

        for fut in as_completed(futures):
            local_votes = fut.result()
            # Merge local votes into global votes
            for idx_flat, ctr in local_votes.items():
                votes[idx_flat].update(ctr)

    print(f"[{hemi_name}] Voting finished.")
    return votes


def apply_majority_vote(
    votes_total: dict,
    vol_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    Convert per-voxel vote counters into a label volume by majority vote.

    Parameters
    ----------
    votes_total : dict[int, Counter]
        Mapping from flattened voxel index to vote Counter(label -> count).
    vol_shape : tuple[int, int, int]
        Shape of the output volume (nx, ny, nz).

    Returns
    -------
    np.ndarray
        3D label volume (int32).
    """
    nx, ny, nz = vol_shape
    labels_vol = np.zeros(vol_shape, dtype=np.int32)

    plane = ny * nz  # used to unflatten indices
    for idx_flat, ctr in votes_total.items():
        if not ctr:
            continue
        # Pick label with maximum vote (Counter.most_common handles ties)
        label, _ = ctr.most_common(1)[0]

        i = idx_flat // plane
        rem = idx_flat % plane
        j = rem // nz
        k = rem % nz

        labels_vol[i, j, k] = label

    return labels_vol


def fill_unlabeled_voxels(
    labels_vol: np.ndarray,
    ribbon_vol: np.ndarray,
    max_iters: int = 3,
    radius: int = 1,
) -> np.ndarray:
    """
    Fill ribbon voxels that remain unlabeled (0) using neighbor majority vote.

    For each iteration:
      - Find voxels where ribbon_vol > 0 and labels_vol == 0.
      - For each such voxel, look at a cubic neighborhood with given radius.
      - Collect labels from neighboring voxels that:
            (1) have label > 0, and
            (2) belong to the same hemisphere (same ribbon value).
      - If any neighbor labels are available, assign the majority label.

    Parameters
    ----------
    labels_vol : np.ndarray
        3D label volume after surface projection.
    ribbon_vol : np.ndarray
        3D ribbon volume with hemisphere coding (e.g., 3 for lh, 42 for rh).
    max_iters : int
        Maximum number of filling iterations.
        Multiple iterations allow labels to propagate further.
    radius : int
        Neighborhood radius in voxels (1 => 3x3x3 window).

    Returns
    -------
    np.ndarray
        Filled label volume (int32).
    """
    filled = labels_vol.copy()
    nx, ny, nz = filled.shape

    for it in range(max_iters):
        # Identify unlabeled ribbon voxels
        unlabeled_mask = (ribbon_vol > 0) & (filled == 0)
        coords = np.array(np.where(unlabeled_mask)).T

        if coords.size == 0:
            print(f"[fill] Iteration {it + 1}: no unlabeled ribbon voxels remain.")
            break

        print(
            f"[fill] Iteration {it + 1}: #unlabeled ribbon voxels = {coords.shape[0]}"
        )

        changes = 0
        new_filled = filled.copy()

        for idx in coords:
            i, j, k = idx
            hemi_val = ribbon_vol[i, j, k]
            if hemi_val == 0:
                # Not in ribbon, skip
                continue

            # Neighborhood bounds
            i0 = max(i - radius, 0)
            i1 = min(i + radius + 1, nx)
            j0 = max(j - radius, 0)
            j1 = min(j + radius + 1, ny)
            k0 = max(k - radius, 0)
            k1 = min(k + radius + 1, nz)

            region_labels = filled[i0:i1, j0:j1, k0:k1]
            region_ribbon = ribbon_vol[i0:i1, j0:j1, k0:k1]

            # Only consider neighbors with:
            #   - label > 0
            #   - same hemisphere value in ribbon
            neighbor_mask = (region_labels > 0) & (region_ribbon == hemi_val)
            if not np.any(neighbor_mask):
                continue

            neighbor_labels = region_labels[neighbor_mask].astype(np.int32)
            # Majority vote among neighbor labels
            label_counts = np.bincount(neighbor_labels)
            majority_label = int(np.argmax(label_counts))

            if majority_label > 0:
                new_filled[i, j, k] = majority_label
                changes += 1

        filled = new_filled

        print(f"[fill] Iteration {it + 1}: filled {changes} voxels.")
        if changes == 0:
            print("[fill] No changes in this iteration, stopping early.")
            break

    return filled


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Project hemi.aparc.label.gii surface labels onto a cortical ribbon "
            "volume (3=lh ribbon, 42=rh ribbon), with multi-threaded voting and "
            "post-hoc neighbor-based filling."
        )
    )
    parser.add_argument(
        "--ribbon",
        required=True,
        help="Input ribbon volume (NIfTI/mgz). 3=lh ribbon, 42=rh ribbon.",
    )

    # Left hemisphere
    parser.add_argument("--lh-white", help="Left hemisphere white surface (.surf.gii).")
    parser.add_argument("--lh-pial", help="Left hemisphere pial surface (.surf.gii).")
    parser.add_argument(
        "--lh-label", help="Left hemisphere label file (hemi.aparc.label.gii)."
    )

    # Right hemisphere
    parser.add_argument("--rh-white", help="Right hemisphere white surface (.surf.gii).")
    parser.add_argument("--rh-pial", help="Right hemisphere pial surface (.surf.gii).")
    parser.add_argument(
        "--rh-label", help="Right hemisphere label file (hemi.aparc.label.gii)."
    )

    parser.add_argument(
        "--n-steps",
        type=int,
        default=10,
        help="Number of samples along white→pial direction (default: 10).",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for vertex voting (default: 4).",
    )
    parser.add_argument(
        "--fill-iters",
        type=int,
        default=3,
        help=(
            "Max iterations for neighbor-based filling of unlabeled ribbon voxels "
            "(default: 3)."
        ),
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output label ribbon volume (NIfTI).",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point:
      1. Load ribbon volume and set up affine.
      2. Collect votes from lh and rh surfaces.
      3. Apply majority vote to get initial label volume.
      4. Fill unlabeled ribbon voxels using local neighborhood voting.
      5. Save final label volume.
    """
    args = parse_args()

    # -------------------------------------------------------------------------
    # 1. Load ribbon volume
    # -------------------------------------------------------------------------
    print(f"[ribbon] Loading: {args.ribbon}")
    ribbon_img = nib.load(args.ribbon)
    ribbon_data = ribbon_img.get_fdata().astype(np.int16)
    affine = ribbon_img.affine
    inv_affine = np.linalg.inv(affine)
    vol_shape = ribbon_data.shape
    print(f"[ribbon] Volume shape: {vol_shape}")

    # -------------------------------------------------------------------------
    # 2. Collect votes for each hemisphere
    # -------------------------------------------------------------------------
    votes_total = defaultdict(Counter)

    # Left hemisphere: ribbon == 3
    votes_lh = collect_votes_for_hemisphere(
        white_path=args.lh_white,
        pial_path=args.lh_pial,
        label_path=args.lh_label,
        ribbon_data=ribbon_data,
        ribbon_value=3,
        inv_affine=inv_affine,
        n_steps=args.n_steps,
        num_threads=args.num_threads,
        hemi_name="lh",
    )
    for idx_flat, ctr in votes_lh.items():
        votes_total[idx_flat].update(ctr)

    # Right hemisphere: ribbon == 42
    votes_rh = collect_votes_for_hemisphere(
        white_path=args.rh_white,
        pial_path=args.rh_pial,
        label_path=args.rh_label,
        ribbon_data=ribbon_data,
        ribbon_value=42,
        inv_affine=inv_affine,
        n_steps=args.n_steps,
        num_threads=args.num_threads,
        hemi_name="rh",
    )
    for idx_flat, ctr in votes_rh.items():
        votes_total[idx_flat].update(ctr)

    # -------------------------------------------------------------------------
    # 3. Majority vote to obtain initial label volume
    # -------------------------------------------------------------------------
    print("[vote] Applying majority vote to obtain initial label volume ...")
    labels_vol = apply_majority_vote(votes_total, vol_shape)

    # -------------------------------------------------------------------------
    # 4. Neighbor-based filling for unlabeled ribbon voxels
    # -------------------------------------------------------------------------
    print("[fill] Filling unlabeled ribbon voxels based on local neighborhoods ...")
    filled_vol = fill_unlabeled_voxels(
        labels_vol=labels_vol,
        ribbon_vol=ribbon_data,
        max_iters=args.fill_iters,
        radius=1,
    )

    # -------------------------------------------------------------------------
    # 5. Save result
    # -------------------------------------------------------------------------
    out_img = nib.Nifti1Image(filled_vol.astype(np.int16), affine, header=ribbon_img.header)
    print(f"[out] Saving final label ribbon volume to: {args.out}")
    nib.save(out_img, args.out)


if __name__ == "__main__":
    main()
