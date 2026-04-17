"""
Build ground-truth spot -> cell labels for the training set.

Why: spots_train.csv has no cell_id column — the GT assignment must be
derived by point-in-polygon against cell_boundaries_train.csv. We do this
once and cache the result so downstream scripts can reuse it.

Outputs:
  cache/gt_spot_labels.parquet
    columns: spot_idx (row index in spots_train.csv), fov, gt_cluster_id

Pixel coordinate convention (verified empirically against precomputed
image_row/image_col in spots_train.csv):
  image_row = IMG_H - (global_x - fov_x) / pixel_size
  image_col =         (global_y - fov_y) / pixel_size
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

IMG_H = 2048
IMG_W = 2048
Z_PLANES = 5

log = logging.getLogger("build_gt_labels")


def load_cell_fov_map(h5ad_path: Path) -> pd.Series:
    """cell_id (str) -> fov (str)."""
    with h5py.File(h5ad_path, "r") as f:
        cell_ids = np.array([x.decode() if isinstance(x, bytes) else x
                             for x in f["obs/_index"][:]])
        codes = f["obs/fov/codes"][:]
        cats = np.array([x.decode() if isinstance(x, bytes) else x
                         for x in f["obs/fov/categories"][:]])
    return pd.Series(cats[codes], index=cell_ids.astype(str), name="fov")


def parse_boundary(s) -> np.ndarray | None:
    """Parse a comma-separated float string from the CSV into a 1D array.

    Returns None if the string is empty/NaN/unparseable."""
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        return np.fromstring(s, sep=",", dtype=np.float64)
    except Exception:
        return None


def rasterize_fov_z(
    cells_in_fov: pd.DataFrame,   # cell_id index, columns [boundaryX_zK, boundaryY_zK]
    z: int,
    fov_x: float,
    fov_y: float,
    pixel_size: float,
) -> tuple[np.ndarray, list[str]]:
    """Rasterize all cell polygons for this FOV at plane z into an int32 mask.

    Returns (mask, cell_ids_by_index) — mask values in [0, N], where 0 = bg
    and k>=1 refers to cell_ids_by_index[k-1].
    """
    mask = np.zeros((IMG_H, IMG_W), dtype=np.int32)
    cell_ids_by_index: list[str] = []
    bx_col = f"boundaryX_z{z}"
    by_col = f"boundaryY_z{z}"

    for cell_id, row in cells_in_fov.iterrows():
        bx = parse_boundary(row.get(bx_col))
        by = parse_boundary(row.get(by_col))
        if bx is None or by is None or len(bx) < 3 or len(bx) != len(by):
            continue
        # Global µm -> image pixel (stage x -> inverted image row, stage y -> col)
        img_row = IMG_H - (bx - fov_x) / pixel_size
        img_col = (by - fov_y) / pixel_size
        pts = np.stack([img_col, img_row], axis=1).round().astype(np.int32)
        cell_ids_by_index.append(str(cell_id))
        cv2.fillPoly(mask, [pts], color=len(cell_ids_by_index))
    return mask, cell_ids_by_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/scratch/pl2820/data/competition")
    ap.add_argument("--output", default="/scratch/tjv235/cell_segmentation/cache/gt_spot_labels.parquet")
    ap.add_argument("--mask_dir", default=None, help="Optional dir to save per-(fov,z) GT masks as .npy")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    data_root = Path(args.data_root)
    spots_path = data_root / "train" / "ground_truth" / "spots_train.csv"
    bounds_path = data_root / "train" / "ground_truth" / "cell_boundaries_train.csv"
    h5ad_path = data_root / "train" / "ground_truth" / "counts_train.h5ad"
    meta_path = data_root / "reference" / "fov_metadata.csv"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    if mask_dir:
        mask_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log.info("Loading cell->FOV map from %s", h5ad_path)
    cell_fov = load_cell_fov_map(h5ad_path)
    log.info("  %d cells across %d FOVs", len(cell_fov), cell_fov.nunique())

    log.info("Loading cell boundaries from %s", bounds_path)
    bounds = pd.read_csv(bounds_path, dtype={"Unnamed: 0": str})
    bounds = bounds.rename(columns={"Unnamed: 0": "cell_id"}).set_index("cell_id")
    log.info("  %d boundary rows", len(bounds))
    assert set(cell_fov.index) == set(bounds.index), "cell_id mismatch between h5ad and boundaries"

    # Attach FOV to boundaries
    bounds["fov"] = cell_fov.reindex(bounds.index).values

    log.info("Loading spots from %s", spots_path)
    spots = pd.read_csv(
        spots_path,
        usecols=["fov", "image_row", "image_col", "global_z"],
        dtype={"fov": "category"},
    )
    spots["z_idx"] = spots["global_z"].round().astype(np.int32).clip(0, Z_PLANES - 1)
    log.info("  %d spots", len(spots))

    log.info("Loading fov_metadata from %s", meta_path)
    meta = pd.read_csv(meta_path).set_index("fov")

    # Prepare output container — one string per spot row
    gt_labels = np.full(len(spots), "background", dtype=object)

    fov_ids = sorted(bounds["fov"].unique())
    log.info("Rasterizing and labeling spots for %d FOVs", len(fov_ids))

    # Index spots by fov for fast lookup (positions in the original frame)
    spots_idx_by_fov: dict[str, np.ndarray] = {}
    for fov_id, grp in spots.groupby("fov", observed=True):
        spots_idx_by_fov[str(fov_id)] = grp.index.to_numpy()

    for fi, fov_id in enumerate(fov_ids):
        fov_meta = meta.loc[fov_id]
        fov_x = float(fov_meta["fov_x"])
        fov_y = float(fov_meta["fov_y"])
        px = float(fov_meta["pixel_size"])

        cells_in_fov = bounds[bounds["fov"] == fov_id]
        pos = spots_idx_by_fov.get(fov_id)
        if pos is None or len(pos) == 0:
            log.warning("  [%02d/%d] %s: no spots, skipping", fi + 1, len(fov_ids), fov_id)
            continue

        fov_spots = spots.loc[pos]
        t_fov = time.time()
        n_labeled_total = 0
        for z in range(Z_PLANES):
            mask, cell_ids_by_index = rasterize_fov_z(
                cells_in_fov, z, fov_x, fov_y, px,
            )
            if mask_dir:
                np.save(mask_dir / f"{fov_id}_z{z}.npy", mask)

            at_z = fov_spots["z_idx"].to_numpy() == z
            if not at_z.any():
                continue
            rows = fov_spots.loc[at_z, "image_row"].to_numpy()
            cols = fov_spots.loc[at_z, "image_col"].to_numpy()
            rows = np.clip(rows, 0, IMG_H - 1)
            cols = np.clip(cols, 0, IMG_W - 1)
            cell_int = mask[rows, cols]

            labels = np.full(len(rows), "background", dtype=object)
            assigned = cell_int > 0
            if assigned.any():
                cell_ids_arr = np.array(cell_ids_by_index, dtype=object)
                assigned_cell_ids = cell_ids_arr[cell_int[assigned] - 1]
                labels[assigned] = np.array(
                    [f"{fov_id}_cell_{cid}" for cid in assigned_cell_ids],
                    dtype=object,
                )
                n_labeled_total += int(assigned.sum())
            # Write back into global gt_labels using the original positions
            target_pos = pos[at_z]
            gt_labels[target_pos] = labels

        log.info(
            "  [%02d/%d] %s: %d cells, %d spots, %d labeled (%.1f%% bg), %.2fs",
            fi + 1, len(fov_ids), fov_id, len(cells_in_fov), len(pos),
            n_labeled_total, 100.0 * (len(pos) - n_labeled_total) / max(len(pos), 1),
            time.time() - t_fov,
        )

    # Assemble output
    out = pd.DataFrame({
        "spot_idx": np.arange(len(spots), dtype=np.int64),
        "fov": spots["fov"].astype(str).values,
        "gt_cluster_id": gt_labels,
    })
    out.to_parquet(out_path, index=False)
    log.info("Wrote %d rows to %s (%.1fs total)", len(out), out_path, time.time() - t0)

    # Quick summary
    n_bg = (out["gt_cluster_id"] == "background").sum()
    log.info("GT background fraction: %d / %d = %.3f",
             n_bg, len(out), n_bg / len(out))


if __name__ == "__main__":
    main()
