"""
Phase 2 local validator. Computes the official mean ARI over (FOV × 4 levels)
on a predicted submission for the held-out val FOVs.

Strategy: build a per-spot ground-truth submission from spots_train.csv
by point-in-polygon-style mask lookup against per-(fov, z) GT label
masks rasterized from cell_boundaries_train.csv. This mirrors the
official metric (each spot must have a label at all 4 levels).

We rasterize once per FOV (not the whole train set) and cache nothing
on disk — for 10 val FOVs this takes ~30s.

Usage:
    python scripts/phase2/local_eval_phase2.py \
        --predicted runs/phase2_baseline/val_submission.csv \
        --val_fovs phase2_val_fovs.txt
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/scratch/pl2820/data/competition_phase2")
sys.path.insert(0, str(DATA_ROOT))   # to import metric.py
from metric import merfish_score, LEVELS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

LABELS_CSV = DATA_ROOT / "train" / "ground_truth" / "cell_labels_train.csv"
BOUNDS_CSV = DATA_ROOT / "train" / "ground_truth" / "cell_boundaries_train.csv"
SPOTS_CSV = DATA_ROOT / "train" / "ground_truth" / "spots_train.csv"
FOV_META_CSV = DATA_ROOT / "reference" / "fov_metadata.csv"

IMG_H, IMG_W = 2048, 2048
N_Z = 5
PIXEL_SIZE = 0.109
LABEL_LEVELS = ["class_label", "subclass_label", "supertype_label", "cluster_label"]
SUB_LEVELS = ["class", "subclass", "supertype", "cluster"]
BG = "background"


def parse_boundary(s) -> np.ndarray | None:
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        return np.fromstring(s, sep=",", dtype=np.float64)
    except Exception:
        return None


def rasterize_fov_z(
    bounds_in_fov: pd.DataFrame, z: int, fov_x: float, fov_y: float
) -> np.ndarray:
    """Returns (H, W) int32 mask: 0 background, 1..N integer cell label.

    The per-cell label is the row index in bounds_in_fov + 1 (1-indexed).
    """
    mask = np.zeros((IMG_H, IMG_W), dtype=np.int32)
    bx_col = f"boundaryX_z{z}"
    by_col = f"boundaryY_z{z}"
    for i, row in enumerate(bounds_in_fov.itertuples(index=False)):
        bx = parse_boundary(getattr(row, bx_col))
        by = parse_boundary(getattr(row, by_col))
        if bx is None or by is None or len(bx) != len(by) or len(bx) < 3:
            continue
        # boundary stored as global stage µm; convert to pixel coords.
        rows = IMG_H - (bx - fov_x) / PIXEL_SIZE
        cols = (by - fov_y) / PIXEL_SIZE
        poly = np.column_stack([cols, rows]).astype(np.int32)  # cv2: (x=col, y=row)
        cv2.fillPoly(mask, [poly], color=int(i + 1))
    return mask


def build_gt_for_fov(
    fov_id: str, fov_meta: pd.DataFrame, labels: pd.DataFrame, bounds: pd.DataFrame,
    spots: pd.DataFrame
) -> pd.DataFrame:
    """Build (spot_id, fov, class, subclass, supertype, cluster) for one FOV's spots."""
    m = fov_meta.loc[fov_id]
    fov_bounds = bounds[bounds["fov"] == fov_id].reset_index(drop=True)
    fov_labels_by_id = labels.set_index("cell_id")
    fov_labels_by_id.index = fov_labels_by_id.index.astype(str)

    fov_spots = spots[spots["fov"] == fov_id].copy().reset_index(drop=True)
    if len(fov_spots) == 0:
        return pd.DataFrame(columns=["spot_id","fov"] + SUB_LEVELS)

    fov_spots["zint"] = np.rint(fov_spots["global_z"]).astype(int).clip(0, N_Z - 1)
    out = pd.DataFrame({
        "spot_id": fov_spots["spot_id"].to_numpy(),  # globally-unique ids set by main()
        "fov": fov_id,
    })
    for lvl in SUB_LEVELS:
        out[lvl] = BG

    # Rasterize per z, look up spots in that z plane.
    cell_id_arr = fov_bounds["cell_id"].astype(str).to_numpy()
    for z in range(N_Z):
        z_spots = fov_spots[fov_spots["zint"] == z]
        if len(z_spots) == 0:
            continue
        mask = rasterize_fov_z(fov_bounds, z, m.fov_x, m.fov_y)
        rows = z_spots["image_row"].to_numpy().clip(0, IMG_H - 1)
        cols = z_spots["image_col"].to_numpy().clip(0, IMG_W - 1)
        cell_idx = mask[rows, cols]
        in_cell = cell_idx > 0
        if not in_cell.any():
            continue
        cell_ids = cell_id_arr[cell_idx[in_cell] - 1]
        for label_lvl, sub_lvl in zip(LABEL_LEVELS, SUB_LEVELS):
            vals = fov_labels_by_id.reindex(cell_ids)[label_lvl].fillna(BG).to_numpy()
            sel = z_spots.index[in_cell]
            out.loc[sel, sub_lvl] = vals

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 local validator (mean ARI over FOVs × levels)")
    ap.add_argument("--predicted", required=True, help="Submission CSV from predict.py")
    ap.add_argument("--val_fovs", default=str(_PROJECT_ROOT / "phase2_val_fovs.txt"))
    args = ap.parse_args()

    pred_path = Path(args.predicted)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    val_fovs = [ln.strip() for ln in open(args.val_fovs) if ln.strip()]
    log.info("Val FOVs (%d): %s", len(val_fovs), val_fovs)

    log.info("Loading reference tables...")
    fov_meta = pd.read_csv(FOV_META_CSV).set_index("fov")
    labels = pd.read_csv(LABELS_CSV, dtype={"cell_id": str})
    # boundaries CSV has unnamed first column = cell_id; attach fov from labels.
    bounds = pd.read_csv(BOUNDS_CSV, dtype={"Unnamed: 0": str})
    bounds = bounds.rename(columns={"Unnamed: 0": "cell_id"})
    cell_to_fov = labels.set_index("cell_id")["fov"]
    bounds["fov"] = bounds["cell_id"].map(cell_to_fov)
    n_orphan = bounds["fov"].isna().sum()
    if n_orphan:
        log.warning("  %d boundary rows have no fov mapping (dropping)", n_orphan)
        bounds = bounds.dropna(subset=["fov"])
    log.info("  cell_labels=%d  cell_boundaries=%d", len(labels), len(bounds))

    log.info("Loading val spots only...")
    t = time.time()
    spots = pd.read_csv(
        SPOTS_CSV,
        usecols=["fov", "image_row", "image_col", "global_z", "target_gene"],
    )
    # Synthesize global spot_ids that match predict.py's "s<global_row_idx>"
    # convention BEFORE filtering, so val and predicted spot_ids align.
    spots["spot_id"] = "s" + spots.index.astype(str)
    spots = spots[spots["fov"].isin(val_fovs)].reset_index(drop=True)
    log.info("  %d val spots loaded in %.1fs", len(spots), time.time() - t)

    log.info("Building per-spot GT (rasterizing %d FOV × %d z = %d masks)...",
             len(val_fovs), N_Z, len(val_fovs) * N_Z)
    t = time.time()
    gt_pieces = []
    for i, fov in enumerate(val_fovs):
        log.info("  [%d/%d] %s", i + 1, len(val_fovs), fov)
        gt_pieces.append(build_gt_for_fov(fov, fov_meta, labels, bounds, spots))
    gt = pd.concat(gt_pieces, ignore_index=True)
    log.info("GT built (%d rows) in %.1fs", len(gt), time.time() - t)

    log.info("Loading predicted submission %s", pred_path)
    pred = pd.read_csv(pred_path)
    pred_val = pred[pred["fov"].isin(val_fovs)].copy()
    log.info("  predicted rows for val FOVs: %d (vs %d GT rows)",
             len(pred_val), len(gt))

    # Align by spot_id; predict.py already uses 's<index>' synthesized ids on
    # train spots, so they should match GT's 's<index>' ids exactly.
    if set(pred_val["spot_id"]) != set(gt["spot_id"]):
        only_pred = set(pred_val["spot_id"]) - set(gt["spot_id"])
        only_gt = set(gt["spot_id"]) - set(pred_val["spot_id"])
        log.warning("  spot_id mismatch: %d in pred only, %d in gt only",
                    len(only_pred), len(only_gt))

    sol = gt.set_index("spot_id")
    sub = pred_val.set_index("spot_id")
    score = merfish_score(sol, sub)
    log.info("=== mean ARI (official metric, %d FOVs × %d levels) = %.4f ===",
             len(val_fovs), len(LEVELS), score)

    # Per-level breakdown
    from sklearn.metrics import adjusted_rand_score
    log.info("Per-level breakdown:")
    for lvl in LEVELS:
        per_fov = []
        for fov in val_fovs:
            mask = sol["fov"] == fov
            if mask.sum() == 0:
                continue
            gt_vals = sol.loc[mask, lvl].astype(str).to_numpy()
            pred_vals = sub.reindex(sol.index)[lvl].astype(str).fillna(BG).to_numpy()[mask]
            ari = adjusted_rand_score(gt_vals, pred_vals)
            per_fov.append((fov, ari))
        per_df = pd.DataFrame(per_fov, columns=["fov","ari"])
        log.info("  %-9s mean=%.4f  per-FOV: %s",
                 lvl, per_df["ari"].mean(),
                 [(f, round(a, 3)) for f, a in per_fov])

    return score


if __name__ == "__main__":
    main()
