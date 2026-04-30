"""
Local ARI evaluator for MERFISH segmentation experiments.

Invokes a pluggable segmenter factory on a fixed list of held-out training FOVs,
assigns spots to cells via mask lookup (matching the Kaggle metric), joins with
cached ground-truth labels from cache/gt_spot_labels.parquet, and reports
per-FOV + mean Adjusted Rand Index.

Example:
    python scripts/local_eval.py \\
        --segmenter scripts.segmenters:build_cellpose_zeroshot \\
        --gpu \\
        --diameter 0  # 0 means None (auto)

Outputs written to runs/<timestamp>/:
    config.json      — cmd-line args and env info
    per_fov.csv      — fov, ari, n_pred_cells, n_gt_cells, pred_bg_pct, gt_bg_pct
    summary.json     — mean_ari, per_fov_ari, timing
    masks/<fov>.npy  — predicted mask (for post-hoc debugging)
    eval.log         — full log

Notes on the segmenter interface:
    --segmenter must point to a FACTORY: module.path:factory_name. Calling the
    factory (with kwargs --gpu, --diameter) returns a segment(fov_dir) callable
    that returns a (2048, 2048) int mask. This lets the caller load a model
    once and reuse it across FOVs.
"""

import argparse
import importlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

# Import the organizer's official metric so our headline number is identical
# to what Kaggle computes. Located outside the project tree.
_OFFICIAL_METRIC_DIR = "/scratch/pl2820/data/competition"
if _OFFICIAL_METRIC_DIR not in sys.path:
    sys.path.insert(0, _OFFICIAL_METRIC_DIR)
from metric import merfish_score as _official_merfish_score  # noqa: E402

log = logging.getLogger("local_eval")

IMG_H = 2048
IMG_W = 2048
N_Z = 5  # valid global_z values are {0..4}


# ---------------------------------------------------------------------------
# Segmenter loading
# ---------------------------------------------------------------------------
def load_segmenter_factory(spec: str) -> Callable:
    """Parse 'module.path:factory_name' and return the factory callable."""
    if ":" not in spec:
        raise ValueError(f"--segmenter must be 'module:factory', got {spec!r}")
    mod_path, func_name = spec.split(":", 1)
    # Make sure project root and scripts/ dir are on sys.path
    project_root = Path(__file__).resolve().parent.parent
    for p in (project_root, project_root / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    mod = importlib.import_module(mod_path)
    if not hasattr(mod, func_name):
        raise AttributeError(f"{mod_path!r} has no attribute {func_name!r}")
    return getattr(mod, func_name)


# ---------------------------------------------------------------------------
# Spot -> cluster_id lookup (matches organizer's generate_submission.py format)
# ---------------------------------------------------------------------------
def predict_cluster_ids(
    mask: np.ndarray,
    fov_id: str,
    rows: np.ndarray,
    cols: np.ndarray,
    zs: np.ndarray | None = None,
) -> np.ndarray:
    """Return per-spot cluster_id string array ('{fov}_cell_{N}' or 'background').

    If `mask` is (H, W) each spot is looked up in that single plane (existing
    behavior). If `mask` is (Z, H, W) each spot is looked up at its own z via
    `zs`; zs is clamped to [0, Z-1] as a defense against out-of-range values.
    """
    rows_c = np.clip(rows, 0, IMG_H - 1)
    cols_c = np.clip(cols, 0, IMG_W - 1)
    if mask.ndim == 3:
        if zs is None:
            raise ValueError("3D mask supplied but zs is None")
        Z = mask.shape[0]
        zs_c = np.clip(zs, 0, Z - 1).astype(np.intp)
        cell_ints = mask[zs_c, rows_c, cols_c]
    elif mask.ndim == 2:
        cell_ints = mask[rows_c, cols_c]
    else:
        raise ValueError(f"Unexpected mask.ndim={mask.ndim}, shape={mask.shape}")
    out = np.full(len(rows), "background", dtype=object)
    assigned = cell_ints > 0
    if assigned.any():
        out[assigned] = np.array(
            [f"{fov_id}_cell_{v}" for v in cell_ints[assigned]],
            dtype=object,
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmenter", required=True,
                    help="Factory spec: 'module.path:factory_name'")
    user = os.environ.get("USER", "dr3432")
    ap.add_argument("--val_fovs", default=f"/scratch/{user}/cell_segmentation/val_fovs.txt")
    ap.add_argument("--data_root", default="/scratch/pl2820/data/competition")
    ap.add_argument("--gt_labels",
                    default=f"/scratch/{user}/cell_segmentation/cache/gt_spot_labels.parquet")
    ap.add_argument("--runs_dir", default=f"/scratch/{user}/cell_segmentation/runs")
    ap.add_argument("--run_name", default=None,
                    help="Override run directory name (default: timestamp)")
    ap.add_argument("--gpu", action="store_true",
                    help="Pass gpu=True to the segmenter factory")
    ap.add_argument("--diameter", type=float, default=None,
                    help="Cell diameter hint (passthrough to factory). "
                         "Negative values are treated as None (auto).")
    ap.add_argument("--save_masks", action="store_true",
                    help="Persist each predicted mask as runs/<name>/masks/<fov>.npy")
    ap.add_argument("--z_filter", type=int, default=None,
                    help="If set, restrict evaluation to spots with global_z == this "
                         "value. Diagnostic for per-z segmentation lift.")
    ap.add_argument("--segmenter_kwargs", default=None,
                    help="Extra keyword args forwarded to the segmenter factory, "
                         "comma-separated key=value pairs. Example: "
                         "'pretrained_model=/path/to/best.pt'. Values are strings; "
                         "factories coerce as needed.")
    args = ap.parse_args()

    # Run dir + logging
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "eval.log"
    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=[fh, sh])

    log.info("Run dir: %s", run_dir)

    # Persist config
    try:
        import torch
        cuda_avail = bool(torch.cuda.is_available())
    except Exception:
        cuda_avail = False
    config = {
        "segmenter": args.segmenter,
        "val_fovs_path": args.val_fovs,
        "data_root": args.data_root,
        "gt_labels_path": args.gt_labels,
        "gpu_flag": args.gpu,
        "diameter": args.diameter,
        "save_masks": args.save_masks,
        "z_filter": args.z_filter,
        "segmenter_kwargs": args.segmenter_kwargs,
        "cuda_available": cuda_avail,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    if args.gpu and not cuda_avail:
        log.warning("--gpu requested but torch.cuda.is_available() is False "
                    "— Cellpose will fall back to CPU and be slow.")

    # Load val FOV list
    val_fovs = [ln.strip() for ln in Path(args.val_fovs).read_text().splitlines()
                if ln.strip()]
    log.info("Val FOVs (%d): %s", len(val_fovs), val_fovs)

    # Load GT labels (only val rows, to keep memory light)
    log.info("Loading GT labels from %s", args.gt_labels)
    gt_all = pd.read_parquet(args.gt_labels)
    gt = gt_all[gt_all["fov"].isin(val_fovs)].copy()
    log.info("  GT rows for val FOVs: %d", len(gt))

    # Load train spots (need image_row/image_col/global_z for prediction)
    log.info("Loading spots_train.csv (val subset only, via chunk filter)")
    spots_path = Path(args.data_root) / "train" / "ground_truth" / "spots_train.csv"
    spots = pd.read_csv(
        spots_path,
        usecols=["fov", "image_row", "image_col", "global_z"],
    )
    spots = spots.reset_index().rename(columns={"index": "spot_idx"})
    spots = spots[spots["fov"].isin(val_fovs)].copy()
    log.info("  %d spots across val FOVs", len(spots))

    # Validate global_z range — per-spot 3D lookup depends on it, and the 2D
    # case also relies on z_filter / frame-map invariants. Out-of-range z is
    # clamped at lookup time, but we log a warning so silent data issues get
    # surfaced.
    z_unique = sorted(spots["global_z"].unique().tolist())
    z_bad = [z for z in z_unique if z < 0 or z >= N_Z]
    if z_bad:
        log.warning("global_z values outside [0,%d): %s — will be clamped",
                    N_Z, z_bad)
    else:
        log.info("  global_z values in spots: %s (all in [0,%d))", z_unique, N_Z)

    if args.z_filter is not None:
        before = len(spots)
        spots = spots[spots["global_z"] == args.z_filter].copy()
        log.info("  --z_filter=%d: kept %d / %d spots (%.1f%%)",
                 args.z_filter, len(spots), before, 100 * len(spots) / max(before, 1))
        gt = gt[gt["spot_idx"].isin(spots["spot_idx"])].copy()

    # Sanity check join
    assert len(gt) == len(spots), \
        f"GT/pred row mismatch for val set: gt={len(gt)} spots={len(spots)}"

    # Build segmenter
    factory_spec = args.segmenter
    factory = load_segmenter_factory(factory_spec)
    diameter = None if (args.diameter is None or args.diameter < 0) else args.diameter
    extra_kwargs: dict[str, str] = {}
    if args.segmenter_kwargs:
        for pair in args.segmenter_kwargs.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"Bad --segmenter_kwargs entry (no '='): {pair!r}")
            k, v = pair.split("=", 1)
            extra_kwargs[k.strip()] = v.strip()
    log.info("Building segmenter via %s (gpu=%s, diameter=%s, extra=%s)",
             factory_spec, args.gpu, diameter, extra_kwargs or "<none>")
    t_model_load = time.time()
    segment = factory(gpu=args.gpu, diameter=diameter, **extra_kwargs)
    log.info("  factory ready (%s) in %.2fs",
             getattr(segment, "name", "<unnamed>"), time.time() - t_model_load)

    # Prepare mask dir
    masks_dir = run_dir / "masks"
    if args.save_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)

    # Loop
    per_fov_rows = []
    gt_by_idx = gt.set_index("spot_idx")["gt_cluster_id"]
    # Accumulate predictions across all FOVs to build a submission-shaped
    # DataFrame and score it with the organizer's official merfish_score.
    all_pred_idx: list[np.ndarray] = []
    all_pred_fov: list[np.ndarray] = []
    all_pred_labels: list[np.ndarray] = []

    for i, fov_id in enumerate(val_fovs):
        log.info("--- [%d/%d] %s ---", i + 1, len(val_fovs), fov_id)
        fov_dir = Path(args.data_root) / "train" / fov_id
        if not fov_dir.exists():
            log.error("FOV dir missing: %s — skipping", fov_dir)
            continue

        t_seg = time.time()
        mask = segment(fov_dir)
        seg_time = time.time() - t_seg
        if mask.ndim == 2:
            if mask.shape != (IMG_H, IMG_W):
                raise ValueError(
                    f"{fov_id}: expected (2048,2048) mask, got {mask.shape}"
                )
        elif mask.ndim == 3:
            if mask.shape[1:] != (IMG_H, IMG_W):
                raise ValueError(
                    f"{fov_id}: expected (Z,2048,2048) mask, got {mask.shape}"
                )
        else:
            raise ValueError(
                f"{fov_id}: mask has unexpected ndim={mask.ndim} shape={mask.shape}"
            )
        n_pred_cells = int(mask.max())
        log.info("  segmenter: %d predicted cells, shape=%s, %.2fs",
                 n_pred_cells, mask.shape, seg_time)

        if args.save_masks:
            np.save(masks_dir / f"{fov_id}.npy", mask.astype(np.int32))

        # Predict cluster_ids for this FOV's spots
        fov_spots = spots[spots["fov"] == fov_id]
        pred = predict_cluster_ids(
            mask, fov_id,
            fov_spots["image_row"].to_numpy(),
            fov_spots["image_col"].to_numpy(),
            zs=fov_spots["global_z"].to_numpy() if mask.ndim == 3 else None,
        )
        gt_this = gt_by_idx.reindex(fov_spots["spot_idx"].to_numpy()).to_numpy()

        # Stats
        pred_bg = (pred == "background").sum()
        gt_bg = (gt_this == "background").sum()
        n_spots = len(fov_spots)
        n_gt_cells = pd.Series(gt_this[gt_this != "background"]).nunique()

        ari = adjusted_rand_score(gt_this.astype(str), pred.astype(str))
        log.info(
            "  spots=%d  pred_bg=%.1f%%  gt_bg=%.1f%%  pred_cells=%d  gt_cells=%d  ARI=%.4f",
            n_spots, 100 * pred_bg / n_spots, 100 * gt_bg / n_spots,
            n_pred_cells, n_gt_cells, ari,
        )

        all_pred_idx.append(fov_spots["spot_idx"].to_numpy())
        all_pred_fov.append(np.full(len(fov_spots), fov_id, dtype=object))
        all_pred_labels.append(pred)

        per_fov_rows.append({
            "fov": fov_id,
            "n_spots": n_spots,
            "n_pred_cells": n_pred_cells,
            "n_gt_cells": int(n_gt_cells),
            "pred_bg_pct": round(100 * pred_bg / n_spots, 2),
            "gt_bg_pct": round(100 * gt_bg / n_spots, 2),
            "ari": round(float(ari), 4),
            "seg_seconds": round(seg_time, 2),
        })

    # Per-FOV summary (computed with inlined sklearn call — for diagnostics)
    per_fov_df = pd.DataFrame(per_fov_rows)
    per_fov_df.to_csv(run_dir / "per_fov.csv", index=False)
    mean_ari_inline = float(per_fov_df["ari"].mean()) if len(per_fov_df) else float("nan")

    # Official metric: build solution+submission DataFrames and call
    # metric.merfish_score verbatim. Uses spot_idx as the stand-in spot_id —
    # merfish_score only requires that both frames share the same index.
    submission = pd.DataFrame({
        "spot_id": np.concatenate(all_pred_idx) if all_pred_idx else np.array([], dtype=np.int64),
        "fov": np.concatenate(all_pred_fov) if all_pred_fov else np.array([], dtype=object),
        "cluster_id": np.concatenate(all_pred_labels) if all_pred_labels else np.array([], dtype=object),
    }).set_index("spot_id")
    solution = (
        gt.rename(columns={"spot_idx": "spot_id", "gt_cluster_id": "gt_cluster_id"})
          [["spot_id", "fov", "gt_cluster_id"]]
          .set_index("spot_id")
    )
    official_mean_ari = float(_official_merfish_score(solution, submission))

    log.info("=" * 60)
    log.info("PER-FOV RESULTS")
    log.info("\n%s", per_fov_df.to_string(index=False))
    log.info("MEAN ARI (inline per-FOV reduction):  %.4f", mean_ari_inline)
    log.info("MEAN ARI (official metric.py call):   %.4f", official_mean_ari)
    if abs(mean_ari_inline - official_mean_ari) > 1e-6:
        log.warning(
            "Inline vs official ARI disagree by %.2e — investigate.",
            abs(mean_ari_inline - official_mean_ari),
        )
    log.info("=" * 60)

    summary = {
        "segmenter": factory_spec,
        "gpu": args.gpu,
        "diameter": diameter,
        "z_filter": args.z_filter,
        "segmenter_kwargs": extra_kwargs or None,
        "val_fovs": val_fovs,
        "mean_ari": round(official_mean_ari, 4),
        "mean_ari_inline": round(mean_ari_inline, 4),
        "per_fov_ari": {r["fov"]: r["ari"] for r in per_fov_rows},
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote %s", run_dir / "summary.json")


if __name__ == "__main__":
    main()
