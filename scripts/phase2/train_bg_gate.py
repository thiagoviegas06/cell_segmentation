"""
Phase 2.9 step 1: train a binary bg-vs-named LightGBM classifier on the
SAME features as v1 baseline (1147 log1p+L2 genes + 2 stage coords).

This is the gate stage of the hierarchical predictor: cells that pass
the gate are routed through v3's existing subclass + cluster-head
pipeline; cells that fail are predicted as all-background.

Outputs runs/phase2_bggate/{model.txt, feature_meta.json,
train_split.json, val_predictions.npz}.

Usage:
    python scripts/phase2/train_bg_gate.py --run_dir runs/phase2_bggate
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as feat  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_NPZ = _PROJECT_ROOT / "cache" / "phase2_train.npz"
VAL_FOVS_PATH = _PROJECT_ROOT / "phase2_val_fovs.txt"
FOV_META_CSV = Path("/scratch/pl2820/data/competition_phase2/reference/fov_metadata.csv")
DEFAULT_RUN_DIR = _PROJECT_ROOT / "runs" / "phase2_bggate"
PIXEL_SIZE = 0.109

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def fov_xy_per_cell(fov_ids, fov_meta):
    fx = np.empty(len(fov_ids), dtype=np.float32)
    fy = np.empty(len(fov_ids), dtype=np.float32)
    for fov in np.unique(fov_ids):
        m = fov_meta.loc[fov]
        idx = np.where(fov_ids == fov)[0]
        fx[idx] = m.fov_x
        fy[idx] = m.fov_y
    return fx, fy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--num_leaves", type=int, default=31)
    ap.add_argument("--min_data_in_leaf", type=int, default=10)
    ap.add_argument("--early_stopping_rounds", type=int, default=30)
    ap.add_argument("--num_threads", type=int, default=8)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "train.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                       datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    log.info("Loading %s", TRAIN_NPZ)
    data = np.load(TRAIN_NPZ, allow_pickle=True)
    X_raw = data["X_train"]
    fov_ids = data["fov_ids"].astype(str)
    centroids = data["centroids"].astype(np.float32)
    y_subclass = data["y_subclass"].astype(str)
    mask_volume_px = data["mask_volume_px"].astype(np.int64)

    val_fovs = [ln.strip() for ln in open(VAL_FOVS_PATH) if ln.strip()]
    fov_meta = pd.read_csv(FOV_META_CSV).set_index("fov")

    # Same feature pipeline as baseline: log1p+L2 + global_x_um, global_y_um
    log.info("Building features (1147 log1p+L2 genes + 2 stage coords)")
    X_norm = feat.normalize_counts(X_raw)
    fov_x, fov_y = fov_xy_per_cell(fov_ids, fov_meta)
    extras_full = feat.build_extra_features(X_raw, centroids, mask_volume_px,
                                             fov_x, fov_y)
    extras, extras_names = feat.select_extras(extras_full,
                                               ["global_x_um", "global_y_um"])
    X = np.concatenate([X_norm, extras], axis=1).astype(np.float32)
    log.info("  X.shape=%s", X.shape)

    # Binary label: 1 = named, 0 = background
    y = (y_subclass != "background").astype(np.int32)
    log.info("  Class balance: named=%d (%.1f%%), bg=%d (%.1f%%)",
             y.sum(), 100*y.mean(), (1-y).sum(), 100*(1-y.mean()))

    # Train/val split (same as baseline)
    val_mask = np.isin(fov_ids, val_fovs)
    train_mask = ~val_mask
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[val_mask], y[val_mask]
    log.info("  train=%d (named=%d), val=%d (named=%d)",
             len(X_tr), int(y_tr.sum()), len(X_va), int(y_va.sum()))

    # Train LightGBM binary
    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_va, label=y_va, reference=train_set)

    params = {
        "objective": "binary",
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "metric": ["binary_logloss", "auc"],
        "verbosity": -1,
        "num_threads": args.num_threads,
    }

    log.info("Training binary LightGBM: lr=%.3f, max_depth=%d, num_leaves=%d",
             args.learning_rate, args.max_depth, args.num_leaves)
    t0 = time.time()
    model = lgb.train(
        params, train_set,
        num_boost_round=args.n_estimators,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=20),
        ],
    )
    log.info("Trained in %.1fs (best_iter=%d, val_logloss=%.4f, val_auc=%.4f)",
             time.time() - t0, model.best_iteration,
             model.best_score["val"]["binary_logloss"],
             model.best_score["val"]["auc"])

    # Val predictions
    p_named = model.predict(X_va, num_iteration=model.best_iteration)
    pred = (p_named >= 0.5).astype(np.int32)
    tp = int(((pred == 1) & (y_va == 1)).sum())
    fp = int(((pred == 1) & (y_va == 0)).sum())
    fn = int(((pred == 0) & (y_va == 1)).sum())
    tn = int(((pred == 0) & (y_va == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    log.info("Val @ T=0.5: TP=%d FP=%d FN=%d TN=%d  precision=%.3f recall=%.3f",
             tp, fp, fn, tn, prec, rec)
    log.info("  Named-call rate: pred=%.1f%% (true=%.1f%%)",
             100*pred.mean(), 100*y_va.mean())

    # Save
    model.save_model(str(run_dir / "model.txt"),
                     num_iteration=model.best_iteration)
    np.savez(run_dir / "val_predictions.npz",
             y_true=y_va, p_named=p_named,
             val_fov_ids=fov_ids[val_mask])
    feat_meta = {
        "n_genes": int(X_norm.shape[1]),
        "n_extra_features": int(extras.shape[1]),
        "extra_feature_names": extras_names,
        "neighbor_K": 0,
        "n_neighbor_features": 0,
        "normalization": "log1p_then_l2",
        "pixel_size_um": PIXEL_SIZE,
        "objective": "binary_bg_vs_named",
    }
    with open(run_dir / "feature_meta.json", "w") as f:
        json.dump(feat_meta, f, indent=2)
    log.info("Saved -> %s", run_dir)


if __name__ == "__main__":
    main()
