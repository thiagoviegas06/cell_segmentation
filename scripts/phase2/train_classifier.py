"""
Phase 2.4 step 1: train a LightGBM multiclass classifier on the FINEST
hierarchy level (cluster). Parents (supertype, subclass, class) are looked
up deterministically at inference time from the training table.

Inputs:
    cache/phase2_train.npz       # X_train (n,1147), y_cluster (n,), fov_ids (n,),
                                 # centroids (n,2), match_dist_px (n,)
    phase2_val_fovs.txt          # 10 FOVs held out of training
    cache/gene_vocab.json        # 1147 gene names (for feature naming)

Feature pipeline:
    1. raw counts -> log1p
    2. per-cell L2 normalize (across the 1147 gene dims only)
    3. append 2 stage-coordinate features:
       global_x = fov_x + (2048 - centroid_row) * pixel_size
       global_y = fov_y +         centroid_col * pixel_size
       (CCF was requested but isn't computable for test cells; stage coords
        play the same brain-region role and ARE computable for test FOVs.)

Outputs:
    runs/phase2_baseline/
        model.txt               # LightGBM Booster save (text)
        label_encoder.pkl       # cluster_label string <-> int
        feature_meta.json       # n_genes, n_extra_features, normalization params
        train_split.json        # train_fovs, val_fovs, n_train, n_val
        train.log               # full stdout
        val_predictions.npz     # y_true, y_pred, val_fov_ids, val_pred_cell_ids
        per_class_report.csv    # precision/recall/f1/support per cluster
"""

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Constants duplicated from scripts/phase2/build_expression.py — kept inline
# to avoid tangling sys.path / package layout for what is effectively a
# script entry point.
LABEL_LEVELS = ["class_label", "subclass_label", "supertype_label", "cluster_label"]
IMG_H = 2048
PIXEL_SIZE = 0.109  # µm/px

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TRAIN_NPZ = _PROJECT_ROOT / "cache" / "phase2_train.npz"
VAL_FOVS_PATH = _PROJECT_ROOT / "phase2_val_fovs.txt"
FOV_META_CSV = Path("/scratch/pl2820/data/competition_phase2/reference/fov_metadata.csv")
LABELS_CSV = Path("/scratch/pl2820/data/competition_phase2/train/ground_truth/cell_labels_train.csv")
DEFAULT_RUN_DIR = _PROJECT_ROOT / "runs" / "phase2_baseline"


def normalize_counts(X: np.ndarray) -> np.ndarray:
    """log1p then per-cell L2 normalize across the 1147 gene dims."""
    X = np.log1p(X.astype(np.float32))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def stage_xy(centroids: np.ndarray, fov_ids: np.ndarray,
             fov_meta: pd.DataFrame) -> np.ndarray:
    """Convert per-cell (image_row, image_col) centroids to (global_x, global_y) µm."""
    out = np.empty((len(centroids), 2), dtype=np.float32)
    for fov in np.unique(fov_ids):
        m = fov_meta.loc[fov]
        idx = np.where(fov_ids == fov)[0]
        rows = centroids[idx, 0]
        cols = centroids[idx, 1]
        out[idx, 0] = m.fov_x + (IMG_H - rows) * PIXEL_SIZE
        out[idx, 1] = m.fov_y + cols * PIXEL_SIZE
    return out


def build_promotion_lookup(labels_df: pd.DataFrame, level: str) -> pd.DataFrame:
    """
    Map each value of `level` to all 4 hierarchy labels.

    For levels above `level`: deterministic parent (strict hierarchy).
    For levels below `level`: most-common child within that group.

    Always inject a 'background' row that maps to all-background.
    """
    if level == "cluster_label":
        # Strict parent lookup, one row per cluster
        out = labels_df[LABEL_LEVELS].drop_duplicates(subset=[level]).set_index(level)
    else:
        rows = []
        for val, grp in labels_df.groupby(level, sort=False):
            row = {}
            for lvl in LABEL_LEVELS:
                if lvl == level:
                    row[lvl] = val
                else:
                    row[lvl] = grp[lvl].mode().iloc[0]   # most common
            rows.append(row)
        out = pd.DataFrame(rows).set_index(level)

    if "background" not in out.index:
        out.loc["background"] = {lvl: "background" for lvl in LABEL_LEVELS}
    # Sanity: strict-hierarchy upward must hold for the chosen level (already verified at cluster)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2.4 step 1: hierarchy classifier")
    ap.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--label_level", default="subclass_label",
                    choices=LABEL_LEVELS,
                    help="Which hierarchy level to train on. cluster_label is too sparse for "
                         "5K cells / 198 classes; subclass_label is the recommended default.")
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--num_leaves", type=int, default=31)
    ap.add_argument("--early_stopping_rounds", type=int, default=30)
    ap.add_argument("--min_data_in_leaf", type=int, default=10)
    ap.add_argument("--num_threads", type=int, default=8)
    ap.add_argument("--use_balanced_weights", action="store_true", default=True,
                    help="Per-row inverse-class-frequency weights (replaces lgb's "
                         "non-existent class_weight param)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Add a file handler so the full log is captured on disk
    fh = logging.FileHandler(run_dir / "train.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                       datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    # ---- Load training data and val split
    log.info("Loading %s", TRAIN_NPZ)
    data = np.load(TRAIN_NPZ, allow_pickle=True)
    X_raw = data["X_train"]                    # (N, 1147) int32
    label_key = "y_" + args.label_level.replace("_label", "")
    y_target = data[label_key].astype(str)     # (N,) string labels
    fov_ids = data["fov_ids"].astype(str)
    centroids = data["centroids"].astype(np.float32)
    log.info("  X_train shape=%s, training on %s (unique=%d), FOVs=%d",
             X_raw.shape, args.label_level, len(np.unique(y_target)), len(np.unique(fov_ids)))

    val_fovs = [ln.strip() for ln in open(VAL_FOVS_PATH) if ln.strip()]
    log.info("Val FOVs (%d): %s", len(val_fovs), val_fovs)

    fov_meta = pd.read_csv(FOV_META_CSV).set_index("fov")

    # ---- Feature pipeline
    log.info("Building features (log1p + L2 normalize + 2 stage coords)...")
    X_norm = normalize_counts(X_raw)                          # (N, 1147)
    stage = stage_xy(centroids, fov_ids, fov_meta)            # (N, 2)
    X = np.concatenate([X_norm, stage], axis=1).astype(np.float32)
    log.info("  X.shape=%s", X.shape)

    # Group split: cells from val FOVs go to val
    val_mask = np.isin(fov_ids, val_fovs)
    train_mask = ~val_mask
    X_tr, y_tr_str = X[train_mask], y_target[train_mask]
    X_va, y_va_str = X[val_mask], y_target[val_mask]
    log.info("  train cells=%d, val cells=%d", len(X_tr), len(X_va))
    log.info("  train labels=%d, val labels=%d (overlap=%d)",
             len(np.unique(y_tr_str)), len(np.unique(y_va_str)),
             len(set(y_tr_str) & set(y_va_str)))

    # ---- Label encoding (fit on TRAIN only so encoder size = train classes)
    le = LabelEncoder()
    le.fit(y_tr_str)
    y_tr = le.transform(y_tr_str)
    # Map val labels: any val cluster not seen in train collapses to 'background' for scoring
    bg_class = "background"
    if bg_class not in le.classes_:
        # Defensive — should never happen given the data
        raise RuntimeError("'background' missing from training clusters; sanity-check failed")
    bg_idx = int(np.where(le.classes_ == bg_class)[0][0])
    y_va = np.array([
        le.transform([y])[0] if y in le.classes_ else bg_idx
        for y in y_va_str
    ])
    n_unmapped = int((np.array([y not in le.classes_ for y in y_va_str])).sum())
    log.info("  val cells whose label wasn't seen in train (mapped -> background for scoring): %d",
             n_unmapped)

    # Per-row sample weights (lgb.train doesn't honor sklearn-style class_weight).
    sample_weight = None
    if args.use_balanced_weights:
        cls_counts = np.bincount(y_tr, minlength=len(le.classes_)).astype(np.float64)
        # sklearn-style: w = n_samples / (n_classes * count)
        weights_per_class = len(y_tr) / (np.maximum(cls_counts, 1) * len(le.classes_))
        sample_weight = weights_per_class[y_tr].astype(np.float32)
        log.info("  balanced sample_weight: per-class weight min=%.3f median=%.3f max=%.3f",
                 weights_per_class.min(), float(np.median(weights_per_class)),
                 weights_per_class.max())

    # ---- Train LightGBM
    n_classes = len(le.classes_)
    log.info("Training LightGBM multiclass: n_classes=%d, n_estimators=%d, lr=%.3f, "
             "max_depth=%d, num_leaves=%d, early_stop=%d",
             n_classes, args.n_estimators, args.learning_rate,
             args.max_depth, args.num_leaves, args.early_stopping_rounds)

    train_set = lgb.Dataset(X_tr, label=y_tr, weight=sample_weight)
    val_set = lgb.Dataset(X_va, label=y_va, reference=train_set)

    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "metric": "multi_logloss",
        "verbosity": -1,
        "num_threads": args.num_threads,
    }

    t0 = time.time()
    model = lgb.train(
        params,
        train_set,
        num_boost_round=args.n_estimators,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=20),
        ],
    )
    log.info("LightGBM trained in %.1fs (best_iter=%d, best_val_logloss=%.4f)",
             time.time() - t0, model.best_iteration, model.best_score["val"]["multi_logloss"])

    # ---- Eval on val
    log.info("Predicting val...")
    val_proba = model.predict(X_va, num_iteration=model.best_iteration)  # (n_val, n_classes)
    val_pred_idx = val_proba.argmax(axis=1)
    val_pred_str = le.classes_[val_pred_idx]

    # Per-class report (string labels, on val)
    log.info("=== Per-class precision/recall (val, cluster level) ===")
    report_dict = classification_report(
        y_va_str, val_pred_str, output_dict=True, zero_division=0
    )
    rep_df = pd.DataFrame(report_dict).T
    rep_df.index.name = "label"
    # Print top 30 by support
    rep_df_sorted = rep_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore").copy()
    rep_df_sorted["support"] = rep_df_sorted["support"].astype(int)
    rep_df_sorted = rep_df_sorted.sort_values("support", ascending=False)
    log.info("Top 30 clusters by val support:")
    for label, row in rep_df_sorted.head(30).iterrows():
        log.info("  %-44s prec=%.2f  recall=%.2f  f1=%.2f  n=%d",
                 label, row["precision"], row["recall"], row["f1-score"], row["support"])
    overall = rep_df.loc["weighted avg"]
    log.info("WEIGHTED AVG  prec=%.3f  recall=%.3f  f1=%.3f",
             overall["precision"], overall["recall"], overall["f1-score"])
    if "accuracy" in rep_df.index:
        log.info("ACCURACY %.3f", float(rep_df.loc["accuracy"]["f1-score"]))

    # ---- Save artifacts
    model_path = run_dir / "model.txt"
    model.save_model(str(model_path), num_iteration=model.best_iteration)
    log.info("Saved model -> %s", model_path)

    with open(run_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    feat_meta = {
        "n_genes": X_norm.shape[1],
        "n_extra_features": stage.shape[1],
        "extra_feature_names": ["global_x_um", "global_y_um"],
        "normalization": "log1p_then_l2",
        "pixel_size_um": PIXEL_SIZE,
    }
    with open(run_dir / "feature_meta.json", "w") as f:
        json.dump(feat_meta, f, indent=2)

    split_meta = {
        "val_fovs": val_fovs,
        "n_train_cells": int(train_mask.sum()),
        "n_val_cells": int(val_mask.sum()),
        "n_train_clusters": int(len(np.unique(y_tr_str))),
        "n_val_clusters": int(len(np.unique(y_va_str))),
        "best_iteration": int(model.best_iteration),
        "best_val_multi_logloss": float(model.best_score["val"]["multi_logloss"]),
    }
    with open(run_dir / "train_split.json", "w") as f:
        json.dump(split_meta, f, indent=2)

    # Save val predictions for the local Phase 2 evaluator to consume
    val_cell_ids = data["cell_ids"][val_mask]
    val_pred_cluster_top1 = val_pred_str
    np.savez(
        run_dir / "val_predictions.npz",
        y_true_cluster=y_va_str,
        y_pred_cluster=val_pred_cluster_top1,
        val_fov_ids=fov_ids[val_mask],
        val_cell_ids=val_cell_ids,
    )
    rep_df.to_csv(run_dir / "per_class_report.csv")
    log.info("Saved per_class_report.csv (%d rows)", len(rep_df))

    # Build + save promotion lookup so predict.py doesn't need cell_labels_train.csv.
    # Maps the predicted label_level value to all 4 hierarchy levels.
    labels_df = pd.read_csv(LABELS_CSV)
    h = build_promotion_lookup(labels_df, args.label_level)
    h.to_csv(run_dir / "promotion_lookup.csv")
    log.info("Saved promotion_lookup.csv (%d %s -> 4 levels)",
             len(h), args.label_level)
    # Persist which level we trained on so predict.py picks the right column.
    feat_meta["trained_label_level"] = args.label_level
    with open(run_dir / "feature_meta.json", "w") as f:
        json.dump(feat_meta, f, indent=2)

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
