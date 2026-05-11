"""
Phase 2.5 piece C: per-subclass cluster head trainer.

The v1 baseline trains one classifier at subclass level and rolls each
prediction up/down via deterministic majority. That makes class /
subclass / supertype / cluster all share a single partition (subclass) —
mathematically capping cluster ARI at subclass ARI.

This script trains a small LightGBM head **per subclass** (subclasses
with at least MIN_CELLS training cells AND >1 cluster), targeting
cluster_label. At inference time, when the subclass classifier predicts
subclass X with high enough confidence, we route the cell through head X
to get a sharper cluster (and supertype, when subclass spans multiple
supertypes) prediction.

Inputs:
    cache/phase2_train.npz    # X_train, y_subclass, y_cluster, ...
    phase2_val_fovs.txt
    /scratch/pl2820/.../cell_labels_train.csv   (for cluster->higher lookup)

Outputs:
    runs/phase2_clusterheads/
        cluster_to_higher.csv             # cluster -> (subclass, supertype, class)
        heads_meta.json                   # subclass -> {head_dir, n_train, n_val, ...}
        heads/<subclass_safe>/
            model.txt
            label_classes.json
            train.log
        train.log                          # outer-loop log

Usage:
    python scripts/phase2/train_cluster_heads.py \
        --min_cells 20 --run_dir runs/phase2_clusterheads
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMG_H = 2048
PIXEL_SIZE = 0.109
LABEL_LEVELS = ["class_label", "subclass_label", "supertype_label", "cluster_label"]

TRAIN_NPZ = _PROJECT_ROOT / "cache" / "phase2_train.npz"
VAL_FOVS_PATH = _PROJECT_ROOT / "phase2_val_fovs.txt"
FOV_META_CSV = Path("/scratch/pl2820/data/competition_phase2/reference/fov_metadata.csv")
LABELS_CSV = Path("/scratch/pl2820/data/competition_phase2/train/ground_truth/cell_labels_train.csv")
DEFAULT_RUN_DIR = _PROJECT_ROOT / "runs" / "phase2_clusterheads"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def normalize_counts(X: np.ndarray) -> np.ndarray:
    X = np.log1p(X.astype(np.float32))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def stage_xy(centroids: np.ndarray, fov_ids: np.ndarray,
             fov_meta: pd.DataFrame) -> np.ndarray:
    out = np.empty((len(centroids), 2), dtype=np.float32)
    for fov in np.unique(fov_ids):
        m = fov_meta.loc[fov]
        idx = np.where(fov_ids == fov)[0]
        rows = centroids[idx, 0]
        cols = centroids[idx, 1]
        out[idx, 0] = m.fov_x + (IMG_H - rows) * PIXEL_SIZE
        out[idx, 1] = m.fov_y + cols * PIXEL_SIZE
    return out


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


def safe_name(s: str) -> str:
    """Filesystem-safe slug for a subclass label."""
    return _SAFE_NAME_RE.sub("_", s).strip("_")


def build_cluster_to_higher(labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Strict-hierarchy lookup: cluster_label -> (subclass_label, supertype_label,
    class_label). Verified strict in Phase 2.1 (no cluster has ambiguous
    parents).
    """
    out = labels_df[LABEL_LEVELS].drop_duplicates(subset=["cluster_label"]).set_index("cluster_label")
    # Sanity check: each cluster should have exactly one parent at each level
    for lvl in ["subclass_label", "supertype_label", "class_label"]:
        n_unique = labels_df.groupby("cluster_label")[lvl].nunique()
        ambiguous = n_unique[n_unique > 1]
        if len(ambiguous) > 0:
            raise RuntimeError(f"Hierarchy not strict at {lvl}: {len(ambiguous)} clusters "
                               f"have multiple parents — first: {ambiguous.head().to_dict()}")
    if "background" not in out.index:
        out.loc["background"] = {lvl: "background" for lvl in LABEL_LEVELS}
    return out


def train_one_head(
    subclass: str,
    X_subcl: np.ndarray,
    y_cluster_subcl: np.ndarray,
    is_val: np.ndarray,
    head_dir: Path,
    args,
) -> dict:
    """Train one cluster head. Returns metadata dict."""
    head_dir.mkdir(parents=True, exist_ok=True)
    head_log = logging.FileHandler(head_dir / "train.log", mode="w")
    head_log.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                             datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(head_log)
    try:
        n_total = len(X_subcl)
        n_val_cells = int(is_val.sum())
        n_train_cells = n_total - n_val_cells

        unique_clusters = np.unique(y_cluster_subcl)
        log.info("[%s] n_total=%d (train=%d, val=%d) n_clusters=%d",
                 subclass, n_total, n_train_cells, n_val_cells, len(unique_clusters))

        # Label encoding using cluster string -> int over the full subset
        # (so we know all classes ahead of time, even if some only appear in val)
        cls_to_idx = {c: i for i, c in enumerate(unique_clusters)}
        y_enc = np.array([cls_to_idx[c] for c in y_cluster_subcl], dtype=np.int32)

        n_classes = len(unique_clusters)
        if n_classes < 2:
            log.info("  [%s] only %d cluster — skipping (no decision)", subclass, n_classes)
            return {"skipped": True, "reason": "single_cluster", "n_total": n_total}

        # Per-row inverse-class-frequency weights (training set only)
        train_idx = np.where(~is_val)[0]
        y_tr = y_enc[train_idx]
        cls_counts = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
        weights_per_class = len(y_tr) / (np.maximum(cls_counts, 1) * n_classes)
        sample_weight = weights_per_class[y_tr].astype(np.float32)

        # Build datasets
        X_tr = X_subcl[train_idx]
        train_set = lgb.Dataset(X_tr, label=y_tr, weight=sample_weight)

        # Use val for early stopping only if it has enough cells and at least
        # 2 distinct classes.
        val_idx = np.where(is_val)[0]
        use_val = (len(val_idx) >= 5 and
                   len(np.unique(y_enc[val_idx])) >= 2)

        params = {
            "objective": "multiclass",
            "num_class": n_classes,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "num_leaves": args.num_leaves,
            "min_data_in_leaf": max(args.min_data_in_leaf, max(1, n_train_cells // 50)),
            "metric": "multi_logloss",
            "verbosity": -1,
            "num_threads": args.num_threads,
        }

        valid_sets = [train_set]
        valid_names = ["train"]
        callbacks = [lgb.log_evaluation(period=50)]
        if use_val:
            X_va = X_subcl[val_idx]
            y_va = y_enc[val_idx]
            val_set = lgb.Dataset(X_va, label=y_va, reference=train_set)
            valid_sets.append(val_set)
            valid_names.append("val")
            callbacks.insert(0, lgb.early_stopping(args.early_stopping_rounds, verbose=False))
            n_rounds = args.n_estimators
        else:
            log.info("  [%s] val too small (%d cells, %d classes) -> training without early stopping",
                     subclass, len(val_idx), len(np.unique(y_enc[val_idx])) if len(val_idx) else 0)
            n_rounds = args.fallback_n_estimators

        t0 = time.time()
        model = lgb.train(
            params, train_set,
            num_boost_round=n_rounds,
            valid_sets=valid_sets, valid_names=valid_names,
            callbacks=callbacks,
        )
        elapsed = time.time() - t0
        best_iter = model.best_iteration if use_val else n_rounds
        log.info("  [%s] trained in %.1fs (best_iter=%d)", subclass, elapsed, best_iter)

        # Eval on val (val accuracy)
        val_acc = None
        if use_val and len(val_idx) > 0:
            X_va = X_subcl[val_idx]
            y_va = y_enc[val_idx]
            val_proba = model.predict(X_va, num_iteration=best_iter)
            val_pred = val_proba.argmax(axis=1)
            val_acc = float((val_pred == y_va).mean())
            log.info("  [%s] val acc=%.3f over %d cells", subclass, val_acc, len(val_idx))

        # Save model + label classes
        model.save_model(str(head_dir / "model.txt"), num_iteration=best_iter)
        (head_dir / "label_classes.json").write_text(
            json.dumps(list(unique_clusters.tolist()), indent=2)
        )
        return {
            "skipped": False,
            "n_total": n_total,
            "n_train": n_train_cells,
            "n_val": n_val_cells,
            "n_clusters": int(n_classes),
            "best_iter": int(best_iter),
            "val_acc": val_acc,
            "trained_with_early_stopping": bool(use_val),
        }
    finally:
        logging.getLogger().removeHandler(head_log)
        head_log.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2.5 piece C: per-subclass cluster heads")
    ap.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--min_cells", type=int, default=20,
                    help="Min training cells per subclass to train a head. Default 20.")
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--fallback_n_estimators", type=int, default=80,
                    help="Used when val is too small for early stopping.")
    ap.add_argument("--early_stopping_rounds", type=int, default=20)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--max_depth", type=int, default=5)
    ap.add_argument("--num_leaves", type=int, default=15)
    ap.add_argument("--min_data_in_leaf", type=int, default=3)
    ap.add_argument("--num_threads", type=int, default=8)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    (run_dir / "heads").mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(run_dir / "train.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                       datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    log.info("Loading %s", TRAIN_NPZ)
    data = np.load(TRAIN_NPZ, allow_pickle=True)
    X_raw = data["X_train"]
    y_subclass = data["y_subclass"].astype(str)
    y_cluster = data["y_cluster"].astype(str)
    fov_ids = data["fov_ids"].astype(str)
    centroids = data["centroids"].astype(np.float32)
    log.info("  N=%d cells, unique_subclasses=%d, unique_clusters=%d",
             len(y_subclass), len(np.unique(y_subclass)), len(np.unique(y_cluster)))

    val_fovs = [ln.strip() for ln in open(VAL_FOVS_PATH) if ln.strip()]
    log.info("Val FOVs (%d): %s", len(val_fovs), val_fovs)

    fov_meta = pd.read_csv(FOV_META_CSV).set_index("fov")

    log.info("Building features (log1p+L2 + stage coords)...")
    X_norm = normalize_counts(X_raw)
    stage = stage_xy(centroids, fov_ids, fov_meta)
    X = np.concatenate([X_norm, stage], axis=1).astype(np.float32)
    log.info("  X.shape=%s", X.shape)

    val_mask_global = np.isin(fov_ids, val_fovs)
    log.info("  global train cells=%d, val cells=%d",
             int((~val_mask_global).sum()), int(val_mask_global.sum()))

    log.info("Building cluster->higher lookup from %s", LABELS_CSV)
    labels_df = pd.read_csv(LABELS_CSV)
    cluster_to_higher = build_cluster_to_higher(labels_df)
    cluster_to_higher.to_csv(run_dir / "cluster_to_higher.csv")
    log.info("  Saved cluster_to_higher.csv (%d clusters)", len(cluster_to_higher))

    # Decide which subclasses get a head: >= min_cells training cells AND >1 cluster
    rows = []
    for subcl in np.unique(y_subclass):
        if subcl == "background":
            continue
        in_subcl = (y_subclass == subcl)
        in_train = in_subcl & ~val_mask_global
        n_train = int(in_train.sum())
        n_clusters = int(np.unique(y_cluster[in_subcl]).size)
        eligible = (n_train >= args.min_cells) and (n_clusters > 1)
        rows.append({
            "subclass": subcl,
            "n_total": int(in_subcl.sum()),
            "n_train": n_train,
            "n_val": int((in_subcl & val_mask_global).sum()),
            "n_clusters": n_clusters,
            "eligible": eligible,
        })
    eligibility = pd.DataFrame(rows).sort_values("n_train", ascending=False)
    log.info("=== Subclass eligibility (min_cells=%d) ===", args.min_cells)
    for _, r in eligibility.iterrows():
        flag = "+" if r["eligible"] else " "
        log.info("  %s %-30s n_train=%4d n_val=%3d n_clusters=%2d",
                 flag, r["subclass"], r["n_train"], r["n_val"], r["n_clusters"])
    n_eligible = int(eligibility["eligible"].sum())
    log.info("Will train %d cluster heads (out of %d named subclasses)",
             n_eligible, len(eligibility))

    # Train one head per eligible subclass
    heads_meta: dict[str, dict] = {}
    t_total = time.time()
    for _, r in eligibility[eligibility["eligible"]].iterrows():
        subcl = r["subclass"]
        in_subcl = (y_subclass == subcl)
        X_sub = X[in_subcl]
        y_cl = y_cluster[in_subcl]
        is_val_sub = val_mask_global[in_subcl]
        head_dir = run_dir / "heads" / safe_name(subcl)
        meta = train_one_head(subcl, X_sub, y_cl, is_val_sub, head_dir, args)
        meta["head_dir"] = str(head_dir.relative_to(_PROJECT_ROOT))
        heads_meta[subcl] = meta

    # Write top-level meta
    out_meta = {
        "min_cells": args.min_cells,
        "n_eligible_subclasses": n_eligible,
        "n_named_subclasses": len(eligibility),
        "feature_pipeline": {
            "n_genes": int(X_norm.shape[1]),
            "n_extra_features": int(stage.shape[1]),
            "extra_feature_names": ["global_x_um", "global_y_um"],
            "normalization": "log1p_then_l2",
            "pixel_size_um": PIXEL_SIZE,
        },
        "subclass_to_head": heads_meta,
    }
    (run_dir / "heads_meta.json").write_text(json.dumps(out_meta, indent=2, default=str))
    log.info("Wrote heads_meta.json with %d entries", len(heads_meta))
    log.info("=== Trained %d heads in %.1fs ===", len(heads_meta), time.time() - t_total)


if __name__ == "__main__":
    main()
