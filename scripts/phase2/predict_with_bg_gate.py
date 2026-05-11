"""
Phase 2.9 step 2: hierarchical predictor — bg gate then v3 pipeline.

For each cell:
  - Compute v1-style features (1147 log1p+L2 genes + 2 stage coords)
  - p_named = bg_gate.predict(X)
  - If p_named < T: predict background at all 4 levels
  - Else: route through v3's existing subclass classifier + cluster heads.
    If the subclass classifier internally predicts 'background', the cell
    gets bg labels (the gate is a NECESSARY but not SUFFICIENT condition
    for being named).

T defaults to 0.5; do not sweep.

Usage:
    python scripts/phase2/predict_with_bg_gate.py \
        --subclass_run_dir runs/phase2_baseline \
        --bggate_run_dir runs/phase2_bggate \
        --cluster_heads_dir runs/phase2_clusterheads \
        --output submissions/phase2_v8_bggate.csv
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as feat  # noqa: E402
import predict as pred_mod  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/scratch/pl2820/data/competition_phase2")
DEFAULT_TEST_SPOTS = DATA_ROOT / "test_spots.csv"
DEFAULT_SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"
DEFAULT_TEST_FOVS = ["FOV_E", "FOV_F", "FOV_G", "FOV_H", "FOV_I", "FOV_J",
                     "FOV_K", "FOV_L", "FOV_M", "FOV_N"]
FOV_META_CSV = DATA_ROOT / "reference" / "fov_metadata.csv"
EXPR_DIR = _PROJECT_ROOT / "cache" / "expression_phase2"
MASK_DIR = _PROJECT_ROOT / "cache" / "masks_phase2"
SUBMISSION_LEVELS = ["class", "subclass", "supertype", "cluster"]
BG = "background"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def predict_fov_with_gate(fov_id, mask, subclass_model, subclass_le,
                           bg_gate, promotion, fov_meta,
                           subclass_feature_cfg, gate_feature_cfg,
                           cluster_heads, T):
    """Returns dict cell_id -> labels."""
    expr_path = EXPR_DIR / f"{fov_id}.npz"
    data = np.load(expr_path, allow_pickle=False)
    matrix = data["matrix"]
    cell_ids = data["cell_ids"]
    centroids = data["centroids"]
    n = len(cell_ids)
    if n == 0:
        return {}

    m = fov_meta.loc[fov_id]
    # subclass classifier features
    X_sub = pred_mod.build_features_for_fov(
        matrix, centroids, mask, m.fov_x, m.fov_y, cell_ids,
        extras_keep=subclass_feature_cfg["extras_keep"],
        neighbor_K=subclass_feature_cfg["neighbor_K"],
        fov_id=fov_id,
    )
    # gate features (same shape as baseline)
    if gate_feature_cfg == subclass_feature_cfg:
        X_gate = X_sub
    else:
        X_gate = pred_mod.build_features_for_fov(
            matrix, centroids, mask, m.fov_x, m.fov_y, cell_ids,
            extras_keep=gate_feature_cfg["extras_keep"],
            neighbor_K=gate_feature_cfg["neighbor_K"],
            fov_id=fov_id,
        )

    p_named = bg_gate.predict(X_gate)
    pass_gate = p_named >= T

    proba = subclass_model.predict(X_sub)
    pred_idx = proba.argmax(axis=1)
    pred_label = subclass_le.classes_[pred_idx]

    n_gate_pass = int(pass_gate.sum())
    n_gate_fail = n - n_gate_pass
    n_subclass_named = int(np.sum(pred_label != BG))
    n_combined_named = int(np.sum(pass_gate & (pred_label != BG)))
    log.info("  %s: n_cells=%d  gate_pass=%d (%.1f%%)  subclass_named=%d  combined_named=%d",
             fov_id, n, n_gate_pass, 100*n_gate_pass/max(n,1),
             n_subclass_named, n_combined_named)

    out = {}
    for i, cid in enumerate(cell_ids):
        if (not pass_gate[i]) or (pred_label[i] == BG) or (pred_label[i] not in promotion.index):
            out[int(cid)] = {sub: BG for sub in SUBMISSION_LEVELS}
        else:
            row = promotion.loc[pred_label[i]]
            out[int(cid)] = {
                "class":     row["class_label"],
                "subclass":  row["subclass_label"],
                "supertype": row["supertype_label"],
                "cluster":   row["cluster_label"],
            }

    if cluster_heads is not None:
        eligible = pass_gate & (pred_label != BG)
        for subclass in np.unique(pred_label[eligible]):
            if not cluster_heads.has_head(subclass):
                continue
            sm = eligible & (pred_label == subclass)
            idx = np.where(sm)[0]
            if len(idx) == 0:
                continue
            cluster_preds, _ = cluster_heads.predict_clusters(subclass, X_sub[idx])
            for cid, cluster in zip(cell_ids[idx], cluster_preds):
                lbls = cluster_heads.labels_from_cluster(cluster)
                if lbls is not None:
                    out[int(cid)] = lbls
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subclass_run_dir", default="runs/phase2_baseline")
    ap.add_argument("--bggate_run_dir", default="runs/phase2_bggate")
    ap.add_argument("--cluster_heads_dir", default="runs/phase2_clusterheads")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--fovs", nargs="*", default=None)
    ap.add_argument("--spots_csv", default=str(DEFAULT_TEST_SPOTS))
    ap.add_argument("--sample_submission", default=str(DEFAULT_SAMPLE_SUB))
    ap.add_argument("--no_sample_align", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sub_dir = Path(args.subclass_run_dir)
    gate_dir = Path(args.bggate_run_dir)

    sub_meta = json.loads((sub_dir / "feature_meta.json").read_text())
    gate_meta = json.loads((gate_dir / "feature_meta.json").read_text())
    with open(sub_dir / "label_encoder.pkl", "rb") as f:
        subclass_le = pickle.load(f)
    subclass_model = lgb.Booster(model_file=str(sub_dir / "model.txt"))
    bg_gate = lgb.Booster(model_file=str(gate_dir / "model.txt"))
    promotion = pd.read_csv(sub_dir / "promotion_lookup.csv").set_index(
        sub_meta["trained_label_level"], drop=False)

    sub_feature_cfg = {"extras_keep": sub_meta["extra_feature_names"],
                        "neighbor_K": int(sub_meta.get("neighbor_K", 0))}
    gate_feature_cfg = {"extras_keep": gate_meta["extra_feature_names"],
                         "neighbor_K": int(gate_meta.get("neighbor_K", 0))}
    log.info("Subclass: %s (extras=%s, K=%d)", sub_dir.name,
             sub_feature_cfg["extras_keep"], sub_feature_cfg["neighbor_K"])
    log.info("BG gate:  %s (extras=%s, K=%d)", gate_dir.name,
             gate_feature_cfg["extras_keep"], gate_feature_cfg["neighbor_K"])
    log.info("Threshold T=%.2f", args.threshold)

    cluster_heads = pred_mod.ClusterHeads(Path(args.cluster_heads_dir)) \
        if args.cluster_heads_dir else None

    fov_meta = pd.read_csv(FOV_META_CSV).set_index("fov")
    fovs = args.fovs if args.fovs else DEFAULT_TEST_FOVS
    spots = pd.read_csv(args.spots_csv)
    if "fov" in spots.columns:
        spots = spots[spots["fov"].isin(fovs)].reset_index(drop=True)

    parts = []
    t0 = time.time()
    for i, fov_id in enumerate(fovs, 1):
        log.info("[%d/%d] %s", i, len(fovs), fov_id)
        mask = np.load(MASK_DIR / f"{fov_id}.npy")
        cell_to_labels = predict_fov_with_gate(
            fov_id, mask, subclass_model, subclass_le, bg_gate,
            promotion, fov_meta, sub_feature_cfg, gate_feature_cfg,
            cluster_heads, args.threshold,
        )
        fov_spots = spots[spots["fov"] == fov_id] if "fov" in spots.columns else spots
        sub = pred_mod.build_submission_for_fov(fov_id, mask, cell_to_labels, fov_spots)
        parts.append(sub)
    log.info("Per-FOV done in %.1fs", time.time() - t0)

    submission = pd.concat(parts, ignore_index=True)
    if not args.no_sample_align:
        sample = pd.read_csv(args.sample_submission)
        submission = submission.set_index("spot_id").reindex(sample["spot_id"]).reset_index()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    log.info("Wrote %s (%d rows)", args.output, len(submission))

    # Distribution summary
    log.info("=== Label distribution per level ===")
    for lvl in SUBMISSION_LEVELS:
        bg_n = int((submission[lvl] == BG).sum())
        n_uniq = submission[lvl].nunique()
        log.info("  %-9s background=%d (%.1f%%)  unique=%d",
                 lvl, bg_n, 100*bg_n/len(submission), n_uniq)


if __name__ == "__main__":
    main()
