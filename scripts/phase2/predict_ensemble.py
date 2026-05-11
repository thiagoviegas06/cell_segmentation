"""
Ensemble prediction: average probability outputs of multiple subclass-level
LightGBM classifiers, then take argmax. Optional cluster-head stage on top.

Each --run_dir contains a model trained at subclass_label with its own
feature_meta.json (n_genes, extras, neighbor_K). All models must share the
same subclass label space (verified at load time).

Usage:
    python scripts/phase2/predict_ensemble.py \
        --run_dirs runs/phase2_baseline runs/phase2_g_dropout \
        --output submissions/phase2_v8_ensemble.csv

    # val pass
    python scripts/phase2/predict_ensemble.py \
        --run_dirs runs/phase2_baseline runs/phase2_g_dropout \
        --fovs $(cat phase2_val_fovs.txt) \
        --spots_csv /path/to/spots_train.csv \
        --output runs/ensemble_val.csv \
        --no_sample_align
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


def load_member(run_dir: Path):
    meta = json.loads((run_dir / "feature_meta.json").read_text())
    with open(run_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    booster = lgb.Booster(model_file=str(run_dir / "model.txt"))
    promotion = pd.read_csv(run_dir / "promotion_lookup.csv").set_index(
        meta["trained_label_level"])
    return {
        "run_dir": run_dir,
        "meta": meta,
        "le": le,
        "model": booster,
        "promotion": promotion,
        "extras_keep": meta["extra_feature_names"],
        "neighbor_K": int(meta.get("neighbor_K", 0)),
    }


def predict_fov_ensemble(fov_id, mask, members, weights, fov_meta,
                          cluster_heads=None, head_feature_member=0):
    """Average probs across members for one FOV. Returns dict cell_id -> labels."""
    expr_path = EXPR_DIR / f"{fov_id}.npz"
    data = np.load(expr_path, allow_pickle=False)
    matrix = data["matrix"]
    cell_ids = data["cell_ids"]
    centroids = data["centroids"]
    n = len(cell_ids)
    if n == 0:
        return {}, None, None

    m = fov_meta.loc[fov_id]
    # Per-member features + probs (pad members may have different feature dims)
    union_classes = members[0]["le"].classes_
    avg_proba = np.zeros((n, len(union_classes)), dtype=np.float64)
    member_feats = []
    for mem, w in zip(members, weights):
        X = pred_mod.build_features_for_fov(matrix, centroids, mask,
                                             m.fov_x, m.fov_y, cell_ids,
                                             extras_keep=mem["extras_keep"],
                                             neighbor_K=mem["neighbor_K"],
                                             fov_id=fov_id)
        proba = mem["model"].predict(X)
        # Map to union class space (all members share class set; verified at load)
        if not np.array_equal(mem["le"].classes_, union_classes):
            # Reorder columns to union order
            idx = np.array([np.where(mem["le"].classes_ == c)[0][0]
                            for c in union_classes])
            proba = proba[:, idx]
        avg_proba += w * proba
        member_feats.append(X)
    avg_proba /= sum(weights)

    pred_idx = avg_proba.argmax(axis=1)
    pred_label = union_classes[pred_idx]
    max_proba = avg_proba.max(axis=1)

    # Use first member's promotion table (all share since trained on same data)
    promotion = members[0]["promotion"]
    out = {}
    for cid, val, p in zip(cell_ids, pred_label, max_proba):
        if val == BG or val not in promotion.index:
            out[int(cid)] = {sub: BG for sub in SUBMISSION_LEVELS}
        else:
            row = promotion.loc[val]
            out[int(cid)] = {
                "class":     row["class_label"],
                "subclass":  row["subclass_label"],
                "supertype": row["supertype_label"],
                "cluster":   row["cluster_label"],
            }

    if cluster_heads is not None:
        # Cluster heads were trained on baseline features — use that member's X.
        X_for_heads = member_feats[head_feature_member]
        eligible = pred_label != BG
        for subclass in np.unique(pred_label[eligible]):
            if not cluster_heads.has_head(subclass):
                continue
            sm = eligible & (pred_label == subclass)
            idx = np.where(sm)[0]
            if len(idx) == 0:
                continue
            cluster_preds, head_proba = cluster_heads.predict_clusters(
                subclass, X_for_heads[idx])
            for cid, cluster in zip(cell_ids[idx], cluster_preds):
                lbls = cluster_heads.labels_from_cluster(cluster)
                if lbls is not None:
                    out[int(cid)] = lbls
    return out, pred_label, cell_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True,
                    help="Two or more model run dirs to ensemble.")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="Per-member weight (default: equal).")
    ap.add_argument("--cluster_heads_dir", default=None,
                    help="Optional path to runs/phase2_clusterheads.")
    ap.add_argument("--head_feature_member", type=int, default=0,
                    help="Which member's features to feed cluster heads "
                         "(must match how heads were trained, default=0).")
    ap.add_argument("--fovs", nargs="*", default=None)
    ap.add_argument("--spots_csv", default=str(DEFAULT_TEST_SPOTS))
    ap.add_argument("--sample_submission", default=str(DEFAULT_SAMPLE_SUB))
    ap.add_argument("--no_sample_align", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    members = [load_member(Path(p)) for p in args.run_dirs]
    weights = args.weights if args.weights else [1.0] * len(members)
    if len(weights) != len(members):
        raise ValueError("len(weights) must equal len(run_dirs)")
    log.info("Ensemble of %d members:", len(members))
    for mem, w in zip(members, weights):
        log.info("  %s: extras=%s neighbor_K=%d weight=%.2f",
                 mem["run_dir"].name, mem["extras_keep"], mem["neighbor_K"], w)

    # Verify shared class space
    base_classes = members[0]["le"].classes_
    for mem in members[1:]:
        if set(mem["le"].classes_) != set(base_classes):
            raise ValueError(f"Class set mismatch between {members[0]['run_dir']} and {mem['run_dir']}")

    fov_meta = pd.read_csv(FOV_META_CSV).set_index("fov")
    fovs = args.fovs if args.fovs else DEFAULT_TEST_FOVS

    cluster_heads = None
    if args.cluster_heads_dir:
        cluster_heads = pred_mod.ClusterHeads(Path(args.cluster_heads_dir))

    # Load spots
    log.info("Loading spots from %s", args.spots_csv)
    spots = pd.read_csv(args.spots_csv)
    if "fov" in spots.columns:
        spots = spots[spots["fov"].isin(fovs)].reset_index(drop=True)

    submission_parts = []
    t0 = time.time()
    for i, fov_id in enumerate(fovs, 1):
        log.info("[%d/%d] %s", i, len(fovs), fov_id)
        mask_path = MASK_DIR / f"{fov_id}.npy"
        mask = np.load(mask_path)
        cell_to_labels, _, _ = predict_fov_ensemble(
            fov_id, mask, members, weights, fov_meta,
            cluster_heads=cluster_heads,
            head_feature_member=args.head_feature_member,
        )
        fov_spots = spots[spots["fov"] == fov_id] if "fov" in spots.columns else spots
        sub = pred_mod.build_submission_for_fov(fov_id, mask, cell_to_labels, fov_spots)
        submission_parts.append(sub)
    log.info("Per-FOV inference done in %.1fs", time.time() - t0)

    submission = pd.concat(submission_parts, ignore_index=True)

    # Optionally align to sample_submission row order
    if not args.no_sample_align:
        sample = pd.read_csv(args.sample_submission)
        submission = submission.set_index("spot_id").reindex(sample["spot_id"]).reset_index()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    log.info("Wrote %s (%d rows)", args.output, len(submission))


if __name__ == "__main__":
    main()
