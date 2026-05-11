"""
Phase 2.4 step 2: build a Kaggle submission CSV from a trained cluster
classifier.

For each FOV in --fovs:
  1. Load the cached expression matrix (cache/expression_phase2/<FOV>.npz)
     and the segmentation mask (cache/masks_phase2/<FOV>.npy).
  2. Build features identically to training (log1p + L2 + 2 stage coords).
  3. Predict cluster_label per cell with the trained model.
  4. Look up (supertype, subclass, class) from runs/.../hierarchy_lookup.csv.
     Cells whose predicted cluster has no hierarchy entry (only possible if
     the classifier emits 'background') get all-background.
  5. For each spot in the spots table for that FOV, look up its cell via
     mask_stack[global_z, image_row, image_col]. cell_id == 0 -> all
     background; else inherit the cell's 4 predicted labels.
  6. Concatenate across FOVs and write the submission CSV in the order of
     the supplied sample_submission file.

Usage:
    # validation pass on the 10 held-out train FOVs
    python scripts/phase2/predict.py \
        --run_dir runs/phase2_baseline \
        --fovs $(cat phase2_val_fovs.txt) \
        --spots_csv /scratch/pl2820/data/competition_phase2/train/ground_truth/spots_train.csv \
        --output runs/phase2_baseline/val_submission.csv \
        --no_sample_align

    # Kaggle submission: all 10 test FOVs
    python scripts/phase2/predict.py \
        --run_dir runs/phase2_baseline \
        --output submissions/phase2_v1_baseline.csv
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

# Same-dir import: scripts/phase2/features.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as feat  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABEL_LEVELS = ["class_label", "subclass_label", "supertype_label", "cluster_label"]
SUBMISSION_LEVELS = ["class", "subclass", "supertype", "cluster"]
BG = "background"

DATA_ROOT = Path("/scratch/pl2820/data/competition_phase2")
DEFAULT_TEST_SPOTS = DATA_ROOT / "test_spots.csv"
DEFAULT_SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"
DEFAULT_TEST_FOVS = ["FOV_E","FOV_F","FOV_G","FOV_H","FOV_I","FOV_J","FOV_K","FOV_L","FOV_M","FOV_N"]
FOV_META_CSV = DATA_ROOT / "reference" / "fov_metadata.csv"
EXPR_DIR = _PROJECT_ROOT / "cache" / "expression_phase2"
MASK_DIR = _PROJECT_ROOT / "cache" / "masks_phase2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def build_features_for_fov(matrix, centroids, mask, fov_x, fov_y, cell_ids,
                            extras_keep, neighbor_K, fov_id):
    """
    Compose the full (n, F) feature matrix for one FOV. Mirrors
    train_classifier.py exactly via features.py — same extras subset, same
    neighbor_K, same ordering.
    """
    n = len(cell_ids)
    max_id = int(mask.max())
    vols_full = feat.compute_mask_volumes(mask, max_id)
    cids = np.asarray(cell_ids).astype(np.int64)
    in_range = (cids >= 1) & (cids <= max_id)
    mask_volume_px = np.zeros(n, dtype=np.int64)
    mask_volume_px[in_range] = vols_full[cids[in_range]]

    fov_x_arr = np.full(n, fov_x, dtype=np.float32)
    fov_y_arr = np.full(n, fov_y, dtype=np.float32)
    X_norm = feat.normalize_counts(matrix)
    extras_full = feat.build_extra_features(matrix, centroids, mask_volume_px,
                                             fov_x_arr, fov_y_arr)
    extras, _ = feat.select_extras(extras_full, extras_keep)
    parts = [X_norm, extras]
    if neighbor_K > 0:
        # All cells here belong to a single FOV, so fov_ids is constant.
        fov_ids_arr = np.full(n, fov_id, dtype=object)
        nbr_mean = feat.build_neighbor_mean(matrix, centroids, fov_ids_arr,
                                             K=neighbor_K)
        parts.append(nbr_mean)
    return np.concatenate(parts, axis=1).astype(np.float32)


class ClusterHeads:
    """Container for per-subclass cluster heads + cluster -> higher-level lookup.

    Built by scripts/phase2/train_cluster_heads.py. Loaded once and consulted
    in predict_fov_cell_labels.
    """

    def __init__(self, heads_dir: Path,
                 only_subclasses: set[str] | None = None) -> None:
        self.heads_dir = heads_dir
        meta = json.loads((heads_dir / "heads_meta.json").read_text())
        self.cluster_to_higher = (
            pd.read_csv(heads_dir / "cluster_to_higher.csv")
            .set_index("cluster_label")
        )
        self.subclass_to_head: dict[str, tuple[lgb.Booster, np.ndarray]] = {}
        for subclass, info in meta["subclass_to_head"].items():
            if info.get("skipped"):
                continue
            if only_subclasses is not None and subclass not in only_subclasses:
                continue
            head_subdir = _PROJECT_ROOT / info["head_dir"]
            booster = lgb.Booster(model_file=str(head_subdir / "model.txt"))
            label_classes = np.array(
                json.loads((head_subdir / "label_classes.json").read_text())
            )
            self.subclass_to_head[subclass] = (booster, label_classes)
        log.info("Loaded %d cluster heads from %s", len(self.subclass_to_head), heads_dir)

    def has_head(self, subclass: str) -> bool:
        return subclass in self.subclass_to_head

    def predict_clusters(self, subclass: str, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (cluster_pred, max_proba) for each row in X."""
        booster, label_classes = self.subclass_to_head[subclass]
        proba = booster.predict(X)
        idx = proba.argmax(axis=1)
        return label_classes[idx], proba.max(axis=1)

    def labels_from_cluster(self, cluster: str) -> dict[str, str] | None:
        if cluster not in self.cluster_to_higher.index:
            return None
        row = self.cluster_to_higher.loc[cluster]
        return {
            "class":     row["class_label"],
            "subclass":  row["subclass_label"],
            "supertype": row["supertype_label"],
            "cluster":   cluster,
        }


def predict_fov_cell_labels(
    fov_id: str,
    mask: np.ndarray,
    model: lgb.Booster,
    label_classes: np.ndarray,
    promotion: pd.DataFrame,
    fov_meta: pd.DataFrame,
    bg_threshold: float = 0.0,
    cluster_heads: ClusterHeads | None = None,
    subclass_threshold: float = 0.0,
    head_threshold: float = 0.0,
    feature_cfg: dict | None = None,
) -> dict:
    """
    Returns dict cell_id -> {class, subclass, supertype, cluster}.
    cell_id is the 1-indexed mask label. The classifier may have been
    trained at any of the 4 hierarchy levels — `promotion` maps each
    predicted value to all 4 columns (deterministic parent + most-common
    child rollouts).

    bg_threshold: if a cell's max-class probability is below this value,
    the cell is forced to all-background. 0.0 = never demote (baseline).

    cluster_heads: optional. When provided, cells whose predicted subclass
    has a cluster head AND whose subclass max_proba >= subclass_threshold
    get their (cluster, supertype, class) labels from the head + the
    cluster->higher lookup instead of the deterministic majority rollout.
    """
    expr_path = EXPR_DIR / f"{fov_id}.npz"
    if not expr_path.exists():
        raise FileNotFoundError(f"Missing {expr_path} — run build_expression.py first")
    data = np.load(expr_path, allow_pickle=False)
    matrix = data["matrix"]
    cell_ids = data["cell_ids"]
    centroids = data["centroids"]
    n = len(cell_ids)
    if n == 0:
        return {}

    m = fov_meta.loc[fov_id]
    X = build_features_for_fov(matrix, centroids, mask, m.fov_x, m.fov_y, cell_ids,
                                extras_keep=feature_cfg["extras_keep"],
                                neighbor_K=feature_cfg["neighbor_K"],
                                fov_id=fov_id)

    proba = model.predict(X)
    pred_idx = proba.argmax(axis=1)
    pred_label = label_classes[pred_idx]
    max_proba = proba.max(axis=1)

    if bg_threshold > 0:
        n_low_conf = int((max_proba < bg_threshold).sum())
        log.info("  %s: %d/%d cells below bg_threshold %.2f -> demoted to background",
                 fov_id, n_low_conf, n, bg_threshold)

    # First pass: deterministic majority rollout (or background) for every cell.
    out: dict[int, dict[str, str]] = {}
    for cid, val, p in zip(cell_ids, pred_label, max_proba):
        if val == BG or val not in promotion.index or p < bg_threshold:
            out[int(cid)] = {sub: BG for sub in SUBMISSION_LEVELS}
        else:
            row = promotion.loc[val]
            out[int(cid)] = {
                "class":     row["class_label"],
                "subclass":  row["subclass_label"],
                "supertype": row["supertype_label"],
                "cluster":   row["cluster_label"],
            }

    # Second pass: route cells through their subclass-specific cluster head when
    # available and the subclass call is confident enough. Batches per subclass
    # so each head's predict() is called once.
    if cluster_heads is not None:
        eligible = (max_proba >= subclass_threshold) & (max_proba >= bg_threshold) & (pred_label != BG)
        for subclass in np.unique(pred_label[eligible]):
            if not cluster_heads.has_head(subclass):
                continue
            subclass_mask = eligible & (pred_label == subclass)
            idx = np.where(subclass_mask)[0]
            if len(idx) == 0:
                continue
            X_sub = X[idx]
            cluster_preds, head_proba = cluster_heads.predict_clusters(subclass, X_sub)
            cell_ids_sub = cell_ids[idx]
            n_routed = 0
            n_head_low_conf = 0
            for cid, cluster, hp in zip(cell_ids_sub, cluster_preds, head_proba):
                if hp < head_threshold:
                    n_head_low_conf += 1
                    continue   # head too uncertain -> keep majority fallback
                lbls = cluster_heads.labels_from_cluster(cluster)
                if lbls is None:
                    continue   # cluster not in lookup — keep majority fallback
                out[int(cid)] = lbls
                n_routed += 1
            log.info("  %s: routed %d cells through cluster head [%s] (k=%d clusters%s)",
                     fov_id, n_routed, subclass,
                     len(cluster_heads.subclass_to_head[subclass][1]),
                     f", {n_head_low_conf} below head_threshold {head_threshold:.2f}"
                     if head_threshold > 0 else "")
    return out


def build_submission_for_fov(
    fov_id: str,
    mask: np.ndarray,
    cell_to_labels: dict[int, dict[str, str]],
    spots: pd.DataFrame,    # already filtered to this FOV
) -> pd.DataFrame:
    Z, H, W = mask.shape
    zs = np.rint(spots["global_z"].to_numpy()).astype(np.int64)
    rows = spots["image_row"].to_numpy().astype(np.int64)
    cols = spots["image_col"].to_numpy().astype(np.int64)
    zs = np.clip(zs, 0, Z - 1)
    rows = np.clip(rows, 0, H - 1)
    cols = np.clip(cols, 0, W - 1)
    cell_ids_per_spot = mask[zs, rows, cols].astype(np.int64)

    # Vectorize the per-spot label lookup.
    n_cells = mask.max()
    # Build (n_cells+1, 4) string array indexed by cell_id (0=background row)
    # Using object dtype so empty/background works.
    lut = np.empty((int(n_cells) + 1, 4), dtype=object)
    lut[0] = [BG, BG, BG, BG]
    for cid in range(1, int(n_cells) + 1):
        labels = cell_to_labels.get(cid)
        if labels is None:
            lut[cid] = [BG, BG, BG, BG]
        else:
            lut[cid] = [labels[k] for k in SUBMISSION_LEVELS]

    spot_labels = lut[cell_ids_per_spot]   # (n_spots, 4)

    out = pd.DataFrame({
        "spot_id": spots["spot_id"].to_numpy() if "spot_id" in spots.columns
                    else np.arange(len(spots)).astype(str),
        "fov": fov_id,
        SUBMISSION_LEVELS[0]: spot_labels[:, 0],
        SUBMISSION_LEVELS[1]: spot_labels[:, 1],
        SUBMISSION_LEVELS[2]: spot_labels[:, 2],
        SUBMISSION_LEVELS[3]: spot_labels[:, 3],
    })
    n_bg = int((out[SUBMISSION_LEVELS[0]] == BG).sum())
    log.info("  %s: %d spots -> %d background (%.1f%%), %d named",
             fov_id, len(out), n_bg, 100*n_bg/len(out), len(out)-n_bg)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2.4 step 2: cluster-classifier inference")
    ap.add_argument("--run_dir", required=True,
                    help="Directory containing model.txt, label_encoder.pkl, hierarchy_lookup.csv")
    ap.add_argument("--output", required=True,
                    help="Submission CSV path")
    ap.add_argument("--fovs", nargs="+", default=None,
                    help="FOV subset (default: 10 test FOVs FOV_E..FOV_N)")
    ap.add_argument("--spots_csv", default=str(DEFAULT_TEST_SPOTS),
                    help="Spot table to predict on (default: test_spots.csv)")
    ap.add_argument("--sample_submission", default=str(DEFAULT_SAMPLE_SUB),
                    help="Optional: for spot_id ordering and row-count verification")
    ap.add_argument("--no_sample_align", action="store_true",
                    help="Skip aligning rows to sample_submission (use for val runs)")
    ap.add_argument("--bg_threshold", type=float, default=0.0,
                    help="Per-cell max-proba threshold below which the cell "
                         "is forced to background at all 4 levels. Default 0.0 "
                         "(no demotion).")
    ap.add_argument("--cluster_heads_dir", default=None,
                    help="If set, route cells through per-subclass cluster heads "
                         "from this dir (e.g. runs/phase2_clusterheads). Default "
                         "off -> baseline majority rollout.")
    ap.add_argument("--subclass_threshold", type=float, default=0.0,
                    help="Min subclass max-proba to route a cell through its "
                         "cluster head. Cells below it use the majority rollout. "
                         "Only meaningful with --cluster_heads_dir.")
    ap.add_argument("--head_threshold", type=float, default=0.0,
                    help="Min cluster-head max-proba to keep the head's cluster "
                         "prediction. Cells where the head is uncertain fall "
                         "back to majority rollout. Only meaningful with "
                         "--cluster_heads_dir.")
    ap.add_argument("--head_subclasses", nargs="+", default=None,
                    help="Optional whitelist of subclass labels for which to "
                         "actually use the head (others fall back to majority "
                         "even if a head exists). Default: use all available heads.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fovs = args.fovs or DEFAULT_TEST_FOVS
    log.info("Predicting %d FOV(s): %s", len(fovs), fovs)

    log.info("Loading model + encoder + promotion lookup from %s", run_dir)
    model = lgb.Booster(model_file=str(run_dir / "model.txt"))
    with open(run_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    label_classes = np.asarray(le.classes_)
    feat_meta = json.loads((run_dir / "feature_meta.json").read_text())
    trained_level = feat_meta.get("trained_label_level", "cluster_label")
    promotion = pd.read_csv(run_dir / "promotion_lookup.csv").set_index(trained_level, drop=False)
    feature_cfg = {
        "extras_keep": feat_meta.get("extra_feature_names", ["global_x_um", "global_y_um"]),
        "neighbor_K": int(feat_meta.get("neighbor_K", 0)),
    }
    log.info("  model classes=%d, promotion entries=%d, trained at %s",
             len(label_classes), len(promotion), trained_level)
    log.info("  feature config: extras=%s, neighbor_K=%d",
             feature_cfg["extras_keep"], feature_cfg["neighbor_K"])

    cluster_heads = None
    if args.cluster_heads_dir:
        only = set(args.head_subclasses) if args.head_subclasses else None
        cluster_heads = ClusterHeads(Path(args.cluster_heads_dir), only_subclasses=only)
        log.info("Cluster head routing enabled (subclass_threshold=%.2f)",
                 args.subclass_threshold)

    fov_meta = pd.read_csv(FOV_META_CSV).set_index("fov")
    log.info("Loading spot table %s", args.spots_csv)
    cols = ["spot_id", "fov", "image_row", "image_col", "global_z", "target_gene"]
    try:
        all_spots = pd.read_csv(args.spots_csv, usecols=cols)
    except ValueError:
        # spots_train.csv has no spot_id column — synthesize one from row index
        all_spots = pd.read_csv(args.spots_csv,
                                usecols=[c for c in cols if c != "spot_id"])
        all_spots["spot_id"] = ["s" + str(i) for i in range(len(all_spots))]
    log.info("  loaded %d spots", len(all_spots))

    pieces: list[pd.DataFrame] = []
    t_total = time.time()
    for i, fov in enumerate(fovs):
        log.info("[%d/%d] %s", i + 1, len(fovs), fov)
        spots_fov = all_spots[all_spots["fov"] == fov].copy()
        if len(spots_fov) == 0:
            log.warning("  %s: 0 spots in spot table", fov)
            continue
        mask_path = MASK_DIR / f"{fov}.npy"
        mask = np.load(mask_path)
        cell_labels = predict_fov_cell_labels(fov, mask, model, label_classes,
                                               promotion, fov_meta,
                                               bg_threshold=args.bg_threshold,
                                               cluster_heads=cluster_heads,
                                               subclass_threshold=args.subclass_threshold,
                                               head_threshold=args.head_threshold,
                                               feature_cfg=feature_cfg)
        sub_fov = build_submission_for_fov(fov, mask, cell_labels, spots_fov)
        pieces.append(sub_fov)
    log.info("Per-FOV inference done in %.1fs", time.time() - t_total)

    submission = pd.concat(pieces, ignore_index=True)

    if not args.no_sample_align:
        sample_path = Path(args.sample_submission)
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample submission missing: {sample_path}")
        sample = pd.read_csv(sample_path, usecols=["spot_id"])
        # Reindex submission to match sample's spot_id order
        submission = (
            submission.set_index("spot_id")
            .reindex(sample["spot_id"])
            .reset_index()
        )
        # Sanity: any row that didn't appear in our pieces will have NaN -> background
        n_missing = int(submission[SUBMISSION_LEVELS[0]].isna().sum())
        if n_missing:
            log.warning("  %d spots had no prediction (filling background)", n_missing)
            for lvl in SUBMISSION_LEVELS:
                submission[lvl] = submission[lvl].fillna(BG)
        submission["fov"] = submission["fov"].fillna("UNK")

    cols_out = ["spot_id", "fov"] + SUBMISSION_LEVELS
    submission = submission[cols_out]
    submission.to_csv(out_path, index=False)
    log.info("Wrote %s (%d rows)", out_path, len(submission))

    # Quick label distribution
    log.info("=== Label distribution per level ===")
    for lvl in SUBMISSION_LEVELS:
        bg = int((submission[lvl] == BG).sum())
        log.info("  %-9s  background=%d (%.1f%%)  unique non-bg=%d",
                 lvl, bg, 100 * bg / len(submission),
                 submission.loc[submission[lvl] != BG, lvl].nunique())


if __name__ == "__main__":
    main()
