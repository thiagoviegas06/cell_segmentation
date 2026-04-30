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

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

class CellTypeClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

LABEL_LEVELS = ["class_label", "subclass_label", "supertype_label", "cluster_label"]
SUBMISSION_LEVELS = ["class", "subclass", "supertype", "cluster"]
IMG_H, IMG_W = 2048, 2048
PIXEL_SIZE = 0.109
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


def normalize_counts(X: np.ndarray) -> np.ndarray:
    X = np.log1p(X.astype(np.float32))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def stage_xy(centroids: np.ndarray, fov_x: float, fov_y: float) -> np.ndarray:
    rows = centroids[:, 0]
    cols = centroids[:, 1]
    out = np.empty((len(centroids), 2), dtype=np.float32)
    out[:, 0] = fov_x + (IMG_H - rows) * PIXEL_SIZE
    out[:, 1] = fov_y + cols * PIXEL_SIZE
    return out


def predict_fov_cell_labels(
    fov_id: str,
    model: nn.Module,
    label_classes: np.ndarray,
    promotion: pd.DataFrame,
    fov_meta: pd.DataFrame,
    device: torch.device,
) -> dict:
    """
    Returns dict cell_id -> {class, subclass, supertype, cluster}.
    """
    expr_path = EXPR_DIR / f"{fov_id}.npz"
    if not expr_path.exists():
        raise FileNotFoundError(f"Missing {expr_path} — run build_expression.py first")
    data = np.load(expr_path, allow_pickle=False)
    matrix = data["matrix"]
    cell_ids = data["cell_ids"]
    centroids = data["centroids"]
    embeddings = data["embeddings"]
    n = len(cell_ids)
    if n == 0:
        return {}

    X_norm = normalize_counts(matrix)
    m = fov_meta.loc[fov_id]
    stage = stage_xy(centroids, m.fov_x, m.fov_y)
    X = np.concatenate([X_norm, stage, embeddings], axis=1).astype(np.float32)

    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        logits = model(X_t)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        
    pred_idx = proba.argmax(axis=1)
    pred_label = label_classes[pred_idx]

    # Promote each predicted label to all 4 hierarchy levels.
    out: dict[int, dict[str, str]] = {}
    for cid, val in zip(cell_ids, pred_label):
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
    return out


def build_submission_for_fov(
    fov_id: str,
    cell_to_labels: dict[int, dict[str, str]],
    spots: pd.DataFrame,    # already filtered to this FOV
) -> pd.DataFrame:
    mask_path = MASK_DIR / f"{fov_id}.npy"
    mask = np.load(mask_path)
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
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fovs = args.fovs or DEFAULT_TEST_FOVS
    log.info("Predicting %d FOV(s): %s", len(fovs), fovs)

    log.info("Loading model + encoder + promotion lookup from %s", run_dir)
    with open(run_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    label_classes = np.asarray(le.classes_)
    feat_meta = json.loads((run_dir / "feature_meta.json").read_text())
    trained_level = feat_meta.get("trained_label_level", "cluster_label")
    promotion = pd.read_csv(run_dir / "promotion_lookup.csv").set_index(trained_level, drop=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = feat_meta["n_genes"] + feat_meta["n_extra_features"]
    model = CellTypeClassifier(input_dim, len(label_classes)).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()
    
    log.info("  model classes=%d, promotion entries=%d, trained at %s",
             len(label_classes), len(promotion), trained_level)

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
        cell_labels = predict_fov_cell_labels(fov, model, label_classes,
                                               promotion, fov_meta, device)
        sub_fov = build_submission_for_fov(fov, cell_labels, spots_fov)
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
