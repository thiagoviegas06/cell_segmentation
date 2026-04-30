"""
Phase 2.3: build per-cell gene-expression vectors from segmentation masks
and spot tables.

Produces, for each of the 70 Phase 2 FOVs, a (n_cells, 1147) integer count
matrix capturing how many spots of each gene fell inside each predicted
cell. Then assembles the 60 training FOVs into a single labeled training
set, with labels assigned by nearest-centroid matching against
cell_labels_train.csv (>30 px → "background").

Outputs:
    cache/gene_vocab.json                # 1147 gene names, ordered as counts.var
    cache/expression_phase2/<fov>.npz    # matrix, cell_ids, fov_id
    cache/phase2_train.npz               # X_train, fov_ids, cell_ids,
                                         # match_dist_px,
                                         # y_class, y_subclass, y_supertype,
                                         # y_cluster, gt_cell_ids

Usage:
    python scripts/phase2/build_expression.py             # all 70 FOVs
    python scripts/phase2/build_expression.py --fovs FOV_E FOV_147   # smoke
    python scripts/phase2/build_expression.py --skip_train_set       # skip step 4
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_ROOT = Path("/scratch/pl2820/data/competition_phase2")
COUNTS_H5AD = DATA_ROOT / "train" / "ground_truth" / "counts_train.h5ad"
LABELS_CSV = DATA_ROOT / "train" / "ground_truth" / "cell_labels_train.csv"
SPOTS_TRAIN_CSV = DATA_ROOT / "train" / "ground_truth" / "spots_train.csv"
SPOTS_TEST_CSV = DATA_ROOT / "test_spots.csv"
FOV_META_CSV = DATA_ROOT / "reference" / "fov_metadata.csv"

MASK_DIR = _PROJECT_ROOT / "cache" / "masks_phase2"
OUT_DIR = _PROJECT_ROOT / "cache" / "expression_phase2"
GENE_VOCAB_PATH = _PROJECT_ROOT / "cache" / "gene_vocab.json"
TRAIN_SET_PATH = _PROJECT_ROOT / "cache" / "phase2_train.npz"

IMG_H, IMG_W = 2048, 2048
N_Z = 5
PIXEL_SIZE = 0.109                # µm/px (constant across FOVs in fov_metadata)
MATCH_THRESHOLD_PX = 30.0         # ~3.3 µm; ~1/3 of median GT cell diameter (87 px)
LABEL_LEVELS = ["class_label", "subclass_label", "supertype_label", "cluster_label"]


# ---------------------------------------------------------------------------
# Step 1: gene vocabulary
# ---------------------------------------------------------------------------
def load_or_build_gene_vocab() -> list[str]:
    """Read counts_train.h5ad var._index; cache to gene_vocab.json. Order matches X."""
    if GENE_VOCAB_PATH.exists():
        with open(GENE_VOCAB_PATH) as f:
            vocab = json.load(f)
        assert len(vocab) == 1147, f"Cached vocab has {len(vocab)} genes, expected 1147"
        return vocab
    with h5py.File(COUNTS_H5AD, "r") as f:
        vocab = [s.decode() if isinstance(s, bytes) else s
                 for s in f["var/_index"][:]]
    assert len(vocab) == 1147, f"counts.var has {len(vocab)} genes, expected 1147"
    GENE_VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GENE_VOCAB_PATH, "w") as f:
        json.dump(vocab, f)
    log.info("Wrote gene vocab (%d genes) to %s", len(vocab), GENE_VOCAB_PATH)
    return vocab


# ---------------------------------------------------------------------------
# Step 2: builder
# ---------------------------------------------------------------------------
def build_expression_matrix(
    mask_stack: np.ndarray,
    spots_df: pd.DataFrame,
    gene_vocab: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate per-spot gene calls into a (n_cells, n_genes) integer matrix.

    Returns
    -------
    matrix : (n_cells, len(gene_vocab)) int32
        Row i corresponds to cell_ids[i]. Cell IDs are 1..n_cells (the
        max cell_id in mask_stack); rows for cells with no in-cell spots
        are all zeros (kept for index alignment with mask_stack).
    cell_ids : (n_cells,) int32
        np.arange(1, n_cells + 1) in current implementation, but returned
        explicitly so downstream code never has to assume that mapping.

    Notes
    -----
    - Spots with cell_id == 0 (extracellular per the mask) are dropped.
    - Spots whose target_gene is not in the 1147-gene vocab (the 93
      blank-* decoys) are dropped with a single aggregate warning.
    """
    if mask_stack.ndim != 3 or mask_stack.shape[1:] != (IMG_H, IMG_W):
        raise ValueError(f"Expected mask shape (Z, {IMG_H}, {IMG_W}), got {mask_stack.shape}")

    n_cells = int(mask_stack.max())
    n_genes = len(gene_vocab)
    matrix = np.zeros((n_cells, n_genes), dtype=np.int32)
    cell_ids = np.arange(1, n_cells + 1, dtype=np.int32)
    if n_cells == 0:
        return matrix, cell_ids

    gene_to_idx = {g: i for i, g in enumerate(gene_vocab)}

    zs = np.rint(spots_df["global_z"].to_numpy()).astype(np.int64)
    rows = spots_df["image_row"].to_numpy().astype(np.int64)
    cols = spots_df["image_col"].to_numpy().astype(np.int64)
    zs = np.clip(zs, 0, mask_stack.shape[0] - 1)
    rows = np.clip(rows, 0, IMG_H - 1)
    cols = np.clip(cols, 0, IMG_W - 1)

    cell_lookup = mask_stack[zs, rows, cols].astype(np.int64)
    in_cell = cell_lookup > 0
    n_extracellular = int((~in_cell).sum())

    genes = spots_df["target_gene"].to_numpy()
    gene_idx_arr = np.array([gene_to_idx.get(g, -1) for g in genes], dtype=np.int64)
    in_vocab = gene_idx_arr >= 0
    n_blank = int((in_cell & ~in_vocab).sum())

    keep = in_cell & in_vocab
    cell_idx = cell_lookup[keep] - 1   # cell_id 1..N -> matrix row 0..N-1
    gene_idx = gene_idx_arr[keep]
    np.add.at(matrix, (cell_idx, gene_idx), 1)

    log.debug(
        "  build_expression: %d spots -> %d kept (%d extracellular, %d blank/unk-gene)",
        len(spots_df), keep.sum(), n_extracellular, n_blank,
    )
    return matrix, cell_ids


# ---------------------------------------------------------------------------
# Centroids (image coords) for predicted cells
# ---------------------------------------------------------------------------
def compute_cell_centroids(mask_stack: np.ndarray) -> np.ndarray:
    """Return (n_cells, 2) array of (image_row, image_col) centroids.

    Centroid is the mean (row, col) over all pixels in any z plane labeled
    with that cell ID. scipy.ndimage.center_of_mass on the 3D mask returns
    (z, row, col) per label; we project to (row, col). Cells with zero
    pixels (shouldn't happen for cells coming out of cellpose, but defend
    against it) get NaN.
    """
    n_cells = int(mask_stack.max())
    if n_cells == 0:
        return np.zeros((0, 2), dtype=np.float32)
    centers = center_of_mass(
        np.ones(mask_stack.shape, dtype=np.float32),
        labels=mask_stack,
        index=range(1, n_cells + 1),
    )
    arr = np.asarray(centers, dtype=np.float32)  # (n_cells, 3) -> (z, r, c)
    return arr[:, 1:3]  # (n_cells, 2) -> (r, c)


# ---------------------------------------------------------------------------
# GT centroid -> image coords (per FOV) and matching
# ---------------------------------------------------------------------------
def gt_centroids_image_coords(
    gt_subset: pd.DataFrame, fov_x: float, fov_y: float
) -> np.ndarray:
    """Convert GT (center_x, center_y) [stage µm] to (image_row, image_col) [px].

    Uses the verified Phase 1 pixel convention:
        image_row = 2048 - (center_x - fov_x) / pixel_size
        image_col =        (center_y - fov_y) / pixel_size
    """
    rows = IMG_H - (gt_subset["center_x"].to_numpy() - fov_x) / PIXEL_SIZE
    cols = (gt_subset["center_y"].to_numpy() - fov_y) / PIXEL_SIZE
    return np.column_stack([rows, cols]).astype(np.float32)


def match_to_gt(
    pred_centroids: np.ndarray,         # (n_pred, 2)
    gt_centroids: np.ndarray,           # (n_gt, 2)
    threshold_px: float = MATCH_THRESHOLD_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each predicted cell, return (gt_idx, distance_px).

    gt_idx == -1 means no GT cell within `threshold_px` (assign background).
    Naive O(N×M) — fine for ~hundreds of cells per FOV.
    """
    if len(gt_centroids) == 0 or len(pred_centroids) == 0:
        n = len(pred_centroids)
        return np.full(n, -1, dtype=np.int32), np.full(n, np.inf, dtype=np.float32)

    diffs = pred_centroids[:, None, :] - gt_centroids[None, :, :]  # (n_pred, n_gt, 2)
    dists = np.linalg.norm(diffs, axis=2)                           # (n_pred, n_gt)
    nearest = dists.argmin(axis=1)
    nearest_d = dists.min(axis=1)
    gt_idx = np.where(nearest_d <= threshold_px, nearest, -1).astype(np.int32)
    return gt_idx, nearest_d.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 3: per-FOV processing
# ---------------------------------------------------------------------------
def process_fov(
    fov_id: str,
    split: str,
    spots_df: pd.DataFrame,    # already filtered to this FOV
    gene_vocab: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (matrix, cell_ids, centroids, embeddings) for one FOV; save to cache/expression_phase2/."""
    mask_path = MASK_DIR / f"{fov_id}.npy"
    embed_path = MASK_DIR / f"{fov_id}_embeddings.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")
    if not embed_path.exists():
        raise FileNotFoundError(f"Missing embeddings: {embed_path} — run segment_all.py with --overwrite")
        
    mask = np.load(mask_path)
    embeddings = np.load(embed_path)
    matrix, cell_ids = build_expression_matrix(mask, spots_df, gene_vocab)
    centroids = compute_cell_centroids(mask)
    
    out_path = OUT_DIR / f"{fov_id}.npz"
    np.savez(out_path, matrix=matrix, cell_ids=cell_ids,
             centroids=centroids, embeddings=embeddings, fov_id=np.array(fov_id))
    return matrix, cell_ids, centroids, embeddings


# ---------------------------------------------------------------------------
# Step 4: consolidated training set with centroid matching
# ---------------------------------------------------------------------------
def build_train_set(
    fov_to_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    gt_labels: pd.DataFrame,
    fov_meta: pd.DataFrame,
) -> dict:
    Xs = []
    fov_ids_out = []
    cell_ids_out = []
    centroids_out = []
    match_dists = []
    gt_cell_ids_out = []
    embeds_out = []
    label_arrays = {lvl: [] for lvl in LABEL_LEVELS}

    summary = []
    for fov_id, (matrix, cell_ids, pred_cent, embeddings) in sorted(fov_to_data.items()):
        gt_subset = gt_labels[gt_labels["fov"] == fov_id]
        meta = fov_meta.loc[fov_id]
        if len(gt_subset) == 0:
            log.warning("  %s: 0 GT cells in cell_labels_train.csv", fov_id)
            gt_cent = np.zeros((0, 2), dtype=np.float32)
        else:
            gt_cent = gt_centroids_image_coords(gt_subset, meta.fov_x, meta.fov_y)
        gt_idx, dists = match_to_gt(pred_cent, gt_cent)

        labels_per_level = {}
        gt_cell_id_arr = np.empty(len(cell_ids), dtype=object)
        for i, gi in enumerate(gt_idx):
            if gi == -1:
                gt_cell_id_arr[i] = ""
            else:
                gt_cell_id_arr[i] = gt_subset.iloc[gi]["cell_id"]
        for lvl in LABEL_LEVELS:
            arr = np.empty(len(cell_ids), dtype=object)
            for i, gi in enumerate(gt_idx):
                if gi == -1:
                    arr[i] = "background"
                else:
                    arr[i] = gt_subset.iloc[gi][lvl]
            labels_per_level[lvl] = arr

        Xs.append(matrix)
        fov_ids_out.extend([fov_id] * len(cell_ids))
        cell_ids_out.append(cell_ids)
        centroids_out.append(pred_cent)
        match_dists.append(dists)
        gt_cell_ids_out.append(gt_cell_id_arr)
        embeds_out.append(embeddings)
        for lvl in LABEL_LEVELS:
            label_arrays[lvl].append(labels_per_level[lvl])

        n_match = int((gt_idx >= 0).sum())
        summary.append({
            "fov": fov_id,
            "n_pred": len(cell_ids),
            "n_gt": len(gt_subset),
            "n_match": n_match,
            "match_pct": 100.0 * n_match / max(len(cell_ids), 1),
            "median_match_d": float(np.median(dists[gt_idx >= 0])) if n_match else float("nan"),
        })

    return {
        "X_train": np.concatenate(Xs, axis=0).astype(np.int32),
        "fov_ids": np.array(fov_ids_out),
        "cell_ids": np.concatenate(cell_ids_out, axis=0).astype(np.int32),
        "centroids": np.concatenate(centroids_out, axis=0).astype(np.float32),
        "match_dist_px": np.concatenate(match_dists, axis=0).astype(np.float32),
        "gt_cell_ids": np.concatenate(gt_cell_ids_out, axis=0),
        "embeddings": np.concatenate(embeds_out, axis=0).astype(np.float32),
        "y_class": np.concatenate(label_arrays["class_label"], axis=0),
        "y_subclass": np.concatenate(label_arrays["subclass_label"], axis=0),
        "y_supertype": np.concatenate(label_arrays["supertype_label"], axis=0),
        "y_cluster": np.concatenate(label_arrays["cluster_label"], axis=0),
        "fov_summary": pd.DataFrame(summary),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2.3 expression-matrix builder")
    ap.add_argument("--fovs", nargs="+", default=None,
                    help="FOV subset for smoke testing. Default: all 70.")
    ap.add_argument("--skip_train_set", action="store_true",
                    help="Skip step 4 (consolidated train set with GT labels).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-process FOVs even if their .npz already exists.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading gene vocabulary...")
    gene_vocab = load_or_build_gene_vocab()

    log.info("Loading FOV metadata...")
    fov_meta = pd.read_csv(FOV_META_CSV).set_index("fov")

    log.info("Loading spot tables...")
    t = time.time()
    train_spots = pd.read_csv(SPOTS_TRAIN_CSV,
                              usecols=["fov", "image_row", "image_col",
                                       "global_z", "target_gene"])
    test_spots = pd.read_csv(SPOTS_TEST_CSV,
                             usecols=["fov", "image_row", "image_col",
                                      "global_z", "target_gene"])
    log.info("  spots loaded in %.1fs (train=%d, test=%d)",
             time.time() - t, len(train_spots), len(test_spots))

    train_groups = train_spots.groupby("fov", sort=False)
    test_groups = test_spots.groupby("fov", sort=False)

    all_fovs = sorted(MASK_DIR.glob("*.npy"))
    fov_ids_present = [p.stem for p in all_fovs]
    if args.fovs is not None:
        fov_ids_present = [f for f in fov_ids_present if f in set(args.fovs)]
        missing = set(args.fovs) - set(fov_ids_present)
        if missing:
            raise ValueError(f"Mask file(s) missing for: {sorted(missing)}")
    log.info("Processing %d FOVs", len(fov_ids_present))

    fov_to_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    t_total = time.time()
    for i, fov_id in enumerate(fov_ids_present):
        out_path = OUT_DIR / f"{fov_id}.npz"
        is_test = fov_id.startswith("FOV_") and not fov_id[4:].isdigit()
        split = "test" if is_test else "train"
        groups = test_groups if is_test else train_groups
        spots_df = groups.get_group(fov_id) if fov_id in groups.groups else None
        if spots_df is None or len(spots_df) == 0:
            log.warning("[%d/%d] %s (%s): NO SPOTS in spot table — empty matrix",
                        i + 1, len(fov_ids_present), fov_id, split)
            spots_df = pd.DataFrame(columns=["image_row", "image_col",
                                             "global_z", "target_gene"])

        if out_path.exists() and not args.overwrite:
            data = np.load(out_path, allow_pickle=False)
            matrix = data["matrix"]
            cell_ids = data["cell_ids"]
            centroids = data["centroids"]
            embeddings = data["embeddings"]
            log.info("[%d/%d] %s (%s): loaded existing  matrix=%s  embed=%s",
                     i + 1, len(fov_ids_present), fov_id, split,
                     matrix.shape, embeddings.shape)
        else:
            t_fov = time.time()
            matrix, cell_ids, centroids, embeddings = process_fov(fov_id, split, spots_df, gene_vocab)
            log.info("[%d/%d] %s (%s): %d cells × %d genes, %d-dim embeddings, %.1fs",
                     i + 1, len(fov_ids_present), fov_id, split,
                     matrix.shape[0], matrix.shape[1], embeddings.shape[1],
                     time.time() - t_fov)
        if split == "train":
            fov_to_data[fov_id] = (matrix, cell_ids, centroids, embeddings)

    log.info("Per-FOV step done in %.1fs", time.time() - t_total)

    # Step 4: consolidated train set
    if args.skip_train_set or not fov_to_data:
        log.info("Skipping step 4 (consolidated train set).")
        return
    log.info("=== Step 4: build consolidated training set ===")
    gt_labels = pd.read_csv(LABELS_CSV, dtype={"cell_id": str})
    out = build_train_set(fov_to_data, gt_labels, fov_meta)

    np.savez(
        TRAIN_SET_PATH,
        X_train=out["X_train"],
        fov_ids=out["fov_ids"],
        cell_ids=out["cell_ids"],
        centroids=out["centroids"],
        match_dist_px=out["match_dist_px"],
        gt_cell_ids=out["gt_cell_ids"],
        embeddings=out["embeddings"],
        y_class=out["y_class"],
        y_subclass=out["y_subclass"],
        y_supertype=out["y_supertype"],
        y_cluster=out["y_cluster"],
    )
    log.info("Saved %s — X_train=%s, embeddings=%s, %d label arrays",
             TRAIN_SET_PATH, out["X_train"].shape, out["embeddings"].shape, len(LABEL_LEVELS))

    # Quick top-line numbers (full report comes from a separate analysis pass).
    n_total = len(out["fov_ids"])
    n_matched = int((out["match_dist_px"] <= MATCH_THRESHOLD_PX).sum())
    log.info("Total predicted train cells: %d", n_total)
    log.info("  matched to GT (≤%.0f px): %d (%.1f%%)",
             MATCH_THRESHOLD_PX, n_matched, 100 * n_matched / n_total)
    log.info("  background (no/far GT)  : %d (%.1f%%)",
             n_total - n_matched, 100 * (n_total - n_matched) / n_total)

    log.info("Class label distribution (predicted train cells):")
    cls_counts = pd.Series(out["y_class"]).value_counts()
    for cls, n in cls_counts.items():
        log.info("  %-22s %5d", cls, n)


if __name__ == "__main__":
    main()
