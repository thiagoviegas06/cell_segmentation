"""
Shared per-cell feature builder for Phase 2 classifiers.

Returns the (n, K) extra-feature array that is concatenated to the
log1p+L2 gene-expression matrix at training and inference time. Both
train_classifier.py and predict.py route their feature construction
through this module so they stay in sync — adding/removing/reordering
a feature here changes the model input everywhere consistently.

Convention:
- The first 1147 features in the model input are the log1p+L2-normalized
  gene-expression columns (in the order of cache/gene_vocab.json).
- The remaining K features are produced by `build_extra_features` below,
  in the order given by `EXTRA_FEATURE_NAMES`.
"""

from __future__ import annotations

import numpy as np

IMG_H = 2048
IMG_W = 2048
PIXEL_SIZE_UM = 0.109

EXTRA_FEATURE_NAMES = [
    "log1p_total_count",
    "log1p_n_genes_detected",
    "log1p_mask_volume_px",
    "global_x_um",
    "global_y_um",
    "image_row_norm",
    "image_col_norm",
]


def normalize_counts(matrix: np.ndarray) -> np.ndarray:
    """log1p then per-cell L2 normalize across the gene dim. (n, n_genes)"""
    X = np.log1p(matrix.astype(np.float32))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def build_extra_features(
    matrix: np.ndarray,
    centroids: np.ndarray,
    mask_volume_px: np.ndarray,
    fov_x_per_cell: np.ndarray,
    fov_y_per_cell: np.ndarray,
) -> np.ndarray:
    """
    matrix: (n, n_genes) raw spot counts per gene per cell
    centroids: (n, 2) (image_row, image_col) in pixels
    mask_volume_px: (n,) total voxels claimed by each cell in the 3D mask
    fov_x_per_cell, fov_y_per_cell: (n,) stage offsets in µm for each cell's FOV

    Returns: (n, K) float32 array of extra (non-gene) features in the order
    of EXTRA_FEATURE_NAMES.
    """
    n = len(matrix)
    if (len(centroids) != n or len(mask_volume_px) != n
            or len(fov_x_per_cell) != n or len(fov_y_per_cell) != n):
        raise ValueError(
            f"feature builder size mismatch: matrix={n}, centroids={len(centroids)}, "
            f"mask_volume={len(mask_volume_px)}, fov_x={len(fov_x_per_cell)}, "
            f"fov_y={len(fov_y_per_cell)}"
        )

    total_count = matrix.sum(axis=1).astype(np.float32)
    n_genes_det = (matrix > 0).sum(axis=1).astype(np.float32)

    rows = centroids[:, 0].astype(np.float32)
    cols = centroids[:, 1].astype(np.float32)
    global_x = fov_x_per_cell.astype(np.float32) + (IMG_H - rows) * PIXEL_SIZE_UM
    global_y = fov_y_per_cell.astype(np.float32) + cols * PIXEL_SIZE_UM
    image_row_norm = rows / IMG_H
    image_col_norm = cols / IMG_W

    extras = np.column_stack([
        np.log1p(total_count),
        np.log1p(n_genes_det),
        np.log1p(mask_volume_px.astype(np.float32)),
        global_x,
        global_y,
        image_row_norm,
        image_col_norm,
    ]).astype(np.float32)
    assert extras.shape == (n, len(EXTRA_FEATURE_NAMES))
    return extras


def compute_mask_volumes(mask: np.ndarray, max_cell_id: int) -> np.ndarray:
    """
    mask: (Z, H, W) uint16 segmentation. 0 = background, 1..max_cell_id = cells.
    Returns: (max_cell_id + 1,) int64 array, where index i = pixels with id i.
    Use as: mask_volume_px[cell_id] for any cell_id in 1..max_cell_id.
    """
    return np.bincount(mask.flatten(), minlength=max_cell_id + 1)


def select_extras(extras: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Subset the extras matrix by EXTRA_FEATURE_NAMES order."""
    if names is None or list(names) == list(EXTRA_FEATURE_NAMES):
        return extras, list(EXTRA_FEATURE_NAMES)
    keep_idx = [EXTRA_FEATURE_NAMES.index(n) for n in names]
    return extras[:, keep_idx], list(names)


def build_neighbor_mean(
    matrix: np.ndarray,
    centroids: np.ndarray,
    fov_ids: np.ndarray,
    K: int = 5,
) -> np.ndarray:
    """
    For each cell, return the mean log1p+L2-normalized expression of its K
    nearest same-FOV neighbors (excluding self). NN distance is computed on
    `centroids` (image_row, image_col) — within each FOV the coord system is
    consistent.

    Returns: (n_cells, n_genes) float32. Cells whose FOV has < 2 cells get a
    zero row (no neighbors available).

    Why per-FOV: cell_y stage coords aren't comparable across FOVs in the
    test set (different brain regions); image-pixel coords inside one FOV are
    isotropic. This keeps train and test in the same coord system.
    """
    from sklearn.neighbors import NearestNeighbors

    n_cells, n_genes = matrix.shape
    if len(centroids) != n_cells or len(fov_ids) != n_cells:
        raise ValueError("matrix/centroids/fov_ids size mismatch in build_neighbor_mean")
    matrix_norm = normalize_counts(matrix)
    out = np.zeros((n_cells, n_genes), dtype=np.float32)

    fov_ids = np.asarray(fov_ids).astype(str)
    for fov in np.unique(fov_ids):
        idx = np.where(fov_ids == fov)[0]
        if len(idx) < 2:
            continue
        coords = centroids[idx].astype(np.float32)
        k_query = min(K + 1, len(idx))
        nn = NearestNeighbors(n_neighbors=k_query).fit(coords)
        _, nn_local = nn.kneighbors(coords)
        # Column 0 is the cell itself (distance=0). Skip it.
        neighbor_cols = nn_local[:, 1:]
        for i in range(len(idx)):
            # neighbor_cols[i] are indices INTO `idx` (local FOV ordering)
            global_nbr_idx = idx[neighbor_cols[i]]
            out[idx[i]] = matrix_norm[global_nbr_idx].mean(axis=0)
    return out
