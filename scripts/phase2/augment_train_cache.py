"""
One-shot: add `mask_volume_px` to cache/phase2_train.npz.

The consolidated training cache built by build_expression.py has per-cell
gene counts and centroids but no cell-volume info, because masks are big
and rebuilding them each training run is wasteful. This script reads each
FOV's mask once, computes per-cell pixel counts (3D = sum over z), and
writes them back into phase2_train.npz as a new `mask_volume_px` array
(positionally aligned with X_train / cell_ids).

Run once after build_expression.py. Idempotent: re-running overwrites
mask_volume_px in place.
"""

import logging
import time
from pathlib import Path

import numpy as np

import features  # local module: scripts/phase2/features.py

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_NPZ = _PROJECT_ROOT / "cache" / "phase2_train.npz"
MASK_DIR = _PROJECT_ROOT / "cache" / "masks_phase2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    log.info("Loading %s", TRAIN_NPZ)
    data = dict(np.load(TRAIN_NPZ, allow_pickle=True))
    fov_ids = data["fov_ids"].astype(str)
    cell_ids = np.asarray(data["cell_ids"]).astype(np.int64)
    n = len(fov_ids)
    log.info("  %d cells across %d FOVs", n, len(np.unique(fov_ids)))

    mask_volume_px = np.zeros(n, dtype=np.int64)

    for i, fov in enumerate(np.unique(fov_ids)):
        mask_path = MASK_DIR / f"{fov}.npy"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")
        t0 = time.time()
        mask = np.load(mask_path)
        max_id = int(mask.max())
        counts = features.compute_mask_volumes(mask, max_id)
        idx = np.where(fov_ids == fov)[0]
        cids = cell_ids[idx]
        # Defensive: cells should always be in [1, max_id]; if any are 0 or
        # out-of-range we leave volume=0 which is what compute_mask_volumes
        # returns for missing ids.
        in_range = (cids >= 1) & (cids <= max_id)
        n_oor = int((~in_range).sum())
        if n_oor:
            log.warning("  %s: %d cell_ids out of mask range [1, %d]",
                        fov, n_oor, max_id)
        mask_volume_px[idx[in_range]] = counts[cids[in_range]]
        log.info("  [%d/%d] %s  cells=%d  median_vol=%.0f  max_vol=%d  (%.1fs)",
                 i + 1, len(np.unique(fov_ids)), fov, len(idx),
                 float(np.median(mask_volume_px[idx])) if len(idx) else 0.0,
                 int(mask_volume_px[idx].max()) if len(idx) else 0,
                 time.time() - t0)

    n_zero = int((mask_volume_px == 0).sum())
    if n_zero:
        log.warning("  %d cells have mask_volume_px=0 (will log1p to 0)", n_zero)
    log.info("Stats: min=%d  median=%.0f  mean=%.1f  max=%d",
             int(mask_volume_px.min()), float(np.median(mask_volume_px)),
             float(mask_volume_px.mean()), int(mask_volume_px.max()))

    data["mask_volume_px"] = mask_volume_px.astype(np.int32)
    np.savez(TRAIN_NPZ, **data)
    log.info("Wrote %s with mask_volume_px field", TRAIN_NPZ)


if __name__ == "__main__":
    main()
