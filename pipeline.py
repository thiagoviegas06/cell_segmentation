"""
MERFISH cell segmentation pipeline.

Loads DAPI + polyT images from raw .dax files, runs Cellpose (cyto2),
assigns spots to cells via mask lookup, and writes submission.csv.

Usage:
    python pipeline.py [--data_root PATH] [--output PATH] [--diameter FLOAT]
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from cellpose import models

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_HEIGHT = 2048
IMG_WIDTH = 2048
DAPI_FRAME = 16      # 0-indexed frame in the multi-channel .dax file
POLY_T_FRAME = 15    # 0-indexed frame in the multi-channel .dax file
STAIN_FILE_PREFIX = "Epi-750s5-635s5-545s1-473s5-408s5"
TEST_FOVS = ["FOV_A", "FOV_B", "FOV_C", "FOV_D"]
BACKGROUND_LABEL = "background"


# ---------------------------------------------------------------------------
# 1. Image loading
# ---------------------------------------------------------------------------
def load_dax_frame(fov_dir: Path, frame_idx: int) -> np.ndarray:
    """
    Load a single 2D frame from the multi-channel staining .dax file.

    The .dax file is a raw uint16 binary stack with shape (n_frames, H, W).
    We use memmap to avoid loading the entire ~220 MB file.

    Parameters
    ----------
    fov_dir : Path
        Directory containing the .dax files for this FOV.
    frame_idx : int
        0-indexed frame number to load (15 = polyT, 16 = DAPI).

    Returns
    -------
    np.ndarray
        Shape (H, W), dtype uint16.
    """
    # Find the staining file (filename pattern differs between train and test)
    candidates = list(fov_dir.glob(f"{STAIN_FILE_PREFIX}_*.dax"))
    if not candidates:
        raise FileNotFoundError(
            f"No staining file matching '{STAIN_FILE_PREFIX}_*.dax' in {fov_dir}"
        )
    dax_path = candidates[0]

    file_bytes = os.path.getsize(dax_path)
    bytes_per_frame = IMG_HEIGHT * IMG_WIDTH * 2  # uint16
    n_frames = file_bytes // bytes_per_frame

    if frame_idx >= n_frames:
        raise ValueError(
            f"frame_idx={frame_idx} out of range for {dax_path.name} ({n_frames} frames)"
        )

    stack = np.memmap(dax_path, dtype=np.uint16, mode="r", shape=(n_frames, IMG_HEIGHT, IMG_WIDTH))
    frame = np.array(stack[frame_idx])  # copy out of memmap before it goes out of scope
    log.debug("Loaded frame %d from %s (min=%d, max=%d, mean=%.1f)",
              frame_idx, dax_path.name, frame.min(), frame.max(), frame.mean())
    return frame


def load_fov_images(fov_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load DAPI and polyT images for a single FOV.

    Returns
    -------
    (dapi, polyt) : each shape (H, W), dtype uint16
    """
    dapi = load_dax_frame(fov_dir, DAPI_FRAME)
    polyt = load_dax_frame(fov_dir, POLY_T_FRAME)
    log.info("Loaded images from %s — DAPI frame %d, polyT frame %d",
             fov_dir.name, DAPI_FRAME, POLY_T_FRAME)
    return dapi, polyt


# ---------------------------------------------------------------------------
# 2. Segmentation
# ---------------------------------------------------------------------------
def normalize_image(img: np.ndarray, pct_low: float = 1.0, pct_high: float = 99.0) -> np.ndarray:
    """
    Percentile-clip and rescale to [0, 1] float32.

    Cellpose expects float32 images; percentile normalization handles
    variable brightness across FOVs robustly.
    """
    lo, hi = np.percentile(img, [pct_low, pct_high])
    img_f = img.astype(np.float32)
    if hi > lo:
        img_f = (img_f - lo) / (hi - lo)
    return np.clip(img_f, 0.0, 1.0)


def segment_fov(
    dapi: np.ndarray,
    polyt: np.ndarray,
    model: "models.CellposeModel",
    diameter: Optional[float] = None,
) -> np.ndarray:
    """
    Run Cellpose (cpsam) segmentation on a single FOV.

    Combines polyT (cytoplasm) and DAPI (nucleus) into a 3-channel image.
    Cellpose v4 (cpsam) expects 3-channel input; we pad with a zero channel.
    Layout: ch0=polyT, ch1=DAPI, ch2=zeros (channel_axis=-1).

    Parameters
    ----------
    dapi : np.ndarray
        DAPI image (H, W), uint16.
    polyt : np.ndarray
        polyT image (H, W), uint16.
    model : models.CellposeModel
        Pre-loaded Cellpose model (reused across FOVs to avoid reload overhead).
    diameter : float, optional
        Expected cell diameter in pixels. None = auto-estimate per FOV.

    Returns
    -------
    np.ndarray
        Integer mask (H, W) where 0 = background, >0 = cell ID.
    """
    # Normalize each channel independently (Cellpose v4 normalizes internally
    # too, but pre-normalizing gives consistent input regardless of model version)
    dapi_norm = normalize_image(dapi)
    polyt_norm = normalize_image(polyt)
    zeros = np.zeros_like(polyt_norm)

    # Build (H, W, 3) image: polyT → cytoplasm, DAPI → nucleus, zeros → padding
    img_3ch = np.stack([polyt_norm, dapi_norm, zeros], axis=-1)  # (H, W, 3)

    # Cellpose v4 (cpsam): channels param is deprecated; model uses all 3 channels.
    # normalize=False because we already normalized above.
    masks, flows, styles = model.eval(
        [img_3ch],
        diameter=diameter,
        channel_axis=2,
        normalize=False,
        do_3D=False,
    )
    mask = masks[0].astype(np.int32)
    n_cells = int(mask.max())
    diam_str = f"{diameter:.1f}" if diameter is not None else "auto"
    log.info("Segmented %d cells (diameter hint=%s px)", n_cells, diam_str)
    return mask


# ---------------------------------------------------------------------------
# 3. Spot assignment
# ---------------------------------------------------------------------------
def assign_spots(
    spots: pd.DataFrame,
    mask: np.ndarray,
    fov_id: str,
) -> pd.DataFrame:
    """
    Assign spots to cells via direct mask lookup.

    Vectorized: no Python loop over individual spots.

    Parameters
    ----------
    spots : pd.DataFrame
        Rows for this FOV only. Must have columns [spot_id, image_row, image_col].
    mask : np.ndarray
        Integer mask (H, W). 0 = background, >0 = cell ID.
    fov_id : str
        FOV identifier string (e.g. 'FOV_A').

    Returns
    -------
    pd.DataFrame
        Columns: [spot_id, fov, cluster_id]
    """
    rows = spots["image_row"].to_numpy()
    cols = spots["image_col"].to_numpy()

    # Clamp to valid pixel range (guard against off-by-one at borders)
    rows_clamped = np.clip(rows, 0, mask.shape[0] - 1)
    cols_clamped = np.clip(cols, 0, mask.shape[1] - 1)

    cell_ids = mask[rows_clamped, cols_clamped]  # vectorized lookup

    # Convert integer cell IDs to cluster labels
    # 0 → "background"; N → "cell_N" (unique within this FOV; ARI is per-FOV)
    cluster_labels = np.where(
        cell_ids == 0,
        BACKGROUND_LABEL,
        np.char.add("cell_", cell_ids.astype(str)),
    )

    n_assigned = (cell_ids > 0).sum()
    n_background = (cell_ids == 0).sum()
    log.info(
        "FOV %s: %d spots → %d in cells, %d background (%.1f%% background)",
        fov_id, len(spots), n_assigned, n_background,
        100.0 * n_background / max(len(spots), 1),
    )

    return pd.DataFrame({
        "spot_id": spots["spot_id"].values,
        "fov": fov_id,
        "cluster_id": cluster_labels,
    })


# ---------------------------------------------------------------------------
# 4. Submission builder
# ---------------------------------------------------------------------------
def build_submission(all_assignments: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate per-FOV assignment DataFrames into final submission.

    Parameters
    ----------
    all_assignments : list of DataFrames
        Each has columns [spot_id, fov, cluster_id].

    Returns
    -------
    pd.DataFrame
        Sorted by spot_id, columns [spot_id, fov, cluster_id].
    """
    submission = pd.concat(all_assignments, ignore_index=True)
    # Numeric sort: extract trailing integer from "spot_N"
    submission = submission.sort_values(
        "spot_id",
        key=lambda s: s.str.extract(r"(\d+)$")[0].astype(int),
    ).reset_index(drop=True)
    return submission


# ---------------------------------------------------------------------------
# 5. Pipeline orchestrator
# ---------------------------------------------------------------------------
def run_pipeline(
    data_root: str | Path,
    spots_path: str | Path,
    output_path: str | Path,
    fov_ids: Optional[list[str]] = None,
    diameter: Optional[float] = None,
    use_gpu: bool = False,
) -> pd.DataFrame:
    """
    End-to-end pipeline: load → segment → assign → save.

    Parameters
    ----------
    data_root : path
        Root directory containing test/FOV_A, test/FOV_B, etc.
    spots_path : path
        Path to test_spots.csv.
    output_path : path
        Where to write submission.csv.
    fov_ids : list of str, optional
        Which FOVs to process. Defaults to TEST_FOVS.
    diameter : float, optional
        Cell diameter hint for Cellpose (pixels). None = auto-estimate per FOV.
    use_gpu : bool
        Whether to use GPU for Cellpose inference.

    Returns
    -------
    pd.DataFrame
        The submission DataFrame (also saved to output_path).
    """
    data_root = Path(data_root)
    spots_path = Path(spots_path)
    output_path = Path(output_path)

    if fov_ids is None:
        fov_ids = TEST_FOVS

    log.info("Loading spots from %s", spots_path)
    all_spots = pd.read_csv(spots_path)
    log.info("Total spots: %d across FOVs: %s", len(all_spots), list(all_spots["fov"].unique()))

    log.info("Loading Cellpose model (cpsam, gpu=%s)", use_gpu)
    model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")

    all_assignments: list[pd.DataFrame] = []

    for fov_id in fov_ids:
        log.info("--- Processing %s ---", fov_id)

        # Locate FOV image directory
        fov_dir = data_root / "test" / fov_id
        if not fov_dir.exists():
            log.error("FOV directory not found: %s — skipping", fov_dir)
            continue

        # 1. Load images
        dapi, polyt = load_fov_images(fov_dir)

        # 2. Segment
        mask = segment_fov(dapi, polyt, model, diameter=diameter)

        # 3. Assign spots
        fov_spots = all_spots[all_spots["fov"] == fov_id].copy()
        if fov_spots.empty:
            log.warning("No spots found for %s", fov_id)
            continue

        assignments = assign_spots(fov_spots, mask, fov_id)
        all_assignments.append(assignments)

    # 4. Build and save submission
    submission = build_submission(all_assignments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    log.info("Saved submission to %s (%d rows)", output_path, len(submission))

    return submission


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MERFISH cell segmentation pipeline")
    parser.add_argument(
        "--data_root",
        default="/scratch/pl2820/data/competition",
        help="Root data directory",
    )
    parser.add_argument(
        "--spots",
        default="/scratch/pl2820/data/competition/test_spots.csv",
        help="Path to test_spots.csv",
    )
    parser.add_argument(
        "--output",
        default="/scratch/dr3432/cell_segmentation/submission.csv",
        help="Output path for submission.csv",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=None,
        help="Expected cell diameter in pixels (default: auto-estimate)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for Cellpose inference",
    )
    parser.add_argument(
        "--fovs",
        nargs="+",
        default=None,
        help="Subset of FOVs to process (default: all test FOVs)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    run_pipeline(
        data_root=args.data_root,
        spots_path=args.spots,
        output_path=args.output,
        fov_ids=args.fovs,
        diameter=args.diameter,
        use_gpu=args.gpu,
    )
