"""
Phase 5 test-set inference: 3D-stitched Cellpose + per-spot z-plane lookup.

Mirrors pipeline.py's load -> segment -> assign -> submit flow, but:
  * segments all 5 z-planes per FOV and stitches 2D masks across z via
    cellpose's stitch_threshold (same path as build_cellpose_finetuned_3d
    in scripts/segmenters.py),
  * looks up each spot at mask_stack[int(global_z), row, col] instead of
    a single z=2 plane,
  * loads weights from a fine-tuned .pt checkpoint (default: phase4_v1_h200
    best.pt — same model used in the Phase 5 local eval runs/phase5_v1_eval
    that scored 0.7544 on the 6-FOV val split).

Frame map (verified in Phase 3 and reused here):
    polyT[z] = 5 + 5*z   (frames 5, 10, 15, 20, 25)
    DAPI[z]  = 6 + 5*z   (frames 6, 11, 16, 21, 26)

Usage:
    python scripts/pipeline_v2.py \\
        --pretrained_model runs/phase4_v1_h200/checkpoints/best.pt \\
        --output submissions/phase5_v1_h200_submission.csv \\
        --gpu
"""

import argparse
import inspect
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline import (  # noqa: E402
    BACKGROUND_LABEL,
    TEST_FOVS,
    build_submission,
    load_dax_frame,
    normalize_image,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

N_Z = 5
POLY_T_FRAMES = [5 + 5 * z for z in range(N_Z)]
DAPI_FRAMES   = [6 + 5 * z for z in range(N_Z)]


def segment_fov_3d(
    fov_dir: Path,
    model: "models.CellposeModel",
    diameter: Optional[float] = None,
    stitch_threshold: float = 0.3,
) -> np.ndarray:
    """Segment all 5 z-planes of a FOV and return a (Z, H, W) stitched int mask.

    Loads polyT+DAPI per z, normalizes each channel per-z, stacks to
    (Z, H, W, 3), then calls model.eval with do_3D=False and stitch_threshold
    so cellpose links 2D masks across z by IoU (one cell -> one id across the
    planes it appears in).
    """
    slices = []
    for z in range(N_Z):
        polyt = load_dax_frame(fov_dir, POLY_T_FRAMES[z])
        dapi = load_dax_frame(fov_dir, DAPI_FRAMES[z])
        polyt_n = normalize_image(polyt)
        dapi_n = normalize_image(dapi)
        zeros = np.zeros_like(polyt_n)
        slices.append(np.stack([polyt_n, dapi_n, zeros], axis=-1))
    stack = np.stack(slices, axis=0)  # (Z, H, W, 3) float32

    masks, _flows, _styles = model.eval(
        stack,
        diameter=diameter,
        channel_axis=-1,
        z_axis=0,
        do_3D=False,
        stitch_threshold=stitch_threshold,
        normalize=False,
    )
    mask = np.asarray(masks).astype(np.int32)
    if mask.ndim != 3 or mask.shape[0] != N_Z:
        raise ValueError(
            f"Expected stitched mask of shape ({N_Z}, H, W), got {mask.shape}"
        )
    return mask


def assign_spots_3d(
    spots: pd.DataFrame,
    mask_stack: np.ndarray,
    fov_id: str,
) -> pd.DataFrame:
    """Per-spot z-plane lookup: mask_stack[global_z, image_row, image_col].

    Vectorized; z is clamped to [0, Z-1] and row/col to valid pixel range as
    defense against any borderline values. All of those clamps are no-ops on
    the actual test_spots schema (global_z ∈ {0,1,2,3,4}; image_row/col in
    [0, 2047]).
    """
    Z, H, W = mask_stack.shape
    zs = spots["global_z"].to_numpy()
    # Test spots have global_z as float (0.0..4.0); cast safely.
    if not np.issubdtype(zs.dtype, np.integer):
        zs = np.rint(zs).astype(np.int64)
    rows = spots["image_row"].to_numpy()
    cols = spots["image_col"].to_numpy()

    zs_c = np.clip(zs, 0, Z - 1)
    rows_c = np.clip(rows, 0, H - 1)
    cols_c = np.clip(cols, 0, W - 1)

    cell_ids = mask_stack[zs_c, rows_c, cols_c]

    cluster_labels = np.where(
        cell_ids == 0,
        BACKGROUND_LABEL,
        np.char.add("cell_", cell_ids.astype(str)),
    )

    n_assigned = int((cell_ids > 0).sum())
    n_background = int((cell_ids == 0).sum())
    n_cells = int(mask_stack.max())
    log.info(
        "FOV %s: %d spots -> %d in cells, %d background (%.1f%%); %d unique cells in mask",
        fov_id, len(spots), n_assigned, n_background,
        100.0 * n_background / max(len(spots), 1), n_cells,
    )

    return pd.DataFrame({
        "spot_id": spots["spot_id"].values,
        "fov": fov_id,
        "cluster_id": cluster_labels,
    })


def run_inference_v2(
    pretrained_model: str | Path,
    data_root: str | Path,
    spots_path: str | Path,
    output_path: str | Path,
    fov_ids: Optional[list[str]] = None,
    diameter: Optional[float] = None,
    stitch_threshold: float = 0.3,
    use_gpu: bool = False,
    sample_submission_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    from cellpose import models

    pretrained_model = Path(pretrained_model)
    data_root = Path(data_root)
    spots_path = Path(spots_path)
    output_path = Path(output_path)

    if not pretrained_model.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_model}")

    # Match the segmenter factory's up-front compatibility check: fail fast if
    # this cellpose build can't stitch — otherwise we'd silently produce
    # unstitched masks and quietly score like Phase 4.
    eval_params = inspect.signature(models.CellposeModel.eval).parameters
    if "stitch_threshold" not in eval_params:
        raise RuntimeError(
            "Installed cellpose model.eval() is missing 'stitch_threshold'; "
            "phase 5 pipeline cannot stitch. Params: "
            + ", ".join(eval_params.keys())
        )

    if fov_ids is None:
        fov_ids = list(TEST_FOVS)

    log.info("Loading spots from %s", spots_path)
    all_spots = pd.read_csv(spots_path)
    log.info(
        "Total spots: %d across FOVs: %s",
        len(all_spots),
        sorted(all_spots["fov"].unique().tolist()),
    )
    log.info(
        "  global_z unique values: %s",
        sorted(all_spots["global_z"].unique().tolist()),
    )

    log.info(
        "Loading fine-tuned Cellpose model (pretrained_model=%s, gpu=%s, stitch=%.2f)",
        pretrained_model, use_gpu, stitch_threshold,
    )
    t0 = time.time()
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(pretrained_model))
    log.info("  model ready in %.2fs", time.time() - t0)

    all_assignments: list[pd.DataFrame] = []
    fov_stats: list[dict] = []

    for fov_id in fov_ids:
        log.info("--- Processing %s ---", fov_id)
        fov_dir = data_root / "test" / fov_id
        if not fov_dir.exists():
            raise FileNotFoundError(f"FOV directory not found: {fov_dir}")

        t_seg = time.time()
        mask_stack = segment_fov_3d(
            fov_dir, model,
            diameter=diameter,
            stitch_threshold=stitch_threshold,
        )
        seg_s = time.time() - t_seg
        n_cells = int(mask_stack.max())
        log.info(
            "  segmented %d cells, shape=%s in %.2fs",
            n_cells, mask_stack.shape, seg_s,
        )

        fov_spots = all_spots[all_spots["fov"] == fov_id].copy()
        if fov_spots.empty:
            raise RuntimeError(f"No spots found in test_spots.csv for {fov_id}")

        assignments = assign_spots_3d(fov_spots, mask_stack, fov_id)
        all_assignments.append(assignments)

        bg = int((assignments["cluster_id"] == BACKGROUND_LABEL).sum())
        fov_stats.append({
            "fov": fov_id,
            "n_spots": len(fov_spots),
            "n_cells": n_cells,
            "n_background": bg,
            "background_pct": 100.0 * bg / max(len(fov_spots), 1),
            "seg_seconds": round(seg_s, 2),
        })

    submission = build_submission(all_assignments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    log.info("Saved submission to %s (%d rows)", output_path, len(submission))

    _verify_submission(
        submission,
        spots_path=spots_path,
        fov_ids=fov_ids,
        sample_submission_path=sample_submission_path,
    )

    log.info("=== Per-FOV summary ===")
    stats_df = pd.DataFrame(fov_stats)
    for line in stats_df.to_string(index=False).splitlines():
        log.info("  %s", line)

    return submission


def _verify_submission(
    submission: pd.DataFrame,
    spots_path: Path,
    fov_ids: list[str],
    sample_submission_path: Optional[str | Path] = None,
) -> None:
    log.info("=== Submission verification ===")

    expected_cols = ["spot_id", "fov", "cluster_id"]
    assert list(submission.columns) == expected_cols, (
        f"Column mismatch: {list(submission.columns)} != {expected_cols}"
    )
    log.info("  columns: %s (OK)", list(submission.columns))

    n_spots = sum(1 for _ in open(spots_path)) - 1
    assert len(submission) == n_spots, (
        f"Row count {len(submission)} != test_spots.csv {n_spots}"
    )
    log.info("  row count: %d matches test_spots.csv (OK)", len(submission))

    nulls = submission["cluster_id"].isna().sum()
    assert nulls == 0, f"{nulls} null cluster_ids"
    log.info("  null cluster_ids: 0 (OK)")

    fovs_present = set(submission["fov"].unique())
    expected_fovs = set(fov_ids)
    assert fovs_present == expected_fovs, (
        f"FOV mismatch: {fovs_present} != {expected_fovs}"
    )
    log.info("  fov values: %s (OK)", sorted(fovs_present))

    if sample_submission_path is not None and Path(sample_submission_path).exists():
        sample = pd.read_csv(sample_submission_path)
        assert len(sample) == len(submission), (
            f"sample_submission row count {len(sample)} != {len(submission)}"
        )
        same_order = (sample["spot_id"].values == submission["spot_id"].values).all()
        assert same_order, "spot_id ordering differs from sample_submission.csv"
        log.info(
            "  spot_id ordering matches sample_submission.csv (%d rows, OK)",
            len(sample),
        )
    else:
        log.warning(
            "  sample_submission not available at %s — skipping ordering check",
            sample_submission_path,
        )

    bg = int((submission["cluster_id"] == BACKGROUND_LABEL).sum())
    log.info(
        "  cluster_id stats: %d unique labels, %d background (%.1f%%)",
        submission["cluster_id"].nunique(),
        bg,
        100.0 * bg / len(submission),
    )
    log.info("=== All checks passed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 3D-stitched test-set inference")
    parser.add_argument("--pretrained_model", required=True,
                        help="Path to fine-tuned .pt checkpoint")
    parser.add_argument("--data_root", default="/scratch/pl2820/data/competition")
    parser.add_argument("--spots", default="/scratch/pl2820/data/competition/test_spots.csv")
    parser.add_argument("--sample_submission",
                        default="/scratch/pl2820/data/competition/sample_submission.csv")
    parser.add_argument("--output", required=True,
                        help="Output path for submission.csv "
                             "(e.g. submissions/phase5_v1_h200_submission.csv)")
    parser.add_argument("--diameter", type=float, default=None,
                        help="Cell diameter hint (px). Omit/negative for auto.")
    parser.add_argument("--stitch_threshold", type=float, default=0.3,
                        help="IoU threshold for linking 2D masks across z (default 0.3)")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--fovs", nargs="+", default=None,
                        help="FOV subset (default: all 4 test FOVs)")
    args = parser.parse_args()

    diameter = args.diameter if (args.diameter is not None and args.diameter >= 0) else None

    run_inference_v2(
        pretrained_model=args.pretrained_model,
        data_root=args.data_root,
        spots_path=args.spots,
        output_path=args.output,
        fov_ids=args.fovs,
        diameter=diameter,
        stitch_threshold=args.stitch_threshold,
        use_gpu=args.gpu,
        sample_submission_path=args.sample_submission,
    )
