"""
Test-set inference with a fine-tuned Cellpose model.

Mirrors pipeline.run_pipeline, but instantiates CellposeModel from a custom
.pt checkpoint instead of the built-in 'cpsam' weights. Writes a Kaggle-format
submission.csv to a new path (default: submissions/<run>_submission.csv) so
the existing 0.62 submission.csv is not overwritten.

Usage:
    python scripts/infer_test.py \
        --pretrained_model runs/phase4_v1_h200/checkpoints/best.pt \
        --output submissions/phase4_v1_h200_submission.csv \
        --gpu
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline import (  # noqa: E402
    TEST_FOVS,
    assign_spots,
    build_submission,
    load_fov_images,
    segment_fov,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_inference(
    pretrained_model: str | Path,
    data_root: str | Path,
    spots_path: str | Path,
    output_path: str | Path,
    fov_ids: Optional[list[str]] = None,
    diameter: Optional[float] = None,
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
        "Loading fine-tuned Cellpose model (pretrained_model=%s, gpu=%s)",
        pretrained_model,
        use_gpu,
    )
    t0 = time.time()
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(pretrained_model))
    log.info("  model ready in %.2fs", time.time() - t0)

    all_assignments: list[pd.DataFrame] = []
    for fov_id in fov_ids:
        log.info("--- Processing %s ---", fov_id)
        fov_dir = data_root / "test" / fov_id
        if not fov_dir.exists():
            raise FileNotFoundError(f"FOV directory not found: {fov_dir}")

        dapi, polyt = load_fov_images(fov_dir)

        t_seg = time.time()
        mask = segment_fov(dapi, polyt, model, diameter=diameter)
        log.info("  segmented in %.2fs", time.time() - t_seg)

        fov_spots = all_spots[all_spots["fov"] == fov_id].copy()
        if fov_spots.empty:
            raise RuntimeError(f"No spots found in test_spots.csv for {fov_id}")

        all_assignments.append(assign_spots(fov_spots, mask, fov_id))

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

    n_spots = sum(1 for _ in open(spots_path)) - 1  # header
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

    bg = (submission["cluster_id"] == "background").sum()
    log.info(
        "  cluster_id stats: %d unique labels, %d background (%.1f%%)",
        submission["cluster_id"].nunique(),
        bg,
        100.0 * bg / len(submission),
    )
    log.info("=== All checks passed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuned test-set inference")
    parser.add_argument("--pretrained_model", required=True,
                        help="Path to fine-tuned .pt checkpoint")
    parser.add_argument("--data_root", default="/scratch/pl2820/data/competition")
    parser.add_argument("--spots", default="/scratch/pl2820/data/competition/test_spots.csv")
    parser.add_argument("--sample_submission",
                        default="/scratch/pl2820/data/competition/sample_submission.csv")
    parser.add_argument("--output", required=True,
                        help="Output path for submission.csv "
                             "(e.g. submissions/phase4_v1_h200_submission.csv)")
    parser.add_argument("--diameter", type=float, default=None,
                        help="Cell diameter hint (px). Omit for auto.")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--fovs", nargs="+", default=None,
                        help="FOV subset (default: all 4 test FOVs)")
    args = parser.parse_args()

    run_inference(
        pretrained_model=args.pretrained_model,
        data_root=args.data_root,
        spots_path=args.spots,
        output_path=args.output,
        fov_ids=args.fovs,
        diameter=args.diameter,
        use_gpu=args.gpu,
        sample_submission_path=args.sample_submission,
    )
