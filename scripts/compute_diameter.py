"""
Calibrate Cellpose's `diameter` hint from ground-truth cell polygons.

Reads cell_boundaries_train.csv (polygon coords in stage-µm), converts polygon
area to pixels², and reports the equivalent-circle diameter per cell:

    d_px = 2 * sqrt(area_px / pi)

Area is computed with the shoelace formula. Since the stage-µm -> image-pixel
transform is an affine map with isotropic scale 1/pixel_size plus a reflection
and translation, area in µm² divided by pixel_size² equals area in pixels²
exactly. We don't need the per-FOV (fov_x, fov_y) offsets.

Default z-plane is 2 — the middle plane, matching where pipeline.segment_fov
currently operates. Override with --z for diagnostics.

Writes the median diameter (in pixels) to reference/diameter_px.txt and
prints the full distribution summary.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

PIXEL_SIZE_UM = 0.109
Z_PLANES = 5

log = logging.getLogger("compute_diameter")


def parse_boundary(s) -> np.ndarray | None:
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        return np.fromstring(s, sep=",", dtype=np.float64)
    except Exception:
        return None


def shoelace_area(x: np.ndarray, y: np.ndarray) -> float:
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def diameters_at_z(bounds: pd.DataFrame, z: int) -> tuple[np.ndarray, int, int]:
    """Return (diameters_px, n_valid, n_missing) for the given z-plane."""
    bx_col = f"boundaryX_z{z}"
    by_col = f"boundaryY_z{z}"
    diameters: list[float] = []
    missing = 0
    for _, row in bounds.iterrows():
        bx = parse_boundary(row.get(bx_col))
        by = parse_boundary(row.get(by_col))
        if bx is None or by is None or len(bx) < 3 or len(bx) != len(by):
            missing += 1
            continue
        area_um2 = shoelace_area(bx, by)
        area_px = area_um2 / (PIXEL_SIZE_UM ** 2)
        diameters.append(2.0 * np.sqrt(area_px / np.pi))
    return np.array(diameters, dtype=np.float64), len(diameters), missing


def summarize(d: np.ndarray) -> dict:
    return {
        "n": int(len(d)),
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "p25": float(np.percentile(d, 25)),
        "p75": float(np.percentile(d, 75)),
        "p90": float(np.percentile(d, 90)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--boundaries",
        default="/scratch/pl2820/data/competition/train/ground_truth/cell_boundaries_train.csv",
    )
    ap.add_argument("--z", type=int, default=2,
                    help="z-plane whose polygon sets each cell's diameter (default: 2)")
    user = os.environ.get("USER", "dr3432")
    ap.add_argument(
        "--output",
        default=f"/scratch/{user}/cell_segmentation/reference/diameter_px.txt",
    )
    ap.add_argument("--all_z", action="store_true",
                    help="Also print per-z summaries for context.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    log.info("Loading boundaries: %s", args.boundaries)
    bounds = pd.read_csv(args.boundaries, dtype={"Unnamed: 0": str})
    log.info("  %d cells", len(bounds))

    if args.all_z:
        for z in range(Z_PLANES):
            d, n, miss = diameters_at_z(bounds, z)
            if n == 0:
                log.info("  z=%d: no valid polygons", z)
                continue
            s = summarize(d)
            log.info("  z=%d: n=%d missing=%d  mean=%.2f median=%.2f "
                     "p25=%.2f p75=%.2f p90=%.2f",
                     z, s["n"], miss, s["mean"], s["median"],
                     s["p25"], s["p75"], s["p90"])

    d, n_valid, missing = diameters_at_z(bounds, args.z)
    if n_valid == 0:
        raise RuntimeError(f"No valid polygons at z={args.z}")
    s = summarize(d)

    log.info("=" * 60)
    log.info("Diameter (pixels) from z=%d:", args.z)
    log.info("  cells with valid polygon: %d / %d (missing %d)",
             n_valid, len(bounds), missing)
    log.info("  mean   = %.2f", s["mean"])
    log.info("  median = %.2f", s["median"])
    log.info("  p25    = %.2f", s["p25"])
    log.info("  p75    = %.2f", s["p75"])
    log.info("  p90    = %.2f", s["p90"])
    log.info("=" * 60)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"{s['median']:.4f}\n")
    log.info("Wrote median diameter %.4f px to %s", s["median"], out_path)


if __name__ == "__main__":
    main()
