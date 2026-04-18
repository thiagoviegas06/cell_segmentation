"""
Phase 3: Prepare Cellpose fine-tuning data from the 40 training FOVs, minus
the 6 held out for validation.

For each training FOV writes training_data/<fov_id>.npz with:
    img  : (H, W, 3) float32 — channels = [polyT, DAPI, zeros]
           (matches pipeline.py's 2-channel normalization but padded to 3 so
            Cellpose can load it with a single configurable channel spec.)
    mask : (H, W) int32      — 0 = background, 1..N = cell index at z=2

Rasterization reuses build_gt_labels.rasterize_fov_z so the masks here are
bit-identical to the GT masks local_eval.py scores against — no silent drift
between "what the model trains on" and "what we grade it with".

Only the z=2 polygon is rasterized (per Phase 3 spec). Cells whose z=2
polygon is empty/degenerate are dropped from the training mask; this is
logged per-FOV so the dropout rate is visible.

Also renders N QC PNGs under training_data/qc/<fov>.png: DAPI grayscale with
z=2 polygon outlines overlaid. Purpose is a visual check that polygons land
on nuclei before training anything.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Project imports — add project root and scripts/ to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline import load_fov_images, normalize_image  # noqa: E402
from build_gt_labels import (  # noqa: E402
    load_cell_fov_map,
    parse_boundary,
    rasterize_fov_z,
)

IMG_H = 2048
IMG_W = 2048
Z_PLANE = 2  # segmentation plane

log = logging.getLogger("prep_training_data")


def render_qc_overlay(
    dapi_norm: np.ndarray,
    cells_in_fov: pd.DataFrame,
    fov_x: float,
    fov_y: float,
    pixel_size: float,
    out_path: Path,
    title: str,
) -> None:
    """DAPI grayscale + z=2 polygon outlines in magenta. PNG."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.imshow(dapi_norm, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    bx_col = f"boundaryX_z{Z_PLANE}"
    by_col = f"boundaryY_z{Z_PLANE}"
    n_drawn = 0
    for _, row in cells_in_fov.iterrows():
        bx = parse_boundary(row.get(bx_col))
        by = parse_boundary(row.get(by_col))
        if bx is None or by is None or len(bx) < 3 or len(bx) != len(by):
            continue
        img_row = IMG_H - (bx - fov_x) / pixel_size
        img_col = (by - fov_y) / pixel_size
        img_row_c = np.append(img_row, img_row[0])
        img_col_c = np.append(img_col, img_col[0])
        ax.plot(img_col_c, img_row_c, linewidth=0.5, color="magenta", alpha=0.85)
        n_drawn += 1
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)  # image orientation: row 0 at top
    ax.set_title(f"{title} — {n_drawn} polygons", fontsize=10)
    ax.set_axis_off()
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/scratch/pl2820/data/competition")
    ap.add_argument(
        "--val_fovs",
        default="/scratch/tjv235/cell_segmentation/val_fovs.txt",
    )
    ap.add_argument(
        "--out_dir",
        default="/scratch/tjv235/cell_segmentation/training_data",
    )
    ap.add_argument(
        "--qc_count", type=int, default=4,
        help="Number of FOVs to render QC overlays for (evenly spaced across "
             "sorted training FOVs by cell count). 0 disables QC.",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Re-save FOVs whose .npz already exists (default: skip).",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    qc_dir = out_dir / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # ---- Shared references ----
    val_fovs = {ln.strip() for ln in Path(args.val_fovs).read_text().splitlines()
                if ln.strip()}
    log.info("Val FOVs to exclude (%d): %s", len(val_fovs), sorted(val_fovs))

    h5ad_path = data_root / "train" / "ground_truth" / "counts_train.h5ad"
    bounds_path = data_root / "train" / "ground_truth" / "cell_boundaries_train.csv"
    meta_path = data_root / "reference" / "fov_metadata.csv"

    log.info("Loading cell->FOV map from %s", h5ad_path)
    cell_fov = load_cell_fov_map(h5ad_path)

    log.info("Loading cell boundaries from %s", bounds_path)
    bounds = pd.read_csv(bounds_path, dtype={"Unnamed: 0": str})
    bounds = bounds.rename(columns={"Unnamed: 0": "cell_id"}).set_index("cell_id")
    bounds["fov"] = cell_fov.reindex(bounds.index).values

    log.info("Loading fov_metadata from %s", meta_path)
    meta = pd.read_csv(meta_path).set_index("fov")

    # ---- Resolve training FOVs (= all FOVs with GT cells, minus val) ----
    all_fovs = sorted(bounds["fov"].dropna().unique())
    train_fovs = [f for f in all_fovs if f not in val_fovs]
    log.info("Training FOVs: %d (of %d total with GT)",
             len(train_fovs), len(all_fovs))

    # ---- Pick QC subset: evenly spaced by cell count (stratified) ----
    cells_per_fov = bounds["fov"].value_counts()
    train_sorted = sorted(train_fovs, key=lambda f: int(cells_per_fov.get(f, 0)))
    qc_set: set[str] = set()
    if args.qc_count > 0 and train_sorted:
        qc_idx = np.linspace(0, len(train_sorted) - 1, args.qc_count).round().astype(int)
        qc_set = {train_sorted[int(i)] for i in qc_idx}
    log.info("QC FOVs (%d): %s", len(qc_set), sorted(qc_set))

    # ---- Loop ----
    t0 = time.time()
    ok_count = 0
    fail_count = 0
    total_cells_rasterized = 0
    total_cells_gt = 0
    failures: list[tuple[str, str]] = []

    for fi, fov_id in enumerate(train_fovs):
        out_path = out_dir / f"{fov_id}.npz"
        cells_in_fov = bounds[bounds["fov"] == fov_id]
        n_cells_gt = len(cells_in_fov)
        total_cells_gt += n_cells_gt

        if out_path.exists() and not args.overwrite:
            log.info("  [%02d/%d] %s: .npz exists — skipping (use --overwrite)",
                     fi + 1, len(train_fovs), fov_id)
            ok_count += 1
            continue

        if fov_id not in meta.index:
            log.warning("  [%02d/%d] %s: missing from fov_metadata — skipping",
                        fi + 1, len(train_fovs), fov_id)
            fail_count += 1
            failures.append((fov_id, "missing fov_metadata"))
            continue

        fov_meta = meta.loc[fov_id]
        fov_x = float(fov_meta["fov_x"])
        fov_y = float(fov_meta["fov_y"])
        px = float(fov_meta["pixel_size"])

        fov_dir = data_root / "train" / fov_id
        if not fov_dir.exists():
            log.warning("  [%02d/%d] %s: FOV dir missing: %s — skipping",
                        fi + 1, len(train_fovs), fov_id, fov_dir)
            fail_count += 1
            failures.append((fov_id, "missing fov_dir"))
            continue

        t_fov = time.time()
        try:
            dapi, polyt = load_fov_images(fov_dir)
        except Exception as e:
            log.error("  [%02d/%d] %s: image load failed: %s",
                      fi + 1, len(train_fovs), fov_id, e)
            fail_count += 1
            failures.append((fov_id, f"image load: {e}"))
            continue

        dapi_n = normalize_image(dapi)
        polyt_n = normalize_image(polyt)

        mask, cell_ids = rasterize_fov_z(cells_in_fov, Z_PLANE, fov_x, fov_y, px)
        n_rasterized = len(cell_ids)
        total_cells_rasterized += n_rasterized

        img = np.stack(
            [polyt_n, dapi_n, np.zeros_like(dapi_n, dtype=np.float32)],
            axis=-1,
        ).astype(np.float32)

        np.savez(out_path, img=img, mask=mask.astype(np.int32))
        ok_count += 1

        if fov_id in qc_set:
            render_qc_overlay(
                dapi_n, cells_in_fov, fov_x, fov_y, px,
                qc_dir / f"{fov_id}.png",
                title=f"{fov_id}  DAPI + z={Z_PLANE} polygons",
            )

        dropout = n_cells_gt - n_rasterized
        log.info(
            "  [%02d/%d] %s: cells_gt=%d  rasterized_z%d=%d  dropout=%d  "
            "mask_max=%d  %.2fs",
            fi + 1, len(train_fovs), fov_id,
            n_cells_gt, Z_PLANE, n_rasterized, dropout,
            int(mask.max()), time.time() - t_fov,
        )

    dt = time.time() - t0
    log.info("=" * 60)
    log.info("Phase 3 training-data prep complete in %.1fs", dt)
    log.info("  FOVs prepared (incl. pre-existing): %d", ok_count)
    log.info("  FOVs failed:   %d", fail_count)
    log.info("  Total GT cells across prepared FOVs:     %d", total_cells_gt)
    log.info("  Total cells rasterized into training masks: %d (dropout = %d, %.1f%%)",
             total_cells_rasterized,
             total_cells_gt - total_cells_rasterized,
             100.0 * (total_cells_gt - total_cells_rasterized) / max(total_cells_gt, 1))
    log.info("  Output dir: %s", out_dir)
    log.info("  QC PNGs:    %s", qc_dir)
    if failures:
        log.warning("Failures (%d):", len(failures))
        for fov, reason in failures:
            log.warning("  %s: %s", fov, reason)


if __name__ == "__main__":
    main()
