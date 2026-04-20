"""
Phase 3 (v2): Prepare Cellpose fine-tuning data from all 5 z-planes.

Writes one .npz per (FOV, z) pair under training_data/:
    FOV_XXX_z0.npz
    FOV_XXX_z1.npz
    FOV_XXX_z2.npz
    FOV_XXX_z3.npz
    FOV_XXX_z4.npz

Each file contains:
    img  : (H, W, 3) float32 — [polyT_z, DAPI_z, zeros]
    mask : (H, W) int32       — 0 = bg, 1..N = cell index at z

The .dax stain file (Epi-750s5-635s5-545s1-473s5-408s5_<fov>.dax) is a
27-frame uint16 stack. Empirically verified layout (2 preamble frames +
5 z-blocks of 5 stain channels each):
    polyT[z] = frame 5 + 5*z
    DAPI[z]  = frame 6 + 5*z
The z=2 frames are 15 (polyT) and 16 (DAPI), matching pipeline.py constants.

Rasterization reuses build_gt_labels.rasterize_fov_z so training masks are
bit-identical to the GT masks local_eval.py scores against.

(FOV, z) pairs with an empty mask (mask.max() == 0 — no cell polygon at
this z in this FOV) are SKIPPED with a warning: empty-mask images teach
the model to under-segment.

QC PNGs still render the z=2 overlay for a handful of FOVs (one per QC
FOV, not one per z-plane).
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

from pipeline import load_dax_frame, normalize_image  # noqa: E402
from build_gt_labels import (  # noqa: E402
    load_cell_fov_map,
    parse_boundary,
    rasterize_fov_z,
)

IMG_H = 2048
IMG_W = 2048
Z_PLANES = [0, 1, 2, 3, 4]
QC_Z = 2  # QC overlay uses the z=2 polygons (reference plane)

log = logging.getLogger("prep_training_data")


def polyT_frame(z: int) -> int:
    return 5 + 5 * z


def DAPI_frame(z: int) -> int:
    return 6 + 5 * z


def render_qc_overlay(
    dapi_norm: np.ndarray,
    cells_in_fov: pd.DataFrame,
    fov_x: float,
    fov_y: float,
    pixel_size: float,
    out_path: Path,
    title: str,
) -> None:
    """DAPI grayscale (z=QC_Z) + z=QC_Z polygon outlines in magenta. PNG."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.imshow(dapi_norm, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    bx_col = f"boundaryX_z{QC_Z}"
    by_col = f"boundaryY_z{QC_Z}"
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
    ax.set_ylim(IMG_H, 0)
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
        help="Re-save .npz files that already exist (default: skip).",
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
    log.info("Training FOVs: %d (of %d total with GT). Z-planes per FOV: %d",
             len(train_fovs), len(all_fovs), len(Z_PLANES))
    log.info("Expected output files (pre-filter): %d", len(train_fovs) * len(Z_PLANES))

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
    written = 0
    skipped_empty: list[tuple[str, int]] = []  # (fov, z) with mask.max() == 0
    failures: list[tuple[str, str]] = []        # (fov, reason) at FOV-level
    total_cells_rasterized = 0

    for fi, fov_id in enumerate(train_fovs):
        t_fov = time.time()
        cells_in_fov = bounds[bounds["fov"] == fov_id]
        n_cells_gt = len(cells_in_fov)

        if fov_id not in meta.index:
            log.warning("  [%02d/%d] %s: missing from fov_metadata — skipping all z",
                        fi + 1, len(train_fovs), fov_id)
            failures.append((fov_id, "missing fov_metadata"))
            continue

        fov_meta = meta.loc[fov_id]
        fov_x = float(fov_meta["fov_x"])
        fov_y = float(fov_meta["fov_y"])
        px = float(fov_meta["pixel_size"])

        fov_dir = data_root / "train" / fov_id
        if not fov_dir.exists():
            log.warning("  [%02d/%d] %s: FOV dir missing: %s — skipping all z",
                        fi + 1, len(train_fovs), fov_id, fov_dir)
            failures.append((fov_id, "missing fov_dir"))
            continue

        per_z_summary: list[str] = []
        dapi_qc: np.ndarray | None = None   # z=QC_Z DAPI for overlay (if FOV in qc_set)

        for z in Z_PLANES:
            out_path = out_dir / f"{fov_id}_z{z}.npz"
            if out_path.exists() and not args.overwrite:
                per_z_summary.append(f"z{z}=exists")
                # still count rasterized cells for bookkeeping? read lazily
                continue

            try:
                polyt_raw = load_dax_frame(fov_dir, polyT_frame(z))
                dapi_raw = load_dax_frame(fov_dir, DAPI_frame(z))
            except Exception as e:
                log.error("  [%02d/%d] %s z=%d: image load failed: %s",
                          fi + 1, len(train_fovs), fov_id, z, e)
                failures.append((f"{fov_id}_z{z}", f"image load: {e}"))
                continue

            polyt_n = normalize_image(polyt_raw)
            dapi_n = normalize_image(dapi_raw)
            if z == QC_Z and fov_id in qc_set:
                dapi_qc = dapi_n

            mask, cell_ids = rasterize_fov_z(cells_in_fov, z, fov_x, fov_y, px)
            n_rasterized = len(cell_ids)

            if int(mask.max()) == 0:
                log.warning("  [%02d/%d] %s z=%d: EMPTY MASK "
                            "(no valid polygons at this z) — skipping file",
                            fi + 1, len(train_fovs), fov_id, z)
                skipped_empty.append((fov_id, z))
                per_z_summary.append(f"z{z}=EMPTY")
                continue

            img = np.stack(
                [polyt_n, dapi_n, np.zeros_like(dapi_n, dtype=np.float32)],
                axis=-1,
            ).astype(np.float32)

            np.savez(out_path, img=img, mask=mask.astype(np.int32))
            written += 1
            total_cells_rasterized += n_rasterized
            per_z_summary.append(f"z{z}={n_rasterized}")

        if fov_id in qc_set and dapi_qc is not None:
            render_qc_overlay(
                dapi_qc, cells_in_fov, fov_x, fov_y, px,
                qc_dir / f"{fov_id}.png",
                title=f"{fov_id}  DAPI + z={QC_Z} polygons",
            )

        log.info(
            "  [%02d/%d] %s: cells_gt=%d  per_z=[%s]  %.2fs",
            fi + 1, len(train_fovs), fov_id, n_cells_gt,
            " ".join(per_z_summary), time.time() - t_fov,
        )

    dt = time.time() - t0
    produced = sorted(out_dir.glob("FOV_*_z*.npz"))
    log.info("=" * 60)
    log.info("Phase 3 (multi-z) training-data prep complete in %.1fs (= %.1f min)",
             dt, dt / 60.0)
    log.info("  .npz files on disk: %d", len(produced))
    log.info("  Newly written this run: %d", written)
    log.info("  Skipped for empty mask (FOV,z): %d", len(skipped_empty))
    for fov_id, z in skipped_empty:
        log.info("    - %s z=%d", fov_id, z)
    log.info("  Cells rasterized (newly-written files): %d", total_cells_rasterized)
    if failures:
        log.warning("FOV-level failures (%d):", len(failures))
        for fov_or_key, reason in failures:
            log.warning("  %s: %s", fov_or_key, reason)
    log.info("  Output dir: %s", out_dir)
    log.info("  QC PNGs:    %s", qc_dir)


if __name__ == "__main__":
    main()
