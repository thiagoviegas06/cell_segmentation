"""
Phase 2.2: produce 3D-stitched segmentation masks for every Phase 2 FOV
(60 train + 10 test = 70 FOVs) using the Phase 1 fine-tuned cpsam checkpoint.

The model itself is unchanged — same brain, same staining, same imaging
setup, so segmentation quality should track Phase 1 (0.83 LB / 0.7544 local).
This script is a one-time mask materializer; downstream Phase 2 work
(per-cell gene vectors, classifier training) will read from the cached
.npy files and not re-run inference.

Outputs:
    cache/masks_phase2/<FOV_ID>.npy      # int mask stack, shape (5, 2048, 2048)
    cache/masks_phase2/cell_counts.csv   # fov, split, n_cells, seconds

Usage:
    python scripts/phase2/segment_all.py --gpu
    python scripts/phase2/segment_all.py --gpu --fovs FOV_E      # smoke test
    python scripts/phase2/segment_all.py --gpu --fovs FOV_E FOV_101 --overwrite

Frame map (verified Phase 2.1 against Phase 1 FOV_A):
    polyT[z] = 5 + 5*z   (frames 5, 10, 15, 20, 25)
    DAPI[z]  = 6 + 5*z   (frames 6, 11, 16, 21, 26)
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
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline import load_dax_frame, normalize_image  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

N_Z = 5
POLY_T_FRAMES = [5 + 5 * z for z in range(N_Z)]
DAPI_FRAMES = [6 + 5 * z for z in range(N_Z)]
IMG_HW = (2048, 2048)
UINT16_MAX = np.iinfo(np.uint16).max  # 65535


def discover_fovs(data_root: Path) -> list[tuple[str, str, Path]]:
    """Return [(fov_id, split, fov_dir), ...] across train + test, sorted."""
    fovs: list[tuple[str, str, Path]] = []
    for split in ("train", "test"):
        split_dir = data_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split dir: {split_dir}")
        for d in sorted(split_dir.iterdir()):
            if d.is_dir() and d.name.startswith("FOV_") and d.name != "ground_truth":
                fovs.append((d.name, split, d))
    return fovs


def load_stack(fov_dir: Path) -> np.ndarray:
    """Build (5, H, W, 3) float32 input: ch0=polyT_norm, ch1=DAPI_norm, ch2=zeros."""
    slices = []
    for z in range(N_Z):
        polyt = load_dax_frame(fov_dir, POLY_T_FRAMES[z])
        dapi = load_dax_frame(fov_dir, DAPI_FRAMES[z])
        polyt_n = normalize_image(polyt)
        dapi_n = normalize_image(dapi)
        zeros = np.zeros_like(polyt_n)
        slices.append(np.stack([polyt_n, dapi_n, zeros], axis=-1))
    return np.stack(slices, axis=0)


def segment_one(model, fov_dir: Path, stitch_threshold: float,
                diameter: Optional[float]) -> tuple[np.ndarray, np.ndarray]:
    stack = load_stack(fov_dir)

    # Register forward hook on the final layer of the backbone to capture feature maps
    feature_maps = []
    def hook_fn(module, inputs, output):
        # inputs[0] is the feature map before the final output convolution
        feature_maps.append(inputs[0].detach().cpu().numpy())

    # In CellposeModel, the resnet backbone is in .net.
    # The output block is usually the last part of the resnet.
    hook = model.net.output.register_forward_hook(hook_fn)

    try:
        masks, _flows, _styles = model.eval(
            stack,
            diameter=diameter,
            channel_axis=-1,
            z_axis=0,
            do_3D=False,
            stitch_threshold=stitch_threshold,
            normalize=False,
        )
    finally:
        hook.remove()

    mask = np.asarray(masks)
    if mask.ndim != 3 or mask.shape != (N_Z, *IMG_HW):
        raise ValueError(
            f"Expected mask of shape (5, 2048, 2048), got {mask.shape}"
        )

    # Extract Per-Cell Embeddings (ROI Pooling)
    # feature_maps might contain multiple batches if Cellpose tiles or chunks the stack.
    feat_map = np.concatenate(feature_maps, axis=0)  # Shape: (Z, C, H, W)
    Z, C, H, W = feat_map.shape

    # If spatial dimensions don't match exactly (e.g. due to padding), resize
    if (H, W) != IMG_HW:
        log.warning("Feature map shape %s != %s, resizing for ROI pooling", (H, W), IMG_HW)
        import cv2
        # Resize each slice/channel
        resized = np.empty((Z, C, *IMG_HW), dtype=feat_map.dtype)
        for z in range(Z):
            for c in range(C):
                resized[z, c] = cv2.resize(feat_map[z, c], (IMG_HW[1], IMG_HW[0]),
                                           interpolation=cv2.INTER_LINEAR)
        feat_map = resized

    n_cells = int(mask.max())
    embeddings = np.zeros((n_cells, C), dtype=np.float32)

    # Flatten spatial dims to vectorize masking
    feat_flat = feat_map.transpose(1, 0, 2, 3).reshape(C, -1)
    mask_flat = mask.reshape(-1)

    for cell_id in range(1, n_cells + 1):
        idx = (mask_flat == cell_id)
        if idx.any():
            embeddings[cell_id - 1] = feat_flat[:, idx].mean(axis=1)

    return mask, embeddings


def to_packed(mask: np.ndarray) -> np.ndarray:
    """uint16 if cell count fits, else int32. uint16 saves ~half the disk."""
    n_cells = int(mask.max())
    if n_cells <= UINT16_MAX:
        return mask.astype(np.uint16, copy=False)
    log.warning("FOV has %d cells > uint16 max — falling back to int32", n_cells)
    return mask.astype(np.int32, copy=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 bulk 3D-stitched segmentation")
    ap.add_argument(
        "--pretrained_model",
        default=str(_PROJECT_ROOT / "runs" / "phase4_v1_h200" / "checkpoints" / "best.pt"),
        help="Path to fine-tuned cpsam .pt checkpoint",
    )
    ap.add_argument(
        "--data_root",
        default="/scratch/pl2820/data/competition_phase2",
        help="Phase 2 dataset root (contains train/, test/)",
    )
    ap.add_argument(
        "--output_dir",
        default=str(_PROJECT_ROOT / "cache" / "masks_phase2"),
        help="Where to save <FOV_ID>.npy and cell_counts.csv",
    )
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--diameter", type=float, default=None,
                    help="Cell diameter hint (px). Omit/negative for auto.")
    ap.add_argument("--stitch_threshold", type=float, default=0.3)
    ap.add_argument(
        "--fovs", nargs="+", default=None,
        help="Optional FOV subset (e.g. FOV_E FOV_101). Default: all 70.",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Re-segment FOVs whose .npy already exists (default: skip).",
    )
    args = ap.parse_args()

    pretrained_model = Path(args.pretrained_model)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    if not pretrained_model.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_model}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    diameter = args.diameter if (args.diameter is not None and args.diameter >= 0) else None

    all_fovs = discover_fovs(data_root)
    if args.fovs is not None:
        wanted = set(args.fovs)
        all_fovs = [t for t in all_fovs if t[0] in wanted]
        missing = wanted - {t[0] for t in all_fovs}
        if missing:
            raise ValueError(f"Requested FOVs not found in data_root: {sorted(missing)}")
    if not all_fovs:
        raise RuntimeError("No FOVs to process")

    log.info("Found %d FOVs: %d train, %d test",
             len(all_fovs),
             sum(1 for _, s, _ in all_fovs if s == "train"),
             sum(1 for _, s, _ in all_fovs if s == "test"))

    # Lazy import — keeps --help fast on machines without the cellpose env.
    from cellpose import models

    eval_params = inspect.signature(models.CellposeModel.eval).parameters
    if "stitch_threshold" not in eval_params:
        raise RuntimeError(
            "Installed cellpose model.eval() is missing 'stitch_threshold'; "
            "3D stitching impossible. Params: " + ", ".join(eval_params.keys())
        )

    log.info("Loading model: pretrained=%s gpu=%s stitch=%.2f diameter=%s",
             pretrained_model, args.gpu, args.stitch_threshold, diameter)
    t0 = time.time()
    model = models.CellposeModel(gpu=args.gpu, pretrained_model=str(pretrained_model))
    log.info("  model ready in %.2fs", time.time() - t0)

    rows: list[dict] = []
    t_start = time.time()
    for i, (fov_id, split, fov_dir) in enumerate(all_fovs):
        out_path = output_dir / f"{fov_id}.npy"
        if out_path.exists() and not args.overwrite:
            existing = np.load(out_path, mmap_mode="r")
            n_cells = int(existing.max())
            log.info("[%d/%d] %s (%s): skip — %s already exists (%d cells)",
                     i + 1, len(all_fovs), fov_id, split, out_path.name, n_cells)
            rows.append({"fov": fov_id, "split": split, "n_cells": n_cells,
                         "seconds": 0.0, "skipped": True})
            continue

        t_fov = time.time()
        try:
            mask, embeddings = segment_one(model, fov_dir, args.stitch_threshold, diameter)
        except Exception as e:
            log.exception("[%d/%d] %s (%s): FAILED — %s", i + 1, len(all_fovs),
                          fov_id, split, e)
            raise
        elapsed = time.time() - t_fov
        n_cells = int(mask.max())
        packed = to_packed(mask)
        np.save(out_path, packed)
        
        embed_path = output_dir / f"{fov_id}_embeddings.npy"
        np.save(embed_path, embeddings)
        
        log.info("[%d/%d] %s (%s): %d cells, saved mask + %d-dim embeddings in %.1fs",
                 i + 1, len(all_fovs), fov_id, split, n_cells,
                 embeddings.shape[1] if n_cells > 0 else 0, elapsed)
        rows.append({"fov": fov_id, "split": split, "n_cells": n_cells,
                     "seconds": round(elapsed, 2), "skipped": False})

    total = time.time() - t_start
    counts_path = output_dir / "cell_counts.csv"
    df = pd.DataFrame(rows).sort_values(["split", "fov"]).reset_index(drop=True)
    df.to_csv(counts_path, index=False)

    log.info("=== Done ===")
    log.info("Total wall time: %.1fs (%.1f min) over %d FOVs",
             total, total / 60.0, len(rows))
    log.info("Cell counts: train sum=%d (60 FOVs), test sum=%d (10 FOVs), grand sum=%d",
             int(df.loc[df["split"] == "train", "n_cells"].sum()),
             int(df.loc[df["split"] == "test", "n_cells"].sum()),
             int(df["n_cells"].sum()))
    log.info("Per-FOV cell counts written to %s", counts_path)


if __name__ == "__main__":
    main()
