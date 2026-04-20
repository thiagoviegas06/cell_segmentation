"""
Segmenter factories used by local_eval.py.

Each factory returns a callable `segment(fov_dir: Path) -> np.ndarray` producing
either a (2048, 2048) int mask (2D, single z-plane) or a (Z, 2048, 2048) int
mask (3D stitched across z). 0 = background, >0 = cell id. Model state is held
in the closure so it's reused across FOVs (one load, many calls).

Reference by '<module>:<factory>' on local_eval.py's --segmenter flag.
"""

import inspect
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure project root (which holds pipeline.py) is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def build_all_background(gpu: bool = False, diameter: Optional[float] = None, **_):
    """Factory: returns a segmenter that labels every pixel as background.

    Used only to smoke-test local_eval.py plumbing without Cellpose/GPU. ARI
    against GT should be ~0 (one cluster == background only, except where GT
    also has a single cell — ARI is degenerate here but computes fine).
    """
    def _segment(fov_dir: Path) -> np.ndarray:
        return np.zeros((2048, 2048), dtype=np.int32)
    _segment.name = "all_background"
    return _segment


def build_cellpose_zeroshot(gpu: bool = False, diameter: Optional[float] = None, **_):
    """Factory: zero-shot Cellpose (cpsam) on polyT+DAPI, single z-plane."""
    from cellpose import models  # imported lazily so non-seg tasks don't need cellpose
    from pipeline import load_fov_images, segment_fov  # reuse existing pipeline funcs

    model = models.CellposeModel(gpu=gpu, pretrained_model="cpsam")

    def _segment(fov_dir: Path) -> np.ndarray:
        dapi, polyt = load_fov_images(fov_dir)
        return segment_fov(dapi, polyt, model, diameter=diameter)

    _segment.name = f"cellpose_zeroshot(gpu={gpu}, diameter={diameter})"
    return _segment


def build_cellpose_finetuned(
    gpu: bool = False,
    diameter: Optional[float] = None,
    pretrained_model: Optional[str] = None,
    **_,
):
    """Factory: Cellpose loaded from a custom fine-tuned weights file.

    `pretrained_model` is required and must point to a .pt checkpoint produced
    by scripts/train_cellpose.py (or any cellpose save_model output). Pass via
    local_eval.py's --segmenter_kwargs flag, e.g.
        --segmenter_kwargs "pretrained_model=/path/to/runs/<name>/best.pt"
    """
    if not pretrained_model:
        raise ValueError(
            "build_cellpose_finetuned requires pretrained_model path "
            "(pass via --segmenter_kwargs pretrained_model=<path>)"
        )
    from cellpose import models
    from pipeline import load_fov_images, segment_fov

    model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)

    def _segment(fov_dir: Path) -> np.ndarray:
        dapi, polyt = load_fov_images(fov_dir)
        return segment_fov(dapi, polyt, model, diameter=diameter)

    _segment.name = (
        f"cellpose_finetuned({Path(pretrained_model).name}, "
        f"gpu={gpu}, diameter={diameter})"
    )
    return _segment


def build_cellpose_finetuned_3d(
    gpu: bool = False,
    diameter: Optional[float] = None,
    pretrained_model: Optional[str] = None,
    stitch_threshold: float = 0.3,
    **_,
):
    """Factory: fine-tuned Cellpose segmenting all 5 z-planes, stitched across z.

    Phase 5: replaces the single-z mask lookup with a per-spot z-plane lookup.
    Loads polyT+DAPI at each z from the .dax, normalizes per-z, stacks to
    (Z=5, H, W, C=3), and calls model.eval with do_3D=False + stitch_threshold
    so 2D masks at adjacent z's get linked by IoU (same cell -> same id where
    linked). Returns a (5, H, W) int mask indexable by each spot's global_z.

    Frame map (Phase 3 verified): polyT[z]=5+5z, DAPI[z]=6+5z.

    `pretrained_model` path (required), `stitch_threshold` default 0.3.
    Pass overrides via local_eval.py's --segmenter_kwargs, e.g.
        --segmenter_kwargs "pretrained_model=/path/best.pt,stitch_threshold=0.25"
    """
    if not pretrained_model:
        raise ValueError(
            "build_cellpose_finetuned_3d requires pretrained_model path "
            "(pass via --segmenter_kwargs pretrained_model=<path>)"
        )
    # segmenter_kwargs values come in as strings; coerce numeric args.
    stitch_threshold = float(stitch_threshold)

    from cellpose import models
    from pipeline import load_dax_frame, normalize_image

    # Fail fast if this cellpose build doesn't expose stitch_threshold — the
    # Phase 5 plan's fallback ("Option A: per-z independent segmentation") is
    # a different factory, so we'd rather error here than silently produce
    # unstitched masks.
    eval_params = inspect.signature(models.CellposeModel.eval).parameters
    if "stitch_threshold" not in eval_params:
        raise RuntimeError(
            "Installed cellpose model.eval() is missing 'stitch_threshold' "
            "— falling back to independent per-z segmentation (Option A) is "
            "required. Parameters found: " + ", ".join(eval_params.keys())
        )

    model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)

    N_Z = 5
    POLY_T_FRAMES = [5 + 5 * z for z in range(N_Z)]   # 5,10,15,20,25
    DAPI_FRAMES   = [6 + 5 * z for z in range(N_Z)]   # 6,11,16,21,26

    def _segment(fov_dir: Path) -> np.ndarray:
        # Build (Z, H, W, 3) input stack.
        slices = []
        for z in range(N_Z):
            polyt = load_dax_frame(fov_dir, POLY_T_FRAMES[z])
            dapi = load_dax_frame(fov_dir, DAPI_FRAMES[z])
            polyt_n = normalize_image(polyt)
            dapi_n = normalize_image(dapi)
            zeros = np.zeros_like(polyt_n)
            slices.append(np.stack([polyt_n, dapi_n, zeros], axis=-1))
        stack = np.stack(slices, axis=0)  # (Z, H, W, 3)

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
        if mask.shape != (N_Z, 2048, 2048):
            raise ValueError(
                f"Expected stitched mask shape ({N_Z}, 2048, 2048), got {mask.shape}"
            )
        return mask

    _segment.name = (
        f"cellpose_finetuned_3d({Path(pretrained_model).name}, "
        f"gpu={gpu}, diameter={diameter}, stitch={stitch_threshold})"
    )
    return _segment
