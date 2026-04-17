"""
Segmenter factories used by local_eval.py.

Each factory returns a callable `segment(fov_dir: Path) -> np.ndarray` producing
a (2048, 2048) int mask where 0 = background, >0 = cell id. Model state is held
in the closure so it's reused across FOVs (one load, many calls).

Reference by '<module>:<factory>' on local_eval.py's --segmenter flag.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure project root (which holds pipeline.py) is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def build_all_background(gpu: bool = False, diameter: Optional[float] = None):
    """Factory: returns a segmenter that labels every pixel as background.

    Used only to smoke-test local_eval.py plumbing without Cellpose/GPU. ARI
    against GT should be ~0 (one cluster == background only, except where GT
    also has a single cell — ARI is degenerate here but computes fine).
    """
    def _segment(fov_dir: Path) -> np.ndarray:
        return np.zeros((2048, 2048), dtype=np.int32)
    _segment.name = "all_background"
    return _segment


def build_cellpose_zeroshot(gpu: bool = False, diameter: Optional[float] = None):
    """Factory: zero-shot Cellpose (cpsam) on polyT+DAPI, single z-plane."""
    from cellpose import models  # imported lazily so non-seg tasks don't need cellpose
    from pipeline import load_fov_images, segment_fov  # reuse existing pipeline funcs

    model = models.CellposeModel(gpu=gpu, pretrained_model="cpsam")

    def _segment(fov_dir: Path) -> np.ndarray:
        dapi, polyt = load_fov_images(fov_dir)
        return segment_fov(dapi, polyt, model, diameter=diameter)

    _segment.name = f"cellpose_zeroshot(gpu={gpu}, diameter={diameter})"
    return _segment
