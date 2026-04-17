"""
Pick 6 held-out validation FOVs from the 40 training FOVs.

Strategy: stratify by cells-per-FOV (a proxy for tissue density). Sort ascending,
then pick at 6 evenly-spaced quantile positions so the chosen FOVs span the full
range of densities (thin tissue -> dense tissue). Deterministic, no randomness.

Writes val_fovs.txt (one FOV_XXX per line) and prints the stratification table.
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

log = logging.getLogger("make_val_split")


def cells_per_fov(h5ad_path: Path) -> pd.Series:
    with h5py.File(h5ad_path, "r") as f:
        codes = f["obs/fov/codes"][:]
        cats = np.array([x.decode() if isinstance(x, bytes) else x
                         for x in f["obs/fov/categories"][:]])
    fov = cats[codes]
    return pd.Series(fov).value_counts().sort_index()


def spots_per_fov(spots_path: Path) -> pd.Series:
    return (
        pd.read_csv(spots_path, usecols=["fov"])["fov"]
        .value_counts()
        .sort_index()
    )


def pick_stratified(sorted_fovs: list[str], n: int) -> list[str]:
    """Pick n FOVs at evenly-spaced positions from a sorted list.

    For n=6 and len=40, picks positions round(linspace(0, 39, 6)) =
    [0, 8, 16, 23, 31, 39] — spanning low to high density.
    """
    L = len(sorted_fovs)
    if n > L:
        raise ValueError(f"n={n} > L={L}")
    idx = np.round(np.linspace(0, L - 1, n)).astype(int)
    # Guard against duplicate indices on small L
    idx = np.unique(idx)
    return [sorted_fovs[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/scratch/pl2820/data/competition")
    ap.add_argument("--output", default="/scratch/tjv235/cell_segmentation/val_fovs.txt")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    data_root = Path(args.data_root)
    h5ad_path = data_root / "train" / "ground_truth" / "counts_train.h5ad"
    spots_path = data_root / "train" / "ground_truth" / "spots_train.csv"
    out_path = Path(args.output)

    cells = cells_per_fov(h5ad_path).rename("n_cells")
    spots = spots_per_fov(spots_path).rename("n_spots")
    df = pd.concat([cells, spots], axis=1).fillna(0).astype(int)
    df["cells_per_spot_x1000"] = (1000 * df["n_cells"] / df["n_spots"].clip(lower=1)).round(2)

    # Sort by cell count ascending; tie-break by spot count ascending for determinism.
    df = df.sort_values(["n_cells", "n_spots"])
    sorted_fovs = df.index.tolist()

    selected = pick_stratified(sorted_fovs, args.n)

    log.info("Full density distribution (sorted ascending by n_cells):")
    for i, fov in enumerate(sorted_fovs):
        marker = "  <-- SELECTED" if fov in selected else ""
        row = df.loc[fov]
        log.info("  [%02d] %s  cells=%3d  spots=%6d  c/s*1000=%5.2f%s",
                 i, fov, row["n_cells"], row["n_spots"],
                 row["cells_per_spot_x1000"], marker)

    log.info("Selected %d val FOVs: %s", len(selected), selected)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for fov in selected:
            f.write(fov + "\n")
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
