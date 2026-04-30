"""
Phase 4: Fine-tune cpsam on the Phase-3 training data.

Loads the .npz files written by prep_training_data.py, splits off a small
in-training val set (2 FOVs, picked deterministically at the 1/3 and 2/3
quantiles of the sorted-by-cell-count training list — distinct from the
Phase 1 val_fovs.txt so those remain untouched for final ARI eval), and
runs cellpose.train.train_seg with the Phase-4 hyperparameters.

Per-epoch train/test loss is logged via cellpose's built-in train_logger to
runs/<name>/train.log. Intermediate checkpoints are saved every
save_every epochs (default 20) as cellpose's save_each=True files, which
this script then copies into a normalized layout at runs/<name>/checkpoints/:
    checkpoints/epoch_NNNN.pt      — one per save_every (weights only)
    checkpoints/final.pt           — last-epoch weights
    checkpoints/best.pt            — argmin(val_loss) over saved epochs
    checkpoints/best_meta.json     — {epoch, train_loss, val_loss, paths}
Caveats of Option A (no per-epoch hook in cellpose.train.train_seg):
  - checkpoints contain MODEL WEIGHTS ONLY (no optimizer state → cannot
    resume mid-run; a crash forces a fresh training job).
  - val_loss is computed by train_seg only at epoch 5 and every 10
    epochs thereafter. best.pt is picked among intersection of
    {save_every epochs} ∩ {eval epochs} — with save_every=20 that's
    exactly the set of saved epochs.
  - train_log.csv is reconstructed post-hoc from train_seg's returned
    arrays; lr is re-derived from the cellpose schedule (pinned to
    cellpose 4.1.1 semantics).

Alignment notes:
  - Inference via pipeline.segment_fov pre-normalizes with normalize_image
    and calls model.eval(..., normalize=False, channel_axis=2). Training must
    match: we pass normalize=False + channel_axis=-1 so we don't double-
    normalize relative to what the model sees at eval time.
  - Training files are FOV_XXX_zK.npz — one per (FOV, z). The val split
    holds out whole FOVs (all their z-planes) to avoid leakage between
    train/val at the image level.
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

FOV_Z_RE = re.compile(r"^(FOV_[^_]+)_z(\d+)$")

# Project imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

log = logging.getLogger("train_cellpose")


def pick_val_fovs(train_fovs: list[str], cells_per_fov: dict[str, int],
                  count: int) -> list[str]:
    """Pick `count` validation FOVs at inner quantile positions of the
    cell-count-sorted training list. Deterministic, reproducible."""
    sorted_fovs = sorted(train_fovs, key=lambda f: int(cells_per_fov.get(f, 0)))
    # inner positions: linspace(0, N-1, count+2)[1:-1]
    positions = np.linspace(0, len(sorted_fovs) - 1, count + 2)[1:-1]
    idx = [int(round(p)) for p in positions]
    return [sorted_fovs[i] for i in idx]


def cell_count_from_mask(mask_path: Path) -> int:
    """Peek at an .npz file and count cells without loading the full img."""
    with np.load(mask_path) as z:
        m = z["mask"]
        return int(m.max())


def parse_fov_z(stem: str) -> tuple[str, int] | None:
    """'FOV_001_z2' -> ('FOV_001', 2). Returns None for unrecognized stems."""
    m = FOV_Z_RE.match(stem)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def compute_lr_schedule(learning_rate: float, n_epochs: int) -> np.ndarray:
    """Replicate cellpose.train.train_seg's internal LR schedule.

    We need the per-epoch LR for train_log.csv because train_seg doesn't
    expose a callback hook. Logic mirrors cellpose/train.py lines 406-415
    (version 4.1.1) — if cellpose changes its schedule, this drifts.
    """
    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
    if n_epochs > 300:
        LR = LR[:-100]
        for _ in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    elif n_epochs > 99:
        LR = LR[:-50]
        for _ in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(5))
    return LR[:n_epochs]


def load_fov(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path) as z:
        return z["img"].astype(np.float32), z["mask"].astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    user = os.environ.get("USER", "dr3432")
    ap.add_argument(
        "--training_data",
        default=f"/scratch/{user}/cell_segmentation/training_data",
    )
    ap.add_argument("--runs_dir", default=f"/scratch/{user}/cell_segmentation/runs")
    ap.add_argument("--run_name", default=None,
                    help="Subdir under runs/ (default: phase4_<timestamp>)")
    ap.add_argument("--model_name", default="finetuned",
                    help="Cellpose save_model filename inside runs/<name>/models/")
    ap.add_argument("--pretrained_model", default="cpsam")
    ap.add_argument("--n_epochs", type=int, default=300)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--bsize", type=int, default=256,
                    help="Train-time crop size (cellpose default: 256)")
    ap.add_argument("--save_every", type=int, default=20,
                    help="Save intermediate checkpoint every N epochs. "
                         "train_seg also evals val loss at epoch 5 and every "
                         "10 epochs thereafter — best.pt is picked among eval "
                         "epochs that also have a saved checkpoint.")
    ap.add_argument("--SGD", action="store_true",
                    help="Requested for API parity; cellpose v4 ignores and "
                         "always uses AdamW.")
    ap.add_argument("--val_count", type=int, default=2,
                    help="Number of training FOVs to hold out as Phase-4 val.")
    args = ap.parse_args()

    # -- Run dir + logging --
    run_name = args.run_name or f"phase4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(name)s:%(levelname)s] %(message)s",
                            "%H:%M:%S")
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    root.addHandler(fh); root.addHandler(sh)

    log.info("Run dir: %s", run_dir)

    # Defer heavy imports (torch/cellpose) until after logging is set up
    import torch  # noqa: E402
    from cellpose import models, train  # noqa: E402

    cuda_avail = bool(torch.cuda.is_available())
    if not cuda_avail:
        log.error("CUDA unavailable — cellpose training on CPU is not viable. Aborting.")
        log.error("Submit this via SLURM (sbatch run_train_cellpose.sh).")
        sys.exit(2)

    # -- Discover training files (FOV_XXX_zK.npz) --
    data_dir = Path(args.training_data)
    npzs = sorted(data_dir.glob("FOV_*_z*.npz"))
    if not npzs:
        log.error("No FOV_*_z*.npz files under %s — run prep_training_data.py first.", data_dir)
        sys.exit(2)

    # Group by FOV ID so val split holds out whole FOVs
    files_by_fov: dict[str, list[Path]] = defaultdict(list)
    for p in npzs:
        parsed = parse_fov_z(p.stem)
        if parsed is None:
            log.warning("Skipping unrecognized filename: %s", p.name)
            continue
        fov_id, _z = parsed
        files_by_fov[fov_id].append(p)

    fov_ids = sorted(files_by_fov.keys())
    n_files = sum(len(v) for v in files_by_fov.values())
    log.info("Found %d .npz files across %d FOVs (avg %.1f z-planes/FOV)",
             n_files, len(fov_ids), n_files / max(len(fov_ids), 1))

    # -- Per-FOV cell-count ranking (use z=2 if available, else max across z) --
    log.info("Computing per-FOV cell counts (for stratified val pick)...")
    cells_per_fov: dict[str, int] = {}
    for fov, paths in files_by_fov.items():
        z2 = next((p for p in paths if p.stem.endswith("_z2")), None)
        if z2 is not None:
            cells_per_fov[fov] = cell_count_from_mask(z2)
        else:
            cells_per_fov[fov] = max(cell_count_from_mask(p) for p in paths)

    # -- Pick val split (NOT the Phase 1 val FOVs — those are final-eval only) --
    val_fovs = pick_val_fovs(fov_ids, cells_per_fov, args.val_count)
    val_set = set(val_fovs)
    train_fovs = [f for f in fov_ids if f not in val_set]
    log.info("Phase-4 val FOVs (%d, inner quantiles by cell count): %s",
             len(val_fovs), val_fovs)
    log.info("  cell counts (z=2): %s",
             {f: cells_per_fov[f] for f in val_fovs})
    log.info("Phase-4 train FOVs: %d (%d z-files)", len(train_fovs),
             sum(len(files_by_fov[f]) for f in train_fovs))
    log.info("Phase-4 val   FOVs: %d (%d z-files)", len(val_fovs),
             sum(len(files_by_fov[f]) for f in val_fovs))

    # -- Load all data into memory --
    log.info("Loading training arrays into memory...")
    t_load = time.time()
    train_imgs, train_masks = [], []
    for f in train_fovs:
        for p in sorted(files_by_fov[f]):
            img, mask = load_fov(p)
            train_imgs.append(img)
            train_masks.append(mask)
    val_imgs, val_masks = [], []
    for f in val_fovs:
        for p in sorted(files_by_fov[f]):
            img, mask = load_fov(p)
            val_imgs.append(img)
            val_masks.append(mask)
    log.info("  loaded in %.1fs — train_imgs=%d, val_imgs=%d",
             time.time() - t_load, len(train_imgs), len(val_imgs))

    # -- Persist config for reproducibility --
    config = {
        "run_name": run_name,
        "pretrained_model": args.pretrained_model,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "SGD": args.SGD,
        "batch_size": args.batch_size,
        "bsize": args.bsize,
        "save_every": args.save_every,
        "model_name": args.model_name,
        "val_count": args.val_count,
        "val_fovs": val_fovs,
        "train_fovs_count": len(train_fovs),
        "train_images_count": len(train_imgs),
        "val_images_count": len(val_imgs),
        "training_data_dir": str(data_dir),
        "cuda_available": cuda_avail,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # -- Load model --
    log.info("Loading pretrained cpsam model (pretrained_model=%r)",
             args.pretrained_model)
    t_model = time.time()
    model = models.CellposeModel(gpu=True, pretrained_model=args.pretrained_model)
    log.info("  model ready in %.1fs — device=%s, dtype=%s",
             time.time() - t_model, model.device, model.net.dtype)

    # -- Train --
    log.info("Calling cellpose.train.train_seg(n_epochs=%d, lr=%.0e, "
             "wd=%.2f, batch=%d, save_every=%d, save_each=True, normalize=False, "
             "channel_axis=-1, bsize=%d)",
             args.n_epochs, args.learning_rate, args.weight_decay,
             args.batch_size, args.save_every, args.bsize)
    t_train = time.time()
    filename, train_losses, test_losses = train.train_seg(
        net=model.net,
        train_data=train_imgs,
        train_labels=train_masks,
        test_data=val_imgs,
        test_labels=val_masks,
        channel_axis=-1,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        SGD=args.SGD,
        n_epochs=args.n_epochs,
        normalize=False,   # prep_training_data already applied normalize_image
        save_path=str(run_dir),
        save_every=args.save_every,
        save_each=True,    # keep intermediate checkpoints so we can pick best
        model_name=args.model_name,
        bsize=args.bsize,
    )
    train_secs = time.time() - t_train
    log.info("Training complete in %.1fs (= %.1f min). Final saved to %s",
             train_secs, train_secs / 60.0, filename)

    # -- train_log.csv (per-epoch) --
    # val_loss is only populated at eval epochs (cellpose evals at iepoch==5
    # and every 10 epochs thereafter); non-eval rows get an empty string.
    # lr is reconstructed from cellpose's schedule — no per-epoch hook.
    lr_schedule = compute_lr_schedule(args.learning_rate, args.n_epochs)
    train_log_path = run_dir / "train_log.csv"
    with open(train_log_path, "w") as fh_out:
        fh_out.write("epoch,train_loss,val_loss,lr\n")
        for e in range(args.n_epochs):
            vl = f"{test_losses[e]:.6f}" if test_losses[e] > 0 else ""
            fh_out.write(f"{e},{train_losses[e]:.6f},{vl},{lr_schedule[e]:.8f}\n")
    log.info("Wrote per-epoch log to %s", train_log_path)

    # -- Checkpoint directory: final.pt, epoch_NNNN.pt, best.pt + best_meta.json --
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    models_dir = run_dir / "models"

    # FINAL: cellpose writes the last-epoch weights to models/<model_name>
    final_src = models_dir / args.model_name
    final_dst = ckpt_dir / "final.pt"
    if final_src.exists():
        shutil.copy(final_src, final_dst)
        log.info("Copied final weights → %s", final_dst)
    else:
        log.error("Final weights not found at %s — training may have failed.",
                  final_src)

    # PERIODIC: copy each intermediate checkpoint to checkpoints/epoch_NNNN.pt.
    # save_each=True + save_every=N gives files named <model_name>_epoch_NNNN.
    periodic_re = re.compile(rf"^{re.escape(args.model_name)}_epoch_(\d+)$")
    periodic_epochs: list[int] = []
    for src in sorted(models_dir.glob(f"{args.model_name}_epoch_*")):
        m = periodic_re.match(src.name)
        if not m:
            continue
        ep = int(m.group(1))
        dst = ckpt_dir / f"epoch_{ep:04d}.pt"
        shutil.copy(src, dst)
        periodic_epochs.append(ep)
    log.info("Copied %d periodic checkpoints: epochs %s",
             len(periodic_epochs), periodic_epochs)

    # BEST: argmin(val_loss) over epochs that are BOTH eval epochs AND saved.
    # Eval epochs: {5, 10, 20, ..., (n_epochs//10)*10}. Saved epochs via
    # save_each: multiples of save_every in (0, n_epochs). With save_every=20,
    # every saved epoch is also an eval epoch — so candidate set = periodic_epochs.
    candidate_epochs = [e for e in periodic_epochs if test_losses[e] > 0]
    best_info: dict = {}
    if candidate_epochs:
        best_epoch = int(min(candidate_epochs, key=lambda e: test_losses[e]))
        best_val = float(test_losses[best_epoch])
        best_train = float(train_losses[best_epoch])
        best_src = ckpt_dir / f"epoch_{best_epoch:04d}.pt"
        best_dst = ckpt_dir / "best.pt"
        shutil.copy(best_src, best_dst)
        log.info("Best checkpoint = epoch %d (val_loss=%.4f). Copied to %s",
                 best_epoch, best_val, best_dst)
        best_info = {
            "epoch": best_epoch,
            "train_loss": best_train,
            "val_loss": best_val,
            "source": str(best_src),
            "path": str(best_dst),
        }
    elif final_src.exists():
        log.warning("No intermediate eval epochs had val_loss — "
                    "using final checkpoint as best.")
        shutil.copy(final_dst, ckpt_dir / "best.pt")
        best_info = {
            "epoch": args.n_epochs - 1,
            "train_loss": float(train_losses[-1]),
            "val_loss": None,
            "source": str(final_dst),
            "path": str(ckpt_dir / "best.pt"),
            "note": "no eval epochs with val_loss; used final",
        }
    else:
        log.error("No best checkpoint candidate available.")

    (ckpt_dir / "best_meta.json").write_text(json.dumps(best_info, indent=2))
    log.info("Wrote %s", ckpt_dir / "best_meta.json")

    # -- Summary --
    eval_losses = [float(test_losses[e]) for e in range(args.n_epochs)
                   if test_losses[e] > 0]
    summary = {
        "run_name": run_name,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "train_seconds": round(train_secs, 1),
        "n_epochs": args.n_epochs,
        "val_fovs": val_fovs,
        "final_train_loss": float(train_losses[-1]),
        "final_test_loss_last_eval": eval_losses[-1] if eval_losses else None,
        "best": best_info,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", run_dir / "summary.json")
    log.info("DONE. Best weights: %s", ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
