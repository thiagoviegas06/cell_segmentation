"""
Phase 4: Fine-tune cpsam on the Phase-3 training data.

Loads the .npz files written by prep_training_data.py, splits off a small
in-training val set (2 FOVs, picked deterministically at the 1/3 and 2/3
quantiles of the sorted-by-cell-count training list — distinct from the
Phase 1 val_fovs.txt so those remain untouched for final ARI eval), and
runs cellpose.train.train_seg with the Phase-4 hyperparameters.

Per-epoch train/test loss is logged via cellpose's built-in train_logger to
runs/<name>/train.log. Intermediate checkpoints are saved every 10 epochs
(save_each=True), and after training the checkpoint whose eval epoch has the
lowest test_loss is promoted to runs/<name>/best.pt. The final checkpoint
(last epoch) is also always kept at runs/<name>/models/<model_name>.

Alignment notes:
  - Inference via pipeline.segment_fov pre-normalizes with normalize_image
    and calls model.eval(..., normalize=False, channel_axis=2). Training must
    match: we pass normalize=False + channel_axis=-1 so we don't double-
    normalize relative to what the model sees at eval time.
  - Masks are rasterized at z=2 in prep_training_data.py, matching the
    single-plane segmentation the model is learning to reproduce.
"""

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

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


def load_fov(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path) as z:
        return z["img"].astype(np.float32), z["mask"].astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--training_data",
        default="/scratch/tjv235/cell_segmentation/training_data",
    )
    ap.add_argument("--runs_dir", default="/scratch/tjv235/cell_segmentation/runs")
    ap.add_argument("--run_name", default=None,
                    help="Subdir under runs/ (default: phase4_<timestamp>)")
    ap.add_argument("--model_name", default="finetuned",
                    help="Cellpose save_model filename inside runs/<name>/models/")
    ap.add_argument("--pretrained_model", default="cpsam")
    ap.add_argument("--n_epochs", type=int, default=100)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--bsize", type=int, default=256,
                    help="Train-time crop size (cellpose default: 256)")
    ap.add_argument("--save_every", type=int, default=10,
                    help="Save intermediate checkpoint every N epochs.")
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

    # -- Discover training FOVs --
    data_dir = Path(args.training_data)
    npzs = sorted(data_dir.glob("FOV_*.npz"))
    if not npzs:
        log.error("No .npz files found under %s — run prep_training_data.py first.", data_dir)
        sys.exit(2)
    fov_ids = [p.stem for p in npzs]
    log.info("Found %d training FOVs: %s...%s", len(fov_ids), fov_ids[:2], fov_ids[-2:])

    # -- Pick val split (NOT the Phase 1 val FOVs — those are final-eval only) --
    log.info("Counting cells per FOV (for stratified val pick)...")
    cells_per_fov = {p.stem: cell_count_from_mask(p) for p in npzs}
    val_fovs = pick_val_fovs(fov_ids, cells_per_fov, args.val_count)
    val_set = set(val_fovs)
    train_fovs = [f for f in fov_ids if f not in val_set]
    log.info("Phase-4 val FOVs (%d, inner quantiles by cell count): %s",
             len(val_fovs), val_fovs)
    log.info("  cell counts: %s",
             {f: cells_per_fov[f] for f in val_fovs})
    log.info("Phase-4 train FOVs: %d", len(train_fovs))

    # -- Load all data into memory (2 GB, fits easily on H100 nodes) --
    log.info("Loading training arrays into memory...")
    t_load = time.time()
    train_imgs, train_masks = [], []
    for f in train_fovs:
        img, mask = load_fov(data_dir / f"{f}.npz")
        train_imgs.append(img)
        train_masks.append(mask)
    val_imgs, val_masks = [], []
    for f in val_fovs:
        img, mask = load_fov(data_dir / f"{f}.npz")
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

    # -- Losses CSV --
    losses_csv = run_dir / "losses.csv"
    with open(losses_csv, "w") as fh_out:
        fh_out.write("epoch,train_loss,test_loss\n")
        for e in range(args.n_epochs):
            # test_losses is 0 at non-eval epochs; report blank rather than 0
            tl = test_losses[e] if test_losses[e] > 0 else ""
            fh_out.write(f"{e},{train_losses[e]:.6f},{tl}\n")
    log.info("Wrote losses to %s", losses_csv)

    # -- Promote best checkpoint --
    # Intermediate checkpoints saved at: save_every, 2*save_every, ..., < n_epochs,
    # each also an eval epoch (iepoch % 10 == 0). test_losses[e] is populated there.
    candidate_epochs = [e for e in range(args.save_every, args.n_epochs, args.save_every)
                        if test_losses[e] > 0]
    best_info = {}
    if candidate_epochs:
        best_epoch = int(min(candidate_epochs, key=lambda e: test_losses[e]))
        best_loss = float(test_losses[best_epoch])
        best_src = run_dir / "models" / f"{args.model_name}_epoch_{best_epoch:04d}"
        best_dst = run_dir / "best.pt"
        if best_src.exists():
            shutil.copy(best_src, best_dst)
            log.info("Best checkpoint = epoch %d (test_loss=%.4f). Copied to %s",
                     best_epoch, best_loss, best_dst)
            best_info = {"epoch": best_epoch, "test_loss": best_loss,
                         "source": str(best_src), "path": str(best_dst)}
        else:
            log.warning("Expected best checkpoint not found at %s — "
                        "falling back to final.", best_src)
            shutil.copy(run_dir / "models" / args.model_name, run_dir / "best.pt")
            best_info = {"epoch": args.n_epochs - 1, "test_loss": None,
                         "source": str(run_dir / "models" / args.model_name),
                         "path": str(run_dir / "best.pt"),
                         "note": "intermediate checkpoint missing; used final"}
    else:
        log.warning("No intermediate eval epochs populated — using final checkpoint as best.")
        shutil.copy(run_dir / "models" / args.model_name, run_dir / "best.pt")
        best_info = {"epoch": args.n_epochs - 1, "test_loss": None,
                     "source": str(run_dir / "models" / args.model_name),
                     "path": str(run_dir / "best.pt"),
                     "note": "no eval epochs with test_loss; used final"}

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
    log.info("DONE. Best weights: %s", run_dir / "best.pt")


if __name__ == "__main__":
    main()
