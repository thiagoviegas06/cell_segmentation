# MERFISH Cell Segmentation

Working directory for the MERFISH cell-segmentation Kaggle task. The original
end-to-end pipeline (`pipeline.py`) produces a zero-shot `submission.csv` for
the public leaderboard; everything under `scripts/`, `cache/`, `runs/`, and
`submissions/` is the local validator + fine-tune + 3D-stitched inference
stack built on top to improve the segmentation and re-score it offline against
a held-out subset of training FOVs using the organizer's official ARI metric.

See `summary.txt` for the running phase-by-phase narrative (findings,
verification, results, outstanding work). This README is a per-file map of
the repo; read `summary.txt` first if you need the motivation behind any
given piece.

Competition data lives outside this tree at
`/scratch/pl2820/data/competition/`.

## First-time setup (replicate from a fresh clone)

If you're cloning this repo into your own `/scratch/<netid>/cell_segmentation/`,
follow these four steps before running any phase. Skipping step 3 is what
causes the common error
`FileNotFoundError: ...cache/gt_spot_labels.parquet` — that file is
gitignored and must be rebuilt locally.

### 1. Prerequisites
- **Competition data** — read access to the shared tree at
  `/scratch/pl2820/data/competition/` (no copy needed; everything points at
  it directly).
- **Singularity image + conda env** — the launchers expect
  `/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif` plus an
  overlay containing the `my_writable_env` conda env (with `cellpose`,
  `h5py`, `cv2`, `pandas`, `numpy`, `pyarrow`). The original author's overlay
  lives at `/scratch/tjv235/neuro.ext3`; either ask to share it, build your
  own, or run the python scripts directly in any equivalent env (the
  launchers are SLURM convenience wrappers — none of the logic is in them).

### 2. Update hardcoded paths
Several scripts and SLURM launchers bake `/scratch/tjv235/cell_segmentation/`
and `/scratch/tjv235/neuro.ext3` into their defaults. After cloning, do a
project-wide search-and-replace of
`/scratch/tjv235/cell_segmentation` → your own checkout root, and update
`OVL=...` in each `run_*.sh` to your overlay path. Files that need
attention:

| File | What to change |
| --- | --- |
| `run_local_eval.sh`, `run_train_cellpose.sh`, `run_infer_test.sh`, `run_infer_test_v2.sh`, `run_pipeline.sh` | `PROJECT=`, `OVL=`, the `cd ...` line, and (for the test-set launchers) `PRETRAINED_MODEL`, `OUTPUT` env-var defaults |
| `scripts/local_eval.py` | `--val_fovs`, `--gt_labels`, `--runs_dir` defaults |
| `scripts/build_gt_labels.py` | `--output` default |
| `scripts/prep_training_data.py` | `--val_fovs`, `--out_dir` defaults |
| `scripts/train_cellpose.py` | `--data_dir`, `--runs_dir` defaults |
| `scripts/compute_diameter.py`, `scripts/make_val_split.py` | `--output` defaults (only matters if you re-run them — outputs are committed) |
| `pipeline.py` | `--output` default |

You can also bypass the defaults by passing explicit `--output`/`--gt_labels`/
`--val_fovs`/etc. flags on every invocation; the search-and-replace is
strictly less error-prone.

### 3. Rebuild gitignored artifacts
What's committed: source, `val_fovs.txt`, `reference/diameter_px.txt`.
What's NOT committed and must be regenerated locally:
`cache/`, `training_data/`, `runs/`, `submissions/`, `submission.csv`,
`logs/`.

Run from the repo root after step 2. Steps 3a is required for *any* local
eval; 3b is only needed if you're going to fine-tune; 3c is a way to skip
fine-tuning by reusing the trained checkpoint.

```bash
# 3a. GT spot labels — REQUIRED for local_eval (~17 s, ~13 MB).
mkdir -p cache
python scripts/build_gt_labels.py     # writes cache/gt_spot_labels.parquet

# 3b. Multi-z fine-tuning inputs — only for Phase 3+ training (~5 min, ~11 GB).
mkdir -p training_data
python scripts/prep_training_data.py  # writes training_data/FOV_*_z{0..4}.npz

# 3c. (Optional) skip fine-tuning by copying the trained checkpoint.
#     The Phase 4 weights aren't in git; ask a teammate to share
#     runs/phase4_v1_h200/checkpoints/best.pt (~few hundred MB) and place it
#     at the same relative path in your checkout. Both Phase 4 and Phase 5
#     test-set inference and 6-FOV eval can run from this single checkpoint.
mkdir -p runs/phase4_v1_h200/checkpoints
# scp <teammate>:/scratch/<their-netid>/cell_segmentation/runs/phase4_v1_h200/checkpoints/best.pt \
#     runs/phase4_v1_h200/checkpoints/best.pt
```

### 4. Sanity-check the install
This reproduces the Phase 1 zero-shot baseline (mean ARI **0.4765** on the
6-FOV val split, ~5 min on an H100):

```bash
sbatch run_local_eval.sh --run_name sanity_zeroshot
# When done:
cat runs/sanity_zeroshot/summary.json     # expect mean_ari ≈ 0.4765
column -s, -t runs/sanity_zeroshot/per_fov.csv | less -S
```

If that matches, you're set up correctly. From here see **Typical workflow**
at the bottom of this README for the per-phase commands.

## Pipeline at a glance

| Phase | What it does | Primary files | Kaggle LB | Local 6-FOV ARI |
| --- | --- | --- | --- | --- |
| Baseline | Zero-shot cpsam, single z-plane | `pipeline.py`, `run_pipeline.sh` | 0.62 | 0.4765 |
| Phase 1 | Local validator + committed 6-FOV val split | `scripts/local_eval.py`, `val_fovs.txt` | — | 0.4765 |
| Phase 2 | Diameter calibration + sweep | `scripts/compute_diameter.py` | — | 0.4608 (best, 1.1× auto) |
| Phase 3 | Multi-z fine-tune data prep (170 `.npz`) | `scripts/prep_training_data.py` | — | — |
| Phase 4 | Fine-tune cpsam → test-set inference (z=2 only) | `scripts/train_cellpose.py`, `scripts/infer_test.py`, `run_infer_test.sh` | 0.76 | 0.6847 |
| Phase 5 | 3D-stitched inference, per-spot z lookup | `scripts/pipeline_v2.py`, `build_cellpose_finetuned_3d` factory, `run_infer_test_v2.sh` | pending | 0.7544 |

## Top level

| File | Purpose |
| --- | --- |
| `pipeline.py` | End-to-end zero-shot inference (frozen): loads DAPI+polyT from `.dax`, runs Cellpose (cpsam) at a single z-plane, assigns spots to cells by 2D mask lookup, writes `submission.csv`. The validator, fine-tune, and 3D test-set scripts all import `load_fov_images`, `load_dax_frame`, `normalize_image`, `segment_fov`, `assign_spots`, `build_submission`, `TEST_FOVS`, `BACKGROUND_LABEL` from here. |
| `run_pipeline.sh` | SLURM launcher that invokes `pipeline.py` inside the singularity image + `my_writable_env` conda env to produce the zero-shot test-set submission. |
| `run_local_eval.sh` | SLURM launcher (H100, 2 h) for `scripts/local_eval.py`. Defaults to the zero-shot cpsam segmenter; extra args after `sbatch run_local_eval.sh …` are forwarded to `local_eval.py` (e.g. `--run_name`, `--diameter`, `--z_filter`, `--segmenter`, `--segmenter_kwargs`). Overlay mounted `:ro` so multiple eval jobs can run in parallel. |
| `run_train_cellpose.sh` | SLURM launcher (H100 on `h100_tandon`, 48 h ceiling) for Phase 4: runs `scripts/train_cellpose.py` then `scripts/local_eval.py` on the fine-tuned weights in one job. Env vars: `RUN_NAME` (default `phase4_<timestamp>`), `EVAL_DIAMETER` (default `-1` = auto, after Phase 2 showed no fixed diameter beats cpsam's per-FOV auto). Override GPU/time at submit time (`sbatch --gres=gpu:h200:1 --time=24:00:00 ...`). |
| `run_infer_test.sh` | Phase 4 test-set inference launcher (L40S, 1 h). Wraps `scripts/infer_test.py` — single-z inference with a fine-tuned checkpoint. Env vars: `PRETRAINED_MODEL`, `OUTPUT`, `DIAMETER`. |
| `run_infer_test_v2.sh` | Phase 5 test-set inference launcher (L40S, 1 h). Wraps `scripts/pipeline_v2.py` — 3D-stitched inference with per-spot z lookup. Env vars: `PRETRAINED_MODEL`, `OUTPUT`, `DIAMETER`, `STITCH_THRESHOLD` (default 0.3). Extra CLI args are forwarded (e.g. `--fovs FOV_A` for single-FOV debugging). |
| `submission.csv` | Pre-existing zero-shot cpsam output used for the 0.62 Kaggle baseline. Kept for reference; do not overwrite. |
| `summary.txt` | Running notes across all phases: data-structure findings, verification steps, deliverables, and the outstanding work queue. Read this first for project context. |
| `val_fovs.txt` | Committed 6-FOV validation split (stratified across cell-density range). Consumed by `local_eval.py` via `--val_fovs`. Distinct from `train_cellpose.py`'s in-training val split so the Phase 1 split stays clean for final ARI eval. |
| `run.log` | Log from an early pipeline run, kept only for reference. |

## `scripts/`

| File | Purpose |
| --- | --- |
| `build_gt_labels.py` | One-off builder: rasterizes `cell_boundaries_train.csv` into a `(2048, 2048)` int mask per `(FOV, z)` with `cv2.fillPoly` and looks up every training spot at `(image_row, image_col, z)` to produce `cache/gt_spot_labels.parquet` (2.66 M rows, ~13 MB, ~17 s). Re-run only if the boundary file or the pixel-conversion formula changes. |
| `make_val_split.py` | Deterministic stratified picker over the 40 training FOVs. Sorts by cells-per-FOV ascending and picks 6 at evenly-spaced quantile positions. Writes `val_fovs.txt`. |
| `compute_diameter.py` | Phase 2: calibrates Cellpose's `diameter` hint from GT polygons. Parses `cell_boundaries_train.csv`, computes per-cell equivalent-circle diameter via the shoelace area of the z=2 polygon (pixel conversion is isotropic so only pixel_size matters, not FOV offsets). Writes median px to `reference/diameter_px.txt`. `--all_z` prints per-z stats. |
| `prep_training_data.py` | Phase 3 (v2 multi-z): materializes fine-tuning inputs at **all 5 z-planes** for the 34 training FOVs (40 − 6 val). Writes `training_data/<fov>_z{0..4}.npz` — 170 files total — with `img (H, W, 3) float32 = [polyT_z, DAPI_z, zeros]` and `mask (H, W) int32`. Uses the verified frame map `polyT[z]=5+5z`, `DAPI[z]=6+5z`. Reuses `build_gt_labels.rasterize_fov_z` so training masks are bit-identical to evaluation GT. Skips any `(FOV, z)` with an empty mask (WARN log) so all-background images don't teach under-segmentation. Renders stratified QC PNGs under `training_data/qc/`. |
| `train_cellpose.py` | Phase 4: fine-tunes cpsam on the Phase 3 `.npz` files. Groups files by FOV id (`FOV_Z_RE = ^(FOV_[^_]+)_z(\d+)$`) so the val split operates on whole FOVs across all 5 z-planes (no cross-z leakage). Holds out 2 inner-quantile FOVs (10 `.npz`) by z=2 mask cell count — distinct from `val_fovs.txt`. Calls `cellpose.train.train_seg` with defaults `n_epochs=200, lr=1e-5, weight_decay=0.1, batch_size=8, save_every=20, save_each=True, normalize=False, channel_axis=-1`. Option-A checkpointing (public API only, weights-only → no mid-run resume): promotes the saved epoch with lowest `val_loss` to `runs/<name>/checkpoints/best.pt` (candidates come from cellpose's eval cadence, so typically epochs `{20, 40, …, 180}` at `n_epochs=200`). Writes `config.json`, `train.log`, `train_log.csv`, `summary.json`, and `checkpoints/{final,best,epoch_NNNN}.pt`. |
| `infer_test.py` | Phase 4 test-set inference: instantiates `CellposeModel` from a fine-tuned `.pt` checkpoint, runs the zero-shot single-z pipeline on each test FOV (`pipeline.load_fov_images` + `segment_fov` + `assign_spots`), writes `submissions/<name>_submission.csv`, and verifies columns, row count, null labels, FOV set, and `spot_id` ordering vs `sample_submission.csv`. |
| `pipeline_v2.py` | Phase 5 test-set inference: loads polyT+DAPI at all 5 z-planes (using the verified frame map), stacks to `(Z, H, W, 3)`, calls `model.eval(..., z_axis=0, do_3D=False, stitch_threshold=0.3, normalize=False)` so Cellpose links 2D masks across z by IoU, then does a vectorized per-spot lookup `mask_stack[global_z, row, col]`. Same post-write verification as `infer_test.py` plus a per-FOV summary table. Fails fast if the installed cellpose's `model.eval` is missing `stitch_threshold` (rather than silently producing unstitched masks). |
| `segmenters.py` | Factory registry for segmentation callables consumed by `local_eval.py`. Each factory takes `(gpu, diameter, **kwargs)` and returns `segment(fov_dir) -> mask`. Current factories: `build_all_background` (plumbing smoke test, returns zeros), `build_cellpose_zeroshot` (wraps `pipeline.segment_fov` with cpsam, 2D single-z), `build_cellpose_finetuned` (same but loads weights from `pretrained_model=<path>`), `build_cellpose_finetuned_3d` (Phase 5: returns a `(5, H, W)` int mask stitched across z; accepts `stitch_threshold=0.3`). Factories are referenced by `'<module>:<factory>'` on `local_eval.py`'s `--segmenter` flag. |
| `local_eval.py` | Main validator. Runs a factory-built segmenter on the val FOVs, assigns spots via mask lookup, joins with cached GT, and reports per-FOV + mean ARI using the organizer's `metric.merfish_score` (with an inline sklearn ARI as a cross-check; disagreements > 1e-6 warn). Accepts both 2D `(H, W)` and 3D `(Z, H, W)` masks — for 3D, each spot's `global_z` selects the plane (clipped to `[0, Z-1]`). Flags: `--segmenter`, `--diameter`, `--z_filter` (restrict val spots + GT to one z-plane in lockstep; diagnostic only), `--segmenter_kwargs "key=val,..."` (forwarded to the factory — used e.g. to pass a fine-tuned weights path or a stitch threshold), `--save_masks`, `--run_name`. Writes everything under `runs/<name>/`. |

## `reference/`

| File | Purpose |
| --- | --- |
| `diameter_px.txt` | Median GT cell diameter in pixels (87.6760), written by `compute_diameter.py` from z=2 polygons. Used for the Phase 2 `--diameter` sweep. Eval default since Phase 2 landed is `-1` (auto) because no fixed diameter beat cpsam's per-FOV estimate on the 6-FOV split. |

## `cache/`

| File | Purpose |
| --- | --- |
| `gt_spot_labels.parquet` | Cached ground-truth `(spot_idx, fov, gt_cluster_id)` for all 2.66 M training spots, produced by `build_gt_labels.py`. ~13 MB; rebuilt in ~17 s if deleted. Background = literal string `"background"`; cell labels = `{fov}_cell_{original_cell_id}`. |

## `training_data/` (gitignored, ~11 GB)

Generated by `scripts/prep_training_data.py`. 170 `.npz` files (34 train FOVs × 5 z-planes), each with `img (H, W, 3) float32` and `mask (H, W) int32` (0 = background, 1..N cells at that z). Any `(FOV, z)` whose mask is empty is skipped with a WARN log (0 skipped on the committed split; 14,401 cells total across all files). `qc/` holds stratified PNG overlays (DAPI + z=2 polygon outlines) for spot-checking rasterization alignment.

## `runs/`

Each validator or training invocation creates `runs/<run_name>/`.

Eval runs contain `config.json` (args + env), `eval.log`, `per_fov.csv`, `summary.json`, and optionally `masks/<fov>.npy` when invoked with `--save_masks`.

Training runs contain `config.json` (hyperparams + val_fovs + train/val image counts), `train.log`, `train_log.csv` ([epoch, train_loss, val_loss, lr]; `val_loss` blank on non-eval rows), `models/<name>` (cellpose's raw final-epoch save), `models/<name>_epoch_NNNN` (intermediate saves), `checkpoints/{final,best,epoch_NNNN}.pt` + `best_meta.json`, and `summary.json`.

| Run | What it was |
| --- | --- |
| `smoke_all_bg/` | All-zeros mask — confirms ARI is 0.0 end-to-end. |
| `smoke_official_metric/` | Verified `merfish_score` (official) agrees bit-exact with the inline per-FOV reduction. |
| `phase1_baseline_zeroshot/` | Zero-shot cpsam on the 6 val FOVs — mean ARI 0.4765. |
| `phase1_zcheck_z2only/` | `--z_filter 2` diagnostic — mean ARI 0.4932 (+0.0167 over all-z baseline; FOV_019 +0.12). Motivated the multi-z retraining and the Phase 5 3D approach. |
| `phase2_diameter_{0.9x,1.1x,1.2x}/` | Diameter sweep at 0.9/1.1/1.2× the median 87.68 px. Best: 1.1× at 0.4608 — still below the auto baseline. Actionable: keep eval diameter at auto. |
| `phase4_v1_h200/` | Phase 4 v1 fine-tune on H200, 84 min, 200 epochs, best epoch 180 (`val_loss = 0.2117`). Source of the Phase 4 + Phase 5 test-set checkpoints. |
| `phase4_v1_h200_eval/` | 6-FOV eval of `phase4_v1_h200/checkpoints/best.pt` — mean ARI 0.6847 (+0.2082 over baseline; clears the 0.62 Kaggle anchor). |
| `phase4_v1_l40s/`, `phase4_v1_l40s_eval/` | Earlier L40S training run kept for reference. |
| `phase5_v1_eval/` | 6-FOV eval with `build_cellpose_finetuned_3d` on `phase4_v1_h200/checkpoints/best.pt` — mean ARI 0.7544 (+0.0697 over Phase 4). Every FOV improved; FOV_019 (densest) gained +0.124. |

## `submissions/`

Kaggle-format outputs written by `infer_test.py` / `pipeline_v2.py` (directory is gitignored).

| File | Produced by | Kaggle LB |
| --- | --- | --- |
| `phase4_v1_h200_submission.csv` | `run_infer_test.sh` on `phase4_v1_h200/checkpoints/best.pt` | 0.76 |
| `phase5_v1_h200_submission.csv` | `run_infer_test_v2.sh` on same checkpoint, 3D-stitched path | pending |

## `logs/`

SLURM stdout/stderr for every job submitted via the launcher scripts.
Filenames follow the SLURM `%x_%j` pattern (`<job-name>_<job-id>.out/err`).
Safe to prune.

## Conventions

- **FOV naming.** `FOV_XXX` on train (directory under `/scratch/pl2820/data/competition/train/`); `FOV_A/B/C/D` on test.
- **Cluster IDs.** Both GT and predictions use `{fov}_cell_{N}` (organizer format). Background spots use the literal string `"background"`. ARI is cluster-ID-independent, so the naming choice doesn't affect the score — it's for compatibility with the organizer's `generate_submission.py`.
- **Pixel conversion.** `image_row = 2048 - (global_x - fov_x) / 0.109`, `image_col = (global_y - fov_y) / 0.109`. Note the stage-x / image-row inversion — a silent-bug source.
- **Z-planes.** Spots span `global_z ∈ {0, 1, 2, 3, 4}`. `pipeline.py` segments a single z-plane (DAPI frame 16 / polyT frame 15 in the `.dax`, which is z=2 under the verified frame map `polyT[z]=5+5z`, `DAPI[z]=6+5z`). GT is rasterized per-z, and Phase 5 inference segments + stitches all 5 z-planes so each spot is looked up at its own z.

## Typical workflow

> If you're running this repo for the first time, do **First-time setup**
> above first — `cache/`, `training_data/`, and `runs/` are all gitignored
> and must be regenerated before any of the commands below will work.

```bash
# One-time setup (see "First-time setup" above for details and path overrides):
python scripts/build_gt_labels.py          # cache/gt_spot_labels.parquet (REQUIRED)
python scripts/prep_training_data.py       # training_data/FOV_*_z{0..4}.npz (only for Phase 3+)
# val_fovs.txt and reference/diameter_px.txt are committed; only re-run if changing them:
# python scripts/make_val_split.py
# python scripts/compute_diameter.py

# Zero-shot / diameter-swept eval (Phases 1–2):
sbatch run_local_eval.sh --run_name phase1_baseline_zeroshot
sbatch run_local_eval.sh --run_name phase2_diameter_median --diameter 87.68

# Fine-tune + eval in one job (Phase 4):
RUN_NAME=phase4_v1 sbatch --gres=gpu:h200:1 --time=24:00:00 run_train_cellpose.sh
# → runs/phase4_v1/       (training artifacts + checkpoints/best.pt)
# → runs/phase4_v1_eval/  (ARI on the 6 Phase 1 val FOVs with best.pt)

# Re-eval arbitrary weights (2D single-z, e.g. a promoted model):
sbatch run_local_eval.sh \
    --run_name phase4_v1_rerun \
    --segmenter scripts.segmenters:build_cellpose_finetuned \
    --segmenter_kwargs "pretrained_model=/scratch/tjv235/cell_segmentation/runs/phase4_v1_h200/checkpoints/best.pt"

# Phase 5 local eval (3D-stitched + per-spot z lookup):
sbatch run_local_eval.sh \
    --run_name phase5_v1_eval \
    --segmenter scripts.segmenters:build_cellpose_finetuned_3d \
    --segmenter_kwargs "pretrained_model=/scratch/tjv235/cell_segmentation/runs/phase4_v1_h200/checkpoints/best.pt"

# Test-set submissions:
sbatch run_infer_test.sh                   # Phase 4, single z (LB 0.76)
sbatch run_infer_test_v2.sh                # Phase 5, 3D stitched (LB pending)

# Inspect results:
cat runs/<name>/summary.json
column -s, -t runs/<name>/per_fov.csv | less -S
```
