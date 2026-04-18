# MERFISH Cell Segmentation

Working directory for the MERFISH cell-segmentation Kaggle task. The original
end-to-end pipeline (`pipeline.py`) produces a `submission.csv` for the public
leaderboard; everything under `scripts/`, `cache/`, and `runs/` is the local
validator stack added to evaluate segmentation changes offline against a held-
out subset of training FOVs using the organizer's official ARI metric.

Competition data lives outside this tree at
`/scratch/pl2820/data/competition/`.

## Top level

| File | Purpose |
| --- | --- |
| `pipeline.py` | End-to-end inference: loads DAPI+polyT from `.dax`, runs Cellpose (cpsam), assigns spots to cells by mask lookup, writes `submission.csv`. Treat as frozen — the local validator reuses `load_fov_images`, `normalize_image`, and `segment_fov`. |
| `run_pipeline.sh` | SLURM launcher (H100, 4 h) that invokes `pipeline.py` inside the singularity image + `my_writable_env` conda env to produce the test-set submission. |
| `run_local_eval.sh` | SLURM launcher (H100, 2 h) for `scripts/local_eval.py`. Defaults to the zero-shot cpsam segmenter; extra args after `sbatch run_local_eval.sh …` are forwarded to `local_eval.py` (e.g. `--run_name`, `--diameter`, `--z_filter`, `--segmenter_kwargs`). Overlay mounted `:ro` so multiple eval jobs can run in parallel. |
| `run_train_cellpose.sh` | SLURM launcher (H100, 6 h) for Phase 4: runs `scripts/train_cellpose.py` then `scripts/local_eval.py` on the fine-tuned weights in one job. Env vars: `RUN_NAME` (default `phase4_<timestamp>`), `EVAL_DIAMETER` (default 87.68 — the Phase 2 median). |
| `submission.csv` | Pre-existing zero-shot cpsam output used for the 0.62 Kaggle baseline. Kept for reference. |
| `summary.txt` | Running notes across all phases: data-structure findings, verification steps, deliverables, and the outstanding work queue. Read this first for project context. |
| `val_fovs.txt` | Committed 6-FOV validation split (stratified across cell-density range). Consumed by `local_eval.py` via `--val_fovs`. |
| `run.log` | Log from an early pipeline run, kept only for reference. |

## `scripts/`

| File | Purpose |
| --- | --- |
| `build_gt_labels.py` | One-off builder: rasterizes `cell_boundaries_train.csv` into a `(2048, 2048)` int mask per `(FOV, z)` and looks up every training spot to produce `cache/gt_spot_labels.parquet`. Re-run only if the boundary file or the pixel-conversion formula changes. |
| `make_val_split.py` | Deterministic stratified picker over the 40 training FOVs. Sorts by cells-per-FOV ascending and picks 6 at evenly-spaced quantile positions. Writes `val_fovs.txt`. |
| `compute_diameter.py` | Phase 2: calibrates Cellpose's `diameter` hint from GT polygons. Parses `cell_boundaries_train.csv`, computes per-cell equivalent-circle diameter via the shoelace area of the z=2 polygon (pixel conversion is isotropic so only pixel_size matters, not FOV offsets). Writes median px to `reference/diameter_px.txt`. `--all_z` prints per-z stats. |
| `prep_training_data.py` | Phase 3: materializes fine-tuning inputs for the 34 training FOVs (40 − 6 val). For each FOV writes `training_data/<fov_id>.npz` with `img (H, W, 3) float32 = [polyT, DAPI, zeros]` and `mask (H, W) int32` (1..N cells at z=2). Reuses `build_gt_labels.rasterize_fov_z` so training masks are bit-identical to evaluation GT. Renders stratified QC PNGs under `training_data/qc/`. |
| `train_cellpose.py` | Phase 4: fine-tunes cpsam on the Phase 3 `.npz` files. Holds out 2 training FOVs (inner-quantile positions) as in-training val — distinct from `val_fovs.txt` so the Phase 1 val set stays clean for final ARI eval. Calls `cellpose.train.train_seg` with `normalize=False` + `channel_axis=-1` to match inference normalization. Promotes the intermediate checkpoint with lowest test_loss to `runs/<name>/best.pt`. |
| `segmenters.py` | Factory registry for segmentation callables. Each factory takes `(gpu, diameter, **kwargs)` and returns `segment(fov_dir) -> (2048, 2048) int mask`. Current factories: `build_all_background` (plumbing smoke test), `build_cellpose_zeroshot` (wraps `pipeline.segment_fov` with cpsam), `build_cellpose_finetuned` (requires `pretrained_model=<path>` via `--segmenter_kwargs`). |
| `local_eval.py` | Main validator. Runs a factory-built segmenter on the val FOVs, assigns spots via mask lookup, joins with cached GT, and reports per-FOV + mean ARI using the organizer's `metric.merfish_score` (with an inline sklearn ARI as a cross-check). Flags: `--diameter`, `--z_filter` (restrict val spots to one z-plane), `--segmenter_kwargs "key=val,..."` (forwarded to the factory — used e.g. to pass a fine-tuned weights path). Writes everything under `runs/<name>/`. |

## `reference/`

| File | Purpose |
| --- | --- |
| `diameter_px.txt` | Median GT cell diameter in pixels (87.6760), written by `compute_diameter.py` from z=2 polygons. Consumed as the Phase 2 `--diameter` value and the Phase 4 eval diameter. |

## `cache/`

| File | Purpose |
| --- | --- |
| `gt_spot_labels.parquet` | Cached ground-truth `(spot_idx, fov, gt_cluster_id)` for all 2.66 M training spots, produced by `build_gt_labels.py`. ~13 MB; rebuilt in ~17 s if deleted. |

## `training_data/` (gitignored, ~2.2 GB)

Generated by `scripts/prep_training_data.py`. 34 files, one per training FOV, each an `.npz` with `img (H, W, 3) float32` and `mask (H, W) int32`. The `qc/` subdirectory holds 4 stratified PNG overlays (DAPI + z=2 polygon outlines) for visual spot-checking of the rasterization alignment. Rebuild in ~40 s.

## `runs/`

Each validator or training invocation creates `runs/<run_name>/`. Eval runs contain `config.json` (args + env), `eval.log`, `per_fov.csv`, `summary.json`, and optionally `masks/<fov>.npy` when invoked with `--save_masks`. Training runs contain `config.json`, `train.log`, `losses.csv`, `models/<name>` (final checkpoint), `models/<name>_epoch_NNNN` (intermediates), `best.pt` (best-by-test_loss copy), and `summary.json`.

| Run | What it was |
| --- | --- |
| `smoke_all_bg/` | All-zeros mask — confirms ARI is 0.0 end-to-end. |
| `smoke_official_metric/` | Verified `merfish_score` (official) agrees bit-exact with the inline per-FOV reduction. |
| `phase1_baseline_zeroshot/` | Zero-shot cpsam on the 6 val FOVs — mean ARI 0.4765 (well below the 0.62 Kaggle baseline; motivates the z-filter diagnostic / widening the split). |

## `logs/`

SLURM stdout/stderr for every job submitted via the two launcher scripts.
Filenames follow the SLURM `%x_%j` pattern (`<job-name>_<job-id>.out/err`).
Safe to prune.

## Conventions

- **FOV naming.** `FOV_XXX` on train (directory under `/scratch/pl2820/data/competition/train/`); `FOV_A/B/C/D` on test.
- **Cluster IDs.** Both GT and predictions use `{fov}_cell_{N}` (organizer format). Background spots use the literal string `"background"`.
- **Pixel conversion.** `image_row = 2048 - (global_x - fov_x) / 0.109`, `image_col = (global_y - fov_y) / 0.109`. Note the stage-x / image-row inversion.
- **Z-planes.** Spots span `global_z ∈ {0..4}`. `pipeline.py` currently segments a single z plane (DAPI frame 16 / polyT frame 15 in the `.dax`). GT is rasterized per-z.

## Typical workflow

```bash
# One-time setup (already done):
python scripts/build_gt_labels.py          # cache/gt_spot_labels.parquet
python scripts/make_val_split.py           # val_fovs.txt
python scripts/compute_diameter.py         # reference/diameter_px.txt
python scripts/prep_training_data.py       # training_data/*.npz

# Zero-shot / diameter-swept eval (Phases 1–2):
sbatch run_local_eval.sh --run_name phase2_diameter_median --diameter 87.68

# Fine-tune + eval in one job (Phase 4):
RUN_NAME=phase4_v1 sbatch run_train_cellpose.sh
# → runs/phase4_v1/       (training artifacts + best.pt)
# → runs/phase4_v1_eval/  (ARI on the 6 Phase 1 val FOVs with best.pt)

# Re-eval arbitrary weights (e.g. a promoted model):
sbatch run_local_eval.sh \
    --run_name finetuned_v1_rerun \
    --segmenter scripts.segmenters:build_cellpose_finetuned \
    --diameter 87.68 \
    --segmenter_kwargs "pretrained_model=/scratch/tjv235/cell_segmentation/models/finetuned_v1.pt"

# Inspect results:
cat runs/<name>/summary.json
column -s, -t runs/<name>/per_fov.csv | less -S
```
