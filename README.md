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
| `pipeline.py` | End-to-end inference: loads DAPI+polyT from `.dax`, runs Cellpose (cpsam), assigns spots to cells by mask lookup, writes `submission.csv`. Treat as frozen — the local validator reuses `load_fov_images` and `segment_fov`. |
| `run_pipeline.sh` | SLURM launcher (H100, 4 h) that invokes `pipeline.py` inside the singularity image + `my_writable_env` conda env to produce the test-set submission. |
| `run_local_eval.sh` | SLURM launcher (H100, 2 h) for `scripts/local_eval.py`. Defaults to the zero-shot cpsam segmenter; extra args after `sbatch run_local_eval.sh …` are forwarded to `local_eval.py` (e.g. `--run_name`, `--diameter`, `--z_filter`). |
| `submission.csv` | Pre-existing zero-shot cpsam output used for the 0.62 Kaggle baseline. Kept for reference. |
| `summary.txt` | Phase 1 notes: data-structure findings, verification steps, deliverables, and the outstanding work queue. Read this first for project context. |
| `val_fovs.txt` | Committed 6-FOV validation split (stratified across cell-density range). Consumed by `local_eval.py` via `--val_fovs`. |
| `run.log` | Log from an early pipeline run, kept only for reference. |

## `scripts/`

| File | Purpose |
| --- | --- |
| `build_gt_labels.py` | One-off builder: rasterizes `cell_boundaries_train.csv` into a `(2048, 2048)` int mask per `(FOV, z)` and looks up every training spot to produce `cache/gt_spot_labels.parquet`. Re-run only if the boundary file or the pixel-conversion formula changes. |
| `make_val_split.py` | Deterministic stratified picker over the 40 training FOVs. Sorts by cells-per-FOV ascending and picks 6 at evenly-spaced quantile positions. Writes `val_fovs.txt`. |
| `segmenters.py` | Factory registry for segmentation callables. Each factory takes `(gpu, diameter)` and returns `segment(fov_dir) -> (2048, 2048) int mask`. Current factories: `build_all_background` (plumbing smoke test), `build_cellpose_zeroshot` (wraps `pipeline.segment_fov` with cpsam). |
| `local_eval.py` | Main validator. Runs a factory-built segmenter on the val FOVs, assigns spots via mask lookup, joins with cached GT, and reports per-FOV + mean ARI using the organizer's `metric.merfish_score` (with an inline sklearn ARI as a cross-check). Writes everything under `runs/<name>/`. |

## `cache/`

| File | Purpose |
| --- | --- |
| `gt_spot_labels.parquet` | Cached ground-truth `(spot_idx, fov, gt_cluster_id)` for all 2.66 M training spots, produced by `build_gt_labels.py`. ~13 MB; rebuilt in ~17 s if deleted. |

## `runs/`

Each validator invocation creates `runs/<run_name>/` containing:
`config.json` (args + env), `eval.log`, `per_fov.csv`, `summary.json`, and
optionally `masks/<fov>.npy` when invoked with `--save_masks`.

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
# (once, already done) build cached GT
python scripts/build_gt_labels.py

# (once, already done) commit val split
python scripts/make_val_split.py

# run a validator experiment
sbatch run_local_eval.sh --run_name <descriptive_name> [extra flags]

# inspect results
cat runs/<descriptive_name>/summary.json
column -s, -t runs/<descriptive_name>/per_fov.csv | less -S
```
