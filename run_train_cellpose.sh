#!/bin/bash
#SBATCH --job-name=cell_seg_train
#SBATCH --account=torch_pr_173_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:h200:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dr3432@nyu.edu

# Phase 4: fine-tune cpsam on Phase 3 training data, then eval on Phase 1 val FOVs.
#
# Default resource request: 1× L40 GPU, 48 h wall time.
#   - L40 is the best GPU currently reliably available on this cluster for
#     Phase 4's 300-epoch fine-tune over 160 train images / 10 val images.
#   - 48 h is a safety ceiling: an L40 is ~1.5-2× slower than an H100 on
#     this workload, so real wall time is expected to be 18-30 h. The
#     ceiling is there because Option-A checkpointing has no resume — a
#     job that gets killed by the time limit loses everything after the
#     last epoch_NNNN.pt snapshot.
#
# Usage:
#   sbatch run_train_cellpose.sh                              # defaults: 300 epochs, lr 1e-5, wd 0.1, bs 8, save_every 20
#   sbatch run_train_cellpose.sh --n_epochs 50                # override a train flag
#   RUN_NAME=phase4_v1 sbatch run_train_cellpose.sh           # named run
#   EVAL_DIAMETER=87.68 sbatch run_train_cellpose.sh          # override the diameter used at eval time
#
#   # Override GPU / wall time at submit time (sbatch CLI takes precedence
#   # over #SBATCH directives). Example for H100 with a tighter ceiling:
#   sbatch --gres=gpu:h100:1 --time=24:00:00 run_train_cellpose.sh
#
# Env vars consumed:
#   RUN_NAME       — subdir under runs/ for this training run (default: phase4_<timestamp>)
#   EVAL_DIAMETER  — diameter (pixels) forwarded to local_eval.py.
#                    Default: -1 (auto). Phase 2 sweep showed no fixed diameter
#                    beat cpsam's auto estimate on the 6-FOV val split, so we
#                    let Cellpose pick per-FOV at eval time. Pass a positive
#                    number to override.
#
# Flags after the script name are forwarded to scripts/train_cellpose.py.

set -euo pipefail

mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SIF="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
OVL="/scratch/$USER/pytorch/pytorch_env.ext3"
PROJECT="/scratch/$USER/cell_segmentation"

RUN_NAME="${RUN_NAME:-phase4_$(date +%Y%m%d_%H%M%S)}"
EVAL_DIAMETER="${EVAL_DIAMETER:--1}"

echo "[launcher] RUN_NAME=$RUN_NAME"
echo "[launcher] EVAL_DIAMETER=$EVAL_DIAMETER"
echo "[launcher] forwarded train args: $*"

singularity exec --nv \
  --overlay "$OVL":ro \
  --fakeroot \
  "$SIF" /bin/bash -s -- "$RUN_NAME" "$EVAL_DIAMETER" "$@" <<'EOF'
set -euo pipefail
source /ext3/env.sh

RUN_NAME="$1"; shift
EVAL_DIAMETER="$1"; shift

cd /scratch/$USER/cell_segmentation

# 1) Train
python -u scripts/train_cellpose.py \
    --run_name "$RUN_NAME" \
    "$@"

BEST="runs/${RUN_NAME}/checkpoints/best.pt"
if [ ! -f "$BEST" ]; then
    echo "[launcher] ERROR: expected best.pt at $BEST not found — aborting eval."
    exit 2
fi

# 2) Eval on Phase 1 val FOVs.
# local_eval.py treats any negative --diameter as "auto" (default for this script).
python -u scripts/local_eval.py \
    --segmenter scripts.segmenters:build_cellpose_finetuned \
    --gpu \
    --diameter "$EVAL_DIAMETER" \
    --segmenter_kwargs "pretrained_model=${PWD}/${BEST}" \
    --run_name "${RUN_NAME}_eval"

echo "[launcher] DONE. Training: runs/${RUN_NAME}/  Eval: runs/${RUN_NAME}_eval/"
EOF
