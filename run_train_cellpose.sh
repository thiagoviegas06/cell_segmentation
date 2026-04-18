#!/bin/bash
#SBATCH --job-name=cell_seg_train
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=tjv235@nyu.edu

# Phase 4: fine-tune cpsam on Phase 3 training data, then eval on Phase 1 val FOVs.
#
# Usage:
#   sbatch run_train_cellpose.sh                              # defaults: 100 epochs, lr 1e-5, wd 0.1, bs 8
#   sbatch run_train_cellpose.sh --n_epochs 50                # override a flag
#   RUN_NAME=phase4_v1 sbatch run_train_cellpose.sh           # named run
#   EVAL_DIAMETER=87.68 sbatch run_train_cellpose.sh          # override the diameter used at eval time
#
# Env vars consumed:
#   RUN_NAME       — subdir under runs/ for this training run (default: phase4_<timestamp>)
#   EVAL_DIAMETER  — diameter (pixels) forwarded to local_eval.py (default: 87.68, Phase-2 median)
#
# Flags after the script name are forwarded to scripts/train_cellpose.py.

set -euo pipefail

mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
OVL="/scratch/tjv235/neuro.ext3"
PROJECT="/scratch/tjv235/cell_segmentation"

RUN_NAME="${RUN_NAME:-phase4_$(date +%Y%m%d_%H%M%S)}"
EVAL_DIAMETER="${EVAL_DIAMETER:-87.68}"

echo "[launcher] RUN_NAME=$RUN_NAME"
echo "[launcher] EVAL_DIAMETER=$EVAL_DIAMETER"
echo "[launcher] forwarded train args: $*"

singularity exec --nv \
  --overlay "$OVL":ro \
  --fakeroot \
  "$SIF" /bin/bash -s -- "$RUN_NAME" "$EVAL_DIAMETER" "$@" <<'EOF'
set -euo pipefail
source /ext3/env.sh
conda activate my_writable_env

RUN_NAME="$1"; shift
EVAL_DIAMETER="$1"; shift

cd /scratch/tjv235/cell_segmentation

# 1) Train
python -u scripts/train_cellpose.py \
    --run_name "$RUN_NAME" \
    "$@"

BEST="runs/${RUN_NAME}/best.pt"
if [ ! -f "$BEST" ]; then
    echo "[launcher] ERROR: expected best.pt at $BEST not found — aborting eval."
    exit 2
fi

# 2) Eval on Phase 1 val FOVs with calibrated diameter
python -u scripts/local_eval.py \
    --segmenter scripts.segmenters:build_cellpose_finetuned \
    --gpu \
    --diameter "$EVAL_DIAMETER" \
    --segmenter_kwargs "pretrained_model=${PWD}/${BEST}" \
    --run_name "${RUN_NAME}_eval"

echo "[launcher] DONE. Training: runs/${RUN_NAME}/  Eval: runs/${RUN_NAME}_eval/"
EOF
