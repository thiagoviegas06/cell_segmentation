#!/bin/bash
#SBATCH --job-name=cell_seg_infer_v2
#SBATCH --account=torch_pr_173_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=dr3432@nyu.edu

# Phase 5 test-set inference: 3D-stitched cellpose + per-spot z lookup.
#
# Env overrides:
#   PRETRAINED_MODEL   path to .pt (default: phase4_v1_h200 best.pt)
#   OUTPUT             submission csv path
#                      (default: submissions/phase5_v1_h200_submission.csv)
#   DIAMETER           cell diameter hint in px; omit / -1 = auto (default: auto)
#   STITCH_THRESHOLD   IoU threshold for cross-z linking (default: 0.3)
#
# Extra args after the script name are forwarded to pipeline_v2.py
# (e.g. --fovs FOV_A to debug on one FOV).
#
# Examples:
#   sbatch run_infer_test_v2.sh
#   PRETRAINED_MODEL=runs/phase4_v2/checkpoints/best.pt \
#     OUTPUT=submissions/phase5_v2_submission.csv \
#     sbatch run_infer_test_v2.sh

set -euo pipefail

mkdir -p logs submissions

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PRETRAINED_MODEL="${PRETRAINED_MODEL:-/scratch/dr3432/cell_segmentation/runs/phase4_v1_h200/checkpoints/best.pt}"
OUTPUT="${OUTPUT:-/scratch/dr3432/cell_segmentation/submissions/phase5_v1_h200_submission.csv}"
DIAMETER="${DIAMETER:-}"
STITCH_THRESHOLD="${STITCH_THRESHOLD:-0.3}"

DIAM_FLAG=""
if [[ -n "$DIAMETER" ]]; then
  if awk "BEGIN {exit !($DIAMETER >= 0)}"; then
    DIAM_FLAG="--diameter $DIAMETER"
  fi
fi

SIF="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
OVL="/scratch/dr3432/pytorch/pytorch_env.ext3"

export PRETRAINED_MODEL OUTPUT DIAM_FLAG STITCH_THRESHOLD

singularity exec --nv \
  --overlay "$OVL":ro \
  --fakeroot \
  "$SIF" /bin/bash -s -- "$@" <<'EOF'
set -euo pipefail
source /ext3/env.sh
conda activate my_writable_env

cd /scratch/dr3432/cell_segmentation
python -u scripts/pipeline_v2.py \
    --pretrained_model  "$PRETRAINED_MODEL" \
    --output            "$OUTPUT" \
    --stitch_threshold  "$STITCH_THRESHOLD" \
    --gpu \
    $DIAM_FLAG \
    "$@"
EOF
