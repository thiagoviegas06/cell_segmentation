#!/bin/bash
#SBATCH --job-name=cell_seg_infer
#SBATCH --account=torch_pr_60_tandon_priority
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

# Test-set inference with a fine-tuned cellpose checkpoint.
#
# Env overrides:
#   PRETRAINED_MODEL  path to .pt (default: phase4_v1_h200 best.pt)
#   OUTPUT            submission csv path (default: submissions/<run>_submission.csv)
#   DIAMETER          cell diameter hint in px; omit / -1 = auto (default: auto)
#
# Extra args after the script name are forwarded to infer_test.py.
#
# Examples:
#   sbatch run_infer_test.sh
#   PRETRAINED_MODEL=runs/phase4_v2/checkpoints/best.pt \
#     OUTPUT=submissions/phase4_v2_submission.csv \
#     sbatch run_infer_test.sh

set -euo pipefail

mkdir -p logs submissions

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PRETRAINED_MODEL="${PRETRAINED_MODEL:-/scratch/dr3432/cell_segmentation/runs/phase4_v1_h200/checkpoints/best.pt}"
OUTPUT="${OUTPUT:-/scratch/dr3432/cell_segmentation/submissions/phase4_v1_h200_submission.csv}"
DIAMETER="${DIAMETER:-}"

DIAM_FLAG=""
if [[ -n "$DIAMETER" ]]; then
  # treat negative as auto (drop the flag)
  if awk "BEGIN {exit !($DIAMETER >= 0)}"; then
    DIAM_FLAG="--diameter $DIAMETER"
  fi
fi

SIF="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
OVL="/scratch/dr3432/pytorch/pytorch_env.ext3"

export PRETRAINED_MODEL OUTPUT DIAM_FLAG

singularity exec --nv \
  --overlay "$OVL":ro \
  --fakeroot \
  "$SIF" /bin/bash -s -- "$@" <<'EOF'
set -euo pipefail
source /ext3/env.sh
conda activate my_writable_env

cd /scratch/dr3432/cell_segmentation
python -u scripts/infer_test.py \
    --pretrained_model "$PRETRAINED_MODEL" \
    --output           "$OUTPUT" \
    --gpu \
    $DIAM_FLAG \
    "$@"
EOF
