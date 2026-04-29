#!/bin/bash
#SBATCH --job-name=seg_all_phase2
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=tjv235@nyu.edu

# Phase 2.2 — bulk 3D-stitched segmentation of all 70 Phase 2 FOVs
# (60 train + 10 test) using the Phase 1 fine-tuned cpsam checkpoint.
# Writes cache/masks_phase2/<FOV_ID>.npy and cell_counts.csv.
#
# Env overrides:
#   PRETRAINED_MODEL   path to .pt (default: phase4_v1_h200 best.pt)
#   DATA_ROOT          Phase 2 dataset root (default: /scratch/pl2820/data/competition_phase2)
#   OUTPUT_DIR         where to save .npy files (default: cache/masks_phase2)
#   STITCH_THRESHOLD   IoU threshold for cross-z linking (default: 0.3)
#   DIAMETER           cell diameter hint in px; omit / -1 = auto (default: auto)
#
# Extra args after the script are forwarded to segment_all.py
# (e.g. --fovs FOV_E for a single-FOV smoke test, --overwrite to redo).
#
# Examples:
#   sbatch run_segment_all_phase2.sh
#   sbatch run_segment_all_phase2.sh --fovs FOV_E
#   sbatch --gres=gpu:h100:1 run_segment_all_phase2.sh

set -euo pipefail

mkdir -p logs cache/masks_phase2

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PRETRAINED_MODEL="${PRETRAINED_MODEL:-/scratch/tjv235/cell_segmentation/runs/phase4_v1_h200/checkpoints/best.pt}"
DATA_ROOT="${DATA_ROOT:-/scratch/pl2820/data/competition_phase2}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/tjv235/cell_segmentation/cache/masks_phase2}"
STITCH_THRESHOLD="${STITCH_THRESHOLD:-0.3}"
DIAMETER="${DIAMETER:-}"

DIAM_FLAG=""
if [[ -n "$DIAMETER" ]]; then
  if awk "BEGIN {exit !($DIAMETER >= 0)}"; then
    DIAM_FLAG="--diameter $DIAMETER"
  fi
fi

SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
OVL="/scratch/tjv235/neuro.ext3"

export PRETRAINED_MODEL DATA_ROOT OUTPUT_DIR STITCH_THRESHOLD DIAM_FLAG

singularity exec --nv \
  --overlay "$OVL":ro \
  --fakeroot \
  "$SIF" /bin/bash -s -- "$@" <<'EOF'
set -euo pipefail
source /ext3/env.sh
conda activate my_writable_env

cd /scratch/tjv235/cell_segmentation
python -u scripts/phase2/segment_all.py \
    --pretrained_model  "$PRETRAINED_MODEL" \
    --data_root         "$DATA_ROOT" \
    --output_dir        "$OUTPUT_DIR" \
    --stitch_threshold  "$STITCH_THRESHOLD" \
    --gpu \
    $DIAM_FLAG \
    "$@"
EOF
