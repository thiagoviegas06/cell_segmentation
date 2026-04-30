#!/bin/bash
#SBATCH --job-name=phase2_full_pipeline
#SBATCH --account=torch_pr_173_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dr3432@nyu.edu

# Phase 2 End-to-End Pipeline:
# 1. Segmentation + Deep Embedding Extraction (ROI Pooling)
# 2. Expression Matrix Construction + Embeddings Integration
# 3. PyTorch MLP Classifier Training (Gene Counts + Stage + Embeddings)
# 4. Inference & Submission Generation
#
# Usage:
#   sbatch run_phase2_full_pipeline.sh
#   sbatch run_phase2_full_pipeline.sh --overwrite  # to redo everything

set -euo pipefail

mkdir -p logs cache/masks_phase2 cache/expression_phase2 submissions runs/phase2_pytorch

# Optimization settings for NumPy/PyTorch on multi-core
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Configurable paths (defaults match the existing scripts)
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/scratch/$USER/cell_segmentation/runs/phase4_v1_h200/checkpoints/best.pt}"
DATA_ROOT="${DATA_ROOT:-/scratch/pl2820/data/competition_phase2}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/cell_segmentation/cache/masks_phase2}"
RUN_DIR="runs/phase2_pytorch"
SUBMISSION_PATH="submissions/pytorch_embeddings_v1.csv"

SIF="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
OVL="/scratch/$USER/pytorch/pytorch_env.ext3"

echo "Starting full Phase 2 pipeline..."
echo "  Pretrained Model: $PRETRAINED_MODEL"
echo "  Data Root:        $DATA_ROOT"
echo "  Output Run Dir:   $RUN_DIR"

singularity exec --nv \
  --overlay "$OVL":ro \
  --fakeroot \
  "$SIF" /bin/bash -s -- "$@" <<'EOF'
set -euo pipefail
source /ext3/env.sh
conda activate my_writable_env

cd /scratch/$USER/cell_segmentation

echo "--- Step 1: Segmentation & Embedding Extraction ---"
python -u scripts/phase2/segment_all.py \
    --pretrained_model "$PRETRAINED_MODEL" \
    --data_root         "$DATA_ROOT" \
    --output_dir        "$OUTPUT_DIR" \
    --gpu \
    "$@"

echo "--- Step 2: Build Expression Matrices & Integrate Embeddings ---"
python -u scripts/phase2/build_expression.py "$@"

echo "--- Step 3: Train PyTorch MLP Classifier ---"
python -u scripts/phase2/train_classifier.py \
    --run_dir "$RUN_DIR" \
    --n_estimators 1000 \
    --learning_rate 0.001

echo "--- Step 4: Generate Submission ---"
python -u scripts/phase2/predict.py \
    --run_dir "$RUN_DIR" \
    --output "$SUBMISSION_PATH"

echo "--- Pipeline Complete ---"
EOF
