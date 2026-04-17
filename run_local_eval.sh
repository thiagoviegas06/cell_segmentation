#!/bin/bash
#SBATCH --job-name=cell_seg_eval
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=tjv235@nyu.edu

# Usage:
#   sbatch run_local_eval.sh                      # defaults: zero-shot cpsam, auto diameter
#   sbatch run_local_eval.sh --diameter 60        # override
#   sbatch run_local_eval.sh --run_name phase1_baseline
#
# Any args after the script are forwarded to local_eval.py (in addition to --gpu).

set -euo pipefail

mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
OVL="/scratch/tjv235/neuro.ext3"
PROJECT="/scratch/tjv235/cell_segmentation"

singularity exec --nv \
  --overlay "$OVL" \
  --fakeroot \
  "$SIF" /bin/bash -s -- "$@" <<'EOF'
set -euo pipefail
source /ext3/env.sh
conda activate my_writable_env
pip install --quiet cellpose

cd /scratch/tjv235/cell_segmentation
python -u scripts/local_eval.py \
    --segmenter scripts.segmenters:build_cellpose_zeroshot \
    --gpu \
    "$@"
EOF
