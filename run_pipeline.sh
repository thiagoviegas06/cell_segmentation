#!/bin/bash
#SBATCH --job-name=cell_seg_pipeline
#SBATCH --account=torch_pr_173_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=dr3432@nyu.edu

set -euo pipefail

mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SIF="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
OVL="/scratch/dr3432/pytorch/pytorch_env.ext3"

singularity exec --nv \
  --overlay "$OVL" \
  --fakeroot \
  "$SIF" /bin/bash <<'EOF'
set -euo pipefail

source /ext3/env.sh
conda activate my_writable_env
pip install cellpose
python -u /scratch/dr3432/cell_segmentation/pipeline.py \
    --data_root /scratch/pl2820/data/competition \
    --spots    /scratch/pl2820/data/competition/test_spots.csv \
    --output   /scratch/dr3432/cell_segmentation/submission.csv \
    --gpu
EOF
