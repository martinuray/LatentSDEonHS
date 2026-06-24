#!/bin/bash
#SBATCH --job-name=exp_qad
#SBATCH --output=/home2/muray/Code/LatentSDEonHS/slurm_logs/exp_qad_%j.log
#SBATCH --error=/home2/muray/Code/LatentSDEonHS/slurm_logs/exp_qad_%j.log
#SBATCH --time=48:00:00
#SBATCH --partition=rtx2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

set -euo pipefail

PROJECT_DIR="/home2/muray/Code/LatentSDEonHS"

# ---- Initialize conda ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate baseline-latent

cd "${PROJECT_DIR}"
mkdir -p logs slurm_logs

echo "Running QAD anomaly detection experiment"
echo "Project dir: ${PROJECT_DIR}"
echo "Dataset: QAD"
echo "Subsample: 0.2"
echo "Window length: 500"

python anomaly_detection.py \
    --dataset QAD \
    --runs 5 \
    --data-dir data_dir \
    --enable-file-logging \
    --log-dir logs \
    --no-enable-checkpointing \
    --final-metrics-csv logs/final_metrics.csv \
    --data-window-length 500 \
    --data-window-overlap 0.0 \
    --batch-size 64 \
    --lr 0.001 \
    --n-epochs 2011 \
    --kl0-weight 0.0001 \
    --klp-weight 10.0 \
    --pxz-weight 1.0 \
    --seed -1 \
    --restart 30 \
    --device cuda \
    --z-dim 64 \
    --h-dim 512 \
    --n-deg 12 \
    --no-learnable-prior \
    --no-freeze-sigma \
    --initial-sigma 0.05 \
    --mc-eval-samples 1 \
    --mc-train-samples 1 \
    --num-max-cpu-worker 10 \
    --eval-every-n-epochs 30 \
    --loglevel debug \
    --no-use-atanh \
    --no-debug \
    --subsample 0.2 \
    --no-normalize-score \
    --data-normalization-strategy min-max \
    --dec-hidden-dim 512 \
    --n-dec-layers 2 \
    --early-stopping-patience 10 \
    --early-stopping-min-delta 0 \
    --non-linear-decoder \
    --delete-processed-data \
    --fixed-subsample-mask

