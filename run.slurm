#!/bin/bash
#SBATCH --job-name=latent_baseline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

# ---- Initialize conda ----
source $(conda info --base)/etc/profile.d/conda.sh

# ---- Activate environment ----
conda activate baseline-latent

# ---- Move to project directory ----
cd /home2/muray/Code/LatentSDEonHS

# ---- Read arguments ----
BENCHMARK=$1
CLASSIFIER=$2

echo "Running classifier=$CLASSIFIER on benchmark=$BENCHMARK"

# ---- Run your experiment ----
python baselines/baseline.py \
	    --benchmarks "$BENCHMARK" \
	        --classifiers "$CLASSIFIER" \
		      --runs 5 --gpu-id 0
