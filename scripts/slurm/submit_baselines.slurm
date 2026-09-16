#!/bin/bash
#SBATCH --job-name=latent_baseline
#SBATCH --output=slurm_logs_baseline/%x_%j.out
#SBATCH --error=slurm_logs_baseline/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=rtx2080ti
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

# Usage: submitt_baselines.slurm BENCHMARK CLASSIFIER [SEED] [RUNS] [OUTPUT_DIR]
#
# Window/decimation/score-smoothing settings are resolved inside
# baselines/baseline.py (BENCHMARK_WINDOW_DEFAULTS), so e.g. QAD automatically
# runs at 10 Hz with 200-step windows, stride 20 and score smoothing 10, in
# line with cfg/anomaly_detection/QAD.json. OUTPUT_DIR defaults to "out"; pass
# a fresh directory when the protocol of a benchmark changed so new rows are
# not appended to CSVs produced under the old protocol.

source $(conda info --base)/etc/profile.d/conda.sh
conda activate baseline-latent

cd /home2/muray/Code/LatentSDEonHS

BENCHMARK=$1
CLASSIFIER=$2
SEED=${3:-42}
RUNS=${4:-5}
OUTPUT_DIR=${5:-out}

echo "Running classifier=$CLASSIFIER on benchmark=$BENCHMARK seed=$SEED runs=$RUNS output_dir=$OUTPUT_DIR"

python baselines/baseline.py \
    --benchmarks "$BENCHMARK" \
    --classifiers "$CLASSIFIER" \
    --runs "$RUNS" \
    --seed "$SEED" \
    --gpu-id 0 \
    --output-dir "$OUTPUT_DIR"
