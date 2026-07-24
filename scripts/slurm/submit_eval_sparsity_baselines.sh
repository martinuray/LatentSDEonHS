#!/bin/bash

################################################################################
# SLURM Script: Run baselines/eval_sparsity_baselines.py in parallel
#
# Sweeps over:
#   - benchmarks:      QAD, PSM
#   - interpolation:   linear, spline (cubic spline)
#   - classifiers:     IForest, KNN, LOF (CPU only)
#                       USAD, TcnED, DeepIF (GPU)
#
# Each classifier runs as its own independent SLURM array job (25 tasks =
# 5 seeds x 5 sparsity levels), writing into its own results directory.
# Aggregation is handled independently elsewhere and is not submitted here.
#
# In total this submits 2 benchmarks x 2 interps x 6 classifiers =
# 24 SLURM array jobs.
#
# Usage:
#   ./scripts/slurm/submit_eval_sparsity_baselines.sh [RESULTS_BASE_DIR]
# Example:
#   ./scripts/slurm/submit_eval_sparsity_baselines.sh
#       -> out/sparsity_baselines_QAD_linear_IForest, out/sparsity_baselines_QAD_linear_KNN, ...
#   ./scripts/slurm/submit_eval_sparsity_baselines.sh out/my_sparsity_sweep
################################################################################

set -euo pipefail

# ---- Configuration ----
PARTITION="gtx1080ti"
CPU_TIMEOUT="3:00:00"     # IForest / KNN / LOF are cheap but KNN/LOF can be slow on large traces
GPU_TIMEOUT="6:00:00"     # USAD / TcnED / DeepIF are deep models; allow more headroom
NUM_CPUS=4
NUM_GPUS=1
CPU_MEMORY="16GB"
GPU_MEMORY="24GB"

BENCHMARKS=(QAD PSM)
INTERPS=(linear spline)

CPU_CLASSIFIERS=(IForest KNN LOF)
GPU_CLASSIFIERS=(USAD TcnED DeepIF)

# Sparsity grid (5 levels) and number of seeds -> 25 tasks per (benchmark, interp, classifier)
SUBSAMPLES="0.001,0.01,0.05,0.1,0.2"
NUM_SUBSAMPLES=5
NUM_SEEDS=5
NUM_TASKS=$(( NUM_SEEDS * NUM_SUBSAMPLES - 1 ))   # 0-based upper bound

RESULTS_BASE_DIR="${1:-out/sparsity_baselines}"

PROJECT_DIR="/home2/muray/Code/LatentSDEonHS"
LOG_DIR="${PROJECT_DIR}/slurm_logs_sparsity_baselines"
mkdir -p "${LOG_DIR}"

# ---- Initialize conda ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate baseline-latent

cd "${PROJECT_DIR}"

submit_classifier_jobs () {
    local benchmark="$1"
    local interp="$2"
    local classifier="$3"
    local device_group="$4"   # "cpu" or "gpu"

    local results_dir="${RESULTS_BASE_DIR}_${benchmark}_${interp}_${classifier}"
    mkdir -p "${results_dir}"

    local timeout mem gpus gpu_id_flag
    if [[ "${device_group}" == "gpu" ]]; then
        timeout="${GPU_TIMEOUT}"
        mem="${GPU_MEMORY}"
        gpus="${NUM_GPUS}"
        gpu_id_flag="--gpu-id 0"
    else
        timeout="${CPU_TIMEOUT}"
        mem="${CPU_MEMORY}"
        gpus=0
        gpu_id_flag=""
    fi

    local array_job_id
    array_job_id=$(sbatch \
        --partition="${PARTITION}" \
        --time="${timeout}" \
        --cpus-per-task="${NUM_CPUS}" \
        --mem="${mem}" \
        --gpus="${gpus}" \
        --job-name="sparsity_bl_${benchmark}_${interp}_${classifier}" \
        --output="${LOG_DIR}/sparsity_bl_${benchmark}_${interp}_${classifier}_%A_%a.log" \
        --error="${LOG_DIR}/sparsity_bl_${benchmark}_${interp}_${classifier}_%A_%a.log" \
        --array="0-${NUM_TASKS}" \
        --wrap="python baselines/eval_sparsity_baselines.py \
            --mode single \
            --task-id \${SLURM_ARRAY_TASK_ID} \
            --benchmark ${benchmark} \
            --classifiers ${classifier} \
            --subsamples '${SUBSAMPLES}' \
            --num-seeds ${NUM_SEEDS} \
            --interp ${interp} \
            ${gpu_id_flag} \
            --results-dir ${results_dir}" \
        --parsable)

    echo "Submitted array job (${benchmark}/${interp}/${classifier}, ${device_group}): ${array_job_id}  (tasks 0-${NUM_TASKS})"
}

for BENCHMARK in "${BENCHMARKS[@]}"; do
    for INTERP in "${INTERPS[@]}"; do
        for CLASSIFIER in "${CPU_CLASSIFIERS[@]}"; do
            submit_classifier_jobs "${BENCHMARK}" "${INTERP}" "${CLASSIFIER}" "cpu"
        done
        for CLASSIFIER in "${GPU_CLASSIFIERS[@]}"; do
            submit_classifier_jobs "${BENCHMARK}" "${INTERP}" "${CLASSIFIER}" "gpu"
        done
        echo ""
    done
done

echo "All jobs submitted. Results will be under: ${RESULTS_BASE_DIR}_<BENCHMARK>_<INTERP>_<CLASSIFIER>/"
echo "Monitor all:  squeue -u \$USER"