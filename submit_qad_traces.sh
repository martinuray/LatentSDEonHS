#!/bin/bash

################################################################################
# SLURM Script: Submit QAD traces as separate jobs and aggregate afterwards
#
# Usage: ./submit_qad_traces.sh
################################################################################

set -euo pipefail

# ---- Parse arguments ----
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <subsample_factor> [trace_ids...]" >&2
    echo "  subsample_factor  fraction of observations to keep, e.g. 0.5" >&2
    echo "  trace_ids         optional subset of traces (default: 1..16)" >&2
    exit 1
fi

SUBSAMPLE="$1"
shift

# SLURM configuration
PARTITION="rtx2080ti"
TIMEOUT="48:00:00"
NUM_GPUS=1
NUM_CPUS=8
MEMORY="40GB"
JOB_NAME_PREFIX="qad"

# QAD configuration
BATCH_SIZE=1024
if [[ $# -gt 0 ]]; then
    TRACES=("$@")
else
    TRACES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
fi

PROJECT_DIR="/home2/muray/Code/LatentSDEonHS"
LOG_DIR="${PROJECT_DIR}/slurm_logs_qad/${SUBSAMPLE}"
TRACE_METRICS_DIR="${PROJECT_DIR}/logs/qad_trace_metrics/${SUBSAMPLE}"
COMBINED_METRICS_CSV="${PROJECT_DIR}/logs/qad_trace_metrics/${SUBSAMPLE}/final_metrics_qad_traces.csv"
AGGREGATED_RESULTS_CSV="${PROJECT_DIR}/logs/qad_aggregated_results_${SUBSAMPLE}.csv"

# ---- Initialize conda ----
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---- Activate environment ----
conda activate baseline-latent

# ---- Move to project directory ----
cd "${PROJECT_DIR}"

# Log / output directories
mkdir -p "${LOG_DIR}" "${PROJECT_DIR}/logs" "${TRACE_METRICS_DIR}"

echo "=================================="
echo "Submitting QAD trace jobs"
echo "=================================="
echo "Total traces: ${#TRACES[@]}"
echo "Batch size: ${BATCH_SIZE}"
echo "Partition: ${PARTITION}"
echo "Timeout: ${TIMEOUT}"
echo "GPUs per job: ${NUM_GPUS}"
echo "CPUs per job: ${NUM_CPUS}"
echo "Memory per job: ${MEMORY}"
echo "Log directory: ${LOG_DIR}"
echo "=================================="
echo ""

JOB_IDS=()

for TRACE in "${TRACES[@]}"; do
    JOB_NAME="${JOB_NAME_PREFIX}_trace${TRACE}"

    echo "Submitting job for trace: ${TRACE}"

    JOB_ID=$(sbatch \
        --partition="${PARTITION}" \
        --time="${TIMEOUT}" \
        --gres="gpu:${NUM_GPUS}" \
        --cpus-per-task="${NUM_CPUS}" \
        --mem="${MEMORY}" \
        --job-name="${JOB_NAME}" \
        --output="${LOG_DIR}/${JOB_NAME}_%j.log" \
        --error="${LOG_DIR}/${JOB_NAME}_%j.log" \
        --wrap="python anomaly_detection.py \
            --dataset QAD \
            --batch-size ${BATCH_SIZE} \
            --trace-ids ${TRACE} \
            --subsample ${SUBSAMPLE} \
            --data-dir data_dir \
            --enable-file-logging \
            --log-dir logs \
            --final-metrics-csv logs/qad_trace_metrics/${SUBSAMPLE}/final_metrics_trace${TRACE}.csv \
            --delete-processed-data" \
        --parsable)

    if [[ -z "${JOB_ID}" ]]; then
        echo "Error: failed to submit trace job for trace ${TRACE}" >&2
        exit 1
    fi

    JOB_IDS+=("${JOB_ID}")
    echo "  → SLURM job ID: ${JOB_ID}"
    sleep 0.5
done

echo ""
echo "=================================="
echo "All ${#TRACES[@]} jobs submitted!"
echo "Job IDs: ${JOB_IDS[*]}"
echo "=================================="
echo ""

DEPS=$(IFS=:; echo "${JOB_IDS[*]}")

echo "Submitting aggregation job (depends on: ${DEPS})"

AGG_JOB_ID=$(sbatch \
    --partition="${PARTITION}" \
    --time="00:10:00" \
    --cpus-per-task=2 \
    --mem="4GB" \
    --job-name="${JOB_NAME_PREFIX}_aggregate" \
    --output="${LOG_DIR}/${JOB_NAME_PREFIX}_aggregate_%j.log" \
    --error="${LOG_DIR}/${JOB_NAME_PREFIX}_aggregate_%j.log" \
    --dependency="afterok:${DEPS}" \
    --wrap="cd ${PROJECT_DIR} && \
        rm -f logs/qad_trace_metrics/${SUBSAMPLE}/final_metrics_qad_traces.csv && \
        first=1; \
        for csv in logs/qad_trace_metrics/${SUBSAMPLE}/final_metrics_trace*.csv; do \
            [ -f \"\$csv\" ] || continue; \
            if [ \"\$first\" -eq 1 ]; then \
                cat \"\$csv\" > logs/qad_trace_metrics/${SUBSAMPLE}/final_metrics_qad_traces.csv; \
                first=0; \
            else \
                tail -n +2 \"\$csv\" >> logs/qad_trace_metrics/${SUBSAMPLE}/final_metrics_qad_traces.csv; \
            fi; \
        done; \
        if [ \"\$first\" -eq 1 ]; then \
            echo 'Error: no per-trace metric files found.' >&2; \
            exit 1; \
        fi; \
        python aggregate_qad_results.py \
            --csv logs/qad_trace_metrics/${SUBSAMPLE}/final_metrics_qad_traces.csv \
            --dataset QAD \
            --n-traces ${#TRACES[@]} \
            --out logs/qad_aggregated_results_${SUBSAMPLE}.csv" \
    --parsable)

if [[ -z "${AGG_JOB_ID}" ]]; then
    echo "Error: failed to submit aggregation job" >&2
    exit 1
fi

echo "Submitted aggregation job: ${AGG_JOB_ID}  (depends on ${DEPS})"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs in: ${LOG_DIR}"
echo "Combined trace metrics: ${COMBINED_METRICS_CSV}"
echo "Aggregated results: ${AGGREGATED_RESULTS_CSV}"
echo ""

