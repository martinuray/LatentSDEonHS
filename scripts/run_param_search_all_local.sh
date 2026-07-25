#!/bin/bash
################################################################################
# Spawn the full baseline_param_search.py grid search on this machine's GPUs
# (TcnED) plus CPU (KNN), all as background processes. No SLURM - just plain
# bash job control.
#
# Shards = (classifier, benchmark) pairs, each run via
# scripts/run_param_search_shard.sh:
#   KNN   (CPU)  x {SWaT, WaDi, PSM, SMD, QAD, MSL, SMAP}  -> 7 shards, all in parallel
#   TcnED (GPU)  x {SWaT, WaDi, PSM, SMD, QAD, MSL, SMAP}  -> 7 shards, bucketed onto GPUs
#
# TcnED's 7 benchmarks are bucketed onto the 4 GPUs by dataset count (SMAP=55,
# SMD=28, MSL=27, QAD=16, SWaT/WaDi/PSM=1 each) so no single GPU gets stuck
# running two heavy multi-trace benchmarks; benchmarks sharing a GPU run
# sequentially (one after another), never two-at-once on the same GPU. SMAP
# alone (55 traces) is the long pole - baseline_param_search.py has no CLI
# flag to split a single benchmark's datasets across GPUs, so that job can't
# be shortened further this way.
#
# Each shard gets its own --output-dir (set inside run_param_search_shard.sh)
# so parallel shards can't corrupt each other's local CSVs. Pull the combined
# results back together afterwards from the shared W&B project rather than
# merging the local CSVs by hand (see doc/get_ablation_results.py /
# get_sparsity_results_table.py for the download-from-W&B pattern to adapt).
#
# Usage:
#   ./scripts/run_param_search_all_local.sh
#
# To keep it running after you log out of the machine:
#   setsid nohup ./scripts/run_param_search_all_local.sh \
#       > logs_param_search/launcher.log 2>&1 < /dev/null &
################################################################################

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARD_SCRIPT="${PROJECT_DIR}/scripts/run_param_search_shard.sh"
LOG_DIR="${PROJECT_DIR}/logs_param_search"
mkdir -p "${LOG_DIR}"

if [[ ! -x "${SHARD_SCRIPT}" ]]; then
    echo "Error: missing or non-executable ${SHARD_SCRIPT}" >&2
    exit 1
fi

# GPU 0..3 -> bucket of benchmarks run sequentially on that GPU. Rebalance
# these lists if your GPU count or dataset sizes change.
GPU_BUCKETS=(
    "SMAP"
    "SMD"
    "MSL"
    "QAD SWaT WaDi PSM"
)
KNN_BENCHMARKS=(SWaT WaDi PSM SMD QAD MSL SMAP)

PIDS=()
LABELS=()

run_sequential_gpu_bucket() {
    local gpu_id="$1"
    shift
    local benchmark
    for benchmark in "$@"; do
        "${SHARD_SCRIPT}" "TcnED" "${benchmark}" "${gpu_id}"
    done
}

echo "=================================="
echo "Param search: local launch (${#GPU_BUCKETS[@]} GPUs + CPU)"
echo "=================================="

# TcnED: one background worker per GPU, running its bucket sequentially.
for gpu_id in "${!GPU_BUCKETS[@]}"; do
    read -ra bucket <<< "${GPU_BUCKETS[$gpu_id]}"
    echo "GPU ${gpu_id}: TcnED on ${bucket[*]}"
    run_sequential_gpu_bucket "${gpu_id}" "${bucket[@]}" > /dev/null 2>&1 &
    PIDS+=("$!")
    LABELS+=("gpu${gpu_id}:TcnED[${bucket[*]}]")
done

# KNN: CPU-only, cheap, all benchmarks in parallel.
for benchmark in "${KNN_BENCHMARKS[@]}"; do
    echo "CPU: KNN on ${benchmark}"
    "${SHARD_SCRIPT}" "KNN" "${benchmark}" > /dev/null 2>&1 &
    PIDS+=("$!")
    LABELS+=("cpu:KNN[${benchmark}]")
done

echo ""
echo "Launched ${#PIDS[@]} background worker(s):"
for i in "${!PIDS[@]}"; do
    echo "  ${LABELS[$i]}  (pid ${PIDS[$i]})"
done
echo ""
echo "Per-shard logs (one per classifier/benchmark, timestamped): ${LOG_DIR}/"
echo "Monitor with: tail -f ${LOG_DIR}/*.log"
echo ""
echo "Waiting for all workers to finish (Ctrl-C here does NOT stop them; they"
echo "keep running detached - re-check with 'jobs' or 'ps' / the log files)..."
wait
echo "All shards finished."