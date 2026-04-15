#!/bin/bash
# Plan B: let the current Full Method Ego3D eval finish, then run only
# baseline_lora/frame_lora on MMSI (skip their Ego3D — too slow).
# Finally, regenerate the summary.
#
# Expected total: ~2-2.5h from now.

set -u
cd /workspace/reframe-vlm
export PYTHONPATH=/workspace/reframe-vlm

BASE_CKPT="checkpoints/baseline_lora/checkpoint-716"
FRAME_CKPT="checkpoints/frame_lora/checkpoint-716"

LOGDIR=logs/evals
mkdir -p "$LOGDIR"
log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/_overnight.log"; }

wait_for_pid() {
    local pid=$1
    local label=$2
    if [[ -z "$pid" ]]; then return; fi
    log "waiting for $label (pid=$pid) ..."
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    log "$label finished"
}

run_eval() {
    local label=$1
    local ckpt=$2
    local bench=$3
    local use_token=$4
    local out="results/${label}/${bench}.json"
    if [[ -f "$out" ]]; then log "[skip] $out exists"; return; fi
    mkdir -p "results/${label}"
    local flag=""
    if [[ "$use_token" == "yes" ]]; then flag="--use_frame_token"; fi
    log "=== eval: $label on $bench (frame_token=$use_token) ==="
    python src/eval/run_benchmark.py \
        --model_path "$ckpt" \
        --benchmark "$bench" \
        --output "$out" \
        $flag > "$LOGDIR/${label}_${bench}.log" 2>&1 \
        && log "OK  $label $bench" \
        || log "FAIL $label $bench (see $LOGDIR/${label}_${bench}.log)"
}

CURRENT_PID=${1:-71013}
wait_for_pid "$CURRENT_PID" "full_method_ep1/ego3d"

# Only MMSI for the baselines (skip their Ego3D per Plan B)
run_eval baseline_lora_ep1 "$BASE_CKPT" mmsi no
run_eval frame_lora_ep1    "$FRAME_CKPT" mmsi yes

log "=== summarizing ==="
python scripts/summarize_results.py --out results/summary.md \
    > "$LOGDIR/_summary.log" 2>&1 \
    && log "summary written to results/summary.md" \
    || log "summary FAILED (see $LOGDIR/_summary.log)"

log "ALL DONE (Plan B)"
