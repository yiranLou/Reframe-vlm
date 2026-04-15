#!/bin/bash
# Run every remaining benchmark eval in sequence, then produce the summary.
# Designed to run overnight in background.
#
# Current assumption: the ViewSpatial eval for checkpoints/full/checkpoint-926
# is already running (PID passed via CURRENT_PID). We wait for it to finish,
# then chain the rest.

set -u
cd /workspace/reframe-vlm
export PYTHONPATH=/workspace/reframe-vlm

FULL_CKPT="checkpoints/full/checkpoint-926"
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
    local label=$1   # result subdir label
    local ckpt=$2
    local bench=$3
    local use_token=$4   # "yes" or "no"
    local out="results/${label}/${bench}.json"
    if [[ -f "$out" ]]; then
        log "[skip] $out exists"
        return
    fi
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

# ── Wait for the already-running full_method ViewSpatial eval ─────
CURRENT_PID=${1:-68636}
wait_for_pid "$CURRENT_PID" "full_method_ep1/viewspatial"

# ── Full method: remaining 2 benchmarks ───────────────────────────
run_eval full_method_ep1 "$FULL_CKPT" mmsi  yes
run_eval full_method_ep1 "$FULL_CKPT" ego3d yes

# ── Baseline LoRA on MMSI + Ego3D ─────────────────────────────────
run_eval baseline_lora_ep1 "$BASE_CKPT" mmsi  no
run_eval baseline_lora_ep1 "$BASE_CKPT" ego3d no

# ── Frame LoRA on MMSI + Ego3D ────────────────────────────────────
run_eval frame_lora_ep1 "$FRAME_CKPT" mmsi  yes
run_eval frame_lora_ep1 "$FRAME_CKPT" ego3d yes

# ── Final: regenerate summary ─────────────────────────────────────
log "=== summarizing ==="
python scripts/summarize_results.py --out results/summary.md \
    > "$LOGDIR/_summary.log" 2>&1 \
    && log "summary written to results/summary.md" \
    || log "summary FAILED (see $LOGDIR/_summary.log)"

log "ALL DONE"
