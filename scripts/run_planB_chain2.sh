#!/bin/bash
# Plan B chain v2 — resumes after the failure. Dry-run already passed, so we
# skip it. Order:
#   1. text-instruction inference-time controls × 4 conditions
#   2. frame_gated_only training (1 epoch)
#   3. eval frame_gated on viewspatial + mmsi
#   4. refresh bootstrap / domain / confound tables

set -u
cd /workspace/reframe-vlm
export PYTHONPATH=/workspace/reframe-vlm

LOGDIR=logs/planB
mkdir -p "$LOGDIR"
log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/_chain.log"; }

log "=== text_instruction_controls (4 conditions) ==="
python scripts/text_instruction_controls.py \
    --ckpt checkpoints/text_instruction_lora/checkpoint-716 \
    --conditions wrong,none,always_camera,always_person \
    --out_dir results/text_instr_controls \
    > "$LOGDIR/text_instr_controls.log" 2>&1 \
    && log "OK  text_instruction_controls" \
    || { log "FAIL text_instruction_controls"; exit 2; }

log "=== frame_gated_only training (1 epoch, ~25 h) ==="
WANDB_MODE=offline python -u src/training/train.py \
    --config configs/train_frame_gated.yaml \
    > "$LOGDIR/train_frame_gated.log" 2>&1
log "training finished"

CKPT=$(ls -td checkpoints/frame_gated_lora/checkpoint-* 2>/dev/null | head -1)
if [[ -z "$CKPT" ]]; then
    log "FATAL: no checkpoint at checkpoints/frame_gated_lora/"
    exit 1
fi
log "checkpoint: $CKPT"

run_eval() {
    local bench=$1
    local out="results/frame_gated_lora_ep1/${bench}.json"
    if [[ -f "$out" ]]; then log "[skip] $out exists"; return; fi
    mkdir -p "$(dirname "$out")"
    log "=== eval frame_gated on $bench ==="
    python src/eval/run_benchmark.py \
        --model_path "$CKPT" \
        --benchmark "$bench" \
        --output "$out" \
        > "$LOGDIR/eval_${bench}.log" 2>&1 \
        && log "OK  $bench" \
        || log "FAIL $bench"
}

run_eval viewspatial
run_eval mmsi

log "=== refreshing bootstrap / domain / confound tables ==="
python scripts/bootstrap_ci.py         > "$LOGDIR/bootstrap_ci.log" 2>&1
python scripts/bootstrap_ci_domain.py  > "$LOGDIR/bootstrap_ci_domain.log" 2>&1
python scripts/diagnostics_domain_split.py > "$LOGDIR/diag_split.log" 2>&1
python scripts/domain_confound_check.py > "$LOGDIR/confound.log" 2>&1
python scripts/summarize_results.py --out results/summary.md \
    > "$LOGDIR/summary.log" 2>&1
log "ALL DONE (Plan B chain v2)"
