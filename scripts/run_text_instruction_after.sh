#!/bin/bash
# Wait for the wrong-frame controls eval to finish, then launch the
# LoRA+text-instruction SFT training. Then eval it on all domains.
# ~17h training + ~2h eval after a ~3.5h wait.

set -u
cd /workspace/reframe-vlm
export PYTHONPATH=/workspace/reframe-vlm

WAIT_PID=${1:-85936}
LOGDIR=logs/text_instr
mkdir -p "$LOGDIR"
log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/_runner.log"; }

log "waiting for wrong-frame controls (pid=$WAIT_PID)..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
log "wrong-frame controls finished"

# ── Train ───────────────────────────────────────────────────────
log "=== launching text_instruction training ==="
WANDB_MODE=offline python -u src/training/train.py \
    --config configs/train_text_instruction.yaml \
    > "$LOGDIR/train.log" 2>&1
log "training finished"

# ── Find the checkpoint (use last saved)
CKPT=$(ls -td checkpoints/text_instruction_lora/checkpoint-* 2>/dev/null | head -1)
if [[ -z "$CKPT" ]]; then
    log "FATAL: no checkpoint found at checkpoints/text_instruction_lora/"
    exit 1
fi
log "checkpoint: $CKPT"

# ── Evaluate on all 3 benchmarks, always with --use_frame_prompt to match training
run_eval() {
    local bench=$1
    local out="results/text_instruction_lora_ep1/${bench}.json"
    if [[ -f "$out" ]]; then log "[skip] $out exists"; return; fi
    mkdir -p "$(dirname "$out")"
    log "=== eval: text_instruction on $bench ==="
    python src/eval/run_benchmark.py \
        --model_path "$CKPT" \
        --benchmark "$bench" \
        --use_frame_prompt \
        --output "$out" > "$LOGDIR/eval_${bench}.log" 2>&1 \
        && log "OK  $bench" || log "FAIL $bench (see $LOGDIR/eval_${bench}.log)"
}

run_eval viewspatial
run_eval mmsi
# skip ego3d baseline/frame per plan B consistency

# ── Refresh bootstrap & domain tables with the new method ────────
log "=== regenerate summary tables ==="
python scripts/bootstrap_ci.py         > "$LOGDIR/bootstrap_ci.log" 2>&1
python scripts/bootstrap_ci_domain.py  > "$LOGDIR/bootstrap_ci_domain.log" 2>&1
python scripts/domain_confound_check.py > "$LOGDIR/domain_confound.log" 2>&1
python scripts/summarize_results.py --out results/summary.md \
    > "$LOGDIR/summary.log" 2>&1
log "ALL DONE (text_instruction full pipeline)"
