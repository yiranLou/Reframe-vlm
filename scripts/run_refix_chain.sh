#!/bin/bash
# Chain: wait for Frame LoRA re-eval -> Full Method re-eval -> custom_train_loop smoke.

set -u
cd /workspace/reframe-vlm
export PYTHONPATH=/workspace/reframe-vlm

LOGDIR=logs/refix
mkdir -p "$LOGDIR"
log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/_chain.log"; }

WAIT_PID=${1:-133909}
log "waiting for Frame LoRA re-eval (pid=$WAIT_PID) ..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
log "Frame LoRA re-eval finished"

# Full Method re-eval
log "=== Full Method re-eval with --use_frame_token ==="
python src/eval/run_benchmark.py \
    --model_path checkpoints/full/checkpoint-926 \
    --benchmark viewspatial \
    --use_frame_token \
    --output results/full_method_ep1_refix/viewspatial.json \
    > "$LOGDIR/refix_full_vs.log" 2>&1 \
    && log "OK  Full Method re-eval" \
    || log "FAIL Full Method re-eval"

# custom_train_loop smoke
log "=== custom_train_loop smoke test (20 samples, 1 epoch) ==="
WANDB_MODE=offline python -u src/training/custom_train_loop.py \
    --config configs/smoke_custom.yaml \
    > "$LOGDIR/smoke_train.log" 2>&1
SMOKE_RC=$?
log "smoke train rc=$SMOKE_RC"

# Inspect saved checkpoints
log "=== smoke-test saved artefacts ==="
for d in checkpoints/smoke_custom/epoch_1 checkpoints/smoke_custom/final; do
    if [[ -d "$d" ]]; then
        log "$d contains:"
        ls -la "$d" | tee -a "$LOGDIR/_chain.log"
    else
        log "MISSING: $d"
    fi
done

log "ALL DONE (refix chain)"
