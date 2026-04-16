#!/bin/bash
# Plan B' execution chain — runs sequentially after the in-flight
# always_person job finishes, freeing the GPU for the next steps.
#
# Order (estimated):
#   1. wait for frame_lora always_person  (~1 h remaining)
#   2. text-instruction inference-time controls × 4 conditions  (~5.5 h)
#   3. dryrun_frame_gated.py                                    (~0.2 h)
#   4. frame_gated_only training (1 epoch)                      (~25 h)
#   5. eval frame_gated_only on viewspatial + mmsi              (~2 h)
#   6. refresh bootstrap / domain / confound tables             (<1 min)
#
# Total ≈ 33 h. Designed to be a fire-and-forget overnight chain.

set -u
cd /workspace/reframe-vlm
export PYTHONPATH=/workspace/reframe-vlm

LOGDIR=logs/planB
mkdir -p "$LOGDIR"
log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/_chain.log"; }

WAIT_PID=${1:-129823}
log "waiting for always_person frame-token control (pid=$WAIT_PID) ..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
log "always_person finished"

# ── Step 2: text-instruction inference-time controls ────────────
log "=== text_instruction_controls (4 conditions) ==="
python scripts/text_instruction_controls.py \
    --ckpt checkpoints/text_instruction_lora/checkpoint-716 \
    --conditions wrong,none,always_camera,always_person \
    --out_dir results/text_instr_controls \
    > "$LOGDIR/text_instr_controls.log" 2>&1 \
    && log "OK  text_instruction_controls" \
    || log "FAIL text_instruction_controls"

# ── Step 3: Frame-Gated LoRA dry-run ────────────────────────────
log "=== dryrun_frame_gated ==="
python scripts/dryrun_frame_gated.py --batch_size 2 --steps 2 \
    > "$LOGDIR/dryrun.log" 2>&1
DRY_RC=$?
if [[ $DRY_RC -ne 0 ]]; then
    log "FAIL dryrun_frame_gated (rc=$DRY_RC). See $LOGDIR/dryrun.log"
    log "Aborting — frame_gated training will not start."
    exit 1
fi
log "OK  dryrun_frame_gated"

# ── Step 4: frame_gated_only training (1 epoch) ─────────────────
log "=== frame_gated_only training (1 epoch, ~25 h) ==="
WANDB_MODE=offline python -u src/training/train.py \
    --config configs/train_frame_gated.yaml \
    > "$LOGDIR/train_frame_gated.log" 2>&1
log "training finished"

# ── Step 5: eval frame_gated on viewspatial + mmsi ──────────────
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

# ── Step 6: refresh tables ──────────────────────────────────────
log "=== refreshing bootstrap / domain / confound tables ==="
python scripts/bootstrap_ci.py         > "$LOGDIR/bootstrap_ci.log" 2>&1
python scripts/bootstrap_ci_domain.py  > "$LOGDIR/bootstrap_ci_domain.log" 2>&1
python scripts/diagnostics_domain_split.py > "$LOGDIR/diag_split.log" 2>&1
python scripts/domain_confound_check.py > "$LOGDIR/confound.log" 2>&1
python scripts/summarize_results.py --out results/summary.md \
    > "$LOGDIR/summary.log" 2>&1
log "ALL DONE (Plan B chain)"
