#!/bin/bash
# Poll for the ep1 checkpoint save, then gracefully stop training.
#
# HF Trainer with save_strategy=epoch and 926 steps/epoch saves to
# checkpoints/full/checkpoint-926 at the end of epoch 1. Once that
# directory contains adapter_model.safetensors (save fully flushed),
# we SIGTERM the training PID so GPU is freed for the next job.

set -u
PID=${1:-49026}
CKPT_DIR="checkpoints/full/checkpoint-926"
TARGET_FILE="$CKPT_DIR/adapter_model.safetensors"

echo "[$(date -Is)] Watching for $TARGET_FILE, will kill PID=$PID on appearance"

while kill -0 "$PID" 2>/dev/null; do
    if [[ -f "$TARGET_FILE" ]]; then
        # Wait 30s extra to let any async flushes finish
        echo "[$(date -Is)] ep1 checkpoint detected, grace 30s..."
        sleep 30
        echo "[$(date -Is)] sending SIGTERM to $PID"
        kill -TERM "$PID"
        # Give it a minute to shut down cleanly
        for i in 1 2 3 4 5 6; do
            sleep 10
            if ! kill -0 "$PID" 2>/dev/null; then break; fi
        done
        if kill -0 "$PID" 2>/dev/null; then
            echo "[$(date -Is)] SIGTERM didn't stop, sending SIGKILL"
            kill -KILL "$PID"
        fi
        echo "[$(date -Is)] training stopped. ep1 checkpoint at $CKPT_DIR"
        exit 0
    fi
    sleep 60
done

echo "[$(date -Is)] PID $PID already exited before ep1 checkpoint seen"
exit 1
