#!/bin/bash
# Fill Table 1: evaluate existing baseline_lora + frame_lora + full checkpoints
# on MMSI-Bench and Ego3D-Bench so all 3 benchmark columns are populated.
#
# Run this AFTER Full Method training completes (or whenever GPU is free).

set -e
cd "$(dirname "$0")/.."

run() {
    local label=$1
    local ckpt=$2
    local bench=$3
    local use_token=$4  # "yes" or "no"
    local out="results/${label}/${bench}.json"
    if [[ -f "$out" ]]; then
        echo "[skip] $out already exists"
        return
    fi
    mkdir -p "results/${label}"
    echo "=== $label on $bench (frame_token=$use_token) ==="
    local flag=""
    if [[ "$use_token" == "yes" ]]; then flag="--use_frame_token"; fi
    python src/eval/run_benchmark.py \
        --model_path "$ckpt" \
        --benchmark "$bench" \
        --output "$out" \
        $flag 2>&1 | tail -5 || echo "(failed: $label $bench)"
}

run baseline_lora_ep1 checkpoints/baseline_lora/checkpoint-716 mmsi  no
run baseline_lora_ep1 checkpoints/baseline_lora/checkpoint-716 ego3d no

run frame_lora_ep1    checkpoints/frame_lora/checkpoint-716    mmsi  yes
run frame_lora_ep1    checkpoints/frame_lora/checkpoint-716    ego3d yes

# Full method (only if checkpoint exists)
if [[ -d checkpoints/full ]]; then
    FULL_CKPT=$(ls -td checkpoints/full/checkpoint-* 2>/dev/null | head -1)
    if [[ -n "$FULL_CKPT" ]]; then
        run full_method_ep1 "$FULL_CKPT" viewspatial yes
        run full_method_ep1 "$FULL_CKPT" mmsi        yes
        run full_method_ep1 "$FULL_CKPT" ego3d       yes
    fi
fi

echo ""
echo "=== Regenerating summary ==="
python scripts/summarize_results.py --out results/summary.md
