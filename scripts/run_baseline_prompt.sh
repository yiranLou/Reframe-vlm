#!/bin/bash
# Text prompt baseline (baseline 2)
# Uses text frame instructions instead of learned tokens
# This number MUST be lower than the frame-conditioned method
set -e

MODEL_PATH="${1:-models/qwen25-vl-7b}"
RESULTS_DIR="results/prompt_baseline"
mkdir -p "$RESULTS_DIR"

echo "=== Text Prompt Baseline ==="
echo "Model: $MODEL_PATH"

echo ""
echo "[1/3] ViewSpatial-Bench (with frame prompts)..."
python src/eval/run_benchmark.py \
    --model_path "$MODEL_PATH" \
    --benchmark viewspatial \
    --use_frame_prompt \
    --output "$RESULTS_DIR/viewspatial.json"

echo ""
echo "[2/3] MMSI-Bench (with frame prompts)..."
python src/eval/run_benchmark.py \
    --model_path "$MODEL_PATH" \
    --benchmark mmsi \
    --use_frame_prompt \
    --output "$RESULTS_DIR/mmsi.json"

echo ""
echo "[3/3] Ego3D-Bench (with frame prompts)..."
python src/eval/run_benchmark.py \
    --model_path "$MODEL_PATH" \
    --benchmark ego3d \
    --use_frame_prompt \
    --output "$RESULTS_DIR/ego3d.json"

echo ""
echo "=== Text Prompt Baseline Results ==="
for f in "$RESULTS_DIR"/*.json; do
    name=$(basename "$f" .json)
    acc=$(python -c "import json; d=json.load(open('$f')); print(f'{d[\"accuracy\"]:.2f}%')")
    echo "  $name: $acc"
done
