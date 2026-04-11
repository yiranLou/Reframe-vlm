#!/bin/bash
# Run zero-shot baseline on all 3 benchmarks
# This is Step 2 - must pass before anything else
set -e

MODEL_PATH="${1:-models/qwen25-vl-7b}"
RESULTS_DIR="results/zeroshot"
mkdir -p "$RESULTS_DIR"

echo "=== Zero-Shot Baseline ==="
echo "Model: $MODEL_PATH"

echo ""
echo "[1/3] ViewSpatial-Bench..."
python src/eval/run_benchmark.py \
    --model_path "$MODEL_PATH" \
    --benchmark viewspatial \
    --output "$RESULTS_DIR/viewspatial.json"

echo ""
echo "[2/3] MMSI-Bench..."
python src/eval/run_benchmark.py \
    --model_path "$MODEL_PATH" \
    --benchmark mmsi \
    --output "$RESULTS_DIR/mmsi.json"

echo ""
echo "[3/3] Ego3D-Bench..."
python src/eval/run_benchmark.py \
    --model_path "$MODEL_PATH" \
    --benchmark ego3d \
    --output "$RESULTS_DIR/ego3d.json"

echo ""
echo "=== Zero-Shot Results ==="
for f in "$RESULTS_DIR"/*.json; do
    name=$(basename "$f" .json)
    acc=$(python -c "import json; d=json.load(open('$f')); print(f'{d[\"accuracy\"]:.2f}%')")
    echo "  $name: $acc"
done
