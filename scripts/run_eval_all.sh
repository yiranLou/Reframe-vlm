#!/bin/bash
# Evaluate a checkpoint on all benchmarks + consistency analysis
# Usage: bash scripts/run_eval_all.sh checkpoints/full [models/qwen25-vl-7b] [yes|no]
set -e

CHECKPOINT="${1:?Usage: $0 <checkpoint_path> [base_model_path] [yes|no]}"
BASE_MODEL="${2:-models/qwen25-vl-7b}"
USE_FRAME_TOKEN="${3:-no}"
RESULTS_DIR="results/$(basename $CHECKPOINT)"
mkdir -p "$RESULTS_DIR"

case "$USE_FRAME_TOKEN" in
    yes|no) ;;
    *)
        echo "USE_FRAME_TOKEN must be 'yes' or 'no'"
        exit 1
        ;;
esac

FRAME_ARGS=()
if [ "$USE_FRAME_TOKEN" = "yes" ]; then
    FRAME_ARGS+=(--use_frame_token)
fi

echo "=== Full Evaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo "Base model: $BASE_MODEL"
echo "Frame token: $USE_FRAME_TOKEN"
echo "Results: $RESULTS_DIR"

# 1. Benchmarks
echo ""
echo "[1/5] ViewSpatial-Bench..."
python src/eval/run_benchmark.py \
    --model_path "$CHECKPOINT" \
    --base_model_path "$BASE_MODEL" \
    --benchmark viewspatial \
    --output "$RESULTS_DIR/viewspatial.json" \
    "${FRAME_ARGS[@]}"

echo ""
echo "[2/5] MMSI-Bench..."
python src/eval/run_benchmark.py \
    --model_path "$CHECKPOINT" \
    --base_model_path "$BASE_MODEL" \
    --benchmark mmsi \
    --output "$RESULTS_DIR/mmsi.json" \
    "${FRAME_ARGS[@]}"

echo ""
echo "[3/5] Ego3D-Bench..."
python src/eval/run_benchmark.py \
    --model_path "$CHECKPOINT" \
    --base_model_path "$BASE_MODEL" \
    --benchmark ego3d \
    --output "$RESULTS_DIR/ego3d.json" \
    "${FRAME_ARGS[@]}"

# 2. Consistency Analysis
echo ""
echo "[4/5] Consistency Analysis..."
python src/eval/consistency_eval.py \
    --results "$RESULTS_DIR/viewspatial.json" \
    --benchmark_data data/processed/viewspatial_test.jsonl \
    --output "$RESULTS_DIR/consistency.json"

# 3. Frame Type Analysis
echo ""
echo "[5/5] Frame Type Analysis..."
python src/eval/frame_type_analysis.py \
    --results "$RESULTS_DIR/viewspatial.json" \
    --benchmark_data data/processed/viewspatial_test.jsonl \
    --output "$RESULTS_DIR/frame_type.json"

# Summary
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
for f in "$RESULTS_DIR"/viewspatial.json "$RESULTS_DIR"/mmsi.json "$RESULTS_DIR"/ego3d.json; do
    if [ -f "$f" ]; then
        name=$(basename "$f" .json)
        acc=$(python -c "import json; d=json.load(open('$f')); print(f'{d[\"accuracy\"]:.2f}%')")
        echo "  $name: $acc"
    fi
done

if [ -f "$RESULTS_DIR/consistency.json" ]; then
    python -c "
import json
d = json.load(open('$RESULTS_DIR/consistency.json'))
print(f\"  FCA: {d['frame_consistency_accuracy']:.2f}%\")
print(f\"  CR: {d['contradiction_rate']:.2f}%\")
pdr = d.get('paired_disagreement_rate')
if pdr is not None:
    print(f\"  PDR: {pdr:.2f}%\")
print(f\"  FG: {d['frame_gap']:.2f}%\")
"
fi

# One-line summary for autoresearch
echo ""
echo "METRIC_SUMMARY: viewspatial=$(python -c "import json; print(json.load(open('$RESULTS_DIR/viewspatial.json'))['accuracy'])") mmsi=$(python -c "import json; print(json.load(open('$RESULTS_DIR/mmsi.json'))['accuracy'])") ego3d=$(python -c "import json; print(json.load(open('$RESULTS_DIR/ego3d.json'))['accuracy'])") cr=$(python -c "import json; print(json.load(open('$RESULTS_DIR/consistency.json'))['contradiction_rate'])" 2>/dev/null || echo 'N/A') pdr=$(python -c "import json; print(json.load(open('$RESULTS_DIR/consistency.json')).get('paired_disagreement_rate', 'N/A'))" 2>/dev/null || echo 'N/A')"
