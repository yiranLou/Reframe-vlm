#!/bin/bash
# Run all ablation experiments
# Each ablation: train + eval on all 3 benchmarks
set -e

BASE_MODEL="${1:-models/qwen25-vl-7b}"

echo "=== Running Ablation Experiments ==="

ABLATIONS=(
    "configs/ablation_no_frame_token.yaml"
    "configs/ablation_no_consistency.yaml"
    "configs/ablation_no_permutation.yaml"
    "configs/ablation_viewspatial_only.yaml"
)

for config in "${ABLATIONS[@]}"; do
    name=$(basename "$config" .yaml)
    echo ""
    echo "=============================="
    echo "Ablation: $name"
    echo "=============================="

    # Train
    echo "Training..."
    python src/training/train.py --config "$config"

    # Get output dir from config
    output_dir=$(python -c "import yaml; print(yaml.safe_load(open('$config'))['output_dir'])")

    # Eval
    echo "Evaluating..."
    bash scripts/run_eval_all.sh "$output_dir" "$BASE_MODEL"

    echo ""
    echo "$name: DONE"
done

# Also eval the baseline (already trained in step 5)
echo ""
echo "=============================="
echo "Baseline LoRA (already trained)"
echo "=============================="
if [ -d "checkpoints/baseline_lora" ]; then
    bash scripts/run_eval_all.sh "checkpoints/baseline_lora" "$BASE_MODEL"
fi

echo ""
echo "=== All Ablations Complete ==="
echo ""
echo "Results summary:"
for dir in results/ablation_* results/baseline_lora results/full; do
    if [ -f "$dir/viewspatial.json" ]; then
        name=$(basename "$dir")
        vs=$(python -c "import json; print(f'{json.load(open(\"$dir/viewspatial.json\"))[\"accuracy\"]:.2f}')")
        echo "  $name: ViewSpatial=$vs%"
    fi
done
