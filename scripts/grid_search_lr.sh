#!/bin/bash
# Learning Rate 网格搜索
# 在最优 lambda 的基础上调 lr
set -e

MODEL_PATH="${1:-models/qwen25-vl-7b}"
LAMBDA="${2:-0.1}"
LRS="1e-5 2e-5 3e-5 5e-5"

echo "============================================"
echo " Learning Rate Grid Search"
echo "============================================"
echo "固定 Lambda=$LAMBDA"
echo "搜索范围: $LRS"
echo ""

for lr in $LRS; do
    echo ""
    echo "━━━ LR = $lr ━━━"

    OUTPUT_DIR="checkpoints/grid_lr_${lr}"
    cat > "/tmp/grid_lr_${lr}.yaml" << EOF
mode: full
model_path: $MODEL_PATH
train_data: data/processed/train.jsonl
consistency_pairs: data/processed/consistency_pairs.jsonl
output_dir: $OUTPUT_DIR

lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05
use_frame_tokens: true
lambda_consistency: $LAMBDA
canonical_dim: 64
view_permutation: true
view_permutation_prob: 0.5

num_epochs: 3
batch_size: 2
gradient_accumulation: 32
learning_rate: $lr
scheduler: cosine
warmup_ratio: 0.05
weight_decay: 0.01
max_length: 2048
bf16: true
gradient_checkpointing: true
num_workers: 4

logging_steps: 10
save_strategy: epoch
save_total_limit: 1
report_to: wandb
run_name: grid_lr_${lr}
seed: 42
EOF

    python src/training/train.py --config "/tmp/grid_lr_${lr}.yaml"

    RESULT_DIR="results/grid_lr_${lr}"
    mkdir -p "$RESULT_DIR"
    python src/eval/run_benchmark.py \
        --model_path "$OUTPUT_DIR" \
        --base_model_path "$MODEL_PATH" \
        --benchmark viewspatial \
        --output "$RESULT_DIR/viewspatial.json"

    ACC=$(python -c "import json; print(json.load(open('$RESULT_DIR/viewspatial.json'))['accuracy'])")
    echo "LR=$lr -> ViewSpatial=$ACC%"
    echo "lr=$lr  viewspatial=$ACC" >> results/grid_search_lr_log.txt
done

echo ""
echo "完整日志: results/grid_search_lr_log.txt"
