#!/bin/bash
# Lambda (consistency loss 权重) 网格搜索
# 固定其他超参，只调 lambda
# 适合用 autoresearch 自动跑
set -e

MODEL_PATH="${1:-models/qwen25-vl-7b}"
LAMBDAS="0.01 0.05 0.1 0.2 0.3"

echo "============================================"
echo " Lambda Grid Search"
echo "============================================"
echo "搜索范围: $LAMBDAS"
echo ""

BEST_ACC=0
BEST_LAMBDA=""

for lambda in $LAMBDAS; do
    echo ""
    echo "━━━ Lambda = $lambda ━━━"

    # 动态生成 config
    OUTPUT_DIR="checkpoints/grid_lambda_${lambda}"
    cat > "/tmp/grid_lambda_${lambda}.yaml" << EOF
mode: full
model_path: $MODEL_PATH
train_data: data/processed/train.jsonl
consistency_pairs: data/processed/consistency_pairs.jsonl
output_dir: $OUTPUT_DIR

lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05

use_frame_tokens: true
lambda_consistency: $lambda
canonical_dim: 64
view_permutation: true
view_permutation_prob: 0.5

num_epochs: 3
batch_size: 2
gradient_accumulation: 32
learning_rate: 2.0e-5
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
run_name: grid_lambda_${lambda}
seed: 42
EOF

    # 训练
    python src/training/train.py --config "/tmp/grid_lambda_${lambda}.yaml"

    # 评测 (只跑 ViewSpatial 加快速度)
    RESULT_DIR="results/grid_lambda_${lambda}"
    mkdir -p "$RESULT_DIR"
    python src/eval/run_benchmark.py \
        --model_path "$OUTPUT_DIR" \
        --base_model_path "$MODEL_PATH" \
        --benchmark viewspatial \
        --output "$RESULT_DIR/viewspatial.json"

    # Consistency 分析
    python src/eval/consistency_eval.py \
        --results "$RESULT_DIR/viewspatial.json" \
        --benchmark_data data/processed/viewspatial_test.jsonl \
        --output "$RESULT_DIR/consistency.json" 2>/dev/null || true

    # 记录结果
    ACC=$(python -c "import json; print(json.load(open('$RESULT_DIR/viewspatial.json'))['accuracy'])")
    CR=$(python -c "import json; print(json.load(open('$RESULT_DIR/consistency.json'))['contradiction_rate'])" 2>/dev/null || echo "N/A")

    echo "Lambda=$lambda -> ViewSpatial=$ACC%, CR=$CR%"

    # 追踪最优
    IS_BETTER=$(python -c "print(1 if $ACC > $BEST_ACC else 0)")
    if [ "$IS_BETTER" = "1" ]; then
        BEST_ACC=$ACC
        BEST_LAMBDA=$lambda
    fi

    # 写入日志
    echo "lambda=$lambda  viewspatial=$ACC  cr=$CR" >> results/grid_search_log.txt
done

echo ""
echo "============================================"
echo " Grid Search 完成"
echo "============================================"
echo "最优 Lambda: $BEST_LAMBDA (ViewSpatial=$BEST_ACC%)"
echo ""
echo "完整日志: results/grid_search_log.txt"
echo ""
echo "结果汇总:"
printf "%-10s %12s %8s\n" "Lambda" "ViewSpatial" "CR"
echo "─────────────────────────────────"
cat results/grid_search_log.txt | while read line; do
    lambda=$(echo $line | awk -F'[ =]+' '{print $2}')
    vs=$(echo $line | awk -F'[ =]+' '{print $4}')
    cr=$(echo $line | awk -F'[ =]+' '{print $6}')
    printf "%-10s %12s %8s\n" "$lambda" "$vs" "$cr"
done
