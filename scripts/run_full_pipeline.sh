#!/bin/bash
# ============================================
# ReFrame-VLM 完整实验流水线
# 按照论文的 Step 1 - Step 12 顺序执行
# ============================================
set -e

MODEL_PATH="${MODEL_ROOT:-models}/qwen25-vl-7b"
DATA_RAW="${DATA_ROOT:-data/raw}"

echo "============================================"
echo " ReFrame-VLM 完整实验流水线"
echo "============================================"
echo "模型: $MODEL_PATH"
echo "数据: $DATA_RAW"
echo ""

# ─── Step 1-2: Zero-shot Baseline ───
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 1-2: Zero-Shot Baselines"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash scripts/run_baseline_zeroshot.sh "$MODEL_PATH"

echo ""
echo ">> Gate 1: 检查 zero-shot 数字是否合理"
echo "   如果和论文报告的差距 > 5%，请先排查 eval pipeline"
echo "   按回车继续，或 Ctrl+C 退出..."
read -r

# ─── Step 3-4: 数据预处理 ───
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 3-4: 数据预处理"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash scripts/preprocess_all.sh

echo ""
echo ">> Gate 2: 检查数据量和 pair 数"
echo "   按回车继续..."
read -r

# ─── Step 5: Baseline LoRA Fine-tune ───
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 5: Baseline LoRA Fine-tune"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/training/train.py --config configs/train_baseline.yaml
bash scripts/run_eval_all.sh checkpoints/baseline_lora "$MODEL_PATH"

echo ""
echo ">> Gate 3: Baseline 应该比 zero-shot 高 3-5%+"
echo "   按回车继续..."
read -r

# ─── Step 6: Frame-Conditioned LoRA ───
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 6: Frame-Conditioned LoRA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/training/train.py --config configs/train_frame.yaml
bash scripts/run_eval_all.sh checkpoints/frame_lora "$MODEL_PATH"

echo ""
echo ">> 检查: Frame LoRA 应该比 baseline 再高一些"
echo "   按回车继续..."
read -r

# ─── Step 7: Text Prompt Baseline ───
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 7: Text Prompt Baseline (关键对照)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash scripts/run_baseline_prompt.sh "$MODEL_PATH"

echo ""
echo ">> 关键检查: Prompt baseline 必须明显低于 Step 6"
echo "   如果差距 < 2%，说明 frame token 设计需要加强"
echo "   按回车继续..."
read -r

# ─── Step 8-9: Full Method ───
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 8-9: Full Method (+ Consistency Loss + View Permutation)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/training/train.py --config configs/train_full.yaml
bash scripts/run_eval_all.sh checkpoints/full "$MODEL_PATH"

# ─── Step 10: Ablations ───
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 10: Ablation 实验"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash scripts/run_ablation.sh "$MODEL_PATH"

# ─── Step 11-12: 分析 ───
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 11-12: Consistency + Frame Type 分析"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 对所有方法做 consistency 分析
for dir in results/zeroshot results/baseline_lora results/frame_lora results/full; do
    if [ -f "$dir/viewspatial.json" ]; then
        name=$(basename "$dir")
        echo ""
        echo "--- Consistency Analysis: $name ---"
        python src/eval/consistency_eval.py \
            --results "$dir/viewspatial.json" \
            --benchmark_data data/processed/viewspatial_test.jsonl \
            --output "$dir/consistency.json" 2>/dev/null || true
    fi
done

# 多方法对比表
echo ""
echo "--- Frame Type 对比表 ---"
RESULT_FILES=""
METHOD_NAMES=""
for dir in results/zeroshot results/baseline_lora results/full; do
    if [ -f "$dir/viewspatial.json" ]; then
        RESULT_FILES="$RESULT_FILES $dir/viewspatial.json"
        METHOD_NAMES="$METHOD_NAMES $(basename $dir)"
    fi
done

if [ -n "$RESULT_FILES" ]; then
    python src/eval/frame_type_analysis.py \
        --results $RESULT_FILES \
        --benchmark_data data/processed/viewspatial_test.jsonl \
        --method_names $METHOD_NAMES \
        --output results/frame_type_comparison.json
fi

# ─── 最终汇总 ───
echo ""
echo "============================================"
echo " 实验全部完成！最终汇总"
echo "============================================"
echo ""
echo "Table 1: 主结果"
echo "─────────────────────────────────────────────"
printf "%-25s %12s %12s %12s\n" "Method" "ViewSpatial" "MMSI" "Ego3D"
echo "─────────────────────────────────────────────"

for dir in results/zeroshot results/prompt_baseline results/baseline_lora results/frame_lora results/full; do
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        vs=$(python -c "import json; print(f'{json.load(open(\"$dir/viewspatial.json\"))[\"accuracy\"]:.2f}')" 2>/dev/null || echo "N/A")
        mmsi=$(python -c "import json; print(f'{json.load(open(\"$dir/mmsi.json\"))[\"accuracy\"]:.2f}')" 2>/dev/null || echo "N/A")
        ego=$(python -c "import json; print(f'{json.load(open(\"$dir/ego3d.json\"))[\"accuracy\"]:.2f}')" 2>/dev/null || echo "N/A")
        printf "%-25s %12s %12s %12s\n" "$name" "$vs" "$mmsi" "$ego"
    fi
done

echo ""
echo "Table 4: Consistency 分析"
echo "─────────────────────────────────────────────"
printf "%-25s %8s %8s %8s\n" "Method" "FCA" "CR" "FG"
echo "─────────────────────────────────────────────"

for dir in results/zeroshot results/baseline_lora results/full; do
    if [ -f "$dir/consistency.json" ]; then
        name=$(basename "$dir")
        python -c "
import json
d = json.load(open('$dir/consistency.json'))
print(f'  {\"$name\":<23} {d[\"frame_consistency_accuracy\"]:>7.2f} {d[\"contradiction_rate\"]:>7.2f} {d[\"frame_gap\"]:>7.2f}')
" 2>/dev/null || true
    fi
done

echo ""
echo "所有结果保存在 results/ 目录下"
echo "下一步: 写论文！"
