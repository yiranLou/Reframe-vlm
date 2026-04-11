#!/bin/bash
# 完整数据预处理流水线
# 从原始数据 -> 统一格式 -> consistency pairs -> LLaMA-Factory 格式
# 每一步都有 gate check
set -e

DATA_RAW="${DATA_ROOT:-data/raw}"
DATA_OUT="data/processed"
LF_OUT="data/llamafactory"

echo "============================================"
echo " ReFrame-VLM 数据预处理流水线"
echo "============================================"
echo "原始数据: $DATA_RAW"
echo "输出目录: $DATA_OUT"
echo ""

# Step 1: 转换 ViewSpatial
echo "[Step 1/6] 转换 ViewSpatial 训练集..."
python data/scripts/convert_viewspatial.py \
    --data_dir "$DATA_RAW/viewspatial" \
    --output_dir "$DATA_OUT" \
    --split train

echo ""
echo "[Step 2/6] 转换 ViewSpatial 测试集..."
python data/scripts/convert_viewspatial.py \
    --data_dir "$DATA_RAW/viewspatial" \
    --output_dir "$DATA_OUT" \
    --split test

# Gate check 1
VS_TRAIN_COUNT=$(wc -l < "$DATA_OUT/viewspatial_train.jsonl")
echo ""
echo ">> Gate Check: ViewSpatial 训练集 $VS_TRAIN_COUNT 条"
if [ "$VS_TRAIN_COUNT" -lt 10000 ]; then
    echo "WARNING: ViewSpatial 样本量偏少 (<10K)，请检查数据目录"
fi

# Step 3: 转换 RoboSpatial
echo ""
echo "[Step 3/6] 转换 RoboSpatial（采样 60K）..."
python data/scripts/convert_robospatial.py \
    --data_dir "$DATA_RAW/robospatial" \
    --output_dir "$DATA_OUT" \
    --target_size 60000

# Step 4: 合并
echo ""
echo "[Step 4/6] 合并训练数据..."
python data/scripts/merge_data.py --output_dir "$DATA_OUT"

# Gate check 2
TOTAL_COUNT=$(wc -l < "$DATA_OUT/train.jsonl")
echo ""
echo ">> Gate Check: 合并后 $TOTAL_COUNT 条"
if [ "$TOTAL_COUNT" -lt 50000 ]; then
    echo "WARNING: 总量 <50K，可能不够"
elif [ "$TOTAL_COUNT" -gt 120000 ]; then
    echo "WARNING: 总量 >120K，可能需要减少采样"
else
    echo "OK: 在目标范围 50K-120K 内"
fi

# Step 5: 提取 consistency pairs
echo ""
echo "[Step 5/6] 提取 frame consistency pairs..."
python data/scripts/build_consistency_pairs.py \
    --input "$DATA_OUT/train.jsonl" \
    --output "$DATA_OUT/consistency_pairs.jsonl"

# Gate check 3
PAIR_COUNT=$(wc -l < "$DATA_OUT/consistency_pairs.jsonl")
echo ""
echo ">> Gate Check: $PAIR_COUNT 个 consistency pairs"
if [ "$PAIR_COUNT" -lt 5000 ]; then
    echo "WARNING: pairs 不足 5K，consistency loss 效果可能有限"
else
    echo "OK: pairs 数量充足"
fi

# Step 6: 转换 LLaMA-Factory 格式
echo ""
echo "[Step 6/6] 转换 LLaMA-Factory 格式..."

# 带 frame token 版本（用于 frame/full 训练）
python data/scripts/convert_to_llamafactory.py \
    --input "$DATA_OUT/train.jsonl" \
    --output "$LF_OUT/reframe_train.json" \
    --use_frame_tokens

# 不带 frame token 版本（用于 baseline 训练）
python data/scripts/convert_to_llamafactory.py \
    --input "$DATA_OUT/train.jsonl" \
    --output "$LF_OUT/reframe_baseline.json" \
    --no_frame_tokens

# 打印完整统计
echo ""
echo "[统计] 完整数据统计..."
python data/scripts/stats.py --data_dir "$DATA_OUT"

echo ""
echo "============================================"
echo " 预处理完成！"
echo "============================================"
echo "文件列表:"
ls -lh "$DATA_OUT"/*.jsonl
ls -lh "$LF_OUT"/*.json
echo ""
echo "下一步: bash scripts/run_baseline_zeroshot.sh"
