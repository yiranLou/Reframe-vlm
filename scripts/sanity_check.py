"""
完整性检查脚本。在服务器上开跑之前，先运行这个确保：
1. 所有代码都能 import
2. 数据文件存在且格式正确
3. 模型能加载
4. 前向传播能跑通（用 dummy 数据）

Usage:
    python scripts/sanity_check.py [--model_path models/qwen25-vl-7b] [--skip_model]
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CHECKS_PASSED = 0
CHECKS_FAILED = 0


def check(name, condition, detail=""):
    global CHECKS_PASSED, CHECKS_FAILED
    if condition:
        print(f"  [PASS] {name}")
        CHECKS_PASSED += 1
    else:
        print(f"  [FAIL] {name}: {detail}")
        CHECKS_FAILED += 1


def check_imports():
    """检查所有模块是否能导入。"""
    print("\n=== 1. Import 检查 ===")

    try:
        import torch
        check("PyTorch", True)
        check(f"CUDA 可用", torch.cuda.is_available(),
              "没有 GPU 也能开发，但训练需要")
    except Exception as e:
        check("PyTorch", False, str(e))

    try:
        import transformers
        check("Transformers", True)
    except Exception as e:
        check("Transformers", False, str(e))

    try:
        import peft
        check("PEFT", True)
    except Exception as e:
        check("PEFT", False, str(e))

    try:
        from src.model.frame_embedding import FrameEmbedding, FrameTokenModule
        check("frame_embedding", True)
    except Exception as e:
        check("frame_embedding", False, str(e))

    try:
        from src.model.frame_lora import (
            FrameGateEmbedding,
            patch_lora_with_frame_gating,
            set_frame_type_ids_for_lora,
        )
        check("frame_lora", True)
    except Exception as e:
        check("frame_lora", False, str(e))

    try:
        from src.model.relation_head import RelationHead, FrameCanonicalProjection
        check("relation_head", True)
    except Exception as e:
        check("relation_head", False, str(e))

    try:
        from src.model.reframe_model import ReFrameVLM, FRAME_SPECIAL_TOKENS
        check("reframe_model", True)
    except Exception as e:
        check("reframe_model", False, str(e))

    try:
        from src.training.dataset import ReFrameDataset
        check("dataset", True)
    except Exception as e:
        check("dataset", False, str(e))

    try:
        from src.training.losses import ReFrameLoss
        check("losses", True)
    except Exception as e:
        check("losses", False, str(e))

    try:
        from src.training.trainer import ReFrameTrainer
        check("trainer", True)
    except Exception as e:
        check("trainer", False, str(e))


def check_data():
    """检查数据文件是否存在、格式正确。"""
    print("\n=== 2. 数据文件检查 ===")

    data_files = {
        "训练数据": "data/processed/train.jsonl",
        "ViewSpatial 训练": "data/processed/viewspatial_train.jsonl",
        "ViewSpatial 测试": "data/processed/viewspatial_test.jsonl",
        "Consistency pairs": "data/processed/consistency_pairs.jsonl",
    }

    for name, path in data_files.items():
        if os.path.exists(path):
            # 检查格式
            try:
                count = 0
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line)
                            count += 1
                            if count == 1:
                                # 验证必要字段
                                if "consistency" in path:
                                    required = ["pair_id", "sample_a", "sample_b"]
                                else:
                                    required = ["id", "question", "answer", "frame_type"]
                                missing = [k for k in required if k not in sample]
                                check(f"{name} 格式", not missing,
                                      f"缺少字段: {missing}")
                check(f"{name} ({count} 条)", True)
            except Exception as e:
                check(f"{name}", False, f"解析错误: {e}")
        else:
            check(f"{name}", False, f"文件不存在: {path}")

    # LLaMA-Factory 格式
    lf_files = [
        "data/llamafactory/reframe_train.json",
        "data/llamafactory/dataset_info.json",
    ]
    for path in lf_files:
        check(f"LLaMA-Factory: {os.path.basename(path)}",
              os.path.exists(path),
              f"不存在: {path}")


def check_model_components():
    """检查模型组件能否正常初始化（用 dummy tensor）。"""
    print("\n=== 3. 模型组件检查 ===")

    try:
        import torch
        from src.model.frame_embedding import FrameEmbedding, FrameTokenModule
        from src.model.relation_head import RelationHead, FrameCanonicalProjection
        from src.training.losses import ReFrameLoss

        hidden_dim = 64  # 小维度用于测试

        # Frame Embedding
        fe = FrameEmbedding(hidden_dim)
        ids = torch.tensor([0, 1, 2, 3])
        out = fe(ids)
        check("FrameEmbedding forward", out.shape == (4, hidden_dim))

        # Frame Token Module
        ftm = FrameTokenModule(hidden_dim, tokens_per_frame=4)
        out = ftm(ids)
        check("FrameTokenModule forward", out.shape == (4, 4, hidden_dim))

        # Relation Head
        rh = RelationHead(hidden_dim)
        hs = torch.randn(4, hidden_dim)
        out = rh(hs)
        check("RelationHead forward", out.shape == (4, 14))

        # Canonical Projection
        cp = FrameCanonicalProjection()
        out = cp(torch.randn(4, 14), ids)
        check("FrameCanonicalProjection forward", out.shape == (4, 64))

        # Loss
        loss_fn = ReFrameLoss(lambda_consistency=0.1)
        qa_loss = torch.tensor(1.0, requires_grad=True)
        r_logits = torch.randn(6, 14, requires_grad=True)
        ft_ids = torch.tensor([0, 1, 0, 2, 1, 3])
        pairs = [(0, 1), (2, 3)]
        result = loss_fn(
            qa_loss=qa_loss,
            relation_logits=r_logits,
            pair_indices=pairs,
            frame_type_ids=ft_ids,
            canonical_proj=cp,
        )
        check("ReFrameLoss forward", "total_loss" in result)
        check("Consistency loss 有梯度",
              result["consistency_loss_raw"].requires_grad)

    except Exception as e:
        check("模型组件测试", False, str(e))


def check_full_model(model_path):
    """检查完整模型能否加载。"""
    print(f"\n=== 4. 完整模型检查 ({model_path}) ===")

    if not os.path.exists(model_path):
        check("模型路径", False, f"不存在: {model_path}")
        return

    try:
        from src.model.reframe_model import ReFrameVLM
        print("  正在加载模型（可能需要几分钟）...")
        model = ReFrameVLM(
            model_path=model_path,
            use_frame_tokens=True,
            use_relation_head=True,
        )
        check("模型加载", True)

        # 检查可训练参数
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        check(f"可训练参数: {trainable:,}", trainable > 0)

    except Exception as e:
        check("模型加载", False, str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/qwen25-vl-7b")
    parser.add_argument("--skip_model", action="store_true",
                        help="跳过完整模型加载（在没有 GPU 时有用）")
    args = parser.parse_args()

    print("=" * 50)
    print(" ReFrame-VLM Sanity Check")
    print("=" * 50)

    check_imports()
    check_data()
    check_model_components()

    if not args.skip_model:
        check_full_model(args.model_path)
    else:
        print("\n=== 4. 完整模型检查 (跳过) ===")

    # 汇总
    print("\n" + "=" * 50)
    total = CHECKS_PASSED + CHECKS_FAILED
    print(f" 结果: {CHECKS_PASSED}/{total} 通过, {CHECKS_FAILED} 失败")
    print("=" * 50)

    if CHECKS_FAILED > 0:
        print("\n有失败项，请修复后再开始实验。")
        sys.exit(1)
    else:
        print("\n全部通过！可以开始实验了。")
        sys.exit(0)


if __name__ == "__main__":
    main()
