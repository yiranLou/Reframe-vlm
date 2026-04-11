"""
Main training entry point for ReFrame-VLM.

Supports multiple training modes:
1. baseline:    Standard LoRA fine-tune (no frame tokens, no consistency)
2. frame:       LoRA + frame special tokens
3. full:        LoRA + frame tokens + consistency loss
4. ablation_*:  Various ablation configurations

Usage:
    python src/training/train.py --config configs/train_full.yaml
    python src/training/train.py --mode baseline --model_path models/qwen25-vl-7b
"""

import argparse
import os
import yaml
import torch
from transformers import AutoProcessor, TrainingArguments
from peft import LoraConfig, TaskType

from src.model.reframe_model import (
    ReFrameVLM,
    add_frame_tokens_to_tokenizer,
    get_default_lora_config,
    FRAME_SPECIAL_TOKENS,
)
from src.training.dataset import ReFrameDataset
from src.training.collator import ReFrameCollator
from src.training.trainer import ReFrameTrainer, BaselineTrainer


def load_config(config_path):
    """Load YAML config."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_training_args(config):
    """Build HuggingFace TrainingArguments from config."""
    return TrainingArguments(
        output_dir=config.get("output_dir", "checkpoints/default"),
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation", 32),
        learning_rate=config.get("learning_rate", 2e-5),
        lr_scheduler_type=config.get("scheduler", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.05),
        weight_decay=config.get("weight_decay", 0.01),
        bf16=config.get("bf16", True),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        logging_steps=config.get("logging_steps", 10),
        save_strategy=config.get("save_strategy", "epoch"),
        save_total_limit=config.get("save_total_limit", 2),
        report_to=config.get("report_to", "wandb"),
        run_name=config.get("run_name", "reframe"),
        dataloader_num_workers=config.get("num_workers", 4),
        remove_unused_columns=False,
        seed=config.get("seed", 42),
        eval_strategy=config.get("eval_strategy", "no"),
    )


def train_baseline(config):
    """
    Baseline LoRA fine-tune: no frame tokens, no consistency loss.
    Produces baseline 3 numbers.
    """
    print("=== Training Mode: Baseline LoRA ===")

    model_path = config["model_path"]
    processor = AutoProcessor.from_pretrained(model_path)

    # Build model with standard LoRA, no frame tokens
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    lora_config = get_default_lora_config(
        rank=config.get("lora_rank", 64),
        alpha=config.get("lora_alpha", 128),
        dropout=config.get("lora_dropout", 0.05),
    )
    from peft import get_peft_model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset (no pairs, no view permutation)
    dataset = ReFrameDataset(
        data_path=config["train_data"],
        mode="qa",
        view_permutation=False,
    )

    collator = ReFrameCollator(
        processor=processor,
        max_length=config.get("max_length", 2048),
        use_frame_tokens=False,
    )

    training_args = build_training_args(config)

    trainer = BaselineTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model()
    print(f"Baseline model saved to {config.get('output_dir')}")


def train_frame(config):
    """
    LoRA + frame special tokens, no consistency loss.
    Tests frame-conditioned adaptation in isolation.
    """
    print("=== Training Mode: Frame-Conditioned LoRA ===")

    model_path = config["model_path"]

    # Build model with frame tokens
    model = ReFrameVLM(
        model_path=model_path,
        lora_config=get_default_lora_config(
            rank=config.get("lora_rank", 64),
            alpha=config.get("lora_alpha", 128),
            dropout=config.get("lora_dropout", 0.05),
        ),
        use_frame_tokens=True,
        use_relation_head=False,  # No consistency loss yet
    )

    dataset = ReFrameDataset(
        data_path=config["train_data"],
        mode="qa",
        view_permutation=config.get("view_permutation", False),
        view_permutation_prob=config.get("view_permutation_prob", 0.5),
    )

    collator = ReFrameCollator(
        processor=model.processor,
        max_length=config.get("max_length", 2048),
        use_frame_tokens=True,
    )

    training_args = build_training_args(config)

    trainer = BaselineTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model()
    print(f"Frame-conditioned model saved to {config.get('output_dir')}")


def train_full(config):
    """
    Full method: LoRA + frame tokens + consistency loss + view permutation.
    """
    print("=== Training Mode: Full ReFrame-VLM ===")

    model_path = config["model_path"]

    model = ReFrameVLM(
        model_path=model_path,
        lora_config=get_default_lora_config(
            rank=config.get("lora_rank", 64),
            alpha=config.get("lora_alpha", 128),
            dropout=config.get("lora_dropout", 0.05),
        ),
        use_frame_tokens=True,
        use_relation_head=True,
        canonical_dim=config.get("canonical_dim", 64),
    )

    # Dataset with consistency pairs and view permutation
    dataset = ReFrameDataset(
        data_path=config["train_data"],
        consistency_pairs_path=config.get("consistency_pairs"),
        mode="both",
        view_permutation=config.get("view_permutation", True),
        view_permutation_prob=config.get("view_permutation_prob", 0.5),
    )

    collator = ReFrameCollator(
        processor=model.processor,
        max_length=config.get("max_length", 2048),
        use_frame_tokens=True,
    )

    training_args = build_training_args(config)

    trainer = ReFrameTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        lambda_consistency=config.get("lambda_consistency", 0.1),
    )

    trainer.train()
    trainer.save_model()
    model.save_auxiliary_modules(config.get("output_dir", "checkpoints/full"))
    print(f"Full model saved to {config.get('output_dir')}")


TRAIN_MODES = {
    "baseline": train_baseline,
    "frame": train_frame,
    "full": train_full,
}


def main():
    parser = argparse.ArgumentParser(description="Train ReFrame-VLM")
    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--mode", type=str, default=None,
                        choices=list(TRAIN_MODES.keys()),
                        help="Training mode (overrides config)")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lambda_consistency", type=float, default=None)
    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = {}

    # CLI overrides
    if args.mode:
        config["mode"] = args.mode
    if args.model_path:
        config["model_path"] = args.model_path
    if args.train_data:
        config["train_data"] = args.train_data
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.lambda_consistency is not None:
        config["lambda_consistency"] = args.lambda_consistency

    # Defaults
    config.setdefault("mode", "full")
    config.setdefault("model_path", "models/qwen25-vl-7b")
    config.setdefault("train_data", "data/processed/train.jsonl")
    config.setdefault("output_dir", f"checkpoints/{config['mode']}")

    mode = config["mode"]
    print(f"\nConfig: {config}\n")

    if mode not in TRAIN_MODES:
        raise ValueError(f"Unknown mode: {mode}. Choose from {list(TRAIN_MODES.keys())}")

    TRAIN_MODES[mode](config)


if __name__ == "__main__":
    main()
