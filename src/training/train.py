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
import inspect
import os
import yaml
import torch
# Workaround for cuDNN initialization issues on some CUDA 12.x setups
torch.backends.cudnn.enabled = False
from transformers import AutoProcessor, TrainingArguments

from src.model.reframe_model import (
    ReFrameVLM,
    get_default_lora_config,
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
    eval_value = config.get(
        "eval_strategy", config.get("evaluation_strategy", "no")
    )
    kwargs = dict(
        output_dir=config.get("output_dir", "checkpoints/default"),
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation", 16),
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
    )
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = eval_value
    elif "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = eval_value
    if "max_steps" in config:
        kwargs["max_steps"] = config["max_steps"]
    return TrainingArguments(**kwargs)


def save_processor(processor, output_dir):
    """Persist processor/tokenizer so eval can reload frame tokens correctly."""
    os.makedirs(output_dir, exist_ok=True)
    processor.save_pretrained(output_dir)
    print(f"Processor/tokenizer saved to {output_dir}")


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
        use_frame_text_prompt=config.get("use_frame_text_prompt", False),
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
    output_dir = config.get("output_dir", "checkpoints/baseline")
    save_processor(processor, output_dir)
    print(f"Baseline model saved to {output_dir}")


def train_frame(config):
    """
    LoRA + frame special tokens, no consistency loss.
    Tests frame-conditioned adaptation in isolation.
    """
    print("=== Training Mode: Frame-Conditioned LoRA ===")

    model_path = config["model_path"]

    use_frame_tokens = config.get("use_frame_tokens", True)

    # Build model with optional frame tokens
    model = ReFrameVLM(
        model_path=model_path,
        lora_config=get_default_lora_config(
            rank=config.get("lora_rank", 64),
            alpha=config.get("lora_alpha", 128),
            dropout=config.get("lora_dropout", 0.05),
        ),
        use_frame_tokens=use_frame_tokens,
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
        use_frame_tokens=use_frame_tokens,
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
    output_dir = config.get("output_dir", "checkpoints/frame")
    model.save_auxiliary_modules(output_dir)
    save_processor(model.processor, output_dir)
    print(f"Frame-conditioned model saved to {output_dir}")


def train_frame_gated(config):
    """
    Frame-Gated LoRA only (no special tokens, no consistency loss).

    Plan B' clean comparison data point: parameter-space frame conditioning
    in isolation, so we can attribute any improvement specifically to the
    gate rather than to the input-side token or to a regulariser.
    """
    print("=== Training Mode: Frame-Gated LoRA (gate only) ===")
    model_path = config["model_path"]

    model = ReFrameVLM(
        model_path=model_path,
        lora_config=get_default_lora_config(
            rank=config.get("lora_rank", 64),
            alpha=config.get("lora_alpha", 128),
            dropout=config.get("lora_dropout", 0.05),
        ),
        use_frame_tokens=False,
        use_relation_head=False,
        use_frame_gated_lora=True,
    )

    dataset = ReFrameDataset(
        data_path=config["train_data"],
        mode="qa",
        view_permutation=False,
    )

    collator = ReFrameCollator(
        processor=model.processor,
        max_length=config.get("max_length", 512),
        use_frame_tokens=False,
        use_frame_text_prompt=False,
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
    output_dir = config.get("output_dir", "checkpoints/frame_gated")
    model.save_auxiliary_modules(output_dir)
    save_processor(model.processor, output_dir)
    print(f"Frame-gated model saved to {output_dir}")


def train_token_gated(config):
    """
    Frame-Gated LoRA + learned frame tokens (combined input + parameter
    conditioning). Run only if frame_gated_only is competitive — otherwise
    skip per Plan B'.
    """
    print("=== Training Mode: Token-Gated (frame token + gate) ===")
    model_path = config["model_path"]

    use_frame_tokens = config.get("use_frame_tokens", True)
    model = ReFrameVLM(
        model_path=model_path,
        lora_config=get_default_lora_config(
            rank=config.get("lora_rank", 64),
            alpha=config.get("lora_alpha", 128),
            dropout=config.get("lora_dropout", 0.05),
        ),
        use_frame_tokens=use_frame_tokens,
        use_relation_head=False,
        use_frame_gated_lora=True,
    )

    dataset = ReFrameDataset(
        data_path=config["train_data"],
        mode="qa",
        view_permutation=False,
    )

    collator = ReFrameCollator(
        processor=model.processor,
        max_length=config.get("max_length", 640),
        use_frame_tokens=use_frame_tokens,
        use_frame_text_prompt=False,
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
    output_dir = config.get("output_dir", "checkpoints/token_gated")
    model.save_auxiliary_modules(output_dir)
    save_processor(model.processor, output_dir)
    print(f"Token-gated model saved to {output_dir}")


def train_full(config):
    """
    Full method: LoRA + frame tokens + consistency loss + view permutation.
    """
    print("=== Training Mode: Full ReFrame-VLM ===")

    model_path = config["model_path"]

    use_frame_tokens = config.get("use_frame_tokens", True)
    model = ReFrameVLM(
        model_path=model_path,
        lora_config=get_default_lora_config(
            rank=config.get("lora_rank", 64),
            alpha=config.get("lora_alpha", 128),
            dropout=config.get("lora_dropout", 0.05),
        ),
        use_frame_tokens=use_frame_tokens,
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
        use_frame_tokens=use_frame_tokens,
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
    output_dir = config.get("output_dir", "checkpoints/full")
    model.save_auxiliary_modules(output_dir)
    save_processor(model.processor, output_dir)
    print(f"Full model saved to {output_dir}")


TRAIN_MODES = {
    "baseline": train_baseline,
    "frame": train_frame,
    "full": train_full,
    "frame_gated": train_frame_gated,
    "token_gated": train_token_gated,
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
