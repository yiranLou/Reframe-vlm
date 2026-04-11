"""
自定义训练循环，不依赖 HF Trainer。

优点：
1. 对 consistency loss 有完全控制（什么时候算、怎么累积）
2. 两个 dataloader 交替使用（QA + consistency pairs）
3. loss annealing 更方便

当 HF Trainer 方案工作正常时，优先用 trainer.py。
这个文件作为备选，当需要更精细控制时使用。

Usage:
    python src/training/custom_train_loop.py --config configs/train_full.yaml
"""

import argparse
import os
import time
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoProcessor
from accelerate import Accelerator

from src.model.reframe_model import ReFrameVLM, get_default_lora_config
from src.training.dataset import ReFrameDataset
from src.training.collator import ReFrameCollator
from src.training.losses import ReFrameLoss


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(config):
    # ── 初始化 ──
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=config.get("gradient_accumulation", 32),
        log_with="wandb" if config.get("report_to") == "wandb" else None,
    )

    if accelerator.is_main_process:
        print(f"Config: {config}")

    # ── 模型 ──
    model = ReFrameVLM(
        model_path=config["model_path"],
        lora_config=get_default_lora_config(
            rank=config.get("lora_rank", 64),
            alpha=config.get("lora_alpha", 128),
            dropout=config.get("lora_dropout", 0.05),
        ),
        use_frame_tokens=config.get("use_frame_tokens", True),
        use_relation_head=True,
        canonical_dim=config.get("canonical_dim", 64),
    )

    # ── 数据 ──
    qa_dataset = ReFrameDataset(
        data_path=config["train_data"],
        mode="qa",
        view_permutation=config.get("view_permutation", True),
        view_permutation_prob=config.get("view_permutation_prob", 0.5),
    )

    consistency_dataset = ReFrameDataset(
        data_path=config["train_data"],
        consistency_pairs_path=config.get("consistency_pairs"),
        mode="consistency",
    )

    collator = ReFrameCollator(
        processor=model.processor,
        max_length=config.get("max_length", 2048),
        use_frame_tokens=config.get("use_frame_tokens", True),
    )

    qa_loader = DataLoader(
        qa_dataset,
        batch_size=config.get("batch_size", 2),
        shuffle=True,
        collate_fn=collator,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    # Consistency loader batch size 小一些（每个 pair 展开成 2 个样本）
    consist_loader = DataLoader(
        consistency_dataset,
        batch_size=max(1, config.get("batch_size", 2) // 2),
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    # ── 优化器 ──
    no_decay = ["bias", "LayerNorm.weight", "layer_norm"]
    optimizer_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.get("weight_decay", 0.01),
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_groups, lr=config.get("learning_rate", 2e-5))

    num_epochs = config.get("num_epochs", 3)
    grad_accum = config.get("gradient_accumulation", 32)
    total_steps = len(qa_loader) * num_epochs // grad_accum

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.get("warmup_ratio", 0.05)),
        num_training_steps=total_steps,
    )

    # ── Loss ──
    loss_fn = ReFrameLoss(
        lambda_consistency=config.get("lambda_consistency", 0.1),
    )

    # Lambda annealing: 前 1/3 训练只用 L_qa，之后线性增加 consistency
    use_annealing = config.get("consistency_annealing", False)

    # ── Accelerate prepare ──
    model, optimizer, qa_loader, consist_loader, scheduler = accelerator.prepare(
        model, optimizer, qa_loader, consist_loader, scheduler
    )

    # ── 训练循环 ──
    consist_iter = iter(consist_loader)
    consist_every = config.get("consistency_every", 3)  # 每 N 个 QA step 做一次 consistency
    global_step = 0
    output_dir = config.get("output_dir", "checkpoints/custom")

    for epoch in range(num_epochs):
        model.train()
        epoch_qa_loss = 0.0
        epoch_consist_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for step, qa_batch in enumerate(qa_loader):
            with accelerator.accumulate(model):
                # ── QA Forward ──
                qa_outputs = model(
                    input_ids=qa_batch["input_ids"],
                    attention_mask=qa_batch["attention_mask"],
                    pixel_values=qa_batch.get("pixel_values"),
                    image_grid_thw=qa_batch.get("image_grid_thw"),
                    labels=qa_batch["labels"],
                    frame_type_ids=qa_batch.get("frame_type_ids"),
                    output_hidden_states=False,
                )
                qa_loss = qa_outputs["loss"]

                # ── Consistency Forward（每 N 步做一次）──
                consist_loss_val = 0.0
                if step % consist_every == 0 and len(consistency_dataset) > 0:
                    try:
                        consist_batch = next(consist_iter)
                    except StopIteration:
                        consist_iter = iter(consist_loader)
                        consist_batch = next(consist_iter)

                    consist_outputs = model(
                        input_ids=consist_batch["input_ids"],
                        attention_mask=consist_batch["attention_mask"],
                        pixel_values=consist_batch.get("pixel_values"),
                        image_grid_thw=consist_batch.get("image_grid_thw"),
                        labels=consist_batch["labels"],
                        frame_type_ids=consist_batch.get("frame_type_ids"),
                        output_hidden_states=True,
                    )

                    losses = loss_fn(
                        qa_loss=consist_outputs["loss"],
                        relation_logits=consist_outputs.get("relation_logits"),
                        pair_indices=consist_batch.get("pair_indices", []),
                        frame_type_ids=consist_batch.get("frame_type_ids"),
                    )

                    # Annealing
                    if use_annealing:
                        progress = global_step / max(total_steps, 1)
                        if progress < 0.33:
                            lambda_scale = 0.0
                        else:
                            lambda_scale = (progress - 0.33) / 0.67
                        total_loss = qa_loss + lambda_scale * losses["consistency_loss"]
                    else:
                        total_loss = qa_loss + loss_fn.lambda_consistency * losses["consistency_loss"]

                    consist_loss_val = losses["consistency_loss"].item()
                else:
                    total_loss = qa_loss

                accelerator.backward(total_loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_qa_loss += qa_loss.item()
                epoch_consist_loss += consist_loss_val
                epoch_steps += 1
                global_step += 1

                # 日志
                if step % config.get("logging_steps", 10) == 0 and accelerator.is_main_process:
                    avg_qa = epoch_qa_loss / epoch_steps
                    avg_cs = epoch_consist_loss / max(1, epoch_steps // consist_every)
                    elapsed = time.time() - t0
                    print(
                        f"  Epoch {epoch+1}/{num_epochs} "
                        f"Step {step}/{len(qa_loader)} "
                        f"| qa_loss={avg_qa:.4f} "
                        f"| consist_loss={avg_cs:.4f} "
                        f"| lr={scheduler.get_last_lr()[0]:.2e} "
                        f"| {elapsed:.0f}s"
                    )

        # Epoch 结束
        if accelerator.is_main_process:
            avg_qa = epoch_qa_loss / epoch_steps
            avg_cs = epoch_consist_loss / max(1, epoch_steps // consist_every)
            print(
                f"\nEpoch {epoch+1} 完成: "
                f"avg_qa_loss={avg_qa:.4f}, "
                f"avg_consist_loss={avg_cs:.4f}\n"
            )

            # 保存 checkpoint
            save_path = os.path.join(output_dir, f"epoch_{epoch+1}")
            accelerator.unwrap_model(model).base_model.save_pretrained(save_path)
            accelerator.unwrap_model(model).save_auxiliary_modules(save_path)
            print(f"Checkpoint 保存到 {save_path}")

    # 保存最终模型
    if accelerator.is_main_process:
        final_path = os.path.join(output_dir, "final")
        accelerator.unwrap_model(model).base_model.save_pretrained(final_path)
        accelerator.unwrap_model(model).save_auxiliary_modules(final_path)
        model.processor.save_pretrained(final_path)
        print(f"\n最终模型保存到 {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
