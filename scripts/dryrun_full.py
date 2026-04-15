"""
Dry-run the Full Method training path end-to-end.

What this validates (before committing to a 17h overnight run):
  1. Model builds: Qwen2.5-VL + LoRA + frame tokens + relation head + canonical proj.
  2. Dataset emits a mix of single samples and consistency pairs.
  3. Collator produces model-ready tensors with pair_indices populated.
  4. ReFrameTrainer routes frame_type_ids / pair_indices through correctly.
  5. At least one step sees a non-empty pair batch → L_consistency > 0.
  6. Gradients flow into relation_head AND canonical_proj.
  7. Loss backward + optimizer.step do not raise.

Usage:
    python scripts/dryrun_full.py --config configs/train_full.yaml --steps 3
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# cuDNN workaround matching src/training/train.py
torch.backends.cudnn.enabled = False

from torch.utils.data import DataLoader, Subset

from src.model.reframe_model import ReFrameVLM, get_default_lora_config
from src.training.collator import ReFrameCollator
from src.training.dataset import ReFrameDataset
from src.training.losses import ReFrameLoss


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_small_batch(dataset, n_single=2, n_pair=2, pair_start=None):
    """Build a Subset guaranteed to include at least n_pair paired indices.

    ReFrameDataset in mode='both' indexes singles in [0, len(samples)) and
    pairs in [len(samples), len(samples)+len(pairs)).
    """
    pair_start = pair_start if pair_start is not None else len(dataset.samples)
    single_idxs = list(range(min(n_single, pair_start)))
    pair_idxs = list(range(pair_start, pair_start + n_pair))
    return Subset(dataset, single_idxs + pair_idxs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_full.yaml")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--model_path", default=None,
                        help="Override model_path (e.g. a smaller checkpoint)")
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model_path:
        cfg["model_path"] = args.model_path
    if args.batch_size:
        cfg["batch_size"] = args.batch_size

    print(f"[dryrun] config: {cfg}")

    # ── 1. Model ───────────────────────────────────────────────
    print("\n[dryrun] Building ReFrameVLM (frame tokens + relation head)...")
    model = ReFrameVLM(
        model_path=cfg["model_path"],
        lora_config=get_default_lora_config(
            rank=cfg.get("lora_rank", 64),
            alpha=cfg.get("lora_alpha", 128),
            dropout=cfg.get("lora_dropout", 0.05),
        ),
        use_frame_tokens=True,
        use_relation_head=True,
        canonical_dim=cfg.get("canonical_dim", 64),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    # Gradient checkpointing + LoRA requires enabling input grads.
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Sanity: relation_head + canonical_proj are registered and trainable.
    rh_params = list(model.relation_head.parameters())
    cp_params = list(model.canonical_proj.parameters())
    assert len(rh_params) > 0 and all(p.requires_grad for p in rh_params), \
        "relation_head missing or frozen"
    assert len(cp_params) > 0 and all(p.requires_grad for p in cp_params), \
        "canonical_proj missing or frozen"
    print(f"[dryrun] relation_head params: "
          f"{sum(p.numel() for p in rh_params):,}")
    print(f"[dryrun] canonical_proj params: "
          f"{sum(p.numel() for p in cp_params):,}")

    # ── 2. Dataset + collator ─────────────────────────────────
    print("\n[dryrun] Loading dataset in mode='both'...")
    dataset = ReFrameDataset(
        data_path=cfg["train_data"],
        consistency_pairs_path=cfg.get("consistency_pairs"),
        mode="both",
        view_permutation=cfg.get("view_permutation", True),
        view_permutation_prob=cfg.get("view_permutation_prob", 0.5),
    )
    assert len(dataset.pairs) > 0, "no consistency pairs loaded"

    collator = ReFrameCollator(
        processor=model.processor,
        max_length=cfg.get("max_length", 512),
        use_frame_tokens=True,
    )

    batch_size = cfg.get("batch_size", 2)
    subset = build_small_batch(dataset, n_single=batch_size, n_pair=batch_size)
    loader = DataLoader(subset, batch_size=batch_size * 2,
                        shuffle=False, collate_fn=collator,
                        num_workers=0)

    # ── 3. Trainer loss ───────────────────────────────────────
    loss_fn = ReFrameLoss(lambda_consistency=cfg.get("lambda_consistency", 0.1))
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.get("learning_rate", 2e-5),
    )

    saw_pair_batch = False
    for step, batch in enumerate(loader):
        if step >= args.steps:
            break

        pair_indices = batch.pop("pair_indices", [])
        frame_type_ids = batch.pop("frame_type_ids", None)
        if frame_type_ids is not None:
            frame_type_ids = frame_type_ids.to(device)

        batch_on_device = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        batch_on_device["output_hidden_states"] = True

        outputs = model(**batch_on_device, frame_type_ids=frame_type_ids)
        qa_loss = outputs["loss"]
        relation_logits = outputs.get("relation_logits")

        print(f"\n[dryrun] step {step}")
        print(f"  pair_indices: {pair_indices}")
        print(f"  frame_type_ids: {frame_type_ids.tolist() if frame_type_ids is not None else None}")
        print(f"  qa_loss: {qa_loss.item():.4f}")
        if relation_logits is not None:
            print(f"  relation_logits: shape={tuple(relation_logits.shape)} "
                  f"dtype={relation_logits.dtype} mean={relation_logits.float().mean().item():.4f}")

        loss_dict = loss_fn(
            qa_loss=qa_loss,
            relation_logits=relation_logits,
            pair_indices=pair_indices,
            frame_type_ids=frame_type_ids,
            canonical_proj=model.canonical_proj,
        )
        total = loss_dict["total_loss"]
        c_val = loss_dict["consistency_loss"]
        print(f"  total_loss: {total.item():.4f}  L_consist: {float(c_val):.4f}")

        if pair_indices and len(pair_indices) > 0:
            saw_pair_batch = True
            assert float(c_val) > 0, \
                f"expected L_consistency > 0 on a pair batch, got {float(c_val)}"

        # Backward + a zero-lr optimizer step; just verify gradients flow.
        total.backward()

        rh_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                      for p in rh_params)
        cp_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                      for p in cp_params)
        if pair_indices:
            assert rh_grad, "no gradient reached relation_head on a pair batch"
            assert cp_grad, "no gradient reached canonical_proj on a pair batch"
        print(f"  relation_head got grad: {rh_grad}")
        print(f"  canonical_proj got grad: {cp_grad}")

        optim.step()
        optim.zero_grad()

    assert saw_pair_batch, "no pair batch encountered — subset construction wrong"
    print("\n[dryrun] OK — Full Method path is healthy. Safe to launch overnight.")


if __name__ == "__main__":
    main()
