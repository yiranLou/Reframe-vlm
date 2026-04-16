"""
Counterfactual inference-time controls for the LoRA + text-instruction SFT
checkpoint.

Counterpart to ``frame_token_controls.py`` but the conditioning signal is the
natural-language frame instruction (FRAME_TEXT_PROMPTS in
src/training/collator.py mirroring src/eval/run_benchmark.py FRAME_PROMPTS).

Five conditions:
  correct       — instruction matches the sample's frame_type (default eval).
  wrong         — instruction is swapped (camera ↔ person) — fall back to
                  person for object/world to keep symmetric.
  none          — no instruction prepended.
  always_camera — always use the camera-perspective instruction.
  always_person — always use the person-perspective instruction.

Together with the Frame LoRA wrong-frame controls, this answers the paper-
critical question: does each conditioning channel act as a *test-time
control* (matters which signal is given at inference) or merely as a
*training-time signal* (model bakes in frame behaviour and ignores the
runtime conditioning)?

Usage::

    python scripts/text_instruction_controls.py \
        --ckpt checkpoints/text_instruction_lora/checkpoint-716 \
        --conditions wrong,none,always_camera,always_person \
        --out_dir results/text_instr_controls
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.run_benchmark import (
    load_model, load_benchmark_data, run_inference, match_answer,
)


FLIP_PERSPECTIVE = {
    "camera": "person",
    "person": "camera",
    "object": "person",
    "world":  "person",
}


def override_frame_for_text_prompt(sample, condition):
    """Returns (frame_type_str, use_frame_prompt)."""
    ft = sample.get("frame_type", "camera")
    if condition == "correct":
        return ft, True
    if condition == "wrong":
        return FLIP_PERSPECTIVE.get(ft, ft), True
    if condition == "always_camera":
        return "camera", True
    if condition == "always_person":
        return "person", True
    if condition == "none":
        return None, False
    raise ValueError(condition)


def evaluate_condition(model, processor, samples, condition, out_path):
    correct = 0
    total = 0
    rows = []
    for s in tqdm(samples, desc=f"[{condition}]"):
        ft_use, inject = override_frame_for_text_prompt(s, condition)
        pred = run_inference(
            model, processor,
            images=s["images"],
            question=s["question"],
            choices=s.get("choices"),
            frame_type=ft_use,
            use_frame_prompt=inject,   # <-- text instruction path
            use_frame_token=False,
            max_new_tokens=32,
        )
        ok = match_answer(pred, s["answer"], s.get("choices"))
        correct += int(ok)
        total += 1
        rows.append({
            "id": s.get("id", f"sample_{total}"),
            "pred": pred,
            "gt": s["answer"],
            "correct": ok,
            "frame_type": s.get("frame_type"),
            "frame_type_used": ft_use,
            "condition": condition,
        })
    acc = correct / total * 100 if total else 0
    result = {
        "benchmark": "viewspatial",
        "condition": condition,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "results": rows,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[{condition}] acc = {acc:.2f}%  ({correct}/{total})  -> {out_path}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="checkpoints/text_instruction_lora/checkpoint-716")
    ap.add_argument("--conditions",
                    default="wrong,none,always_camera,always_person")
    ap.add_argument("--out_dir", default="results/text_instr_controls")
    ap.add_argument("--bench_path",
                    default="data/processed/viewspatial_test.jsonl")
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    print(f"Running conditions: {conds} on {args.ckpt}")

    model, processor = load_model(args.ckpt)
    samples = load_benchmark_data("viewspatial", args.bench_path)

    summary = {}
    for cond in conds:
        out = os.path.join(args.out_dir, f"viewspatial_{cond}.json")
        if os.path.exists(out):
            print(f"[skip] {out} exists")
            with open(out) as f:
                d = json.load(f)
            summary[cond] = d["accuracy"]
            continue
        summary[cond] = evaluate_condition(model, processor, samples, cond, out)

    print("\n=== Summary ===")
    for c, a in summary.items():
        print(f"  {c}: {a:.2f}%")


if __name__ == "__main__":
    main()
