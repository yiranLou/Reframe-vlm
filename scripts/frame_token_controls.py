"""
Counterfactual frame-token inference controls.

Uses the Frame LoRA checkpoint (frame_lora/checkpoint-716) and re-evaluates
ViewSpatial under 4 conditions that differ only in the frame token prepended
to each question at inference time:

    correct       — original frame_type (baseline, already computed)
    wrong         — flip camera <-> person (dominant test frame types)
    none          — no frame token at all (same as use_frame_token=False)
    always_camera — always prepend <frame_camera>

If the model truly uses frame tokens, `wrong` should underperform `correct`.
If `none` matches `correct`, the token was decorative. If `always_camera`
approaches `correct` (close to 50% of test set is camera-perspective), the
model may just have learned to use any prepended token as a generic prefix.

Usage:
    python scripts/frame_token_controls.py \
        --ckpt checkpoints/frame_lora/checkpoint-716 \
        --conditions wrong,none,always_camera \
        --out_dir results/frame_controls

Takes ~1-2h per condition on A100 (5712 ViewSpatial samples).
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
    "object": "person",   # no object in ViewSpatial, but map if it appears
    "world":  "person",
}


def override_frame_type(sample, condition):
    ft = sample.get("frame_type", "camera")
    if condition == "correct":
        return ft, True
    if condition == "wrong":
        return FLIP_PERSPECTIVE.get(ft, ft), True
    if condition == "always_camera":
        return "camera", True
    if condition == "none":
        return None, False   # no token injected at all
    raise ValueError(condition)


def evaluate_condition(model, processor, samples, condition, out_path):
    correct = 0
    total = 0
    rows = []
    for s in tqdm(samples, desc=f"[{condition}]"):
        ft_use, inject = override_frame_type(s, condition)
        pred = run_inference(
            model, processor,
            images=s["images"],
            question=s["question"],
            choices=s.get("choices"),
            frame_type=ft_use,
            use_frame_prompt=False,
            use_frame_token=inject,
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
    ap.add_argument("--ckpt", default="checkpoints/frame_lora/checkpoint-716")
    ap.add_argument("--conditions", default="wrong,none,always_camera",
                    help="comma-separated subset of {correct,wrong,none,always_camera}")
    ap.add_argument("--out_dir", default="results/frame_controls")
    ap.add_argument("--bench_path", default="data/processed/viewspatial_test.jsonl")
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
            with open(out) as f: d = json.load(f)
            summary[cond] = d["accuracy"]
            continue
        summary[cond] = evaluate_condition(model, processor, samples, cond, out)

    print("\n=== Summary ===")
    for c, a in summary.items():
        print(f"  {c}: {a:.2f}%")


if __name__ == "__main__":
    main()
