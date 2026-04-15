"""
Smoke-test MMSI + Ego3D eval on 5 samples each with Qwen2.5-VL-7B zero-shot.

Validates:
  1. JSONL loading via BENCHMARK_DEFAULT_PATHS
  2. run_inference handles embedded Options, absolute paths
  3. match_answer correctly scores letter / numeric GT
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
torch.backends.cudnn.enabled = False

from src.eval.run_benchmark import (
    load_model, load_benchmark_data, run_inference, match_answer,
)


def smoke(model, processor, benchmark, n=5):
    data = load_benchmark_data(benchmark)[:n]
    print(f"\n[{benchmark}] smoke test on {len(data)} samples")
    correct = 0
    for s in data:
        pred = run_inference(
            model, processor,
            images=s["images"],
            question=s["question"],
            choices=s.get("choices"),
            frame_type=s.get("frame_type"),
            max_new_tokens=32,
        )
        ok = match_answer(pred, s["answer"], s.get("choices"))
        correct += int(ok)
        print(f"  [{s['id']}] gt={s['answer']!r}  pred={pred[:60]!r}  -> {ok}")
    print(f"[{benchmark}] smoke acc: {correct}/{len(data)}")


def main():
    model, processor = load_model("models/qwen25-vl-7b")
    smoke(model, processor, "mmsi", n=5)
    smoke(model, processor, "ego3d", n=5)


if __name__ == "__main__":
    main()
