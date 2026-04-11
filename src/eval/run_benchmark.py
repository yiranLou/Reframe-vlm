"""
Unified evaluation entry point for all benchmarks.

Supports:
1. ViewSpatial-Bench
2. MMSI-Bench
3. Ego3D-Bench

Usage:
    python src/eval/run_benchmark.py \
        --model_path checkpoints/full \
        --benchmark viewspatial \
        --output results/full_viewspatial.json

    python src/eval/run_benchmark.py \
        --model_path models/qwen25-vl-7b \
        --benchmark all \
        --output_dir results/zeroshot/
"""

import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel


def load_model(model_path, base_model_path=None):
    """
    Load model for evaluation.
    If model_path is a LoRA checkpoint, loads base + adapter.
    If model_path is a full model, loads directly.
    """
    # Check if it's a LoRA checkpoint
    adapter_config = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config):
        if base_model_path is None:
            # Try to read base model from adapter config
            with open(adapter_config) as f:
                cfg = json.load(f)
            base_model_path = cfg.get("base_model_name_or_path", model_path)

        print(f"Loading base model from {base_model_path}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        print(f"Loading LoRA adapter from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        processor = AutoProcessor.from_pretrained(base_model_path)
    else:
        print(f"Loading model from {model_path}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(model_path)

    model.eval()
    return model, processor


def run_inference(model, processor, images, question, choices=None,
                  frame_type=None, use_frame_prompt=False, max_new_tokens=64):
    """
    Run single-sample inference.

    Args:
        images: list of image paths
        question: question string
        choices: optional list of answer choices
        frame_type: optional frame type for prompt baseline
        use_frame_prompt: whether to add text frame prompt (baseline 2)
    """
    content = []
    for img_path in images:
        content.append({"type": "image", "image": f"file://{img_path}"})

    # Build question text
    q_parts = []

    if use_frame_prompt and frame_type:
        FRAME_PROMPTS = {
            "camera": "Answer the following spatial question from the camera's perspective (i.e., left/right refers to the viewer's left/right as seen in the image).",
            "person": "Answer the following spatial question from the person's perspective in the scene (i.e., left/right refers to the person's own left/right, which may be opposite to the viewer's).",
            "object": "Answer the following spatial question relative to the specified reference object's orientation.",
            "world": "Answer the following spatial question using absolute/world coordinates.",
        }
        prompt = FRAME_PROMPTS.get(frame_type, "")
        if prompt:
            q_parts.append(prompt)

    q_parts.append(question)

    if choices:
        opts = ", ".join(str(c) for c in choices)
        q_parts.append(f"\nOptions: {opts}\nAnswer with the correct option only.")
    else:
        q_parts.append("\nAnswer concisely.")

    q_text = "\n".join(q_parts)
    content.append({"type": "text", "text": q_text})

    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


def match_answer(pred, gt, choices=None):
    """
    Match prediction against ground truth.
    Handles multi-choice and open-ended answers.
    """
    pred_lower = pred.lower().strip()
    gt_lower = gt.lower().strip()

    # Exact match
    if pred_lower == gt_lower:
        return True

    # Check if prediction contains the ground truth
    if gt_lower in pred_lower:
        return True

    # For multi-choice, check if prediction starts with the answer
    if choices:
        # Try matching first word/letter
        pred_first = pred_lower.split()[0] if pred_lower else ""
        gt_first = gt_lower.split()[0] if gt_lower else ""
        if pred_first == gt_first:
            return True

        # Check option letter matching (A, B, C, D)
        for i, choice in enumerate(choices):
            letter = chr(ord('A') + i)
            if gt_lower == choice.lower():
                if pred_lower.startswith(letter.lower()) or letter.lower() in pred_lower[:3]:
                    return True

    return False


def load_benchmark_data(benchmark_name, data_dir=None):
    """
    Load benchmark data. Tries multiple locations.
    """
    if data_dir:
        candidates = [data_dir]
    else:
        candidates = [
            f"data/processed/viewspatial_test.jsonl",
            f"data/raw/viewspatial/test",
            f"data/raw/mmsi-bench",
            f"data/raw/ego3d-bench",
        ]

    # Try jsonl format first
    for path in candidates:
        if os.path.exists(path) and path.endswith(".jsonl"):
            samples = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
            print(f"Loaded {len(samples)} samples from {path}")
            return samples

    # Try json format
    for path in candidates:
        json_path = path if path.endswith(".json") else path + ".json"
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                print(f"Loaded {len(data)} samples from {json_path}")
                return data

    raise FileNotFoundError(
        f"Could not find benchmark data for '{benchmark_name}'. "
        f"Tried: {candidates}"
    )


def evaluate(model, processor, benchmark_data, benchmark_name,
             use_frame_prompt=False, max_new_tokens=64):
    """Run evaluation on a benchmark."""
    correct = 0
    total = 0
    results = []

    for sample in tqdm(benchmark_data, desc=f"Evaluating {benchmark_name}"):
        pred = run_inference(
            model, processor,
            images=sample["images"],
            question=sample["question"],
            choices=sample.get("choices"),
            frame_type=sample.get("frame_type"),
            use_frame_prompt=use_frame_prompt,
            max_new_tokens=max_new_tokens,
        )

        gt = sample["answer"]
        is_correct = match_answer(pred, gt, sample.get("choices"))

        correct += int(is_correct)
        total += 1

        results.append({
            "id": sample.get("id", f"sample_{total}"),
            "pred": pred,
            "gt": gt,
            "correct": is_correct,
            "frame_type": sample.get("frame_type", "unknown"),
        })

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n[{benchmark_name}] Accuracy: {accuracy:.2f}% ({correct}/{total})")

    return {
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ReFrame-VLM")
    parser.add_argument("--model_path", required=True,
                        help="Model or checkpoint path")
    parser.add_argument("--base_model_path", default=None,
                        help="Base model path (if model_path is LoRA adapter)")
    parser.add_argument("--benchmark", required=True,
                        choices=["viewspatial", "mmsi", "ego3d", "all"],
                        help="Benchmark to evaluate on")
    parser.add_argument("--data_dir", default=None,
                        help="Custom benchmark data path")
    parser.add_argument("--output", default=None,
                        help="Output results file")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (for --benchmark all)")
    parser.add_argument("--use_frame_prompt", action="store_true",
                        help="Use text frame prompt (baseline 2)")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    model, processor = load_model(args.model_path, args.base_model_path)

    benchmarks = (
        ["viewspatial", "mmsi", "ego3d"]
        if args.benchmark == "all"
        else [args.benchmark]
    )

    for bench_name in benchmarks:
        bench_data = load_benchmark_data(bench_name, args.data_dir)

        result = evaluate(
            model, processor, bench_data, bench_name,
            use_frame_prompt=args.use_frame_prompt,
            max_new_tokens=args.max_new_tokens,
        )

        # Save results
        if args.output:
            output_path = args.output
        elif args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, f"{bench_name}.json")
        else:
            output_path = f"results/{bench_name}.json"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
