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
# Workaround for cuDNN initialization issues on some CUDA 12.x setups
torch.backends.cudnn.enabled = False
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel


FRAME_TYPE_TO_TOKEN = {
    "camera": "<frame_camera>",
    "person": "<frame_person>",
    "object": "<frame_object>",
    "world": "<frame_world>",
}
FRAME_SPECIAL_TOKENS = list(FRAME_TYPE_TO_TOKEN.values())
FRAME_TYPE_TO_ID = {
    "camera": 0,
    "person": 1,
    "object": 2,
    "world": 3,
}


def _load_processor_with_fallback(primary_path, fallback_path=None):
    """Load processor from primary path first, then fallback path."""
    paths = [p for p in (primary_path, fallback_path) if p]
    last_error = None
    for path in paths:
        try:
            return AutoProcessor.from_pretrained(path)
        except Exception as e:
            last_error = e
    raise RuntimeError(
        f"Failed to load processor from {paths}. Last error: {last_error}"
    )


def _ensure_frame_tokens(processor, model):
    """
    Ensure special frame tokens exist in tokenizer and model embedding table.
    Returns number of newly added tokens.
    """
    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()
    missing = [tok for tok in FRAME_SPECIAL_TOKENS if tok not in vocab]
    if not missing:
        return 0
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": missing}
    )
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    return num_added


def load_model(model_path, base_model_path=None, use_frame_tokens=False):
    """
    Load model for evaluation.
    Three cases:
    1. LoRA adapter checkpoint (has adapter_config.json) - load base + adapter
    2. Full ReFrameVLM checkpoint (has model.safetensors with double-wrapped state) - reconstruct
    3. Full HF model - load directly
    """
    adapter_config = os.path.join(model_path, "adapter_config.json")
    full_safetensors = os.path.join(model_path, "model.safetensors")

    # Detect ReFrameVLM full save (has double-wrapped state dict)
    is_reframe_full_save = False
    if os.path.exists(full_safetensors) and not os.path.exists(adapter_config):
        from safetensors import safe_open
        with safe_open(full_safetensors, framework="pt") as f:
            first_keys = list(f.keys())[:5]
        is_reframe_full_save = any("base_model.base_model" in k for k in first_keys)

    if is_reframe_full_save:
        if base_model_path is None:
            base_model_path = "models/qwen25-vl-7b"
        print(f"Detected ReFrameVLM full save. Reconstructing from {model_path}")
        # Build fresh ReFrameVLM with frame tokens (so tokenizer gets resized)
        from src.model.reframe_model import ReFrameVLM, get_default_lora_config
        wrapper = ReFrameVLM(
            model_path=base_model_path,
            lora_config=get_default_lora_config(),
            use_frame_tokens=True,
            use_relation_head=False,
        )
        # Load saved state into wrapper
        from safetensors.torch import load_file
        state_dict = load_file(full_safetensors)
        # The keys in state_dict have "base_model." prefix matching wrapper.base_model.*
        # Strip the wrapper-level "base_model." since we'll load into wrapper.base_model
        stripped = {}
        for k, v in state_dict.items():
            if k.startswith("base_model."):
                stripped[k[len("base_model."):]] = v
            else:
                stripped[k] = v
        missing, unexpected = wrapper.base_model.load_state_dict(stripped, strict=False)
        print(f"Loaded state. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing examples: {missing[:3]}")
        if unexpected:
            print(f"  Unexpected examples: {unexpected[:3]}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wrapper = wrapper.to(device)
        # Merge LoRA for inference speed
        wrapper.base_model = wrapper.base_model.merge_and_unload()
        model = wrapper.base_model
        processor = wrapper.processor
        if use_frame_tokens:
            added = _ensure_frame_tokens(processor, model)
            if added:
                print(f"Added {added} missing frame tokens to tokenizer")
        model.eval()
        return model, processor

    if os.path.exists(adapter_config):
        if base_model_path is None:
            with open(adapter_config) as f:
                cfg = json.load(f)
            base_model_path = cfg.get("base_model_name_or_path", model_path)
        processor = _load_processor_with_fallback(model_path, base_model_path)
        print(f"Loading base model from {base_model_path}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        if use_frame_tokens:
            added = _ensure_frame_tokens(processor, model)
            if added:
                print(f"Added {added} missing frame tokens to tokenizer")
        elif len(processor.tokenizer) != model.get_input_embeddings().num_embeddings:
            model.resize_token_embeddings(len(processor.tokenizer))

        print(f"Loading LoRA adapter from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        gates_path = os.path.join(model_path, "frame_lora_gates.pt")
        if os.path.exists(gates_path):
            from src.model.frame_lora import (
                load_frame_gates,
                patch_lora_with_frame_gating,
            )
            patched = patch_lora_with_frame_gating(
                model,
                num_frames=len(FRAME_TYPE_TO_ID),
                dtype=torch.bfloat16,
            )
            loaded = load_frame_gates(model, model_path)
            model._reframe_uses_frame_gated_lora = True
            print(
                f"Loaded Frame-Gated LoRA gates: "
                f"patched={len(patched)}, loaded={loaded}"
            )
        else:
            model = model.merge_and_unload()
    else:
        print(f"Loading model from {model_path}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = _load_processor_with_fallback(model_path, base_model_path)

    if use_frame_tokens and not os.path.exists(adapter_config):
        added = _ensure_frame_tokens(processor, model)
        if added:
            print(f"Added {added} missing frame tokens to tokenizer")

    model.eval()
    return model, processor


MAX_IMAGES = 4  # Limit images per sample to control inference speed


def run_inference(model, processor, images, question, choices=None,
                  frame_type=None, use_frame_prompt=False,
                  use_frame_token=False, max_new_tokens=64):
    """
    Run single-sample inference.

    Args:
        images: list of image paths
        question: question string
        choices: optional list of answer choices
        frame_type: optional frame type for prompt baseline / frame token
        use_frame_prompt: whether to add text frame prompt (baseline 2)
        use_frame_token: whether to prepend learned <frame_*> special token (frame/full method)
    """
    # Limit number of images to avoid slow multi-image inference
    if len(images) > MAX_IMAGES:
        # Evenly sample to preserve coverage
        step = len(images) / MAX_IMAGES
        images = [images[int(i * step)] for i in range(MAX_IMAGES)]

    content = []
    for img_path in images:
        content.append({"type": "image", "image": f"file://{img_path}"})

    # Build question text
    q_parts = []

    # Frame special token (for frame-conditioned models)
    if use_frame_token and frame_type:
        token = FRAME_TYPE_TO_TOKEN.get(frame_type)
        if token:
            q_parts.append(token)

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

    # Avoid appending options twice: some benchmarks (MMSI, Ego3D) already
    # embed "Options: ..." in the question text itself.
    already_has_options = "Options:" in question or "options:" in question
    if choices and not already_has_options:
        opts = ", ".join(str(c) for c in choices)
        q_parts.append(f"\nOptions: {opts}\nAnswer with the correct option only.")
    elif choices and already_has_options:
        q_parts.append("\nAnswer with the correct option letter only.")
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

    if getattr(model, "_reframe_uses_frame_gated_lora", False):
        from src.model.frame_lora import set_frame_type_ids_for_lora
        if frame_type is None:
            frame_type_ids = None
        else:
            frame_type_ids = torch.tensor(
                [FRAME_TYPE_TO_ID.get(frame_type, 0)],
                dtype=torch.long,
                device=inputs.input_ids.device,
            )
        set_frame_type_ids_for_lora(model, frame_type_ids)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


import re as _re


CHOICE_RE = _re.compile(r"^\s*\(?([A-Da-d])[\).:]?\s*(.*)$")


def _extract_first_number(s: str):
    """Extract the first signed decimal number from a string, or None."""
    m = _re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def _first_letter(s: str):
    """Return uppercase A-D if the string starts with a choice letter."""
    s = s.strip()
    if not s:
        return None
    m = _re.match(r"^\(?([A-Da-d])\b", s)
    return m.group(1).upper() if m else None


def _normalize_choice_text(text: str):
    text = str(text or "").strip()
    m = CHOICE_RE.match(text)
    if m and m.group(2):
        text = m.group(2)
    text = text.lower().replace("-", " ")
    text = _re.sub(r"[^a-z0-9. ]+", " ", text)
    return _re.sub(r"\s+", " ", text).strip()


def _choice_text(choices, letter):
    if not choices or not letter:
        return None
    idx = ord(letter.upper()) - ord("A")
    if idx < 0 or idx >= len(choices):
        return None
    return _normalize_choice_text(choices[idx])


def _matches_choice_content(pred, choice_text):
    pred_text = _normalize_choice_text(pred)
    if not pred_text or not choice_text:
        return False
    return (
        pred_text == choice_text
        or choice_text in pred_text
        or (len(pred_text) >= 3 and pred_text in choice_text)
    )


def match_answer(pred, gt, choices=None):
    """
    Match prediction against ground truth.

    Three cases:
      1. Single-letter GT (MMSI / Ego3D MC): compare leading letter of pred
         against GT letter.
      2. Numeric GT (Ego3D distance): extract first number, accept within
         10% relative error or 1.0 absolute.
      3. Textual GT: loose substring / first-word match.
    """
    if pred is None:
        return False
    pred_s = pred.strip()
    gt_s = str(gt).strip()
    pred_l = pred_s.lower()
    gt_l = gt_s.lower()

    # 1. Letter-only GT
    if _re.fullmatch(r"[A-Da-d]", gt_s):
        gt_letter = gt_s.upper()
        return (
            _first_letter(pred_s) == gt_letter
            or _matches_choice_content(pred_s, _choice_text(choices, gt_letter))
        )

    # 2. Numeric GT
    if _re.fullmatch(r"-?\d+(?:\.\d+)?", gt_s):
        gt_val = float(gt_s)
        pred_val = _extract_first_number(pred_s)
        if pred_val is None:
            return False
        abs_err = abs(pred_val - gt_val)
        rel_err = abs_err / max(abs(gt_val), 1e-6)
        return abs_err <= 1.0 or rel_err <= 0.10

    pred_rel = _normalize_choice_text(pred_s)
    gt_rel = _normalize_choice_text(gt_s)

    # 3. Exact / substring textual match
    if pred_l == gt_l or gt_l in pred_l or (gt_rel and (pred_rel == gt_rel or gt_rel in pred_rel)):
        return True

    if choices:
        pred_first = pred_rel.split()[0] if pred_rel else ""
        gt_first = gt_rel.split()[0] if gt_rel else ""
        if pred_first == gt_first and gt_first:
            return True
        for i, choice in enumerate(choices):
            letter = chr(ord("A") + i)
            choice_text = _choice_text(choices, letter)
            if gt_l == str(choice).lower() or gt_rel == choice_text:
                if (
                    _first_letter(pred_s) == letter
                    or _matches_choice_content(pred_s, choice_text)
                ):
                    return True

    return False


BENCHMARK_DEFAULT_PATHS = {
    "viewspatial": "data/processed/viewspatial_test.jsonl",
    "mmsi":        "data/processed/mmsi_test.jsonl",
    "ego3d":       "data/processed/ego3d_test.jsonl",
}


def load_benchmark_data(benchmark_name, data_dir=None):
    """Load a unified-format JSONL benchmark file."""
    if data_dir:
        candidates = [data_dir]
    else:
        default = BENCHMARK_DEFAULT_PATHS.get(benchmark_name)
        candidates = [default] if default else []

    for path in candidates:
        if path and os.path.exists(path) and path.endswith(".jsonl"):
            samples = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
            print(f"Loaded {len(samples)} samples from {path}")
            return samples

    raise FileNotFoundError(
        f"Could not find benchmark data for '{benchmark_name}'. "
        f"Tried: {candidates}. "
        f"Run data/scripts/convert_{benchmark_name}_bench.py first."
    )


def evaluate(model, processor, benchmark_data, benchmark_name,
             use_frame_prompt=False, use_frame_token=False, max_new_tokens=64):
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
            use_frame_token=use_frame_token,
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
    parser.add_argument("--use_frame_token", action="store_true",
                        help="Prepend learned <frame_*> special token (frame/full method)")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    model, processor = load_model(
        args.model_path,
        args.base_model_path,
        use_frame_tokens=args.use_frame_token,
    )

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
            use_frame_token=args.use_frame_token,
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

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
