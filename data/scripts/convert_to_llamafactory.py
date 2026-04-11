"""
Convert unified format to LLaMA-Factory's sharegpt format.

LLaMA-Factory expects:
{
    "messages": [
        {"role": "user", "content": "<image>...<image>\nquestion text"},
        {"role": "assistant", "content": "answer"}
    ],
    "images": ["path1.jpg", "path2.jpg"]
}

Usage:
    python data/scripts/convert_to_llamafactory.py \
        --input data/processed/train.jsonl \
        --output data/llamafactory/reframe_train.json \
        --use_frame_tokens
"""

import argparse
import json
import os

from src.model.reframe_model import FRAME_TYPE_TO_TOKEN


def convert_sample(sample, use_frame_tokens=True):
    """Convert a single sample to LLaMA-Factory sharegpt format."""
    images = sample["images"]

    # Image tags: one <image> per image
    image_tags = "".join(["<image>" for _ in images])

    # Build user content
    parts = [image_tags]

    # Frame token (as text, will be a special token after tokenizer extension)
    if use_frame_tokens:
        frame_type = sample.get("frame_type", "camera")
        frame_token = FRAME_TYPE_TO_TOKEN.get(frame_type, "<frame_camera>")
        parts.append(frame_token)

    # Question
    parts.append(sample["question"])

    # Choices
    if sample.get("choices"):
        opts = ", ".join(str(c) for c in sample["choices"])
        parts.append(f"\nOptions: {opts}\nAnswer:")
    else:
        parts.append("\nAnswer:")

    user_content = "\n".join(parts)

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": sample["answer"]},
        ],
        "images": images,
    }


def convert_file(input_path, output_path, use_frame_tokens=True):
    """Convert a jsonl file to LLaMA-Factory format."""
    converted = []
    skipped = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            try:
                item = convert_sample(sample, use_frame_tokens)
                converted.append(item)
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"Warning: Skipped sample {sample.get('id')}: {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted)} samples to {output_path}")
    if skipped:
        print(f"Skipped {skipped} samples")

    return len(converted)


def create_dataset_info(output_dir, dataset_name="reframe_train",
                        train_file="reframe_train.json"):
    """
    Create LLaMA-Factory dataset_info.json.
    This file tells LLaMA-Factory where to find your dataset.
    """
    dataset_info = {
        dataset_name: {
            "file_name": train_file,
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }

    info_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Created {info_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/train.jsonl")
    parser.add_argument("--output", default="data/llamafactory/reframe_train.json")
    parser.add_argument("--use_frame_tokens", action="store_true", default=True)
    parser.add_argument("--no_frame_tokens", action="store_true",
                        help="Disable frame tokens (for baseline)")
    args = parser.parse_args()

    use_frame_tokens = args.use_frame_tokens and not args.no_frame_tokens

    print(f"Converting to LLaMA-Factory format")
    print(f"Frame tokens: {use_frame_tokens}")

    convert_file(args.input, args.output, use_frame_tokens)

    output_dir = os.path.dirname(args.output)
    create_dataset_info(output_dir)


if __name__ == "__main__":
    main()
