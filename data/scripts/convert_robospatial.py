"""
Convert RoboSpatial data to unified format with stratified sampling.

RoboSpatial has ~1M images and ~3M spatial relations.
We sample 50K-80K with balanced frame_type distribution.

Usage:
    python data/scripts/convert_robospatial.py \
        --data_dir data/raw/robospatial \
        --output_dir data/processed \
        --target_size 60000
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path


def load_robospatial_raw(data_dir):
    """
    Load RoboSpatial data. Tries multiple formats.
    RoboSpatial may have annotation types:
    - spatial context, spatial configuration, spatial compatibility
    """
    all_samples = []

    # Try single annotation file
    for fname in ["train.json", "data.json", "annotations.json",
                  "train.jsonl", "data.jsonl"]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            print(f"Loading from {path}")
            if fname.endswith(".jsonl"):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_samples.append(json.loads(line))
            else:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_samples.extend(data)
                    elif isinstance(data, dict):
                        if "data" in data:
                            all_samples.extend(data["data"])
                        else:
                            all_samples.extend(data.values())
            break

    # Try scanning subdirectories
    if not all_samples:
        for dirpath, dirnames, filenames in os.walk(data_dir):
            for fname in sorted(filenames):
                if not fname.endswith((".json", ".jsonl")):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    if fname.endswith(".jsonl"):
                        with open(fpath, encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    all_samples.append(json.loads(line))
                    else:
                        with open(fpath, encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                all_samples.extend(data)
                except Exception as e:
                    print(f"Warning: Failed to load {fpath}: {e}")

            if len(all_samples) > 10000:
                break  # enough for sampling

    if not all_samples:
        raise FileNotFoundError(
            f"Could not find RoboSpatial data in {data_dir}. "
            "Check data directory structure."
        )

    return all_samples


def infer_frame_type(sample):
    """
    Infer frame_type from RoboSpatial sample.
    Heuristic based on text content and annotation fields.
    """
    # Check explicit annotations
    for field in ["reference_frame", "frame_type", "perspective", "spatial_type"]:
        val = sample.get(field, "")
        if not val:
            continue
        val = str(val).lower()
        if "camera" in val or "robot" in val or "ego" in val:
            return "camera"
        if "object" in val:
            return "object"
        if "world" in val or "global" in val:
            return "world"

    # Heuristic from text
    text = " ".join(str(sample.get(k, "")) for k in
                    ["question", "description", "caption", "text"]).lower()

    if any(kw in text for kw in ["robot", "camera", "viewer", "your"]):
        return "camera"
    if any(kw in text for kw in ["relative to the", "from the perspective of",
                                  "with respect to"]):
        return "object"
    if any(kw in text for kw in ["world", "global", "absolute", "north",
                                  "south", "east", "west"]):
        return "world"

    return "camera"  # default


def extract_images(sample, data_dir):
    """Extract image paths."""
    images = sample.get("images", [])
    if not images:
        img = sample.get("image", "")
        if isinstance(img, str) and img:
            images = [img]
        elif isinstance(img, list):
            images = img

    if isinstance(images, str):
        images = [images]

    resolved = []
    for img in images:
        if os.path.isabs(img):
            resolved.append(img)
        else:
            full = os.path.join(data_dir, img)
            resolved.append(full)

    return resolved


def stratified_sample(samples, target_size, seed=42):
    """
    Stratified sampling by frame_type.
    Target distribution: camera 30%, object 40%, world 30%
    """
    random.seed(seed)

    # Assign frame types
    for s in samples:
        s["_frame_type"] = infer_frame_type(s)

    groups = defaultdict(list)
    for s in samples:
        groups[s["_frame_type"]].append(s)

    allocation = {
        "camera": int(target_size * 0.30),
        "object": int(target_size * 0.40),
        "world": int(target_size * 0.30),
        "person": 0,  # RoboSpatial unlikely to have person-frame
    }

    sampled = []
    shortfall = 0

    for frame_type, count in allocation.items():
        if count == 0:
            continue
        pool = groups.get(frame_type, [])
        if len(pool) >= count:
            sampled.extend(random.sample(pool, count))
        else:
            sampled.extend(pool)
            shortfall += count - len(pool)
            print(f"Warning: {frame_type} has {len(pool)} samples, "
                  f"needed {count} (shortfall: {count - len(pool)})")

    # Fill shortfall from largest pool
    if shortfall > 0:
        largest_type = max(groups.keys(), key=lambda k: len(groups[k]))
        remaining = [s for s in groups[largest_type] if s not in sampled]
        fill = min(shortfall, len(remaining))
        sampled.extend(random.sample(remaining, fill))
        print(f"Filled {fill} shortfall from {largest_type}")

    random.shuffle(sampled)
    return sampled


def convert_robospatial(data_dir, output_dir, target_size=60000):
    print(f"\n=== Converting RoboSpatial (target: {target_size}) ===")
    raw = load_robospatial_raw(data_dir)
    print(f"Raw samples loaded: {len(raw)}")

    sampled = stratified_sample(raw, target_size)
    print(f"After sampling: {len(sampled)}")

    converted = []
    for i, sample in enumerate(sampled):
        images = extract_images(sample, data_dir)
        question = sample.get("question", sample.get("text", ""))
        answer = sample.get("answer", sample.get("response", ""))

        if not question or not answer:
            continue

        choices = sample.get("choices", sample.get("options"))
        if choices and not isinstance(choices, list):
            choices = None

        item = {
            "id": f"robospatial_{i:06d}",
            "source": "robospatial",
            "images": images,
            "num_views": len(images),
            "question": question,
            "answer_type": "multi_choice" if choices else "open",
            "choices": choices,
            "answer": str(answer),
            "frame_type": sample["_frame_type"],
            "pair_id": None,  # RoboSpatial pairs handled separately if available
            "relation_label": None,
            "split": "train",
        }
        converted.append(item)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "robospatial_train.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    frame_counts = defaultdict(int)
    for item in converted:
        frame_counts[item["frame_type"]] += 1

    print(f"Output: {output_path}")
    print(f"Total: {len(converted)}")
    print(f"Frame types: {dict(frame_counts)}")

    return converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw/robospatial")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--target_size", type=int, default=60000)
    args = parser.parse_args()

    convert_robospatial(args.data_dir, args.output_dir, args.target_size)
