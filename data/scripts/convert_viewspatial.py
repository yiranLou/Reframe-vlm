"""
Convert ViewSpatial raw data to unified training format.

Key tasks:
1. Extract frame_type from perspective annotations
2. Build pair_id for same-scene cross-frame samples
3. Output unified jsonl

Usage:
    python data/scripts/convert_viewspatial.py \
        --data_dir data/raw/viewspatial \
        --output_dir data/processed \
        --split train
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def load_viewspatial_raw(data_dir, split):
    """
    Load raw ViewSpatial data.
    Tries multiple possible file structures since the exact format
    may vary across dataset versions.
    """
    candidates = [
        os.path.join(data_dir, split, "annotations.json"),
        os.path.join(data_dir, f"{split}.json"),
        os.path.join(data_dir, f"{split}_annotations.json"),
        os.path.join(data_dir, "annotations", f"{split}.json"),
    ]

    # Also try jsonl
    jsonl_candidates = [
        os.path.join(data_dir, split, "data.jsonl"),
        os.path.join(data_dir, f"{split}.jsonl"),
    ]

    for path in candidates:
        if os.path.exists(path):
            print(f"Loading from {path}")
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            if isinstance(data, list):
                return data
            return list(data.values())

    for path in jsonl_candidates:
        if os.path.exists(path):
            print(f"Loading from {path}")
            data = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data

    # Try loading all json files in the split directory
    split_dir = os.path.join(data_dir, split)
    if os.path.isdir(split_dir):
        data = []
        for fname in sorted(os.listdir(split_dir)):
            if fname.endswith(".json"):
                with open(os.path.join(split_dir, fname), encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        data.extend(content)
                    else:
                        data.append(content)
        if data:
            print(f"Loaded {len(data)} samples from {split_dir}/*.json")
            return data

    raise FileNotFoundError(
        f"Could not find ViewSpatial data. Tried:\n"
        + "\n".join(f"  - {p}" for p in candidates + jsonl_candidates)
        + f"\n  - {os.path.join(data_dir, split)}/*.json"
        + "\n\nPlease check your data directory structure."
    )


def extract_frame_type(sample):
    """
    Extract frame_type from ViewSpatial sample.
    ViewSpatial distinguishes camera vs human perspective.
    Field names may vary - try common possibilities.
    """
    for field in ["perspective", "viewpoint", "frame", "frame_type",
                  "spatial_type", "reference_frame", "view_type"]:
        val = sample.get(field, "")
        if not val:
            continue
        val = str(val).lower()
        if "camera" in val or "viewer" in val or "egocentric" in val:
            return "camera"
        if "human" in val or "person" in val or "actor" in val:
            return "person"
        if "object" in val:
            return "object"
        if "world" in val or "global" in val or "absolute" in val:
            return "world"

    # Check question text for clues
    question = sample.get("question", "").lower()
    if "from the camera" in question or "as seen in the image" in question:
        return "camera"
    if "from the person" in question or "person's perspective" in question:
        return "person"

    return "camera"  # default


def extract_scene_id(sample):
    """Extract scene identifier for pairing."""
    for field in ["scene_id", "scene", "scene_name", "video_id", "image_set_id"]:
        val = sample.get(field)
        if val:
            return str(val)

    # Infer from image paths: common prefix
    images = sample.get("images", sample.get("image", []))
    if isinstance(images, str):
        images = [images]
    if images:
        basename = os.path.basename(images[0])
        # Remove camera/view suffix to get scene id
        # e.g., scene001_cam0.jpg -> scene001
        parts = basename.rsplit("_", 1)
        if len(parts) > 1:
            return parts[0]
        return os.path.splitext(basename)[0]

    return None


def extract_images(sample, data_dir):
    """Extract image paths, ensuring they are absolute."""
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
            # Try relative to data_dir
            full = os.path.join(data_dir, img)
            if os.path.exists(full):
                resolved.append(full)
            else:
                # Try under images/ subdirectory
                full2 = os.path.join(data_dir, "images", img)
                if os.path.exists(full2):
                    resolved.append(full2)
                else:
                    resolved.append(full)  # keep as-is, will be caught later

    return resolved


def extract_choices(sample):
    """Extract answer choices if present."""
    for field in ["choices", "options", "candidates"]:
        val = sample.get(field)
        if val and isinstance(val, list):
            return val
    return None


def build_pair_index(converted_samples):
    """
    Build cross-frame pair index.
    Pairs samples from same scene with different frame_types.
    """
    # Group by scene_id
    scene_groups = defaultdict(list)
    for s in converted_samples:
        sid = s.get("_scene_id")
        if sid:
            scene_groups[sid].append(s)

    pair_map = {}

    for scene_id, group in scene_groups.items():
        # Group by frame_type within scene
        by_frame = defaultdict(list)
        for s in group:
            by_frame[s["frame_type"]].append(s)

        # Pair camera <-> person (primary)
        cam = by_frame.get("camera", [])
        person = by_frame.get("person", [])
        for i in range(min(len(cam), len(person))):
            pair_map[cam[i]["id"]] = person[i]["id"]
            pair_map[person[i]["id"]] = cam[i]["id"]

        # Pair camera <-> object
        obj = by_frame.get("object", [])
        remaining_cam = cam[len(person):]
        for i in range(min(len(remaining_cam), len(obj))):
            pair_map[remaining_cam[i]["id"]] = obj[i]["id"]
            pair_map[obj[i]["id"]] = remaining_cam[i]["id"]

        # Pair person <-> object (leftover)
        remaining_person = person[len(cam):]
        remaining_obj = obj[len(remaining_cam) if remaining_cam else 0:]
        for i in range(min(len(remaining_person), len(remaining_obj))):
            pair_map[remaining_person[i]["id"]] = remaining_obj[i]["id"]
            pair_map[remaining_obj[i]["id"]] = remaining_person[i]["id"]

    return pair_map


def convert_viewspatial(data_dir, output_dir, split="train"):
    print(f"\n=== Converting ViewSpatial {split} ===")
    raw_data = load_viewspatial_raw(data_dir, split)
    print(f"Raw samples loaded: {len(raw_data)}")

    converted = []
    skipped = 0

    for i, sample in enumerate(raw_data):
        images = extract_images(sample, data_dir)
        question = sample.get("question", sample.get("text", ""))
        answer = sample.get("answer", sample.get("response", ""))

        if not question or not answer:
            skipped += 1
            continue

        item = {
            "id": f"viewspatial_{split}_{i:06d}",
            "source": "viewspatial",
            "images": images,
            "num_views": len(images),
            "question": question,
            "answer_type": "multi_choice" if extract_choices(sample) else "open",
            "choices": extract_choices(sample),
            "answer": str(answer),
            "frame_type": extract_frame_type(sample),
            "pair_id": None,
            "relation_label": None,
            "split": split,
            # Internal field for pairing, removed before output
            "_scene_id": extract_scene_id(sample),
        }
        converted.append(item)

    if skipped:
        print(f"Skipped {skipped} samples (missing question/answer)")

    # Build pairs
    pair_map = build_pair_index(converted)
    for item in converted:
        item["pair_id"] = pair_map.get(item["id"])

    # Remove internal fields
    for item in converted:
        item.pop("_scene_id", None)

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"viewspatial_{split}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Stats
    frame_counts = defaultdict(int)
    pair_count = sum(1 for item in converted if item["pair_id"] is not None)
    for item in converted:
        frame_counts[item["frame_type"]] += 1

    print(f"Output: {output_path}")
    print(f"Total: {len(converted)}")
    print(f"Frame types: {dict(frame_counts)}")
    print(f"Paired samples: {pair_count} ({pair_count // 2} pairs)")

    return converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw/viewspatial")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--split", default="train", choices=["train", "test", "val"])
    args = parser.parse_args()

    convert_viewspatial(args.data_dir, args.output_dir, args.split)
