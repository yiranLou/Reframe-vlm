"""
Convert Ego3D-Bench arrow dataset to our unified JSONL eval format.

Input:
  /workspace/datasets/ego3d-data/test/data-00000-of-00001.arrow
  /workspace/datasets/ego3d-data/raw_images/     (extracted jpegs)

Output:
  data/processed/ego3d_test.jsonl

Ego3D fields:
  idx, source ('nuscenes' / 'argoverse' / 'waymo'),
  category (e.g. 'Ego_Centric_Absolute_Distance', 'Object_Centric_Motion_Reasoning'),
  images (dict: view_name -> filename),
  question (with literal '<image>' placeholders per view),
  options (None for numeric, list of 'A. yes'-style strings for MC),
  answer (letter 'A'/'B'/...' for MC, or number-like string for numeric).

Mapping:
  Ego_Centric_*    -> frame_type=camera
  Object_Centric_* -> frame_type=object
  Localization     -> camera
  Travel_Time      -> camera
"""

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset


VIEW_ORDER = ["Front_Left", "Front", "Front_Right",
              "Back_Right", "Back", "Back_Left",
              "Side_Left", "Side_Right"]


def infer_frame_type(category: str) -> str:
    c = category.lower()
    if c.startswith("ego_centric"):
        return "camera"
    if c.startswith("object_centric"):
        return "object"
    return "camera"


def resolve_images(images_dict, images_root):
    """Return a list of image paths in deterministic view order."""
    if not images_dict:
        return []
    paths = []
    # Ordered views first
    for v in VIEW_ORDER:
        fn = images_dict.get(v)
        if fn:
            paths.append(os.path.join(images_root, fn))
    # Any remaining views not in our canonical ordering
    seen = {v for v in VIEW_ORDER if v in images_dict}
    for v, fn in images_dict.items():
        if v not in seen:
            paths.append(os.path.join(images_root, fn))
    return paths


def parse_choices(options):
    """options is None or a list of 'A. yes'/'B. no' strings. Return that list."""
    if not options:
        return None
    return list(options)


def convert(arrow_path: str, images_root: str, output_jsonl: str):
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    ds = Dataset.from_file(arrow_path)
    print(f"Loaded {len(ds)} rows from {arrow_path}")

    missing_count = 0
    written = 0
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for row in ds:
            paths = resolve_images(row["images"], images_root)
            # Sanity check: skip samples whose image files aren't on disk.
            existing = [p for p in paths if os.path.exists(p)]
            if not existing:
                missing_count += 1
                continue
            if len(existing) < len(paths):
                missing_count += len(paths) - len(existing)

            choices = parse_choices(row.get("options"))
            answer = str(row["answer"])
            is_numeric = choices is None

            item = {
                "id": f"ego3d_{row['idx']}",
                "source": f"ego3d_{row.get('source', 'unknown')}",
                "images": existing,
                "num_views": len(existing),
                "question": row["question"],
                "choices": choices,
                "answer": answer,
                "answer_type": "open" if is_numeric else "multi_choice",
                "frame_type": infer_frame_type(row["category"]),
                "category": row["category"],
                "pair_id": None,
                "split": "test",
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} samples to {output_jsonl}")
    if missing_count:
        print(f"Warning: {missing_count} image paths missing on disk")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arrow",
                   default="/workspace/datasets/ego3d-data/test/data-00000-of-00001.arrow")
    p.add_argument("--images_root",
                   default="/workspace/datasets/ego3d-data/raw_images")
    p.add_argument("--output", default="data/processed/ego3d_test.jsonl")
    args = p.parse_args()
    convert(args.arrow, args.images_root, args.output)


if __name__ == "__main__":
    main()
