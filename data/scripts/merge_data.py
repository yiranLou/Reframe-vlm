"""
Merge ViewSpatial + RoboSpatial into unified train.jsonl.

Usage:
    python data/scripts/merge_data.py --output_dir data/processed
"""

import argparse
import json
import os
from collections import defaultdict


def merge_train_data(processed_dir):
    """Merge all training data sources."""
    all_data = []
    sources = [
        "viewspatial_train.jsonl",
        "robospatial_train.jsonl",
    ]

    for fname in sources:
        path = os.path.join(processed_dir, fname)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping.")
            continue
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_data.append(json.loads(line))
                    count += 1
        print(f"Loaded {fname}: {count} samples")

    # Write merged file
    output_path = os.path.join(processed_dir, "train.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Stats
    frame_counts = defaultdict(int)
    source_counts = defaultdict(int)
    pair_count = 0
    for item in all_data:
        frame_counts[item["frame_type"]] += 1
        source_counts[item["source"]] += 1
        if item.get("pair_id"):
            pair_count += 1

    print(f"\n=== Merged Train Data ===")
    print(f"Total: {len(all_data)}")
    print(f"Sources: {dict(source_counts)}")
    print(f"Frame types: {dict(frame_counts)}")
    print(f"Samples with pair_id: {pair_count}")
    print(f"Output: {output_path}")

    # Validation checks
    camera_ratio = frame_counts.get("camera", 0) / len(all_data) if all_data else 0
    if camera_ratio > 0.5:
        print(f"\nWARNING: camera frame ratio is {camera_ratio:.1%}, "
              f"target is <= 50%. Consider rebalancing.")

    if len(all_data) < 50000:
        print(f"\nWARNING: Total only {len(all_data)}, target is 50K-120K.")
    elif len(all_data) > 120000:
        print(f"\nWARNING: Total is {len(all_data)}, target is 50K-120K. "
              f"Consider reducing sampling.")

    return all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()

    merge_train_data(args.output_dir)
