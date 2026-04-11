"""
Print statistics for processed data files.

Usage:
    python data/scripts/stats.py --data_dir data/processed
"""

import argparse
import json
import os
from collections import defaultdict


def stats_for_jsonl(path):
    """Print detailed stats for a jsonl file."""
    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"{'='*60}")

    if not os.path.exists(path):
        print("  File not found!")
        return

    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Total samples: {len(samples)}")

    if not samples:
        return

    # Frame type distribution
    frame_counts = defaultdict(int)
    source_counts = defaultdict(int)
    answer_type_counts = defaultdict(int)
    pair_count = 0
    view_counts = defaultdict(int)
    missing_images = 0

    for s in samples:
        frame_counts[s.get("frame_type", "unknown")] += 1
        source_counts[s.get("source", "unknown")] += 1
        answer_type_counts[s.get("answer_type", "unknown")] += 1
        view_counts[s.get("num_views", 0)] += 1
        if s.get("pair_id"):
            pair_count += 1
        # Check image existence
        for img in s.get("images", []):
            if not os.path.exists(img):
                missing_images += 1

    print(f"\nFrame types:")
    for k, v in sorted(frame_counts.items()):
        pct = v / len(samples) * 100
        print(f"  {k}: {v} ({pct:.1f}%)")

    print(f"\nSources:")
    for k, v in sorted(source_counts.items()):
        print(f"  {k}: {v}")

    print(f"\nAnswer types:")
    for k, v in sorted(answer_type_counts.items()):
        print(f"  {k}: {v}")

    print(f"\nNum views distribution:")
    for k, v in sorted(view_counts.items()):
        print(f"  {k} views: {v}")

    print(f"\nPaired samples: {pair_count} ({pair_count // 2} pairs)")

    if missing_images > 0:
        print(f"\nWARNING: {missing_images} image paths not found on disk!")

    # Sample a few examples
    print(f"\nSample entries (first 2):")
    for s in samples[:2]:
        print(f"  id={s['id']}, frame={s.get('frame_type')}, "
              f"answer={s['answer'][:50]}, "
              f"images={len(s.get('images', []))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    args = parser.parse_args()

    files = [
        "viewspatial_train.jsonl",
        "viewspatial_test.jsonl",
        "robospatial_train.jsonl",
        "train.jsonl",
        "consistency_pairs.jsonl",
    ]

    for fname in files:
        path = os.path.join(args.data_dir, fname)
        if os.path.exists(path):
            stats_for_jsonl(path)

    # Special stats for consistency pairs
    pairs_path = os.path.join(args.data_dir, "consistency_pairs.jsonl")
    if os.path.exists(pairs_path):
        print(f"\n{'='*60}")
        print(f"Consistency Pairs Details")
        print(f"{'='*60}")
        pairs = []
        with open(pairs_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))

        type_counts = defaultdict(int)
        for p in pairs:
            key = f"{p.get('frame_a', '?')} <-> {p.get('frame_b', '?')}"
            type_counts[key] += 1

        print(f"Total pairs: {len(pairs)}")
        for k, v in sorted(type_counts.items()):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
