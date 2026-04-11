"""
Build frame-switch consistency pairs from training data.

Extracts all samples with pair_id and creates the consistency_pairs.jsonl
used for Frame Consistency Loss during training.

Usage:
    python data/scripts/build_consistency_pairs.py \
        --input data/processed/train.jsonl \
        --output data/processed/consistency_pairs.jsonl
"""

import argparse
import json
from collections import defaultdict


def load_data(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def extract_pairs(samples):
    """Extract all cross-frame pairs from samples with pair_id."""
    id_to_item = {s["id"]: s for s in samples}

    pairs = []
    seen = set()

    for item in samples:
        if not item.get("pair_id"):
            continue
        if item["id"] in seen:
            continue

        pair_item = id_to_item.get(item["pair_id"])
        if not pair_item:
            continue
        if pair_item["id"] in seen:
            continue

        # Ensure they have different frame types
        if item["frame_type"] == pair_item["frame_type"]:
            continue

        pairs.append({
            "pair_id": f"pair_{len(pairs):06d}",
            "sample_a": {
                "id": item["id"],
                "source": item["source"],
                "images": item["images"],
                "num_views": item["num_views"],
                "question": item["question"],
                "answer_type": item["answer_type"],
                "choices": item.get("choices"),
                "answer": item["answer"],
                "frame_type": item["frame_type"],
            },
            "sample_b": {
                "id": pair_item["id"],
                "source": pair_item["source"],
                "images": pair_item["images"],
                "num_views": pair_item["num_views"],
                "question": pair_item["question"],
                "answer_type": pair_item["answer_type"],
                "choices": pair_item.get("choices"),
                "answer": pair_item["answer"],
                "frame_type": pair_item["frame_type"],
            },
            "frame_a": item["frame_type"],
            "frame_b": pair_item["frame_type"],
        })

        seen.add(item["id"])
        seen.add(pair_item["id"])

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/train.jsonl")
    parser.add_argument("--output", default="data/processed/consistency_pairs.jsonl")
    args = parser.parse_args()

    print(f"=== Building Consistency Pairs ===")
    samples = load_data(args.input)
    print(f"Total samples: {len(samples)}")
    print(f"Samples with pair_id: {sum(1 for s in samples if s.get('pair_id'))}")

    pairs = extract_pairs(samples)

    with open(args.output, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Stats
    type_counts = defaultdict(int)
    for p in pairs:
        key = f"{p['frame_a']} <-> {p['frame_b']}"
        type_counts[key] += 1

    print(f"\nOutput: {args.output}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Pair types:")
    for k, v in sorted(type_counts.items()):
        print(f"  {k}: {v}")

    if len(pairs) < 5000:
        print(f"\nWARNING: Only {len(pairs)} pairs found. "
              f"Target is 5K-10K. Consider improving pair extraction logic.")


if __name__ == "__main__":
    main()
