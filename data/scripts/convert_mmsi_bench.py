"""
Convert MMSI-Bench parquet to our unified JSONL eval format.

Input:  /workspace/datasets/mmsi-data/MMSI_Bench.parquet
Output: data/processed/mmsi_test.jsonl  (+ data/processed/mmsi_images/*.jpg)

MMSI fields:
  id, images (list of raw JPEG bytes), question_type, question, answer, thought,
  mean_normed_duration_seconds, difficulty

The question string already embeds "Options: A: ..., B: ..., C: ..., D: ...".
We keep the text as-is and set choices=["A","B","C","D"] so the matcher
accepts "A"/"A: Left...", etc. Answer is a letter.

frame_type heuristic (mapping question_type -> our 4-way taxonomy):
  Motion (Cam.)                    -> camera
  Motion (Obj.)                    -> object
  Positional Relationship (Cam.-*) -> camera
  Positional Relationship (Obj.-*) -> object
  Positional Relationship (Reg.-*) -> world
  Attribute (*)                    -> camera (fallback)
  MSR (Multi-Step Reasoning)       -> camera (fallback)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq


def infer_frame_type(question_type: str) -> str:
    t = question_type.lower()
    if "cam" in t:
        return "camera"
    if "obj" in t:
        return "object"
    if "reg" in t:  # Region
        return "world"
    return "camera"


def parse_choices(question_text: str):
    """Return list of full choice strings if present, else None.

    Expected format: '... Options: A: ..., B: ..., C: ..., D: ...'
    """
    m = re.search(r"Options:\s*(.*)", question_text, flags=re.DOTALL)
    if not m:
        return None
    body = m.group(1)
    # Split on ", A:" style delimiters but keep the letters.
    parts = re.split(r",\s*(?=[A-D]:)", body.strip())
    choices = []
    for p in parts:
        p = p.strip().rstrip(".")
        if re.match(r"^[A-D]:", p):
            choices.append(p)
    return choices or None


def convert(parquet_path: str, output_jsonl: str, images_dir: str):
    images_dir = os.path.abspath(images_dir)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    print(f"Loaded {len(rows)} rows from {parquet_path}")

    written = 0
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for row in rows:
            sid = str(row["id"])
            # Write images to disk
            img_paths = []
            for i, raw in enumerate(row["images"]):
                out = os.path.join(images_dir, f"mmsi_{sid}_{i}.jpg")
                if not os.path.exists(out):
                    with open(out, "wb") as img_f:
                        img_f.write(raw)
                img_paths.append(out)

            choices = parse_choices(row["question"])

            item = {
                "id": f"mmsi_{sid}",
                "source": "mmsi",
                "images": img_paths,
                "num_views": len(img_paths),
                "question": row["question"],
                "choices": choices,
                "answer": row["answer"],
                "frame_type": infer_frame_type(row["question_type"]),
                "question_type": row["question_type"],
                "difficulty": row.get("difficulty"),
                "pair_id": None,
                "split": "test",
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} samples to {output_jsonl}")
    print(f"Images in {images_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet",
                   default="/workspace/datasets/mmsi-data/MMSI_Bench.parquet")
    p.add_argument("--output", default="data/processed/mmsi_test.jsonl")
    p.add_argument("--images_dir", default="data/processed/mmsi_images")
    args = p.parse_args()
    convert(args.parquet, args.output, args.images_dir)


if __name__ == "__main__":
    main()
