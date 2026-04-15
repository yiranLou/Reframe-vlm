"""
Scene leakage audit — does training data overlap with ViewSpatial-Bench test scenes?

Outputs scene-level and image-level overlap statistics so we know whether to
frame the paper as "scene-adaptive training" or "generalization to new scenes".

Extracts scene id from image paths; ScanNet paths look like:
    /.../scannetv2_val/scene0011_00/original_images/280.jpg
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


SCENE_RE = re.compile(r"/(scene\d{4}_\d{2})/")


def scene_id(path: str):
    m = SCENE_RE.search(path)
    return m.group(1) if m else None


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def summarize(samples, label):
    scenes = set()
    images = set()
    for s in samples:
        imgs = s.get("images") or []
        for p in imgs:
            images.add(p)
            sid = scene_id(p)
            if sid:
                scenes.add(sid)
    return {"label": label, "n_samples": len(samples),
            "n_scenes": len(scenes), "n_images": len(images),
            "scenes": scenes, "images": images}


def main():
    train = load_jsonl("data/processed/train.jsonl")
    test  = load_jsonl("data/processed/viewspatial_test.jsonl")
    pairs = load_jsonl("data/processed/consistency_pairs.jsonl")

    t_stat = summarize(train, "train")
    te_stat = summarize(test, "viewspatial_test")

    print(f"\n=== Scene leakage audit ===")
    for s in (t_stat, te_stat):
        print(f"  {s['label']}: {s['n_samples']} samples, "
              f"{s['n_scenes']} unique scenes, {s['n_images']} unique images")

    shared_scenes = t_stat["scenes"] & te_stat["scenes"]
    shared_images = t_stat["images"] & te_stat["images"]
    union_scenes = t_stat["scenes"] | te_stat["scenes"]

    print(f"\n  scene overlap:  {len(shared_scenes)} shared / "
          f"{len(te_stat['scenes'])} test scenes "
          f"({len(shared_scenes) / max(len(te_stat['scenes']), 1) * 100:.1f}% of test)")
    print(f"  image overlap:  {len(shared_images)} shared / "
          f"{len(te_stat['images'])} test images "
          f"({len(shared_images) / max(len(te_stat['images']), 1) * 100:.1f}% of test)")

    # By source
    by_source = defaultdict(list)
    for s in train:
        by_source[s.get("source", "?")].append(s)
    print(f"\n  train sources:")
    for src, items in by_source.items():
        src_stat = summarize(items, src)
        src_scenes = src_stat["scenes"]
        sc_ov = src_scenes & te_stat["scenes"]
        print(f"    {src}: {src_stat['n_samples']} samples / "
              f"{src_stat['n_scenes']} scenes / "
              f"{len(sc_ov)} overlap w/ test")

    # Frame-level: does training set contain actual test questions verbatim?
    test_qs = {(s["question"], tuple(s.get("images") or [])) for s in test}
    q_overlap = sum(
        1 for s in train
        if (s["question"], tuple(s.get("images") or [])) in test_qs
    )
    print(f"\n  question-image-pair overlap: {q_overlap} "
          f"(same (question, images) in both train and test)")

    # Verdict
    print()
    frac_scenes = len(shared_scenes) / max(len(te_stat["scenes"]), 1)
    if frac_scenes > 0.5:
        verdict = ("HIGH OVERLAP — paper must be framed as scene-adaptive "
                   "training on held-out frames/views, not unseen-scene generalization.")
    elif frac_scenes > 0.1:
        verdict = ("PARTIAL OVERLAP — need to clearly label both conditions or "
                   "redo a scene-disjoint split.")
    else:
        verdict = ("LOW OVERLAP — scene-level generalization claim is defensible.")
    print(f"  VERDICT: {verdict}")

    # Write machine-readable audit for the paper
    out = {
        "train": {k: v for k, v in t_stat.items() if k not in ("scenes", "images")},
        "test":  {k: v for k, v in te_stat.items() if k not in ("scenes", "images")},
        "overlap": {
            "scenes": len(shared_scenes),
            "images": len(shared_images),
            "question_image_pairs": q_overlap,
            "test_scene_overlap_fraction": frac_scenes,
        },
        "shared_scenes_sample": sorted(shared_scenes)[:20],
        "verdict": verdict,
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/scene_leakage_audit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  wrote results/scene_leakage_audit.json")


if __name__ == "__main__":
    main()
