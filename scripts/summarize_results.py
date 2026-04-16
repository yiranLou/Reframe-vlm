"""
Aggregate all eval results into paper-ready markdown tables.

Scans results/ for any subdirectory containing {viewspatial,mmsi,ego3d}.json
and emits:

  Table 1 — Main results across the 3 benchmarks (overall accuracy)
  Table 3 — ViewSpatial per-frame-type breakdown (Camera / Person / Overall)
  Table 4 — Frame-switch consistency (FCA / CR / FG) on ViewSpatial

The method labels come from directory names; rename dirs if you want
different row labels in the paper (e.g. 'full_method_ep1').

Usage:
    python scripts/summarize_results.py
    python scripts/summarize_results.py --out results/summary.md
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


VIEWSPATIAL_BENCH = "data/processed/viewspatial_test.jsonl"


# ── Ordering for paper tables ─────────────────────────────────
# Directories listed here render in this order; unlisted dirs append after.
PREFERRED_ORDER = [
    "zeroshot",
    "prompt_baseline",
    "baseline_lora_ep1",
    "text_instruction_lora_ep1",
    "frame_lora_ep1",
    "frame_gated_lora_ep1",
    "token_gated_lora_ep1",
    "full_method_ep1",
]
NICE_LABEL = {
    "zeroshot": "Qwen2.5-VL zero-shot",
    "prompt_baseline": "Qwen2.5-VL + text prompt (inference only)",
    "baseline_lora_ep1": "Naive LoRA (ep1)",
    "text_instruction_lora_ep1": "LoRA + text instruction SFT (ep1)",
    "frame_lora_ep1": "+ frame token (ep1)",
    "frame_gated_lora_ep1": "Frame-Gated LoRA (gate only, ep1)",
    "token_gated_lora_ep1": "Token + Gated LoRA (ep1)",
    "full_method_ep1": "Full (frame + consistency + perm, ep1)",
}


def _load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_if_exists(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _discover_runs(results_dir):
    """Find subdirs under results_dir; return list of (label, dir_path)."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    runs = []
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and any(
            (d / f"{b}.json").exists() for b in ("viewspatial", "mmsi", "ego3d")
        ):
            runs.append((d.name, str(d)))
    # Reorder: preferred first, then the rest alphabetically.
    rank = {name: i for i, name in enumerate(PREFERRED_ORDER)}
    runs.sort(key=lambda kv: (rank.get(kv[0], 10_000), kv[0]))
    return runs


# ── Table 1: main results across 3 benchmarks ─────────────────

def build_table_main(runs):
    lines = ["## Table 1 — Main results (overall accuracy %)",
             "",
             "| Method | ViewSpatial | MMSI | Ego3D |",
             "|---|---:|---:|---:|"]
    for name, path in runs:
        row = [NICE_LABEL.get(name, name)]
        for bench in ("viewspatial", "mmsi", "ego3d"):
            data = _load_if_exists(os.path.join(path, f"{bench}.json"))
            if data is None:
                row.append("—")
            else:
                row.append(f"{data['accuracy']:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ── Table 3: per-frame-type breakdown on ViewSpatial ─────────

def _per_frame_accuracy(results_json_path, bench_samples):
    """Return dict frame_type -> (correct, total). bench_samples is a list
    of unified-format dicts with 'id' and 'frame_type'."""
    data = _load_if_exists(results_json_path)
    if data is None:
        return None
    # Results may carry frame_type inline; fall back to benchmark file lookup.
    bench_map = {b["id"]: b for b in bench_samples}
    per = defaultdict(lambda: [0, 0])
    for r in data["results"]:
        ft = r.get("frame_type")
        if not ft or ft == "unknown":
            b = bench_map.get(r["id"])
            ft = b.get("frame_type", "unknown") if b else "unknown"
        per[ft][1] += 1
        per[ft][0] += int(r["correct"])
    return dict(per)


def build_table_frametype(runs, bench_samples):
    all_types = set()
    cache = {}
    for name, path in runs:
        p = _per_frame_accuracy(os.path.join(path, "viewspatial.json"),
                                bench_samples)
        cache[name] = p
        if p:
            all_types.update(p.keys())

    type_order = [t for t in ("camera", "person", "object", "world")
                  if t in all_types]

    lines = ["## Table 3 — ViewSpatial per-frame-type accuracy (%)",
             "",
             "| Method | " + " | ".join(t.capitalize() for t in type_order)
             + " | Overall |",
             "|---|" + "|".join(["---:"] * (len(type_order) + 1)) + "|"]

    for name, path in runs:
        p = cache[name]
        if not p:
            continue
        row = [NICE_LABEL.get(name, name)]
        total_c = total_n = 0
        for t in type_order:
            c, n = p.get(t, [0, 0])
            row.append(f"{(c/n*100 if n else 0):.2f}")
            total_c += c
            total_n += n
        row.append(f"{(total_c/total_n*100 if total_n else 0):.2f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ── Table 4: FCA / CR / FG on ViewSpatial pairs ──────────────

def _build_bench_pairs(bench_samples):
    """Extract same-scene cross-frame pairs from the benchmark file.

    Preference order:
      1. Explicit `pair_id` fields if present.
      2. Fallback: group samples sharing identical image paths, then
         pair any camera vs non-camera sample in the group.
    """
    by_id = {s["id"]: s for s in bench_samples}
    pairs = []
    seen = set()
    for s in bench_samples:
        pid = s.get("pair_id")
        if pid and s["id"] not in seen and pid in by_id:
            pairs.append((s, by_id[pid]))
            seen.add(s["id"])
            seen.add(pid)
    if pairs:
        return pairs

    # Image-level fallback grouping.
    from collections import defaultdict
    by_img = defaultdict(list)
    for s in bench_samples:
        by_img[tuple(s.get("images", []))].append(s)

    for group in by_img.values():
        fts = {g["frame_type"] for g in group}
        if len(fts) < 2:
            continue
        cams = [g for g in group if g["frame_type"] == "camera"
                and g["id"] not in seen]
        others = [g for g in group if g["frame_type"] != "camera"
                  and g["id"] not in seen]
        for a, b in zip(cams, others):
            pairs.append((a, b))
            seen.add(a["id"])
            seen.add(b["id"])
    return pairs


def build_table_consistency(runs, bench_samples):
    pairs = _build_bench_pairs(bench_samples)
    header = ["## Table 4 — Frame-switch consistency on ViewSpatial",
              "",
              f"*{len(pairs)} cross-frame pairs extracted from the test set.*",
              "",
              "| Method | FCA ↑ | CR ↓ | Camera Acc | Non-Cam Acc | FG |",
              "|---|---:|---:|---:|---:|---:|"]
    if not pairs:
        return "\n".join(header + ["*No pairs found — check `pair_id` in the benchmark jsonl.*"])

    for name, path in runs:
        data = _load_if_exists(os.path.join(path, "viewspatial.json"))
        if data is None:
            continue
        pred_map = {r["id"]: r for r in data["results"]}

        both_ok = contra = 0
        cam_c = cam_n = other_c = other_n = 0
        n_used = 0
        for sa, sb in pairs:
            pa = pred_map.get(sa["id"])
            pb = pred_map.get(sb["id"])
            if not pa or not pb:
                continue
            n_used += 1
            if pa["correct"] and pb["correct"]:
                both_ok += 1
            if pa["correct"] != pb["correct"]:
                contra += 1
            for s, pr in ((sa, pa), (sb, pb)):
                if s.get("frame_type") == "camera":
                    cam_n += 1; cam_c += int(pr["correct"])
                else:
                    other_n += 1; other_c += int(pr["correct"])

        if n_used == 0:
            continue
        fca = both_ok / n_used * 100
        cr = contra / n_used * 100
        cam_acc = cam_c / cam_n * 100 if cam_n else 0
        other_acc = other_c / other_n * 100 if other_n else 0
        fg = cam_acc - other_acc

        header.append(
            "| {m} | {fca:.2f} | {cr:.2f} | {ca:.2f} | {oa:.2f} | {fg:+.2f} |".format(
                m=NICE_LABEL.get(name, name),
                fca=fca, cr=cr, ca=cam_acc, oa=other_acc, fg=fg,
            )
        )
    return "\n".join(header)


# ── Entry point ───────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--bench_path", default=VIEWSPATIAL_BENCH)
    ap.add_argument("--out", default=None,
                    help="If set, also write markdown to this file.")
    args = ap.parse_args()

    runs = _discover_runs(args.results_dir)
    if not runs:
        print(f"No result subdirs found under {args.results_dir}")
        return

    bench_samples = _load_jsonl(args.bench_path) if os.path.exists(args.bench_path) else []

    blocks = [
        build_table_main(runs),
        "",
        build_table_frametype(runs, bench_samples),
        "",
        build_table_consistency(runs, bench_samples),
    ]
    md = "\n".join(blocks) + "\n"
    print(md)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
