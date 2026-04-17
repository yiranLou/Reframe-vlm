"""Aggregate canonical eval results into paper-ready markdown tables."""

import argparse
import json
import os
import re
from collections import defaultdict

from result_registry import RUN_LABELS, active_runs, resolve_result_path


VIEWSPATIAL_BENCH = "data/processed/viewspatial_test.jsonl"
CHOICE_RE = re.compile(r"^\s*\(?([A-Da-d])[\).:]?\s*(.*)$")


def normalize_relation_text(text):
    text = str(text or "").strip()
    match = CHOICE_RE.match(text)
    if match and match.group(2):
        text = match.group(2)
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9. ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def first_choice_letter(text):
    match = CHOICE_RE.match(str(text or "").strip())
    return match.group(1).upper() if match else None


def choice_relation(choices, letter):
    if not choices or not letter:
        return None
    idx = ord(letter.upper()) - ord("A")
    if idx < 0 or idx >= len(choices):
        return None
    return normalize_relation_text(choices[idx])


def target_relation(answer, choices=None):
    letter = first_choice_letter(answer)
    return choice_relation(choices, letter) or normalize_relation_text(answer)


def pred_relation(pred, choices=None):
    letter = first_choice_letter(pred)
    return choice_relation(choices, letter) or normalize_relation_text(pred)


def is_frame_swap_contradiction(sample_a, sample_b, pred_a, pred_b):
    gt_a = target_relation(sample_a.get("answer"), sample_a.get("choices"))
    gt_b = target_relation(sample_b.get("answer"), sample_b.get("choices"))
    if not gt_a or not gt_b or gt_a == gt_b:
        return False

    pred_a_rel = pred_relation(pred_a.get("pred"), sample_a.get("choices"))
    pred_b_rel = pred_relation(pred_b.get("pred"), sample_b.get("choices"))
    a_correct = bool(pred_a.get("correct"))
    b_correct = bool(pred_b.get("correct"))

    if a_correct and not b_correct:
        return pred_b_rel == gt_a
    if b_correct and not a_correct:
        return pred_a_rel == gt_b
    if not a_correct and not b_correct:
        return pred_a_rel == gt_b and pred_b_rel == gt_a
    return False


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_result(run, bench):
    path = resolve_result_path(run, bench)
    if path is None:
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_table_main(runs):
    lines = [
        "## Table 1 — Main results (overall accuracy %)",
        "",
        "| Method | ViewSpatial | MMSI | Ego3D |",
        "|---|---:|---:|---:|",
    ]
    for run in runs:
        row = [RUN_LABELS.get(run, run)]
        for bench in ("viewspatial", "mmsi", "ego3d"):
            data = load_result(run, bench)
            row.append("—" if data is None else f"{data['accuracy']:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def per_frame_accuracy(run, bench_samples):
    data = load_result(run, "viewspatial")
    if data is None:
        return None
    bench_map = {sample["id"]: sample for sample in bench_samples}
    per_frame = defaultdict(lambda: [0, 0])
    for row in data["results"]:
        frame_type = row.get("frame_type")
        if not frame_type or frame_type == "unknown":
            bench_row = bench_map.get(row["id"])
            frame_type = bench_row.get("frame_type", "unknown") if bench_row else "unknown"
        per_frame[frame_type][1] += 1
        per_frame[frame_type][0] += int(row["correct"])
    return dict(per_frame)


def build_table_frametype(runs, bench_samples):
    all_types = set()
    cache = {}
    for run in runs:
        stats = per_frame_accuracy(run, bench_samples)
        cache[run] = stats
        if stats:
            all_types.update(stats.keys())

    type_order = [frame_type for frame_type in ("camera", "person", "object", "world") if frame_type in all_types]
    lines = [
        "## Table 3 — ViewSpatial per-frame-type accuracy (%)",
        "",
        "| Method | " + " | ".join(frame_type.capitalize() for frame_type in type_order) + " | Overall |",
        "|---|" + "|".join(["---:"] * (len(type_order) + 1)) + "|",
    ]

    for run in runs:
        stats = cache[run]
        if not stats:
            continue
        row = [RUN_LABELS.get(run, run)]
        total_c = total_n = 0
        for frame_type in type_order:
            correct, total = stats.get(frame_type, [0, 0])
            row.append(f"{(correct / total * 100 if total else 0):.2f}")
            total_c += correct
            total_n += total
        row.append(f"{(total_c / total_n * 100 if total_n else 0):.2f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_bench_pairs(bench_samples):
    by_id = {sample["id"]: sample for sample in bench_samples}
    pairs = []
    seen = set()
    for sample in bench_samples:
        pair_id = sample.get("pair_id")
        if pair_id and sample["id"] not in seen and pair_id in by_id:
            pairs.append((sample, by_id[pair_id]))
            seen.add(sample["id"])
            seen.add(pair_id)
    if pairs:
        return pairs

    by_img = defaultdict(list)
    for sample in bench_samples:
        by_img[tuple(sample.get("images", []))].append(sample)

    for group in by_img.values():
        if len({item["frame_type"] for item in group}) < 2:
            continue
        cams = [item for item in group if item["frame_type"] == "camera" and item["id"] not in seen]
        others = [item for item in group if item["frame_type"] != "camera" and item["id"] not in seen]
        for sample_a, sample_b in zip(cams, others):
            pairs.append((sample_a, sample_b))
            seen.add(sample_a["id"])
            seen.add(sample_b["id"])
    return pairs


def build_table_consistency(runs, bench_samples, bench_path_exists):
    lines = [
        "## Table 4 — Frame-switch consistency on ViewSpatial",
        "",
    ]
    if not bench_path_exists:
        lines.append(
            "*Benchmark metadata unavailable locally. Re-run with `--bench_path` pointing at the unified ViewSpatial test jsonl to compute FCA / CR / PDR / FG.*"
        )
        return "\n".join(lines)

    pairs = build_bench_pairs(bench_samples)
    lines.extend([
        f"*{len(pairs)} cross-frame pairs extracted from the test set.*",
        "",
        "| Method | FCA ↑ | CR ↓ | PDR ↓ | Camera Acc | Non-Cam Acc | FG |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    if not pairs:
        return "\n".join(lines + ["*No pairs found — check `pair_id` in the benchmark jsonl.*"])

    for run in runs:
        data = load_result(run, "viewspatial")
        if data is None:
            continue
        pred_map = {row["id"]: row for row in data["results"]}

        both_ok = contradictions = disagreements = 0
        cam_c = cam_n = other_c = other_n = 0
        used = 0
        for sample_a, sample_b in pairs:
            pred_a = pred_map.get(sample_a["id"])
            pred_b = pred_map.get(sample_b["id"])
            if pred_a is None or pred_b is None:
                continue
            used += 1
            a_correct = bool(pred_a.get("correct"))
            b_correct = bool(pred_b.get("correct"))
            if a_correct and b_correct:
                both_ok += 1
            if is_frame_swap_contradiction(sample_a, sample_b, pred_a, pred_b):
                contradictions += 1
            if a_correct != b_correct:
                disagreements += 1
            for sample, pred in ((sample_a, pred_a), (sample_b, pred_b)):
                if sample.get("frame_type") == "camera":
                    cam_n += 1
                    cam_c += int(pred.get("correct", 0))
                else:
                    other_n += 1
                    other_c += int(pred.get("correct", 0))

        if used == 0:
            continue
        cam_acc = cam_c / cam_n * 100 if cam_n else 0.0
        other_acc = other_c / other_n * 100 if other_n else 0.0
        lines.append(
            "| {method} | {fca:.2f} | {cr:.2f} | {pdr:.2f} | {cam:.2f} | {other:.2f} | {fg:+.2f} |".format(
                method=RUN_LABELS.get(run, run),
                fca=both_ok / used * 100,
                cr=contradictions / used * 100,
                pdr=disagreements / used * 100,
                cam=cam_acc,
                other=other_acc,
                fg=cam_acc - other_acc,
            )
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_path", default=VIEWSPATIAL_BENCH)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    runs = active_runs()
    if not runs:
        print("No active result runs found under results/")
        return

    bench_path_exists = os.path.exists(args.bench_path)
    bench_samples = load_jsonl(args.bench_path) if bench_path_exists else []
    blocks = [
        build_table_main(runs),
        "",
        build_table_frametype(runs, bench_samples),
        "",
        build_table_consistency(runs, bench_samples, bench_path_exists),
    ]
    md = "\n".join(blocks) + "\n"
    print(md)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(md)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
