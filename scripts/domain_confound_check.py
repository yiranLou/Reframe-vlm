"""Check whether the ViewSpatial domain split is confounded with question type."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from result_registry import resolve_result_path


DEFAULT_BENCH_PATH = "data/processed/viewspatial_test.jsonl"
SCENE_RE = re.compile(r"/(scene\d{4}_\d{2})/")
RUNS = [
    ("zeroshot", "ZS"),
    ("prompt_baseline", "Pr"),
    ("baseline_lora_ep1", "Naive"),
    ("text_instruction_lora_ep1", "TextInstr"),
    ("frame_lora_ep1", "Frame"),
    ("frame_gated_lora_ep1", "FrameGated"),
    ("token_gated_lora_ep1", "TokenGated"),
    ("full_method_ep1", "Full"),
]


def domain_of(sample):
    images = sample.get("images") or []
    if not images:
        return "unknown"
    path = images[0]
    if SCENE_RE.search(path):
        return "scannet"
    if "val2017" in path or "/coco" in path.lower():
        return "coco"
    return "other"


def load_bench(bench_path):
    path = Path(bench_path)
    if not path.exists():
        raise SystemExit(
            f"ViewSpatial benchmark metadata not found at {bench_path}. "
            "Pass --bench_path to point at the unified jsonl used for evaluation."
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_run(run):
    path = resolve_result_path(run, "viewspatial")
    if path is None:
        return None
    return {row["id"]: int(row["correct"]) for row in json.loads(path.read_text())["results"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_path", default=DEFAULT_BENCH_PATH)
    args = parser.parse_args()

    samples = load_bench(args.bench_path)
    out = []
    out.append(f"total test samples: {len(samples)}\n")

    out.append("## Table A: Domain × question_type × frame_type (n samples)\n")
    counts = defaultdict(lambda: defaultdict(Counter))
    for sample in samples:
        counts[domain_of(sample)][sample.get("question_type", "?")][sample.get("frame_type", "?")] += 1

    for domain, qtypes in counts.items():
        domain_total = sum(sum(counter.values()) for counter in qtypes.values())
        out.append(f"### {domain.upper()} (n={domain_total})")
        out.append("")
        for question_type, frames in qtypes.items():
            total = sum(frames.values())
            breakdown = ", ".join(f"{frame}={count}" for frame, count in frames.items())
            out.append(f"- {question_type}: {total} [{breakdown}]")
        out.append("")

    run_maps = {run: load_run(run) for run, _ in RUNS}
    groups = defaultdict(list)
    for sample in samples:
        groups[(domain_of(sample), sample.get("question_type", "?"))].append(sample["id"])

    out.append("## Table B: Accuracy by domain × question_type (%)\n")
    out.append("| domain | question_type | n | " + " | ".join(label for _, label in RUNS) + " |")
    out.append("|---|---|---:|" + "|".join(["---:"] * len(RUNS)) + "|")
    for (domain, question_type), ids in sorted(groups.items()):
        row = [domain, question_type, str(len(ids))]
        for run, _ in RUNS:
            pred_map = run_maps[run]
            if pred_map is None:
                row.append("—")
                continue
            values = [pred_map[sample_id] for sample_id in ids if sample_id in pred_map]
            row.append(f"{(sum(values) / len(values) * 100):.2f}" if values else "—")
        out.append("| " + " | ".join(row) + " |")

    out.append("\n## Table C: Frame LoRA − Naive LoRA per group (pp)\n")
    naive = run_maps.get("baseline_lora_ep1")
    frame = run_maps.get("frame_lora_ep1")
    out.append("| domain | question_type | n | Δ | Naive | Frame |")
    out.append("|---|---|---:|---:|---:|---:|")
    if naive is not None and frame is not None:
        for (domain, question_type), ids in sorted(groups.items(), key=lambda item: -len(item[1])):
            common = [sample_id for sample_id in ids if sample_id in naive and sample_id in frame]
            if not common:
                continue
            naive_acc = sum(naive[sample_id] for sample_id in common) / len(common) * 100
            frame_acc = sum(frame[sample_id] for sample_id in common) / len(common) * 100
            out.append(
                f"| {domain} | {question_type} | {len(common)} | {frame_acc - naive_acc:+.2f} | {naive_acc:.2f} | {frame_acc:.2f} |"
            )

    text_instr = run_maps.get("text_instruction_lora_ep1")
    if naive is not None and text_instr is not None:
        out.append("\n## Table D: text-instr SFT − Naive LoRA per group (pp)\n")
        out.append("| domain | question_type | n | Δ | Naive | TextInstr |")
        out.append("|---|---|---:|---:|---:|---:|")
        for (domain, question_type), ids in sorted(groups.items(), key=lambda item: -len(item[1])):
            common = [sample_id for sample_id in ids if sample_id in naive and sample_id in text_instr]
            if not common:
                continue
            naive_acc = sum(naive[sample_id] for sample_id in common) / len(common) * 100
            ti_acc = sum(text_instr[sample_id] for sample_id in common) / len(common) * 100
            out.append(
                f"| {domain} | {question_type} | {len(common)} | {ti_acc - naive_acc:+.2f} | {naive_acc:.2f} | {ti_acc:.2f} |"
            )

    if frame is not None and text_instr is not None:
        out.append("\n## Table E: text-instr SFT − Frame LoRA per group (pp)\n")
        out.append("| domain | question_type | n | Δ | Frame | TextInstr |")
        out.append("|---|---|---:|---:|---:|---:|")
        for (domain, question_type), ids in sorted(groups.items(), key=lambda item: -len(item[1])):
            common = [sample_id for sample_id in ids if sample_id in frame and sample_id in text_instr]
            if not common:
                continue
            frame_acc = sum(frame[sample_id] for sample_id in common) / len(common) * 100
            ti_acc = sum(text_instr[sample_id] for sample_id in common) / len(common) * 100
            out.append(
                f"| {domain} | {question_type} | {len(common)} | {ti_acc - frame_acc:+.2f} | {frame_acc:.2f} | {ti_acc:.2f} |"
            )

    md = "\n".join(out) + "\n"
    print(md)
    Path("results/domain_confound.md").write_text(md, encoding="utf-8")
    print("\nsaved to results/domain_confound.md")


if __name__ == "__main__":
    main()
