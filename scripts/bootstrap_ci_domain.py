"""Per-domain bootstrap CI on ViewSpatial-Bench."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from result_registry import RUNS, RUN_LABELS, resolve_result_path


DEFAULT_BENCH_PATH = "data/processed/viewspatial_test.jsonl"
COMPARISONS = [
    ("text_instruction_lora_ep1", "frame_lora_ep1"),
    ("frame_gated_lora_ep1", "text_instruction_lora_ep1"),
    ("frame_gated_lora_ep1", "frame_lora_ep1"),
    ("text_instruction_lora_ep1", "baseline_lora_ep1"),
    ("frame_lora_ep1", "baseline_lora_ep1"),
    ("full_method_ep1", "frame_lora_ep1"),
    ("baseline_lora_ep1", "zeroshot"),
]
N_BOOT = 10_000
RNG = np.random.default_rng(0)
SCENE_RE = re.compile(r"/(scene\d{4}_\d{2})/")


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


def scene_of(sample):
    for path in sample.get("images") or []:
        match = SCENE_RE.search(path)
        if match:
            return match.group(1)
    return None


def load_run(run):
    path = resolve_result_path(run, "viewspatial")
    if path is None:
        return None
    data = json.loads(path.read_text())
    return {row["id"]: int(row["correct"]) for row in data["results"]}


def load_bench(bench_path):
    path = Path(bench_path)
    if not path.exists():
        raise SystemExit(
            f"ViewSpatial benchmark metadata not found at {bench_path}. "
            "Pass --bench_path to point at the unified jsonl used for evaluation."
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_groups(samples, domain):
    chosen = [sample for sample in samples if domain_of(sample) == domain]
    if domain == "scannet":
        buckets = defaultdict(list)
        for sample in chosen:
            scene = scene_of(sample)
            if scene:
                buckets[scene].append(sample["id"])
        return list(buckets.values()), len(chosen)
    return [[sample["id"]] for sample in chosen], len(chosen)


def paired_boot(a, b, groups):
    groups = [[sample_id for sample_id in group if sample_id in a and sample_id in b] for group in groups]
    groups = [group for group in groups if group]
    n = sum(len(group) for group in groups)
    if n == 0:
        return None
    boot = np.empty(N_BOOT, dtype=np.float64)
    group_indices = np.arange(len(groups))
    for i in range(N_BOOT):
        picked = RNG.choice(group_indices, size=len(groups), replace=True)
        total_a = total_b = count = 0
        for group_idx in picked:
            for sample_id in groups[group_idx]:
                total_a += a[sample_id]
                total_b += b[sample_id]
                count += 1
        boot[i] = (total_a - total_b) / count * 100
    flat_a = [a[sample_id] for group in groups for sample_id in group]
    flat_b = [b[sample_id] for group in groups for sample_id in group]
    delta = (np.mean(flat_a) - np.mean(flat_b)) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": n,
        "k": len(groups),
        "delta": delta,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_pos": float((boot > 0).mean()),
    }


def acc_ci(run, groups):
    results = load_run(run)
    if results is None:
        return None
    groups = [[sample_id for sample_id in group if sample_id in results] for group in groups]
    groups = [group for group in groups if group]
    values = [results[sample_id] for group in groups for sample_id in group]
    if not values:
        return None
    point = np.mean(values) * 100
    boot = np.empty(N_BOOT, dtype=np.float64)
    group_indices = np.arange(len(groups))
    for i in range(N_BOOT):
        picked = RNG.choice(group_indices, size=len(groups), replace=True)
        sampled = [results[sample_id] for group_idx in picked for sample_id in groups[group_idx]]
        boot[i] = np.mean(sampled) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": len(values),
        "k": len(groups),
        "acc": point,
        "ci_lo": lo,
        "ci_hi": hi,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_path", default=DEFAULT_BENCH_PATH)
    args = parser.parse_args()

    samples = load_bench(args.bench_path)
    out = []
    for domain, unit_label in (("scannet", "scenes"), ("coco", "samples")):
        groups, total = build_groups(samples, domain)
        out.append(f"\n## Domain = {domain} (unit: {unit_label}; k={len(groups)}, n={total})\n")
        out.append("### Per-method accuracy with 95% CI")
        out.append("| Method | n | acc | 95% CI |")
        out.append("|---|---:|---:|---|")
        for run, label in RUNS:
            stats = acc_ci(run, groups)
            if stats is None:
                out.append(f"| {label} | — | — | — |")
            else:
                out.append(
                    f"| {label} | {stats['n']} | {stats['acc']:.2f} | [{stats['ci_lo']:.2f}, {stats['ci_hi']:.2f}] |"
                )

        out.append("\n### Paired Δ with 95% CI")
        out.append("| Comparison | n | Δ (pp) | 95% CI | P(Δ>0) |")
        out.append("|---|---:|---:|---|---:|")
        for run_a, run_b in COMPARISONS:
            a = load_run(run_a)
            b = load_run(run_b)
            if a is None or b is None:
                continue
            stats = paired_boot(a, b, groups)
            if stats is None:
                continue
            out.append(
                f"| {RUN_LABELS[run_a]} − {RUN_LABELS[run_b]} | {stats['n']} | {stats['delta']:+.2f} | "
                f"[{stats['ci_lo']:+.2f}, {stats['ci_hi']:+.2f}] | {stats['p_pos']:.3f} |"
            )

    md = "\n".join(out) + "\n"
    print(md)
    Path("results/bootstrap_ci_domain.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
