"""Scene-level paired bootstrap CI."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from result_registry import RUN_LABELS, resolve_result_path


DEFAULT_VIEWSPATIAL_PATH = "data/processed/viewspatial_test.jsonl"
COMPARISONS = [
    ("text_instruction_lora_ep1", "frame_lora_ep1"),
    ("frame_lora_ep1", "baseline_lora_ep1"),
    ("full_method_ep1", "frame_lora_ep1"),
    ("text_instruction_lora_ep1", "baseline_lora_ep1"),
    ("baseline_lora_ep1", "zeroshot"),
]
BENCHES = ("viewspatial", "mmsi", "ego3d")
N_BOOT = 10_000
RNG = np.random.default_rng(0)
SCENE_RE = re.compile(r"/(scene\d{4}_\d{2})/")


def scene_of(sample):
    for path in sample.get("images") or []:
        match = SCENE_RE.search(path)
        if match:
            return match.group(1)
    return None


def load_run_results(run, bench):
    path = resolve_result_path(run, bench)
    if path is None:
        return None
    data = json.loads(path.read_text())
    return {row["id"]: int(row["correct"]) for row in data["results"]}


def load_viewspatial_bench(bench_path):
    path = Path(bench_path)
    if not path.exists():
        raise SystemExit(
            f"ViewSpatial benchmark metadata not found at {bench_path}. "
            "Pass --viewspatial_path to point at the unified jsonl used for evaluation."
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def scene_cluster_viewspatial(bench_path):
    samples = load_viewspatial_bench(bench_path)
    buckets = defaultdict(list)
    orphan = []
    for sample in samples:
        scene = scene_of(sample)
        (buckets[scene] if scene else orphan).append(sample["id"])
    groups = list(buckets.values())
    if orphan:
        groups.extend([[sample_id] for sample_id in orphan])
    return groups


def paired_scene_boot(run_a, run_b, bench, viewspatial_path):
    a = load_run_results(run_a, bench)
    b = load_run_results(run_b, bench)
    if a is None or b is None:
        return None

    if bench == "viewspatial":
        base_groups = scene_cluster_viewspatial(viewspatial_path)
        groups = [[sample_id for sample_id in group if sample_id in a and sample_id in b] for group in base_groups]
        groups = [group for group in groups if group]
        unit = "scenes"
    else:
        ids = sorted(set(a) & set(b))
        groups = [[sample_id] for sample_id in ids]
        unit = "samples"

    if not groups:
        return None

    all_a = {sample_id: a[sample_id] for group in groups for sample_id in group}
    all_b = {sample_id: b[sample_id] for group in groups for sample_id in group}
    delta = (np.mean(list(all_a.values())) - np.mean(list(all_b.values()))) * 100

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

    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "unit": unit,
        "k": len(groups),
        "n_items": sum(len(group) for group in groups),
        "delta": delta,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_pos": float((boot > 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewspatial_path", default=DEFAULT_VIEWSPATIAL_PATH)
    args = parser.parse_args()

    rows = []
    rows.append("## Scene-level paired bootstrap CI\n")
    rows.append("_Resampling units: **scenes** on ViewSpatial; **samples** on MMSI/Ego3D._\n")
    rows.append("| Comparison | Benchmark | unit | k | n | Δ (pp) | 95% CI | P(Δ>0) |")
    rows.append("|---|---|---|---:|---:|---:|---|---:|")
    for run_a, run_b in COMPARISONS:
        for bench in BENCHES:
            stats = paired_scene_boot(run_a, run_b, bench, args.viewspatial_path)
            if stats is None:
                continue
            rows.append(
                f"| {RUN_LABELS[run_a]} − {RUN_LABELS[run_b]} | {bench} | {stats['unit']} | "
                f"{stats['k']} | {stats['n_items']} | {stats['delta']:+.2f} | "
                f"[{stats['ci_lo']:+.2f}, {stats['ci_hi']:+.2f}] | {stats['p_pos']:.3f} |"
            )
    md = "\n".join(rows) + "\n"
    print(md)
    Path("results/bootstrap_ci_scene.md").write_text(md, encoding="utf-8")
    print("\nsaved to results/bootstrap_ci_scene.md")


if __name__ == "__main__":
    main()
