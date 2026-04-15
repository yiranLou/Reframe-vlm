"""
Scene-level paired bootstrap CI.

Because 98.6% of ViewSpatial-Bench test scenes also appear in training, a
sample-level bootstrap treats every test QA as an independent observation
and likely understates uncertainty. The more reviewer-defensible protocol
is to resample *scenes* — draw a scene id uniformly with replacement, and
include every test QA that belongs to that scene in the bootstrap sample.

This widens CIs (scenes are the de-facto unit of variation) but the
resulting intervals answer the right question: does the improvement
generalize across scenes?

For MMSI and Ego3D we fall back to sample-level bootstrap because those
benchmarks don't share a scene universe with our training data — scene
resampling there would be overkill and not well-defined.
"""

import json
import re
import numpy as np
from collections import defaultdict
from pathlib import Path


RUNS = [
    ("zeroshot",          "Qwen2.5-VL zero-shot"),
    ("prompt_baseline",   "+ text prompt"),
    ("baseline_lora_ep1", "Naive LoRA"),
    ("frame_lora_ep1",    "Frame LoRA"),
    ("full_method_ep1",   "Full Method"),
]
COMPARISONS = [
    ("frame_lora_ep1",    "baseline_lora_ep1"),
    ("full_method_ep1",   "frame_lora_ep1"),
    ("baseline_lora_ep1", "zeroshot"),
]

BENCHES = ["viewspatial", "mmsi", "ego3d"]
N_BOOT = 10_000
RNG = np.random.default_rng(0)

SCENE_RE = re.compile(r"/(scene\d{4}_\d{2})/")


def scene_of(sample):
    imgs = sample.get("images") or []
    for p in imgs:
        m = SCENE_RE.search(p)
        if m:
            return m.group(1)
    return None


def load_run_results(run, bench):
    p = Path(f"results/{run}/{bench}.json")
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return {r["id"]: int(r["correct"]) for r in data["results"]}


def load_bench(bench):
    # Only viewspatial has scene paths we care about.
    if bench == "viewspatial":
        p = Path("data/processed/viewspatial_test.jsonl")
    else:
        p = Path(f"data/processed/{bench}_test.jsonl")
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def scene_cluster_viewspatial():
    """Return list of lists of sample_ids, grouped by scene."""
    samples = load_bench("viewspatial")
    buckets = defaultdict(list)
    orphan = []
    for s in samples:
        sc = scene_of(s)
        (buckets[sc] if sc else orphan).append(s["id"])
    groups = list(buckets.values())
    if orphan:
        groups.extend([[i] for i in orphan])
    return groups


def paired_scene_boot(run_a, run_b, bench):
    a = load_run_results(run_a, bench)
    b = load_run_results(run_b, bench)
    if a is None or b is None:
        return None

    if bench == "viewspatial":
        scene_groups = scene_cluster_viewspatial()
        groups = [[i for i in g if i in a and i in b] for g in scene_groups]
        groups = [g for g in groups if g]
        unit = "scenes"
    else:
        ids = sorted(set(a) & set(b))
        groups = [[i] for i in ids]
        unit = "samples"

    k = len(groups)
    if k == 0:
        return None

    va_all = {i: a[i] for g in groups for i in g}
    vb_all = {i: b[i] for g in groups for i in g}
    point_a = np.mean(list(va_all.values())) * 100
    point_b = np.mean(list(vb_all.values())) * 100
    delta = point_a - point_b

    boot = np.empty(N_BOOT, dtype=np.float64)
    group_idxs = np.arange(k)
    for r in range(N_BOOT):
        pick = RNG.choice(group_idxs, size=k, replace=True)
        va_sum = vb_sum = n = 0
        for gi in pick:
            g = groups[gi]
            for sid in g:
                va_sum += a[sid]
                vb_sum += b[sid]
                n += 1
        boot[r] = (va_sum - vb_sum) / n * 100

    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_pos = float((boot > 0).mean())
    return dict(unit=unit, k=k, n_items=sum(len(g) for g in groups),
                delta=delta, ci_lo=lo, ci_hi=hi, p_pos=p_pos)


def main():
    rows = []
    rows.append("## Scene-level paired bootstrap CI\n")
    rows.append("_Resampling units: **scenes** on ViewSpatial (reviewer-defensible, accounts for scene overlap with train); samples on MMSI/Ego3D._\n")
    rows.append("| Comparison | Benchmark | unit | k | n | Δ (pp) | 95% CI | P(Δ>0) |")
    rows.append("|---|---|---|---:|---:|---:|---|---:|")
    for a, b in COMPARISONS:
        for bench in BENCHES:
            r = paired_scene_boot(a, b, bench)
            if r is None:
                continue
            la = dict(RUNS)[a]; lb = dict(RUNS)[b]
            rows.append(
                f"| {la} − {lb} | {bench} | {r['unit']} | "
                f"{r['k']} | {r['n_items']} | {r['delta']:+.2f} | "
                f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['p_pos']:.3f} |"
            )
    md = "\n".join(rows) + "\n"
    print(md)
    Path("results/bootstrap_ci_scene.md").write_text(md, encoding="utf-8")
    print("\nsaved to results/bootstrap_ci_scene.md")


if __name__ == "__main__":
    main()
