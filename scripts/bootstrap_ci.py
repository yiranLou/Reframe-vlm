"""
Bootstrap confidence intervals for method-vs-method accuracy differences.

For each benchmark, we compare key method pairs and report:
  - mean of the paired accuracy difference (pp)
  - 95% bootstrap CI
  - p(Δ > 0) — how often the bootstrap sample shows the expected sign
  - n samples used (intersection of IDs across the two runs)

Paired bootstrap on shared sample IDs: for each draw, resample indices with
replacement, compute the per-method accuracy on the resample, report Δ.

Usage:
    python scripts/bootstrap_ci.py
"""

import json
import numpy as np
from pathlib import Path


RUNS = [
    ("zeroshot",                  "Qwen2.5-VL zero-shot"),
    ("prompt_baseline",           "+ text prompt"),
    ("baseline_lora_ep1",         "Naive LoRA"),
    ("text_instruction_lora_ep1", "LoRA+text-instr SFT"),
    ("frame_lora_ep1",            "Frame LoRA"),
    ("frame_gated_lora_ep1",      "Frame-Gated LoRA"),
    ("token_gated_lora_ep1",      "Token+Gated LoRA"),
    ("full_method_ep1",           "Full Method"),
]
COMPARISONS = [
    # Primary mechanism comparisons
    ("text_instruction_lora_ep1", "frame_lora_ep1"),       # text vs token
    ("frame_gated_lora_ep1",      "text_instruction_lora_ep1"),  # gate vs text
    ("frame_gated_lora_ep1",      "frame_lora_ep1"),       # gate vs token
    ("frame_gated_lora_ep1",      "baseline_lora_ep1"),    # gate vs naive
    ("text_instruction_lora_ep1", "baseline_lora_ep1"),    # text vs naive
    ("frame_lora_ep1",            "baseline_lora_ep1"),    # token vs naive
    ("full_method_ep1",           "frame_lora_ep1"),       # Full vs Frame
    ("baseline_lora_ep1",         "zeroshot"),             # SFT effect
]

BENCHES = ["viewspatial", "mmsi", "ego3d"]
N_BOOT = 10_000
RNG = np.random.default_rng(0)


def load_correct(run, bench):
    p = Path(f"results/{run}/{bench}.json")
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return {r["id"]: int(r["correct"]) for r in data["results"]}


def bench_ci_for_run(run, bench):
    d = load_correct(run, bench)
    if d is None:
        return None
    v = np.fromiter(d.values(), dtype=np.int8)
    n = len(v)
    if n == 0:
        return None
    point = v.mean() * 100
    idxs = RNG.integers(0, n, size=(N_BOOT, n))
    boot = v[idxs].mean(axis=1) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=n, acc=point, ci_lo=lo, ci_hi=hi)


def paired_diff_ci(run_a, run_b, bench):
    a = load_correct(run_a, bench)
    b = load_correct(run_b, bench)
    if a is None or b is None:
        return None
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    va = np.array([a[i] for i in ids], dtype=np.int8)
    vb = np.array([b[i] for i in ids], dtype=np.int8)
    n = len(ids)
    point = (va.mean() - vb.mean()) * 100
    idxs = RNG.integers(0, n, size=(N_BOOT, n))
    boot = (va[idxs].mean(axis=1) - vb[idxs].mean(axis=1)) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_pos = float((boot > 0).mean())
    return dict(n=n, delta=point, ci_lo=lo, ci_hi=hi, p_pos=p_pos)


def main():
    rows = []
    rows.append("## Per-method accuracy with 95% CI\n")
    rows.append("| Method | ViewSpatial | MMSI | Ego3D |")
    rows.append("|---|---:|---:|---:|")
    for run, label in RUNS:
        cells = [label]
        for bench in BENCHES:
            s = bench_ci_for_run(run, bench)
            if s is None:
                cells.append("—")
            else:
                cells.append(f"{s['acc']:.2f} [{s['ci_lo']:.2f}, {s['ci_hi']:.2f}]")
        rows.append("| " + " | ".join(cells) + " |")

    rows.append("")
    rows.append("## Paired bootstrap: Δ = A − B (pp), 95% CI, P(Δ>0)\n")
    rows.append("| Comparison | Benchmark | n | Δ | 95% CI | P(Δ>0) |")
    rows.append("|---|---|---:|---:|---|---:|")
    for a, b in COMPARISONS:
        for bench in BENCHES:
            r = paired_diff_ci(a, b, bench)
            if r is None:
                continue
            la = dict(RUNS)[a]; lb = dict(RUNS)[b]
            rows.append(
                f"| {la} − {lb} | {bench} | {r['n']} | "
                f"{r['delta']:+.2f} | [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | "
                f"{r['p_pos']:.3f} |"
            )

    md = "\n".join(rows) + "\n"
    print(md)
    out = Path("results/bootstrap_ci.md")
    out.write_text(md, encoding="utf-8")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
