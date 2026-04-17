"""Bootstrap confidence intervals for key accuracy differences.

Uses the canonical run registry so corrected re-evaluation outputs (for example
``*_refix/viewspatial.json``) are preferred automatically.
"""

import json
from pathlib import Path

import numpy as np

from result_registry import RUNS, RUN_LABELS, BENCHES, resolve_result_path


COMPARISONS = [
    ("text_instruction_lora_ep1", "frame_lora_ep1"),
    ("frame_gated_lora_ep1", "text_instruction_lora_ep1"),
    ("frame_gated_lora_ep1", "frame_lora_ep1"),
    ("frame_gated_lora_ep1", "baseline_lora_ep1"),
    ("text_instruction_lora_ep1", "baseline_lora_ep1"),
    ("frame_lora_ep1", "baseline_lora_ep1"),
    ("full_method_ep1", "frame_lora_ep1"),
    ("baseline_lora_ep1", "zeroshot"),
]
N_BOOT = 10_000
RNG = np.random.default_rng(0)


def load_correct(run, bench):
    path = resolve_result_path(run, bench)
    if path is None:
        return None
    data = json.loads(path.read_text())
    return {row["id"]: int(row["correct"]) for row in data["results"]}


def bench_ci_for_run(run, bench):
    correct = load_correct(run, bench)
    if correct is None:
        return None
    values = np.fromiter(correct.values(), dtype=np.int8)
    n = len(values)
    if n == 0:
        return None
    point = values.mean() * 100
    idxs = RNG.integers(0, n, size=(N_BOOT, n))
    boot = values[idxs].mean(axis=1) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": n, "acc": point, "ci_lo": lo, "ci_hi": hi}


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
    return {
        "n": n,
        "delta": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_pos": float((boot > 0).mean()),
    }


def main():
    rows = []
    rows.append("## Per-method accuracy with 95% CI\n")
    rows.append("| Method | ViewSpatial | MMSI | Ego3D |")
    rows.append("|---|---:|---:|---:|")
    for run, label in RUNS:
        cells = [label]
        for bench in BENCHES:
            stats = bench_ci_for_run(run, bench)
            if stats is None:
                cells.append("—")
            else:
                cells.append(
                    f"{stats['acc']:.2f} [{stats['ci_lo']:.2f}, {stats['ci_hi']:.2f}]"
                )
        rows.append("| " + " | ".join(cells) + " |")

    rows.append("")
    rows.append("## Paired bootstrap: Δ = A − B (pp), 95% CI, P(Δ>0)\n")
    rows.append("| Comparison | Benchmark | n | Δ | 95% CI | P(Δ>0) |")
    rows.append("|---|---|---:|---:|---|---:|")
    for run_a, run_b in COMPARISONS:
        for bench in BENCHES:
            stats = paired_diff_ci(run_a, run_b, bench)
            if stats is None:
                continue
            rows.append(
                f"| {RUN_LABELS[run_a]} − {RUN_LABELS[run_b]} | {bench} | {stats['n']} | "
                f"{stats['delta']:+.2f} | [{stats['ci_lo']:+.2f}, {stats['ci_hi']:+.2f}] | "
                f"{stats['p_pos']:.3f} |"
            )

    md = "\n".join(rows) + "\n"
    print(md)
    out = Path("results/bootstrap_ci.md")
    out.write_text(md, encoding="utf-8")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
