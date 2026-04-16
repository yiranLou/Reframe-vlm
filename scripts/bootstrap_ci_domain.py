"""
Per-domain bootstrap CI on ViewSpatial-Bench.

ViewSpatial-Bench has two domains with different relations to our training set:
  - ScanNet portion   (~2878 samples, 279 scenes; train shares 98.6% of scenes)
  - COCO val2017      (~2834 samples, no scene overlap — truly OOD for us)

We report the same method comparisons, split by domain. This lets the paper
claim:
  * in-domain scene-adaptive gains on ScanNet (meaningful but limited),
  * true out-of-distribution gains on COCO.

Units:
  - ScanNet: resample scenes (reviewer-defensible).
  - COCO:    resample samples (no natural cluster).
"""

import json
import re
import numpy as np
from collections import defaultdict
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
    ("text_instruction_lora_ep1", "frame_lora_ep1"),
    ("frame_gated_lora_ep1",      "text_instruction_lora_ep1"),
    ("frame_gated_lora_ep1",      "frame_lora_ep1"),
    ("text_instruction_lora_ep1", "baseline_lora_ep1"),
    ("frame_lora_ep1",            "baseline_lora_ep1"),
    ("full_method_ep1",           "frame_lora_ep1"),
    ("baseline_lora_ep1",         "zeroshot"),
]

N_BOOT = 10_000
RNG = np.random.default_rng(0)

SCENE_RE = re.compile(r"/(scene\d{4}_\d{2})/")


def domain_of(sample):
    imgs = sample.get("images") or []
    if not imgs:
        return "unknown"
    p = imgs[0]
    if SCENE_RE.search(p):
        return "scannet"
    if "val2017" in p or "/coco" in p.lower():
        return "coco"
    return "other"


def scene_of(sample):
    imgs = sample.get("images") or []
    for p in imgs:
        m = SCENE_RE.search(p)
        if m:
            return m.group(1)
    return None


def load_run(run):
    p = Path(f"results/{run}/viewspatial.json")
    if not p.exists():
        return None
    return {r["id"]: int(r["correct"]) for r in json.loads(p.read_text())["results"]}


def load_bench():
    return [json.loads(l) for l in
            Path("data/processed/viewspatial_test.jsonl").read_text().splitlines()
            if l.strip()]


def build_groups(samples, domain):
    """Return list of lists of sample_ids. Scene clusters for scannet,
    singletons for coco."""
    chosen = [s for s in samples if domain_of(s) == domain]
    if domain == "scannet":
        buckets = defaultdict(list)
        for s in chosen:
            sc = scene_of(s)
            if sc:
                buckets[sc].append(s["id"])
        return list(buckets.values()), len(chosen)
    else:
        return [[s["id"]] for s in chosen], len(chosen)


def paired_boot(a, b, groups):
    k = len(groups)
    groups = [[i for i in g if i in a and i in b] for g in groups]
    groups = [g for g in groups if g]
    n = sum(len(g) for g in groups)
    if n == 0:
        return None
    boot = np.empty(N_BOOT, dtype=np.float64)
    idxs = np.arange(len(groups))
    for r in range(N_BOOT):
        pick = RNG.choice(idxs, size=len(groups), replace=True)
        va = vb = m = 0
        for gi in pick:
            for sid in groups[gi]:
                va += a[sid]; vb += b[sid]; m += 1
        boot[r] = (va - vb) / m * 100
    # point estimate (unresampled)
    all_a = [a[i] for g in groups for i in g]
    all_b = [b[i] for g in groups for i in g]
    delta = (np.mean(all_a) - np.mean(all_b)) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=n, k=len(groups), delta=delta, ci_lo=lo, ci_hi=hi,
                p_pos=float((boot > 0).mean()))


def acc_ci(run, groups):
    r = load_run(run)
    if r is None:
        return None
    groups = [[i for i in g if i in r] for g in groups]
    groups = [g for g in groups if g]
    all_v = [r[i] for g in groups for i in g]
    if not all_v:
        return None
    point = np.mean(all_v) * 100
    boot = np.empty(N_BOOT, dtype=np.float64)
    idxs = np.arange(len(groups))
    for n in range(N_BOOT):
        pick = RNG.choice(idxs, size=len(groups), replace=True)
        vs = [r[i] for gi in pick for i in groups[gi]]
        boot[n] = np.mean(vs) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=len(all_v), k=len(groups), acc=point, ci_lo=lo, ci_hi=hi)


def main():
    samples = load_bench()
    out = []
    for domain, unit_label in [("scannet", "scenes"), ("coco", "samples")]:
        groups, total = build_groups(samples, domain)
        out.append(f"\n## Domain = {domain} (unit: {unit_label}; k={len(groups)}, n={total})\n")
        out.append("### Per-method accuracy with 95% CI")
        out.append("| Method | n | acc | 95% CI |")
        out.append("|---|---:|---:|---|")
        for run, label in RUNS:
            s = acc_ci(run, groups)
            if s is None:
                out.append(f"| {label} | — | — | — |")
            else:
                out.append(f"| {label} | {s['n']} | {s['acc']:.2f} | [{s['ci_lo']:.2f}, {s['ci_hi']:.2f}] |")

        out.append("\n### Paired Δ with 95% CI")
        out.append("| Comparison | n | Δ (pp) | 95% CI | P(Δ>0) |")
        out.append("|---|---:|---:|---|---:|")
        for aa, bb in COMPARISONS:
            a = load_run(aa); b = load_run(bb)
            if a is None or b is None:
                continue
            r = paired_boot(a, b, groups)
            if r is None:
                continue
            la = dict(RUNS)[aa]; lb = dict(RUNS)[bb]
            out.append(f"| {la} − {lb} | {r['n']} | {r['delta']:+.2f} | "
                       f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['p_pos']:.3f} |")

    md = "\n".join(out) + "\n"
    print(md)
    Path("results/bootstrap_ci_domain.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
