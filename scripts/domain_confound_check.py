"""
Check whether the COCO / ScanNet domain split is confounded with question_type
and/or frame_type in ViewSpatial-Bench.

If COCO subset is ~100% Camera-Object-View-Orientation questions, the claim
"+2.86pp OOD" must be worded as "+2.86pp on OOD object-view orientation" —
not as blanket out-of-distribution generalization.

We emit:
  1. A domain × question_type × frame_type contingency table.
  2. Per-(domain, question_type) accuracy for every run, so we can see where
     Frame LoRA's gain actually lives.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


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


def load_bench():
    return [json.loads(l) for l in
            Path("data/processed/viewspatial_test.jsonl").read_text().splitlines()
            if l.strip()]


def load_run(run):
    p = Path(f"results/{run}/viewspatial.json")
    if not p.exists():
        return None
    return {r["id"]: int(r["correct"]) for r in json.loads(p.read_text())["results"]}


def main():
    samples = load_bench()
    print(f"\ntotal test samples: {len(samples)}")

    # 1. Contingency table
    print("\n=== Table A: Domain × question_type × frame_type (n samples) ===")
    counts = defaultdict(lambda: defaultdict(Counter))
    for s in samples:
        d = domain_of(s)
        qt = s.get("question_type", "?")
        ft = s.get("frame_type", "?")
        counts[d][qt][ft] += 1

    for dom, qts in counts.items():
        dom_total = sum(sum(c.values()) for c in qts.values())
        print(f"\n  {dom.upper()} (n={dom_total})")
        for qt, fts in qts.items():
            total = sum(fts.values())
            breakdown = ", ".join(f"{k}={v}" for k, v in fts.items())
            print(f"    {qt:55s} {total:>5d}   [{breakdown}]")

    # 2. Per-(domain, question_type) accuracy across runs
    runs = [
        ("zeroshot",                  "ZS"),
        ("prompt_baseline",           "Pr"),
        ("baseline_lora_ep1",         "Naive"),
        ("text_instruction_lora_ep1", "TextInstr"),
        ("frame_lora_ep1",            "Frame"),
        ("full_method_ep1",           "Full"),
    ]
    run_maps = {r: load_run(r) for r, _ in runs}

    print("\n=== Table B: Accuracy by domain × question_type (%) ===")
    # gather groups
    groups = defaultdict(list)
    for s in samples:
        key = (domain_of(s), s.get("question_type", "?"))
        groups[key].append(s["id"])

    header = "| domain | question_type | n | " + " | ".join(lbl for _, lbl in runs) + " |"
    print(header)
    print("|---|---|---:|" + "|".join(["---:"] * len(runs)) + "|")
    for (dom, qt), ids in sorted(groups.items()):
        row = [dom, qt, str(len(ids))]
        for rn, _ in runs:
            m = run_maps[rn]
            if m is None:
                row.append("—")
            else:
                vals = [m[i] for i in ids if i in m]
                row.append(f"{(sum(vals)/len(vals)*100):.2f}" if vals else "—")
        print("| " + " | ".join(row) + " |")

    # 3. Where does Frame LoRA win over Naive LoRA, per group?
    print("\n=== Table C: Frame LoRA − Naive LoRA per group (pp) ===")
    na = run_maps["baseline_lora_ep1"]; fr = run_maps["frame_lora_ep1"]
    print("| domain | question_type | n | Δ | Naive | Frame |")
    print("|---|---|---:|---:|---:|---:|")
    for (dom, qt), ids in sorted(groups.items(), key=lambda x: -len(x[1])):
        ids2 = [i for i in ids if i in na and i in fr]
        if not ids2:
            continue
        na_acc = sum(na[i] for i in ids2) / len(ids2) * 100
        fr_acc = sum(fr[i] for i in ids2) / len(ids2) * 100
        print(f"| {dom} | {qt} | {len(ids2)} | {fr_acc-na_acc:+.2f} | {na_acc:.2f} | {fr_acc:.2f} |")

    # 4. Where does text-instruction SFT win over Naive LoRA, per group?
    ti = run_maps.get("text_instruction_lora_ep1")
    if ti is not None:
        print("\n=== Table D: text-instr SFT − Naive LoRA per group (pp) ===")
        print("| domain | question_type | n | Δ | Naive | TextInstr |")
        print("|---|---|---:|---:|---:|---:|")
        for (dom, qt), ids in sorted(groups.items(), key=lambda x: -len(x[1])):
            ids2 = [i for i in ids if i in na and i in ti]
            if not ids2:
                continue
            na_acc = sum(na[i] for i in ids2) / len(ids2) * 100
            ti_acc = sum(ti[i] for i in ids2) / len(ids2) * 100
            print(f"| {dom} | {qt} | {len(ids2)} | {ti_acc-na_acc:+.2f} | "
                  f"{na_acc:.2f} | {ti_acc:.2f} |")

    # 5. Direct head-to-head: text-instr vs Frame LoRA per group
    if ti is not None:
        print("\n=== Table E: text-instr SFT − Frame LoRA per group (pp) ===")
        print("| domain | question_type | n | Δ | Frame | TextInstr |")
        print("|---|---|---:|---:|---:|---:|")
        for (dom, qt), ids in sorted(groups.items(), key=lambda x: -len(x[1])):
            ids2 = [i for i in ids if i in fr and i in ti]
            if not ids2:
                continue
            fr_acc = sum(fr[i] for i in ids2) / len(ids2) * 100
            ti_acc = sum(ti[i] for i in ids2) / len(ids2) * 100
            print(f"| {dom} | {qt} | {len(ids2)} | {ti_acc-fr_acc:+.2f} | "
                  f"{fr_acc:.2f} | {ti_acc:.2f} |")


if __name__ == "__main__":
    main()
