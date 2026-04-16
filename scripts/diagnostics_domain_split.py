"""
Domain-stratified frame-switch diagnostics: FCA / CR / FG split by ScanNet
in-domain vs COCO OOD.

For paper Table 4-domain. Uses same pair-extraction logic as
``summarize_results.py`` but partitions the 996 cross-frame pairs by the
domain of their image (ScanNet path → "scannet"; COCO val2017 path → "coco").

Output:
  - prints a markdown table to stdout
  - writes to results/diagnostics_domain_split.md
"""

import json
import re
from collections import defaultdict
from pathlib import Path


SCENE_RE = re.compile(r"/(scene\d{4}_\d{2})/")

RUNS = [
    ("zeroshot",                  "Qwen2.5-VL zero-shot"),
    ("prompt_baseline",           "+ text prompt (infer)"),
    ("baseline_lora_ep1",         "Naive LoRA"),
    ("text_instruction_lora_ep1", "LoRA + text-instr SFT"),
    ("frame_lora_ep1",            "Frame LoRA"),
    ("full_method_ep1",           "Full Method"),
]


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
    return {r["id"]: int(r["correct"]) for r in
            json.loads(p.read_text())["results"]}


def build_pairs(samples):
    """Same logic as summarize_results.py: pair samples sharing identical
    image paths whose frame_types differ. Returns list of (sample_a, sample_b)."""
    by_img = defaultdict(list)
    for s in samples:
        by_img[tuple(s.get("images", []))].append(s)
    pairs = []
    seen = set()
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


def stratified_diagnostics(pairs, run_map, label):
    """Compute FCA / CR / Camera / Non-Cam / FG for one method, on a given
    pair list (already domain-filtered)."""
    n = len(pairs)
    if n == 0:
        return None
    both_ok = contra = 0
    cam_c = cam_n = other_c = other_n = 0
    used = 0
    for sa, sb in pairs:
        pa = run_map.get(sa["id"])
        pb = run_map.get(sb["id"])
        if pa is None or pb is None:
            continue
        used += 1
        if pa and pb:
            both_ok += 1
        if pa != pb:
            contra += 1
        for s, pr in ((sa, pa), (sb, pb)):
            if s["frame_type"] == "camera":
                cam_n += 1; cam_c += pr
            else:
                other_n += 1; other_c += pr
    if used == 0:
        return None
    fca = both_ok / used * 100
    cr = contra / used * 100
    cam_acc = cam_c / cam_n * 100 if cam_n else 0
    other_acc = other_c / other_n * 100 if other_n else 0
    return dict(label=label, n=used, fca=fca, cr=cr,
                camera_acc=cam_acc, non_cam_acc=other_acc,
                fg=cam_acc - other_acc)


def main():
    samples = load_bench()
    all_pairs = build_pairs(samples)

    by_domain = {"scannet": [], "coco": []}
    for sa, sb in all_pairs:
        d_a = domain_of(sa)
        d_b = domain_of(sb)
        if d_a == d_b and d_a in by_domain:
            by_domain[d_a].append((sa, sb))

    out = []
    out.append("## Frame-switch diagnostics by domain\n")
    out.append(f"Total cross-frame pairs: **{len(all_pairs)}**  "
               f"(ScanNet pairs: **{len(by_domain['scannet'])}**, "
               f"COCO pairs: **{len(by_domain['coco'])}**)\n")

    for domain in ("scannet", "coco"):
        pairs = by_domain[domain]
        out.append(f"\n### Domain = {domain}  (n_pairs = {len(pairs)})\n")
        out.append("| Method | n | FCA ↑ | CR ↓ | Camera | Non-Cam | FG |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for run_name, label in RUNS:
            run_map = load_run(run_name)
            if run_map is None:
                continue
            r = stratified_diagnostics(pairs, run_map, label)
            if r is None:
                continue
            out.append(
                f"| {r['label']} | {r['n']} | {r['fca']:.2f} | {r['cr']:.2f} | "
                f"{r['camera_acc']:.2f} | {r['non_cam_acc']:.2f} | "
                f"{r['fg']:+.2f} |"
            )

    md = "\n".join(out) + "\n"
    print(md)
    Path("results/diagnostics_domain_split.md").write_text(md, encoding="utf-8")
    print("\nsaved to results/diagnostics_domain_split.md")


if __name__ == "__main__":
    main()
