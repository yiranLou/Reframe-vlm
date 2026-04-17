"""Domain-stratified frame-switch diagnostics on ViewSpatial."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from result_registry import RUN_LABELS, resolve_result_path


DEFAULT_BENCH_PATH = "data/processed/viewspatial_test.jsonl"
SCENE_RE = re.compile(r"/(scene\d{4}_\d{2})/")
CHOICE_RE = re.compile(r"^\s*\(?([A-Da-d])[\).:]?\s*(.*)$")
FOCUSED_RUNS = [
    "zeroshot",
    "prompt_baseline",
    "baseline_lora_ep1",
    "text_instruction_lora_ep1",
    "frame_lora_ep1",
    "full_method_ep1",
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
    return {row["id"]: row for row in json.loads(path.read_text())["results"]}


def build_pairs(samples):
    by_img = defaultdict(list)
    for sample in samples:
        by_img[tuple(sample.get("images", []))].append(sample)
    pairs = []
    seen = set()
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

    pred_rel_a = pred_relation(pred_a.get("pred"), sample_a.get("choices"))
    pred_rel_b = pred_relation(pred_b.get("pred"), sample_b.get("choices"))
    a_correct = bool(pred_a.get("correct"))
    b_correct = bool(pred_b.get("correct"))

    if a_correct and not b_correct:
        return pred_rel_b == gt_a
    if b_correct and not a_correct:
        return pred_rel_a == gt_b
    if not a_correct and not b_correct:
        return pred_rel_a == gt_b and pred_rel_b == gt_a
    return False


def stratified_diagnostics(pairs, pred_map, label):
    if not pairs:
        return None
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
        return None
    cam_acc = cam_c / cam_n * 100 if cam_n else 0.0
    other_acc = other_c / other_n * 100 if other_n else 0.0
    return {
        "label": label,
        "n": used,
        "fca": both_ok / used * 100,
        "cr": contradictions / used * 100,
        "pdr": disagreements / used * 100,
        "camera_acc": cam_acc,
        "non_cam_acc": other_acc,
        "fg": cam_acc - other_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_path", default=DEFAULT_BENCH_PATH)
    args = parser.parse_args()

    samples = load_bench(args.bench_path)
    all_pairs = build_pairs(samples)

    by_domain = {"scannet": [], "coco": []}
    for sample_a, sample_b in all_pairs:
        dom_a = domain_of(sample_a)
        dom_b = domain_of(sample_b)
        if dom_a == dom_b and dom_a in by_domain:
            by_domain[dom_a].append((sample_a, sample_b))

    out = []
    out.append("## Frame-switch diagnostics by domain\n")
    out.append(
        f"Total cross-frame pairs: **{len(all_pairs)}**  (ScanNet pairs: **{len(by_domain['scannet'])}**, COCO pairs: **{len(by_domain['coco'])}**)\n"
    )

    for domain in ("scannet", "coco"):
        pairs = by_domain[domain]
        out.append(f"\n### Domain = {domain}  (n_pairs = {len(pairs)})\n")
        out.append("| Method | n | FCA ↑ | CR ↓ | PDR ↓ | Camera | Non-Cam | FG |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for run in FOCUSED_RUNS:
            pred_map = load_run(run)
            if pred_map is None:
                continue
            stats = stratified_diagnostics(pairs, pred_map, RUN_LABELS[run])
            if stats is None:
                continue
            out.append(
                f"| {stats['label']} | {stats['n']} | {stats['fca']:.2f} | {stats['cr']:.2f} | {stats['pdr']:.2f} | "
                f"{stats['camera_acc']:.2f} | {stats['non_cam_acc']:.2f} | {stats['fg']:+.2f} |"
            )

    md = "\n".join(out) + "\n"
    print(md)
    Path("results/diagnostics_domain_split.md").write_text(md, encoding="utf-8")
    print("\nsaved to results/diagnostics_domain_split.md")


if __name__ == "__main__":
    main()
