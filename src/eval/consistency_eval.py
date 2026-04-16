"""
Frame-Switch Consistency Analysis.

Computes 3 metrics from ViewSpatial-Bench paired predictions:
1. Frame Consistency Accuracy (FCA): Both paired questions answered correctly
2. Contradiction Rate (CR): Paired answers contradict each other
3. Frame Gap (FG): Camera accuracy - Non-camera accuracy

These metrics elevate the paper from "score chasing" to "mechanism verification".
No extra annotation needed - computed from existing benchmark data.

Usage:
    python src/eval/consistency_eval.py \
        --results results/full_viewspatial.json \
        --benchmark_data data/processed/viewspatial_test.jsonl \
        --output results/consistency_analysis.json
"""

import argparse
import json
import re
from collections import defaultdict


CHOICE_RE = re.compile(r"^\s*\(?([A-Da-d])[\).:]?\s*(.*)$")


def _normalize_relation_text(text):
    """Normalize an answer/prediction to relation content, not option letter."""
    text = str(text or "").strip()
    m = CHOICE_RE.match(text)
    if m and m.group(2):
        text = m.group(2)
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9. ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _first_choice_letter(text):
    m = CHOICE_RE.match(str(text or "").strip())
    return m.group(1).upper() if m else None


def _choice_relation(choices, letter):
    if not choices or not letter:
        return None
    idx = ord(letter.upper()) - ord("A")
    if idx < 0 or idx >= len(choices):
        return None
    return _normalize_relation_text(choices[idx])


def _target_relation(answer, choices=None):
    letter = _first_choice_letter(answer)
    mapped = _choice_relation(choices, letter)
    return mapped or _normalize_relation_text(answer)


def _pred_relation(pred, choices=None):
    letter = _first_choice_letter(pred)
    mapped = _choice_relation(choices, letter)
    return mapped or _normalize_relation_text(pred)


def load_results(path):
    """Load evaluation results."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "results" in data:
        return data["results"]
    return data


def load_benchmark(path):
    """Load benchmark data with pair information."""
    samples = []
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                samples = data
            elif "data" in data:
                samples = data["data"]
    return samples


def find_pairs(benchmark_data):
    """
    Extract all cross-frame pairs from benchmark data.
    Returns list of (sample_a, sample_b) tuples.
    """
    id_to_sample = {s["id"]: s for s in benchmark_data}
    pairs = []
    seen = set()

    for sample in benchmark_data:
        pair_id = sample.get("pair_id")
        if not pair_id or sample["id"] in seen:
            continue

        pair_sample = id_to_sample.get(pair_id)
        if not pair_sample or pair_sample["id"] in seen:
            continue

        # Verify different frame types
        ft_a = sample.get("frame_type", "camera")
        ft_b = pair_sample.get("frame_type", "camera")
        if ft_a == ft_b:
            continue

        pairs.append((sample, pair_sample))
        seen.add(sample["id"])
        seen.add(pair_sample["id"])

    return pairs


def is_contradiction(
    pred_a,
    gt_a,
    pred_b,
    gt_b,
    correct_a,
    correct_b,
    choices_a=None,
    choices_b=None,
):
    """
    Detect if paired answers are contradictory.

    A contradiction occurs when:
    - The two frames have different correct relations, AND
    - A prediction uses the relation that is correct for the opposite frame.

    Example: Camera says "left" (correct), Person also says "left"
    but geometrically it should be "right" from person's perspective.
    """
    gt_a_rel = _target_relation(gt_a, choices_a)
    gt_b_rel = _target_relation(gt_b, choices_b)
    if not gt_a_rel or not gt_b_rel or gt_a_rel == gt_b_rel:
        return False

    pred_a_rel = _pred_relation(pred_a, choices_a)
    pred_b_rel = _pred_relation(pred_b, choices_b)

    if correct_a and not correct_b:
        return pred_b_rel == gt_a_rel
    if correct_b and not correct_a:
        return pred_a_rel == gt_b_rel
    if not correct_a and not correct_b:
        return pred_a_rel == gt_b_rel and pred_b_rel == gt_a_rel
    return False


def compute_metrics(results, benchmark_data):
    """
    Compute FCA, CR, and FG from results and benchmark pairs.
    """
    pred_map = {}
    for r in results:
        pred_map[r["id"]] = r

    pairs = find_pairs(benchmark_data)

    if not pairs:
        print("WARNING: No pairs found in benchmark data.")
        print("  Check that benchmark data has 'pair_id' fields.")
        return None

    total_pairs = 0
    both_correct = 0
    contradictions = 0
    paired_disagreements = 0

    # Per-frame accuracy tracking
    frame_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for sa, sb in pairs:
        pred_a = pred_map.get(sa["id"])
        pred_b = pred_map.get(sb["id"])

        if not pred_a or not pred_b:
            continue

        total_pairs += 1

        a_correct = pred_a.get("correct", False)
        b_correct = pred_b.get("correct", False)

        # FCA
        if a_correct and b_correct:
            both_correct += 1

        # CR
        if is_contradiction(
            pred_a.get("pred", ""), sa["answer"],
            pred_b.get("pred", ""), sb["answer"],
            a_correct, b_correct,
            sa.get("choices"), sb.get("choices"),
        ):
            contradictions += 1
        if a_correct != b_correct:
            paired_disagreements += 1

        # Per-frame stats
        for sample, pred in [(sa, pred_a), (sb, pred_b)]:
            ft = sample.get("frame_type", "camera")
            frame_stats[ft]["total"] += 1
            frame_stats[ft]["correct"] += int(pred.get("correct", False))

    if total_pairs == 0:
        print("WARNING: No valid pairs with predictions found.")
        return None

    # Compute metrics
    fca = both_correct / total_pairs * 100
    cr = contradictions / total_pairs * 100
    pdr = paired_disagreements / total_pairs * 100

    cam_stats = frame_stats.get("camera", {"correct": 0, "total": 0})
    cam_acc = (
        cam_stats["correct"] / cam_stats["total"] * 100
        if cam_stats["total"] > 0 else 0
    )

    non_cam_correct = sum(
        v["correct"] for k, v in frame_stats.items() if k != "camera"
    )
    non_cam_total = sum(
        v["total"] for k, v in frame_stats.items() if k != "camera"
    )
    non_cam_acc = (
        non_cam_correct / non_cam_total * 100
        if non_cam_total > 0 else 0
    )

    fg = cam_acc - non_cam_acc

    metrics = {
        "total_pairs": total_pairs,
        "frame_consistency_accuracy": round(fca, 2),
        "contradiction_rate": round(cr, 2),
        "paired_disagreement_rate": round(pdr, 2),
        "camera_accuracy": round(cam_acc, 2),
        "non_camera_accuracy": round(non_cam_acc, 2),
        "frame_gap": round(fg, 2),
        "per_frame_stats": {
            k: {
                "accuracy": round(v["correct"] / v["total"] * 100, 2)
                if v["total"] > 0 else 0,
                "correct": v["correct"],
                "total": v["total"],
            }
            for k, v in frame_stats.items()
        },
    }

    print(f"\n{'='*50}")
    print(f"Frame Consistency Analysis")
    print(f"{'='*50}")
    print(f"Total pairs:                    {total_pairs}")
    print(f"Frame Consistency Accuracy (FCA): {fca:.2f}%")
    print(f"Contradiction Rate (CR):          {cr:.2f}%")
    print(f"Paired Disagreement Rate (PDR):   {pdr:.2f}%")
    print(f"Camera Accuracy:                  {cam_acc:.2f}%")
    print(f"Non-Camera Accuracy:              {non_cam_acc:.2f}%")
    print(f"Frame Gap (FG):                   {fg:.2f}%")
    print(f"\nPer-frame breakdown:")
    for ft, stats in sorted(frame_stats.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {ft}: {acc:.2f}% ({stats['correct']}/{stats['total']})")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Path to evaluation results JSON")
    parser.add_argument("--benchmark_data", required=True,
                        help="Path to benchmark data with pair_id")
    parser.add_argument("--output", default=None,
                        help="Output path for metrics JSON")
    args = parser.parse_args()

    results = load_results(args.results)
    benchmark_data = load_benchmark(args.benchmark_data)

    metrics = compute_metrics(results, benchmark_data)

    if metrics and args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    import os
    main()
