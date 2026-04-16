"""
Per-frame-type accuracy analysis.

Produces Table 3 data:
| Method | Camera | Person | Object | Overall |

Expected pattern: ReFrame-VLM should show largest gains on
Person and Object frames, since frame-conditioned LoRA specifically
targets non-camera reference frames.

Usage:
    python src/eval/frame_type_analysis.py \
        --results results/full_viewspatial.json \
        --benchmark_data data/processed/viewspatial_test.jsonl \
        --output results/frame_type_analysis.json
"""

import argparse
import json
import os
from collections import defaultdict


def analyze_by_frame_type(results_path, benchmark_data_path):
    """
    Split results by frame_type and compute per-type accuracy.
    """
    # Load results
    with open(results_path, encoding="utf-8") as f:
        result_data = json.load(f)

    results = result_data.get("results", result_data)
    pred_map = {r["id"]: r for r in results}

    # Load benchmark data (for frame_type info if not in results)
    benchmark = []
    if benchmark_data_path:
        if benchmark_data_path.endswith(".jsonl"):
            with open(benchmark_data_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        benchmark.append(json.loads(line))
        else:
            with open(benchmark_data_path, encoding="utf-8") as f:
                data = json.load(f)
                benchmark = data if isinstance(data, list) else data.get("data", [])

    # Merge frame_type from benchmark into results
    bench_map = {s["id"]: s for s in benchmark}

    frame_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = 0
    overall_total = 0

    for r in results:
        # Get frame_type from result or benchmark
        ft = r.get("frame_type", "unknown")
        if ft == "unknown" and r["id"] in bench_map:
            ft = bench_map[r["id"]].get("frame_type", "unknown")

        is_correct = r.get("correct", False)
        frame_stats[ft]["total"] += 1
        frame_stats[ft]["correct"] += int(is_correct)
        overall_total += 1
        overall_correct += int(is_correct)

    # Print table
    print(f"\n{'='*60}")
    print(f"Accuracy by Frame Type")
    print(f"{'='*60}")
    print(f"{'Frame Type':<15} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print(f"{'-'*45}")

    analysis = {}
    for ft in ["camera", "person", "object", "world", "unknown"]:
        stats = frame_stats.get(ft)
        if not stats or stats["total"] == 0:
            continue
        acc = stats["correct"] / stats["total"] * 100
        print(f"{ft:<15} {acc:>9.2f}% {stats['correct']:>10} {stats['total']:>10}")
        analysis[ft] = {
            "accuracy": round(acc, 2),
            "correct": stats["correct"],
            "total": stats["total"],
        }

    overall_acc = overall_correct / overall_total * 100 if overall_total > 0 else 0
    print(f"{'-'*45}")
    print(f"{'Overall':<15} {overall_acc:>9.2f}% {overall_correct:>10} {overall_total:>10}")

    analysis["overall"] = {
        "accuracy": round(overall_acc, 2),
        "correct": overall_correct,
        "total": overall_total,
    }

    return analysis


def compare_methods(result_files, benchmark_data_path, method_names=None):
    """
    Compare multiple methods' per-frame-type accuracy.
    Produces a comparison table for the paper.

    Args:
        result_files: list of result JSON paths
        benchmark_data_path: benchmark data for frame_type info
        method_names: list of method names (optional)
    """
    if method_names is None:
        method_names = [os.path.basename(f).replace(".json", "") for f in result_files]

    all_analyses = {}
    for name, fpath in zip(method_names, result_files):
        print(f"\n--- {name} ---")
        analysis = analyze_by_frame_type(fpath, benchmark_data_path)
        all_analyses[name] = analysis

    # Print comparison table
    frame_types = ["camera", "person", "object", "overall"]
    print(f"\n\n{'='*70}")
    print(f"Comparison Table (for paper)")
    print(f"{'='*70}")

    header = f"{'Method':<25}"
    for ft in frame_types:
        header += f" {ft.capitalize():>10}"
    print(header)
    print("-" * 70)

    for name in method_names:
        row = f"{name:<25}"
        for ft in frame_types:
            acc = all_analyses[name].get(ft, {}).get("accuracy", "-")
            if isinstance(acc, (int, float)):
                row += f" {acc:>9.2f}%"
            else:
                row += f" {str(acc):>10}"
        print(row)

    return all_analyses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, nargs="+",
                        help="Result JSON file(s)")
    parser.add_argument("--benchmark_data", default=None,
                        help="Benchmark data with frame_type")
    parser.add_argument("--method_names", nargs="+", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if len(args.results) == 1:
        analysis = analyze_by_frame_type(args.results[0], args.benchmark_data)
    else:
        analysis = compare_methods(
            args.results, args.benchmark_data, args.method_names
        )

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to {args.output}")


if __name__ == "__main__":
    main()
