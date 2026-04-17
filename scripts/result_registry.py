"""Canonical result-run registry for analysis scripts.

Some methods have corrected re-evaluation outputs only for a subset of
benchmarks, e.g. ``frame_lora_ep1_refix/viewspatial.json`` while MMSI still
lives in ``frame_lora_ep1/mmsi.json``. Analysis scripts should always read the
best available file per benchmark instead of hard-coding a single directory.
"""

from pathlib import Path


RESULTS_ROOT = Path("results")
BENCHES = ("viewspatial", "mmsi", "ego3d")

RUNS = [
    ("zeroshot", "Qwen2.5-VL zero-shot"),
    ("prompt_baseline", "Qwen2.5-VL + text prompt (inference only)"),
    ("baseline_lora_ep1", "Naive LoRA (ep1)"),
    ("text_instruction_lora_ep1", "LoRA + text instruction SFT (ep1)"),
    ("frame_lora_ep1", "+ frame token (ep1)"),
    ("frame_gated_lora_ep1", "Frame-Gated LoRA (gate only, ep1)"),
    ("token_gated_lora_ep1", "Token + Gated LoRA (ep1)"),
    ("full_method_ep1", "Full (frame + consistency + perm, ep1)"),
]
RUN_LABELS = dict(RUNS)

RUN_CANDIDATES = {
    "frame_lora_ep1": ("frame_lora_ep1_refix", "frame_lora_ep1"),
    "full_method_ep1": ("full_method_ep1_refix", "full_method_ep1"),
}


def candidate_dirs(run):
    return RUN_CANDIDATES.get(run, (run,))


def resolve_result_path(run, bench):
    for candidate in candidate_dirs(run):
        path = RESULTS_ROOT / candidate / f"{bench}.json"
        if path.exists():
            return path
    return None


def has_any_results(run):
    return any(resolve_result_path(run, bench) is not None for bench in BENCHES)


def active_runs():
    return [run for run, _ in RUNS if has_any_results(run)]
