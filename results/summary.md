## Table 1 — Main results (overall accuracy %)

| Method | ViewSpatial | MMSI | Ego3D |
|---|---:|---:|---:|
| Qwen2.5-VL zero-shot | 36.45 | — | — |
| Qwen2.5-VL + text prompt (inference only) | 38.27 | — | — |
| Naive LoRA (ep1) | 47.48 | 25.10 | — |
| LoRA + text instruction SFT (ep1) | 50.82 | 25.70 | — |
| + frame token (ep1) | 52.75 | 26.60 | — |
| Full (frame + consistency + perm, ep1) | 53.10 | 25.50 | 39.92 |

## Table 3 — ViewSpatial per-frame-type accuracy (%)

| Method | Camera | Person | Overall |
|---|---:|---:|---:|
| Qwen2.5-VL zero-shot | 39.40 | 33.67 | 36.45 |
| Qwen2.5-VL + text prompt (inference only) | 40.77 | 35.92 | 38.27 |
| Naive LoRA (ep1) | 44.64 | 50.15 | 47.48 |
| LoRA + text instruction SFT (ep1) | 46.70 | 54.71 | 50.82 |
| + frame token (ep1) | 47.60 | 57.59 | 52.75 |
| Full (frame + consistency + perm, ep1) | 48.18 | 57.73 | 53.10 |

## Table 4 — Frame-switch consistency on ViewSpatial

*Benchmark metadata unavailable locally. Re-run with `--bench_path` pointing at the unified ViewSpatial test jsonl to compute FCA / CR / PDR / FG.*
