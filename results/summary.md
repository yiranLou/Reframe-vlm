## Table 1 — Main results (overall accuracy %)

| Method | ViewSpatial | MMSI | Ego3D |
|---|---:|---:|---:|
| Qwen2.5-VL zero-shot | 36.45 | — | — |
| Qwen2.5-VL + text prompt | 38.27 | — | — |
| Naive LoRA (ep1) | 47.48 | 25.10 | — |
| + frame token (ep1) | 49.12 | 26.60 | — |
| Full (frame + consistency + perm, ep1) | 48.09 | 25.50 | 39.92 |

## Table 3 — ViewSpatial per-frame-type accuracy (%)

| Method | Camera | Person | Overall |
|---|---:|---:|---:|
| Qwen2.5-VL zero-shot | 39.40 | 33.67 | 36.45 |
| Qwen2.5-VL + text prompt | 40.77 | 35.92 | 38.27 |
| Naive LoRA (ep1) | 44.64 | 50.15 | 47.48 |
| + frame token (ep1) | 46.23 | 51.85 | 49.12 |
| Full (frame + consistency + perm, ep1) | 46.01 | 50.05 | 48.09 |

## Table 4 — Frame-switch consistency on ViewSpatial

*996 cross-frame pairs extracted from the test set.*

| Method | FCA ↑ | CR ↓ | Camera Acc | Non-Cam Acc | FG |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL zero-shot | 6.22 | 56.43 | 28.41 | 40.46 | -12.05 |
| Qwen2.5-VL + text prompt | 9.24 | 57.03 | 29.82 | 45.68 | -15.86 |
| Naive LoRA (ep1) | 11.24 | 61.85 | 19.78 | 64.56 | -44.78 |
| + frame token (ep1) | 13.35 | 63.65 | 21.99 | 68.37 | -46.39 |
| Full (frame + consistency + perm, ep1) | 10.34 | 62.85 | 21.59 | 61.95 | -40.36 |
