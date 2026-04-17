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

*996 cross-frame pairs extracted from the test set.*

| Method | FCA ↑ | CR ↓ | PDR ↓ | Camera Acc | Non-Cam Acc | FG |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL zero-shot | 6.22 | 24.60 | 56.43 | 28.41 | 40.46 | -12.05 |
| Qwen2.5-VL + text prompt (inference only) | 9.24 | 25.40 | 57.03 | 29.82 | 45.68 | -15.86 |
| Naive LoRA (ep1) | 11.24 | 32.53 | 61.85 | 19.78 | 64.56 | -44.78 |
| LoRA + text instruction SFT (ep1) | 19.28 | 25.40 | 62.55 | 29.02 | 72.09 | -43.07 |
| + frame token (ep1) | 14.26 | 33.94 | 68.17 | 22.69 | 74.00 | -51.31 |
| Full (frame + consistency + perm, ep1) | 11.45 | 32.73 | 68.57 | 22.49 | 68.98 | -46.49 |
