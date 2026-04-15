## Per-method accuracy with 95% CI

| Method | ViewSpatial | MMSI | Ego3D |
|---|---:|---:|---:|
| Qwen2.5-VL zero-shot | 36.45 [35.19, 37.71] | — | — |
| + text prompt | 38.27 [37.03, 39.55] | — | — |
| Naive LoRA | 47.48 [46.20, 48.77] | 25.10 [22.50, 27.80] | — |
| Frame LoRA | 49.12 [47.81, 50.40] | 26.60 [23.90, 29.30] | — |
| Full Method | 48.09 [46.78, 49.39] | 25.50 [22.90, 28.30] | 42.02 [38.47, 45.70] |

## Paired bootstrap: Δ = A − B (pp), 95% CI, P(Δ>0)

| Comparison | Benchmark | n | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| Frame LoRA − Naive LoRA | viewspatial | 5712 | +1.65 | [+0.93, +2.36] | 1.000 |
| Frame LoRA − Naive LoRA | mmsi | 1000 | +1.50 | [-0.30, +3.30] | 0.945 |
| Frame LoRA − + text prompt | viewspatial | 5712 | +10.85 | [+9.40, +12.43] | 1.000 |
| Full Method − Frame LoRA | viewspatial | 5712 | -1.03 | [-2.00, -0.05] | 0.019 |
| Full Method − Frame LoRA | mmsi | 1000 | -1.10 | [-3.40, +1.20] | 0.155 |
| Naive LoRA − Qwen2.5-VL zero-shot | viewspatial | 5712 | +11.03 | [+9.49, +12.52] | 1.000 |
