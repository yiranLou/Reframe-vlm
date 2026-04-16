## Per-method accuracy with 95% CI

| Method | ViewSpatial | MMSI | Ego3D |
|---|---:|---:|---:|
| Qwen2.5-VL zero-shot | 36.45 [35.19, 37.71] | — | — |
| + text prompt | 38.27 [37.03, 39.55] | — | — |
| Naive LoRA | 47.48 [46.20, 48.77] | 25.10 [22.50, 27.80] | — |
| LoRA+text-instr SFT | 50.82 [49.51, 52.08] | 25.70 [23.00, 28.40] | — |
| Frame LoRA | 49.12 [47.83, 50.46] | 26.60 [23.90, 29.40] | — |
| Full Method | 48.09 [46.76, 49.37] | 25.50 [22.80, 28.20] | 42.02 [38.47, 45.57] |

## Paired bootstrap: Δ = A − B (pp), 95% CI, P(Δ>0)

| Comparison | Benchmark | n | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| Frame LoRA − LoRA+text-instr SFT | viewspatial | 5712 | -1.70 | [-2.73, -0.68] | 0.000 |
| Frame LoRA − LoRA+text-instr SFT | mmsi | 1000 | +0.90 | [-1.40, +3.20] | 0.763 |
| Frame LoRA − Naive LoRA | viewspatial | 5712 | +1.65 | [+0.95, +2.35] | 1.000 |
| Frame LoRA − Naive LoRA | mmsi | 1000 | +1.50 | [-0.30, +3.30] | 0.943 |
| LoRA+text-instr SFT − Naive LoRA | viewspatial | 5712 | +3.34 | [+2.33, +4.41] | 1.000 |
| LoRA+text-instr SFT − Naive LoRA | mmsi | 1000 | +0.60 | [-1.70, +2.80] | 0.691 |
| Frame LoRA − + text prompt | viewspatial | 5712 | +10.85 | [+9.31, +12.34] | 1.000 |
| Full Method − Frame LoRA | viewspatial | 5712 | -1.03 | [-2.01, -0.05] | 0.017 |
| Full Method − Frame LoRA | mmsi | 1000 | -1.10 | [-3.30, +1.20] | 0.157 |
| Naive LoRA − Qwen2.5-VL zero-shot | viewspatial | 5712 | +11.03 | [+9.52, +12.54] | 1.000 |
