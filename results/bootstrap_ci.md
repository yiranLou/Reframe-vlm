## Per-method accuracy with 95% CI

| Method | ViewSpatial | MMSI | Ego3D |
|---|---:|---:|---:|
| Qwen2.5-VL zero-shot | 36.45 [35.19, 37.71] | — | — |
| Qwen2.5-VL + text prompt (inference only) | 38.27 [37.03, 39.55] | — | — |
| Naive LoRA (ep1) | 47.48 [46.20, 48.77] | 25.10 [22.50, 27.80] | — |
| LoRA + text instruction SFT (ep1) | 50.82 [49.51, 52.08] | 25.70 [23.00, 28.40] | — |
| + frame token (ep1) | 52.75 [51.44, 54.04] | 26.60 [23.90, 29.40] | — |
| Frame-Gated LoRA (gate only, ep1) | — | — | — |
| Token + Gated LoRA (ep1) | — | — | — |
| Full (frame + consistency + perm, ep1) | 53.10 [51.80, 54.39] | 25.50 [22.80, 28.20] | 42.02 [38.47, 45.57] |

## Paired bootstrap: Δ = A − B (pp), 95% CI, P(Δ>0)

| Comparison | Benchmark | n | Δ | 95% CI | P(Δ>0) |
|---|---|---:|---:|---|---:|
| LoRA + text instruction SFT (ep1) − + frame token (ep1) | viewspatial | 5712 | -1.93 | [-3.03, -0.84] | 0.000 |
| LoRA + text instruction SFT (ep1) − + frame token (ep1) | mmsi | 1000 | -0.90 | [-3.20, +1.40] | 0.211 |
| LoRA + text instruction SFT (ep1) − Naive LoRA (ep1) | viewspatial | 5712 | +3.34 | [+2.33, +4.39] | 1.000 |
| LoRA + text instruction SFT (ep1) − Naive LoRA (ep1) | mmsi | 1000 | +0.60 | [-1.70, +2.90] | 0.682 |
| + frame token (ep1) − Naive LoRA (ep1) | viewspatial | 5712 | +5.27 | [+4.48, +6.09] | 1.000 |
| + frame token (ep1) − Naive LoRA (ep1) | mmsi | 1000 | +1.50 | [-0.30, +3.30] | 0.943 |
| Full (frame + consistency + perm, ep1) − + frame token (ep1) | viewspatial | 5712 | +0.35 | [-0.63, +1.31] | 0.754 |
| Full (frame + consistency + perm, ep1) − + frame token (ep1) | mmsi | 1000 | -1.10 | [-3.30, +1.10] | 0.159 |
| Naive LoRA (ep1) − Qwen2.5-VL zero-shot | viewspatial | 5712 | +11.03 | [+9.51, +12.54] | 1.000 |
