
## Domain = scannet (unit: scenes; k=279, n=2878)

### Per-method accuracy with 95% CI
| Method | n | acc | 95% CI |
|---|---:|---:|---|
| Qwen2.5-VL zero-shot | 2878 | 38.08 | [36.08, 40.09] |
| Qwen2.5-VL + text prompt (inference only) | 2878 | 39.54 | [37.47, 41.67] |
| Naive LoRA (ep1) | 2878 | 54.41 | [52.15, 56.58] |
| LoRA + text instruction SFT (ep1) | 2878 | 55.39 | [53.01, 57.79] |
| + frame token (ep1) | 2878 | 58.69 | [56.54, 60.82] |
| Frame-Gated LoRA (gate only, ep1) | — | — | — |
| Token + Gated LoRA (ep1) | — | — | — |
| Full (frame + consistency + perm, ep1) | 2878 | 60.53 | [58.63, 62.40] |

### Paired Δ with 95% CI
| Comparison | n | Δ (pp) | 95% CI | P(Δ>0) |
|---|---:|---:|---|---:|
| LoRA + text instruction SFT (ep1) − + frame token (ep1) | 2878 | -3.30 | [-4.98, -1.62] | 0.000 |
| LoRA + text instruction SFT (ep1) − Naive LoRA (ep1) | 2878 | +0.97 | [-0.68, +2.59] | 0.877 |
| + frame token (ep1) − Naive LoRA (ep1) | 2878 | +4.27 | [+3.13, +5.43] | 1.000 |
| Full (frame + consistency + perm, ep1) − + frame token (ep1) | 2878 | +1.84 | [+0.27, +3.45] | 0.989 |
| Naive LoRA (ep1) − Qwen2.5-VL zero-shot | 2878 | +16.33 | [+13.84, +18.78] | 1.000 |

## Domain = coco (unit: samples; k=2834, n=2834)

### Per-method accuracy with 95% CI
| Method | n | acc | 95% CI |
|---|---:|---:|---|
| Qwen2.5-VL zero-shot | 2834 | 34.79 | [33.03, 36.49] |
| Qwen2.5-VL + text prompt (inference only) | 2834 | 36.98 | [35.25, 38.74] |
| Naive LoRA (ep1) | 2834 | 40.44 | [38.60, 42.24] |
| LoRA + text instruction SFT (ep1) | 2834 | 46.19 | [44.39, 48.09] |
| + frame token (ep1) | 2834 | 46.72 | [44.81, 48.59] |
| Frame-Gated LoRA (gate only, ep1) | — | — | — |
| Token + Gated LoRA (ep1) | — | — | — |
| Full (frame + consistency + perm, ep1) | 2834 | 45.55 | [43.75, 47.39] |

### Paired Δ with 95% CI
| Comparison | n | Δ (pp) | 95% CI | P(Δ>0) |
|---|---:|---:|---|---:|
| LoRA + text instruction SFT (ep1) − + frame token (ep1) | 2834 | -0.53 | [-2.05, +1.02] | 0.247 |
| LoRA + text instruction SFT (ep1) − Naive LoRA (ep1) | 2834 | +5.75 | [+4.27, +7.23] | 1.000 |
| + frame token (ep1) − Naive LoRA (ep1) | 2834 | +6.28 | [+5.12, +7.48] | 1.000 |
| Full (frame + consistency + perm, ep1) − + frame token (ep1) | 2834 | -1.16 | [-2.51, +0.14] | 0.040 |
| Naive LoRA (ep1) − Qwen2.5-VL zero-shot | 2834 | +5.65 | [+3.67, +7.69] | 1.000 |
