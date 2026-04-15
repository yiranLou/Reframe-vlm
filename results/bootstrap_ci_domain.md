
## Domain = scannet (unit: scenes; k=279, n=2878)

### Per-method accuracy with 95% CI
| Method | n | acc | 95% CI |
|---|---:|---:|---|
| Qwen2.5-VL zero-shot | 2878 | 38.08 | [36.08, 40.09] |
| + text prompt | 2878 | 39.54 | [37.47, 41.67] |
| Naive LoRA | 2878 | 54.41 | [52.15, 56.58] |
| Frame LoRA | 2878 | 54.86 | [52.59, 57.17] |
| Full Method | 2878 | 54.97 | [53.00, 56.92] |

### Paired Δ with 95% CI
| Comparison | n | Δ (pp) | 95% CI | P(Δ>0) |
|---|---:|---:|---|---:|
| Frame LoRA − Naive LoRA | 2878 | +0.45 | [-0.52, +1.44] | 0.808 |
| Full Method − Frame LoRA | 2878 | +0.10 | [-1.36, +1.58] | 0.549 |
| Naive LoRA − Qwen2.5-VL zero-shot | 2878 | +16.33 | [+13.84, +18.82] | 1.000 |

## Domain = coco (unit: samples; k=2834, n=2834)

### Per-method accuracy with 95% CI
| Method | n | acc | 95% CI |
|---|---:|---:|---|
| Qwen2.5-VL zero-shot | 2834 | 34.79 | [32.99, 36.59] |
| + text prompt | 2834 | 36.98 | [35.22, 38.74] |
| Naive LoRA | 2834 | 40.44 | [38.60, 42.20] |
| Frame LoRA | 2834 | 43.30 | [41.43, 45.17] |
| Full Method | 2834 | 41.11 | [39.31, 42.91] |

### Paired Δ with 95% CI
| Comparison | n | Δ (pp) | 95% CI | P(Δ>0) |
|---|---:|---:|---|---:|
| Frame LoRA − Naive LoRA | 2834 | +2.86 | [+1.80, +3.88] | 1.000 |
| Full Method − Frame LoRA | 2834 | -2.19 | [-3.53, -0.85] | 0.000 |
| Naive LoRA − Qwen2.5-VL zero-shot | 2834 | +5.65 | [+3.60, +7.69] | 1.000 |
