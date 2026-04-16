
## Domain = scannet (unit: scenes; k=279, n=2878)

### Per-method accuracy with 95% CI
| Method | n | acc | 95% CI |
|---|---:|---:|---|
| Qwen2.5-VL zero-shot | 2878 | 38.08 | [36.08, 40.09] |
| + text prompt | 2878 | 39.54 | [37.47, 41.67] |
| Naive LoRA | 2878 | 54.41 | [52.15, 56.58] |
| LoRA+text-instr SFT | 2878 | 55.39 | [53.01, 57.79] |
| Frame LoRA | 2878 | 54.86 | [52.58, 57.12] |
| Full Method | 2878 | 54.97 | [52.98, 56.90] |

### Paired Δ with 95% CI
| Comparison | n | Δ (pp) | 95% CI | P(Δ>0) |
|---|---:|---:|---|---:|
| Frame LoRA − LoRA+text-instr SFT | 2878 | -0.52 | [-2.13, +1.05] | 0.253 |
| Frame LoRA − Naive LoRA | 2878 | +0.45 | [-0.51, +1.43] | 0.816 |
| LoRA+text-instr SFT − Naive LoRA | 2878 | +0.97 | [-0.64, +2.59] | 0.875 |
| Full Method − Frame LoRA | 2878 | +0.10 | [-1.37, +1.59] | 0.549 |
| Naive LoRA − Qwen2.5-VL zero-shot | 2878 | +16.33 | [+13.84, +18.78] | 1.000 |

## Domain = coco (unit: samples; k=2834, n=2834)

### Per-method accuracy with 95% CI
| Method | n | acc | 95% CI |
|---|---:|---:|---|
| Qwen2.5-VL zero-shot | 2834 | 34.79 | [33.03, 36.49] |
| + text prompt | 2834 | 36.98 | [35.25, 38.74] |
| Naive LoRA | 2834 | 40.44 | [38.60, 42.24] |
| LoRA+text-instr SFT | 2834 | 46.19 | [44.39, 48.09] |
| Frame LoRA | 2834 | 43.30 | [41.39, 45.10] |
| Full Method | 2834 | 41.11 | [39.34, 42.91] |

### Paired Δ with 95% CI
| Comparison | n | Δ (pp) | 95% CI | P(Δ>0) |
|---|---:|---:|---|---:|
| Frame LoRA − LoRA+text-instr SFT | 2834 | -2.89 | [-4.34, -1.48] | 0.000 |
| Frame LoRA − Naive LoRA | 2834 | +2.86 | [+1.80, +3.92] | 1.000 |
| LoRA+text-instr SFT − Naive LoRA | 2834 | +5.75 | [+4.23, +7.23] | 1.000 |
| Full Method − Frame LoRA | 2834 | -2.19 | [-3.49, -0.88] | 0.000 |
| Naive LoRA − Qwen2.5-VL zero-shot | 2834 | +5.65 | [+3.67, +7.69] | 1.000 |
