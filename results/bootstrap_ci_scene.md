## Scene-level paired bootstrap CI

_Resampling units: **scenes** on ViewSpatial (reviewer-defensible, accounts for scene overlap with train); samples on MMSI/Ego3D._

| Comparison | Benchmark | unit | k | n | Δ (pp) | 95% CI | P(Δ>0) |
|---|---|---|---:|---:|---:|---|---:|
| Frame LoRA − Naive LoRA | viewspatial | scenes | 3113 | 5712 | +1.65 | [+0.93, +2.39] | 1.000 |
| Frame LoRA − Naive LoRA | mmsi | samples | 1000 | 1000 | +1.50 | [-0.30, +3.30] | 0.947 |
| Full Method − Frame LoRA | viewspatial | scenes | 3113 | 5712 | -1.03 | [-2.03, -0.02] | 0.023 |
| Full Method − Frame LoRA | mmsi | samples | 1000 | 1000 | -1.10 | [-3.40, +1.10] | 0.151 |
| Naive LoRA − Qwen2.5-VL zero-shot | viewspatial | scenes | 3113 | 5712 | +11.03 | [+9.35, +12.69] | 1.000 |
