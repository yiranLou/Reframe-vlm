## Scene-level paired bootstrap CI

_Resampling units: **scenes** on ViewSpatial; **samples** on MMSI/Ego3D._

| Comparison | Benchmark | unit | k | n | Δ (pp) | 95% CI | P(Δ>0) |
|---|---|---|---:|---:|---:|---|---:|
| LoRA + text instruction SFT (ep1) − + frame token (ep1) | viewspatial | scenes | 3113 | 5712 | -1.93 | [-3.07, -0.78] | 0.000 |
| LoRA + text instruction SFT (ep1) − + frame token (ep1) | mmsi | samples | 1000 | 1000 | -0.90 | [-3.20, +1.40] | 0.205 |
| + frame token (ep1) − Naive LoRA (ep1) | viewspatial | scenes | 3113 | 5712 | +5.27 | [+4.46, +6.12] | 1.000 |
| + frame token (ep1) − Naive LoRA (ep1) | mmsi | samples | 1000 | 1000 | +1.50 | [-0.30, +3.30] | 0.942 |
| Full (frame + consistency + perm, ep1) − + frame token (ep1) | viewspatial | scenes | 3113 | 5712 | +0.35 | [-0.68, +1.38] | 0.738 |
| Full (frame + consistency + perm, ep1) − + frame token (ep1) | mmsi | samples | 1000 | 1000 | -1.10 | [-3.30, +1.20] | 0.160 |
| LoRA + text instruction SFT (ep1) − Naive LoRA (ep1) | viewspatial | scenes | 3113 | 5712 | +3.34 | [+2.22, +4.45] | 1.000 |
| LoRA + text instruction SFT (ep1) − Naive LoRA (ep1) | mmsi | samples | 1000 | 1000 | +0.60 | [-1.70, +2.90] | 0.680 |
| Naive LoRA (ep1) − Qwen2.5-VL zero-shot | viewspatial | scenes | 3113 | 5712 | +11.03 | [+9.36, +12.67] | 1.000 |
