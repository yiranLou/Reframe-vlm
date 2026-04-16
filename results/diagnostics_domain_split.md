## Frame-switch diagnostics by domain

Total cross-frame pairs: **996**  (ScanNet pairs: **0**, COCO pairs: **996**)


### Domain = scannet  (n_pairs = 0)

| Method | n | FCA ↑ | CR ↓ | Camera | Non-Cam | FG |
|---|---:|---:|---:|---:|---:|---:|

### Domain = coco  (n_pairs = 996)

| Method | n | FCA ↑ | CR ↓ | Camera | Non-Cam | FG |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL zero-shot | 996 | 6.22 | 56.43 | 28.41 | 40.46 | -12.05 |
| + text prompt (infer) | 996 | 9.24 | 57.03 | 29.82 | 45.68 | -15.86 |
| Naive LoRA | 996 | 11.24 | 61.85 | 19.78 | 64.56 | -44.78 |
| LoRA + text-instr SFT | 996 | 19.28 | 62.55 | 29.02 | 72.09 | -43.07 |
| Frame LoRA | 996 | 13.35 | 63.65 | 21.99 | 68.37 | -46.39 |
| Full Method | 996 | 10.34 | 62.85 | 21.59 | 61.95 | -40.36 |
