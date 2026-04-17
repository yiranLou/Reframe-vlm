## Frame-switch diagnostics by domain

Total cross-frame pairs: **996**  (ScanNet pairs: **0**, COCO pairs: **996**)


### Domain = scannet  (n_pairs = 0)

| Method | n | FCA ↑ | CR ↓ | PDR ↓ | Camera | Non-Cam | FG |
|---|---:|---:|---:|---:|---:|---:|---:|

### Domain = coco  (n_pairs = 996)

| Method | n | FCA ↑ | CR ↓ | PDR ↓ | Camera | Non-Cam | FG |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL zero-shot | 996 | 6.22 | 24.60 | 56.43 | 28.41 | 40.46 | -12.05 |
| Qwen2.5-VL + text prompt (inference only) | 996 | 9.24 | 25.40 | 57.03 | 29.82 | 45.68 | -15.86 |
| Naive LoRA (ep1) | 996 | 11.24 | 32.53 | 61.85 | 19.78 | 64.56 | -44.78 |
| LoRA + text instruction SFT (ep1) | 996 | 19.28 | 25.40 | 62.55 | 29.02 | 72.09 | -43.07 |
| + frame token (ep1) | 996 | 14.26 | 33.94 | 68.17 | 22.69 | 74.00 | -51.31 |
| Full (frame + consistency + perm, ep1) | 996 | 11.45 | 32.73 | 68.57 | 22.49 | 68.98 | -46.49 |
