# ReFrame-VLM — Resume Guide

Last verified: 2026-04-16 21:30 UTC. Read this when resuming after a
RunPod terminate / restart.

---

## 1. Environment recovery (~3 min)

```bash
cd /workspace/reframe-vlm
pip install -r requirements.txt
pip install flash-attn --no-build-isolation   # optional, ~5 min compile

# Verify critical packages
python -c "import peft; print('peft:', peft.__version__)"           # need >=0.17
python -c "from peft import LoraConfig; import inspect; \
  print('trainable_token_indices:', \
  'trainable_token_indices' in inspect.signature(LoraConfig.__init__).parameters)"  # must be True
python -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('OK')"

# Git identity (repo-local, not global)
git config user.name "LouYiRan"
git config user.email "delusion.lou@gmail.com"
```

**Exact versions that were working (2026-04-16)**:
- torch 2.5.1+cu124
- transformers 5.5.3
- peft 0.19.1
- accelerate 1.13.0
- flash-attn 2.8.3
- safetensors 0.7.0

---

## 2. Disk layout (all on Network Volume `/workspace`)

```
/workspace/
├── models/qwen25-vl-7b/           16 GB   Qwen2.5-VL-7B base weights
├── datasets/
│   ├── viewspatial/                4.4 GB  ViewSpatial raw (ScanNet scenes + images)
│   ├── mmsi-data/                  1.7 GB  MMSI-Bench parquet + extracted images
│   ├── ego3d-data/                 943 MB  Ego3D-Bench arrow + raw_images
│   ├── mmsi-bench/                 108 MB  MMSI repo (code only)
│   ├── ego3d-bench/                31 MB   Ego3D repo (code only)
│   └── robospatial/                126 MB  RoboSpatial subset (not used)
└── reframe-vlm/                    repo root
    ├── checkpoints/
    │   ├── baseline_lora/checkpoint-716/        2.2 GB  (adapter)
    │   ├── frame_lora/checkpoint-716/           18 GB   (full model save)
    │   ├── full/checkpoint-926/                 18 GB   (full model save)
    │   └── text_instruction_lora/checkpoint-716/ 2.9 GB  (adapter)
    ├── data/processed/                          833 MB
    │   ├── train.jsonl                  45.8K training QA
    │   ├── consistency_pairs.jsonl      13.4K pairs
    │   ├── viewspatial_test.jsonl       5712 test
    │   ├── mmsi_test.jsonl              1000 MMSI
    │   ├── ego3d_test.jsonl             8675 Ego3D
    │   └── mmsi_images/                 extracted MMSI images
    └── results/                         27 MB  all eval results (JSON + MD)
```

---

## 3. Current state (Pivot 4)

### Headline numbers (ViewSpatial-Bench, fixed eval)

| Method | ViewSpatial | MMSI |
|---|---:|---:|
| Zero-shot | 36.45 | — |
| + text prompt | 38.27 | — |
| Naive LoRA | 47.48 | 25.10 |
| text-instruction SFT | 50.82 | 25.70 |
| **Frame LoRA** | **52.75** | 26.60 (old eval, needs re-run) |
| **Full Method** | **53.10** | 25.50 (old eval, needs re-run) |

### What happened
The eval pipeline was broken — frame-token embeddings were loaded as random
noise instead of trained values. After fixing (PEFT upgrade + save_processor
+ --use_frame_token), Frame LoRA jumped +3.63 pp and Full Method +5.01 pp.
All domain-split, wrong-frame, FCA/CR/FG analyses from before the fix are
invalid and must be re-run.

### What needs to happen next (priority order)
1. Re-run ALL diagnostics with fixed eval predictions
2. Re-run MMSI eval with --use_frame_token for Frame LoRA + Full Method
3. Re-run wrong-frame 5-condition counterfactuals (now with trained embeddings)
4. Apply vision-encoder skip to frame_lora.py (needed for Frame-Gated training)
5. Fix custom_train_loop smoke test (max_length truncation bug)
6. Decide paper direction based on new numbers

See PROGRESS.md §4 and §11 for the full TODO list.

---

## 4. Key bugs to remember

1. **`trainable_token_indices`** — PEFT <0.17 silently freezes added special
   token embeddings. Must use >=0.17 and pass the indices in LoraConfig.
2. **`save_processor()`** — every training mode must save the tokenizer
   alongside the adapter so eval can reload the resized vocabulary.
3. **Vision-encoder LoRA patching** — `patch_lora_with_frame_gating()` must
   skip modules whose name contains "visual" / "vision_tower". The ViT's
   input shape doesn't match the (batch, seq, hidden) assumption of the gate
   broadcast. **This fix is written but not yet merged into the main branch.**
4. **`truncation=True` + multi-image** — Qwen2.5-VL processor rejects
   truncated inputs if `<image>` token count mismatches. Use
   `max_length>=2048` or disable truncation for multi-image samples.
5. **cuDNN** — `torch.backends.cudnn.enabled = False` is required at the top
   of train.py and run_benchmark.py (cuDNN 9.19 vs CUDA 12.x driver
   incompatibility on RunPod).

---

## 5. Quick-start commands

```bash
# Evaluate a checkpoint on ViewSpatial
PYTHONPATH=/workspace/reframe-vlm python src/eval/run_benchmark.py \
    --model_path checkpoints/frame_lora/checkpoint-716 \
    --benchmark viewspatial --use_frame_token \
    --output results/frame_lora_ep1_refix/viewspatial.json

# Run bootstrap CI
python scripts/bootstrap_ci.py

# Run domain-split bootstrap
python scripts/bootstrap_ci_domain.py

# Run wrong-frame controls on Frame LoRA
python scripts/frame_token_controls.py \
    --ckpt checkpoints/frame_lora/checkpoint-716 \
    --conditions correct,wrong,none,always_camera,always_person

# Run text-instruction controls
python scripts/text_instruction_controls.py \
    --ckpt checkpoints/text_instruction_lora/checkpoint-716 \
    --conditions correct,wrong,none,always_camera,always_person

# Train Frame-Gated LoRA (after vision-skip fix)
WANDB_MODE=offline python src/training/train.py \
    --config configs/train_frame_gated.yaml

# Refresh summary tables
python scripts/summarize_results.py --out results/summary.md
```

---

## 6. Scene leakage (permanent finding, not affected by eval fix)

- 98.6% of ViewSpatial test scenes overlap with training scenes
- 0% image overlap (different frames from same ScanNet scenes)
- COCO portion of ViewSpatial (2834 samples) is true OOD
- All 996 cross-frame pairs are in COCO (ScanNet has 0 pairs)
- Paper must split ScanNet vs COCO and report both

---

## 7. GitHub

```
git@github.com:Shengguang-Zhou/reframe-vlm.git
Latest: 8679a03 "Pivot 4: eval-fix confirms Frame LoRA 52.75% and Full Method 53.10%"
```
