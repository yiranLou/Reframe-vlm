# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ReFrame-VLM is a research project for improving spatial reasoning in Vision Language Models. It fine-tunes Qwen2.5-VL (7B) using LoRA with frame-type-aware training: learnable frame tokens (`<frame_camera>`, `<frame_person>`, `<frame_object>`, `<frame_world>`), a consistency loss across viewing frames, and view permutation augmentation.

Primary metric: ViewSpatial-Bench accuracy. Secondary: Contradiction Rate (lower better), MMSI accuracy, Ego3D accuracy.

## Commands

```bash
# Setup
bash setup.sh                    # Create conda env + install deps

# Training
python src/training/train.py --config configs/train_full.yaml
bash scripts/run_train.sh configs/train_baseline.yaml  # wrapper script

# Evaluation (all benchmarks + analysis)
bash scripts/run_eval_all.sh <checkpoint_path> models/qwen25-vl-7b | grep "METRIC_SUMMARY"
# Output: viewspatial=XX.X mmsi=XX.X ego3d=XX.X cr=XX.X

# Evaluate single benchmark
python src/eval/run_benchmark.py --model_path <checkpoint> --benchmark viewspatial --output results/out.json

# Sanity check (verify imports, data, model components)
python scripts/sanity_check.py --skip_model   # without GPU
python scripts/sanity_check.py                 # full check with model loading

# Hyperparameter search
bash scripts/grid_search_lambda.sh
bash scripts/grid_search_lr.sh
bash scripts/run_ablation.sh

# Data preprocessing
bash scripts/preprocess_all.sh
python data/scripts/generate_training_data.py
python data/scripts/build_consistency_pairs.py
```

## Constraints (from program.md)

- Training must finish within 4 hours on 1xA100 80GB
- Do NOT modify eval scripts (`src/eval/*`) or data preprocessing (`data/scripts/*`)
- LoRA rank must stay between 32-128
- Total training data must stay under 120K samples
- Log experiments to `results/experiment_log.md` using the format in `program.md`

## Architecture

### Three training modes (set via `mode` in YAML config)

1. **baseline** - Standard LoRA fine-tune of Qwen2.5-VL
2. **frame** - LoRA + frame special tokens prepended to inputs
3. **full** - Frame tokens + consistency loss + view permutation augmentation

### Source layout

- `src/model/reframe_model.py` - Main `ReFrameVLM` class; assembles base model + frame tokens + LoRA + relation head
- `src/model/frame_embedding.py` - `FrameEmbedding` (per-frame-type vectors) and `FrameTokenModule` (learnable prefix tokens)
- `src/model/relation_head.py` - `RelationHead` (14-dim: 8 directions + 3 vertical + 3 distance) and `FrameCanonicalProjection`
- `src/model/frame_lora.py` - Frame-conditioned LoRA variant (optional)
- `src/training/train.py` - Training entry point; accepts `--config` or `--mode`
- `src/training/trainer.py` - `ReFrameTrainer` extending HuggingFace Trainer
- `src/training/dataset.py` - `ReFrameDataset` supporting QA / consistency / mixed modes
- `src/training/collator.py` - Builds Qwen2.5-VL conversation format with frame token prepends
- `src/training/losses.py` - `ReFrameLoss`: L_total = L_qa + lambda * L_consistency (MSE in canonical space between paired frames)
- `src/eval/run_benchmark.py` - Evaluation entry point (viewspatial / mmsi / ego3d)
- `src/eval/consistency_eval.py` - Frame-switch consistency analysis (FCA, CR, FG metrics)

### Training flow

1. Load Qwen2.5-VL, add frame tokens to tokenizer
2. Apply LoRA to transformer layers
3. Optionally add relation head + canonical projection
4. Build ReFrameDataset from JSONL (single + paired samples)
5. Collator formats conversations with frame token prepends
6. Trainer computes L_qa + lambda * L_consistency
7. Save checkpoint with adapter config after each epoch

### Data format

Training data is JSONL in `data/processed/`. Each sample has `id`, `images`, `question`, `choices`, `answer`, `frame_type`. Consistency pairs (`consistency_pairs.jsonl`) have `pair_id`, `sample_a`, `sample_b`.

### Key config parameters (in `configs/*.yaml`)

| Parameter | Default (full) | Range |
|---|---|---|
| `lora_rank` | 64 | 32-128 |
| `lora_alpha` | 128 | 2 * rank |
| `lambda_consistency` | 0.1 | 0.01-0.5 |
| `canonical_dim` | 64 | 32-128 |
| `view_permutation_prob` | 0.5 | 0.0-1.0 |
| `learning_rate` | 2e-5 | 1e-5 to 5e-5 |
| `num_epochs` | 3 | 2-5 |
| `batch_size` | 4 | per-device |
| `gradient_accumulation` | 16 | - |

### Frame type system

```python
FRAME_TYPE_MAP = {"camera": 0, "person": 1, "object": 2, "world": 3}
```

Tokens: `<frame_camera>`, `<frame_person>`, `<frame_object>`, `<frame_world>`

### External paths

- Base model: `models/qwen25-vl-7b/`
- Checkpoints: `checkpoints/{baseline|frame|full}/`
- Results: `results/`
- Logging: Weights & Biases (wandb)
