# ReFrame-VLM Autoresearch Program

## Goal
Maximize accuracy on ViewSpatial-Bench while maintaining low Contradiction Rate.
Secondary: generalize to MMSI-Bench and Ego3D-Bench.

## Eval command
```bash
bash scripts/run_eval_all.sh <checkpoint_path> models/qwen25-vl-7b | grep "METRIC_SUMMARY"
```
Output format: `viewspatial=XX.X mmsi=XX.X ego3d=XX.X cr=XX.X`

## Primary metric
ViewSpatial accuracy (higher is better)

## Secondary metrics
- Contradiction Rate (lower is better)
- MMSI accuracy (higher is better)
- Ego3D accuracy (higher is better)

## Train command
```bash
python src/training/train.py --config <config_file>
```

## Constraints
- Training must finish within 4 hours on 1xA100 80GB
- Do not modify eval scripts (src/eval/*)
- Do not modify data preprocessing (data/scripts/*)
- LoRA rank must stay between 32-128
- Total training data must stay under 120K samples

## Current best
(Fill in after initial experiments)
- Baseline zero-shot: ???
- Baseline LoRA: ???
- Frame LoRA: ???
- Full method: ???

## Hyperparameter search space

### Priority 1: Consistency loss weight (lambda)
Values: 0.01, 0.05, 0.1, 0.2, 0.3, 0.5
Config key: lambda_consistency
Expected: 0.05-0.2 range is sweet spot

### Priority 2: Learning rate
Values: 1e-5, 2e-5, 3e-5, 5e-5
Config key: learning_rate

### Priority 3: LoRA rank
Values: 32, 64, 128
Config key: lora_rank (also adjust lora_alpha = 2 * rank)

### Priority 4: View permutation probability
Values: 0.0, 0.3, 0.5, 0.7, 1.0
Config key: view_permutation_prob

### Priority 5: Canonical dim for consistency
Values: 32, 64, 128
Config key: canonical_dim

### Priority 6: Number of epochs
Values: 2, 3, 5
Config key: num_epochs

## Architecture variants to try
1. Frame token count: 1 vs 4 vs 8 tokens per frame type
2. Relation head size: 7 vs 14 vs 28 dimensions
3. Consistency loss annealing: constant vs linear warmup
4. Frame embedding init: random vs from text token "camera"/"person"

## Experiment log format
After each experiment, append to results/experiment_log.md:
```
## Experiment N: <description>
- Config: <path>
- Changes: <what changed>
- ViewSpatial: XX.X%
- MMSI: XX.X%
- Ego3D: XX.X%
- CR: XX.X%
- Training time: Xh Xm
- Notes: <observations>
```

## Decision rules
- If lambda > 0.2 hurts QA accuracy, try annealing (0 for first epoch, then ramp up)
- If frame token ablation shows < 1% difference, focus paper story on consistency loss
- If MMSI/Ego3D don't improve, that's fine - ViewSpatial is primary
- Stop early if training loss diverges
