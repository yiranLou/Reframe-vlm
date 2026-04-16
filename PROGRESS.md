# ReFrame-VLM — Execution Progress & Paper Status

**Target venue**: EMNLP 2026 main conference (ARR 2026-05-25 cycle)
**Backbone**: Qwen2.5-VL-7B + LoRA
**Environment**: RunPod A100-SXM4-80GB, PyTorch 2.5.1, CUDA 12.8, PEFT 0.19.1
**Latest update**: 2026-04-16 21:30 UTC (post eval-fix pivot 4)

---

## 1. Executive summary — where we are today

### Critical eval-fix discovery (Pivot 4, 2026-04-16 evening)

**All results from Pivot 2 and Pivot 3 were based on a buggy eval pipeline.**
The eval code was not correctly loading trained frame-token embeddings.
Root causes (fixed by user commit `13e6d88`):

1. **PEFT 0.13 did not support `trainable_token_indices`** — the 4 added
   `<frame_*>` special tokens were resized into the embedding table but their
   rows were **frozen during training** (PEFT didn't know to include them in
   the LoRA adapter's trainable set). Upgraded to PEFT 0.19.1; the new
   `get_default_lora_config()` passes `trainable_token_indices` so frame-token
   embeddings are properly trained.
2. **Eval did not save/load the processor** — the resized tokenizer (with 4
   extra tokens) was not persisted alongside the checkpoint. Added
   `save_processor()` to all training modes and `_load_processor_with_fallback`
   + `_ensure_frame_tokens` to the eval loader.
3. **`--use_frame_token` flag at inference** — the eval script now correctly
   prepends `<frame_*>` tokens when this flag is set, matching what the model
   saw during training.

### Re-eval results (with fix)

| Method | Old eval (buggy) | **Fixed eval** | Δ |
|---|---:|---:|---:|
| Naive LoRA | 47.48 | 47.48 | 0 |
| text_instruction SFT | 50.82 | 50.82 | 0 |
| Frame LoRA | 49.12 | **52.75** | **+3.63** |
| **Full Method** | 48.09 | **53.10** | **+5.01** |

**Full Method (53.10%) is now the highest on ViewSpatial.** Frame LoRA (52.75%)
also surpasses text-instruction SFT (50.82%) by +1.93 pp.

### What this means for the paper

Every analysis that used the old Frame LoRA / Full Method predictions —
domain-split bootstrap, FCA/CR/FG, wrong-frame counterfactuals, qtype
breakdown — is **invalidated** and must be re-run with the fixed eval.

The paper direction is **no longer locked to Plan B' (analysis paper)**. With
Full Method at 53.10%, the original method story may be revived:

- **If re-run domain-split confirms Frame LoRA / Full Method gain on OOD
  COCO**: the paper can claim learned frame tokens + consistency loss is a
  genuine method contribution.
- **If wrong-frame counterfactuals now show real causal effect** (because
  trained embeddings are no longer random noise): the "vestigial at inference"
  finding flips.
- **Consistency loss** may no longer be a negative finding — Full Method
  53.10 > Frame LoRA 52.75 = +0.35 pp. Small but *positive*, not negative.

**Decision pending**: re-run all diagnostics with fixed eval before committing
to any paper direction.

---

## 2. Pivot history

1. **Original plan** (pre-training): Full Method is the method.
2. **Pivot 1** (2026-04-14): consistency loss regressed. Frame LoRA only.
3. **Pivot 2** (2026-04-15): scene-leakage + domain split. Frame LoRA OOD only.
4. **Pivot 3** (2026-04-16 AM): text-instruction SFT beat Frame LoRA. Plan B' analysis paper.
5. **Pivot 4** (2026-04-16 PM): **eval fix invalidates pivots 2-3**. Frame
   LoRA 52.75, Full Method 53.10, both beat text-instruction 50.82. All
   diagnostics must be re-run.

**Pivots 2 and 3 were artifacts of a broken eval pipeline, not real empirical
findings.** The underlying training was sound; the eval incorrectly loaded
random-init embeddings instead of trained frame-token embeddings.

---

## 3. Current headline numbers (ViewSpatial-Bench overall)

| Method | ViewSpatial | MMSI | Notes |
|---|---:|---:|---|
| Qwen2.5-VL zero-shot | 36.45 | — | |
| + text prompt (inference only) | 38.27 | — | |
| Naive LoRA | 47.48 | 25.10 | |
| LoRA + text-instr SFT | 50.82 | 25.70 | |
| **Frame LoRA** | **52.75** | **26.60** | fixed eval |
| **Full Method** | **53.10** | 25.50 | fixed eval; **BEST** |

MMSI numbers for Frame LoRA and Full Method are from the **old eval** and
likely also need re-running with `--use_frame_token`. The +3-5 pp jump on
ViewSpatial suggests MMSI may shift too.

---

## 4. What must be re-done with fixed eval

### P0 — blocking for any paper claim

| Task | Status | Why |
|---|---|---|
| Re-run domain-split bootstrap (ScanNet vs COCO) for Frame LoRA + Full Method | TODO | Old numbers invalidated |
| Re-run wrong-frame / none / always-* counterfactuals on Frame LoRA | TODO | Old controls used random-init embeddings |
| Re-run FCA / CR / FG on Frame LoRA + Full Method | TODO | Old Table 4 is invalid |
| Re-run qtype breakdown with new Frame / Full predictions | TODO | Old confound tables invalid |
| Re-run MMSI eval for Frame LoRA + Full Method with `--use_frame_token` | TODO | Old MMSI may be wrong |
| Re-run text-instruction counterfactuals (failed earlier due to PeftModel scoping bug, then chain aborted) | TODO | Never completed |

### P1 — important but not blocking

| Task | Status | Why |
|---|---|---|
| Fix `frame_lora.py` to skip vision-encoder LoRA layers | TODO | Dryrun crashed on tensor shape mismatch in ViT rotary embeddings |
| Fix smoke custom_train_loop (truncation ValueError) | TODO | `max_length=512` + `truncation=True` clips `<image>` tokens |
| Frame-Gated LoRA training | BLOCKED on vision-skip fix | Identity-init dryrun passed on LLM-only layers |

### P2 — deferred

| Task | Status |
|---|---|
| Frame-Gated LoRA eval + comparison | Blocked on P1 |
| Pair-Coupled LoRA | Abandoned |
| Consistency loss λ grid | Abandoned |

---

## 5. Scene leakage audit (still valid)

| Metric | Value |
|---|---:|
| Train scenes | 308 |
| Test scenes | 279 |
| Scene overlap | 275 / 279 = 98.6 % |
| Image overlap | 0 / 5 250 = 0 % |

ScanNet = same-scene held-out-view. COCO = true OOD. This finding is
independent of the eval-fix and remains valid.

---

## 6. Wrong-frame counterfactual controls (OLD — must re-run)

These numbers used **random-init frame embeddings** and are NOT valid.
Listed here only for the historical record.

| Condition | Acc (old) |
|---|---:|
| correct | 49.12 |
| wrong | 48.81 |
| none | 48.60 |
| always_camera | 49.11 |
| always_person | 48.90 |

After the eval fix, correct=52.75. If the trained embeddings carry real
frame semantics, `wrong` and `none` should now drop **significantly more**
than 0.3-0.5 pp. This is the single most important diagnostic to re-run.

---

## 7. Bugs fixed in this cycle (complete log)

### Eval-critical (user commit `13e6d88`, 2026-04-16)
1. `trainable_token_indices` added to `LoraConfig` via `get_default_lora_config()`.
2. `save_processor()` persists tokenizer with frame tokens alongside every checkpoint.
3. `_ensure_frame_tokens()` + `_load_processor_with_fallback()` in eval loader.
4. `--use_frame_token` flag correctly prepends `<frame_*>` during inference.
5. PDR (Pair Disagreement Rate) separated from CR in `summarize_results.py`.

### Training-infrastructure (earlier Claude commits)
6. Consistency loss projections moved from Trainer to model (`canonical_proj`).
7. bf16/fp32 dtype alignment in `relation_head` and `FrameCanonicalProjection`.
8. `BaselineTrainer` try/except for `frame_type_ids` kwarg.
9. wandb auth → `WANDB_MODE=offline`.

### Eval-infrastructure (Claude commits)
10. PeftModel scoping bug in `load_model()` (redundant inner import).
11. Vision-encoder LoRA layers must be skipped in `patch_lora_with_frame_gating`
    (tensor shape mismatch on ViT rotary PE). **Fix written but not yet merged
    into user's codebase.**

### Environment
12. PEFT upgraded: 0.13.0 → 0.19.1 (required for `trainable_token_indices`).

---

## 8. Frame-Gated LoRA implementation status

Core module (`src/model/frame_lora.py`) is rewritten:
- `FrameGateEmbedding`: per-LoRA-layer `nn.Embedding(4, out_features)` init to 0.
- Gate formula: `g_f = 1 + tanh(E_f)` → identity at init.
- `patch_lora_with_frame_gating()`: monkey-patches PEFT LoraLinear forward.
  **Needs vision-encoder skip filter re-applied** (user's pull reverted it).
- `set_frame_type_ids_for_lora()`: sets per-sample frame ids before forward.
- `save_frame_gates()` / `load_frame_gates()`: persist to `frame_lora_gates.pt`.

`ReFrameVLM` extended with `use_frame_gated_lora` param. Training modes
`frame_gated` and `token_gated` wired in `train.py`. Eval loader in
`run_benchmark.py` detects `frame_lora_gates.pt` and patches + loads without
merge-and-unload.

Dryrun (`scripts/dryrun_frame_gated.py`) **passed all 6 checks** after
vision-encoder skip was applied:
- 196 LLM LoRA layers patched, 5.56 M gate params
- Identity init verified (max|cam − none| = 0)
- Save/load round-trip OK
- Gradient checkpointing recompute OK

**Training not yet launched** — waiting for vision-skip fix to be merged and
P0 re-diagnostics to complete first.

---

## 9. Updated execution plan (2026-04-16 → 05-25)

| Date | Task |
|---|---|
| **2026-04-17** | Re-run ALL diagnostics with fixed eval: domain-split bootstrap, FCA/CR/FG, wrong-frame 5-condition controls, qtype breakdown, MMSI re-eval. Re-apply vision-encoder skip to frame_lora.py. |
| **2026-04-18** | Assess new numbers. If Frame LoRA / Full Method gains hold on OOD COCO AND wrong-frame controls now show real causal effect → paper direction clears (method paper or analysis paper, decided by magnitude). |
| **2026-04-18-19** | Fix smoke custom_train_loop (max_length issue). If decided, launch Frame-Gated LoRA training (~25h). |
| **2026-04-20-25** | Finalise experiments, lock results, start writing. |
| **2026-04-26 → 05-15** | Full paper draft + figures + appendix. |
| **2026-05-16 → 05-22** | Internal review, polish, anonymise. |
| **2026-05-23 → 05-25** | ARR upload. |

---

## 10. Appendix — historical notes

### 10.1 Full Method training notes (ep1)
* Started: 2026-04-13 18:46 UTC. Stopped: 2026-04-14 22:20 UTC.
* 926 optimizer steps, ~107 s/step. QA loss 14.8 → 3.36.
* Checkpoint: `checkpoints/full/checkpoint-926/` (17.3 GB full-model save).

### 10.2 text-instruction SFT training (ep1)
* Started: 2026-04-15 15:06 UTC. Finished: 2026-04-16 08:01 UTC.
* 716 optimizer steps, ~85 s/step. Loss 0.26 → 0.22.
* Checkpoint: `checkpoints/text_instruction_lora/checkpoint-716/`.

### 10.3 Data pipeline
* Training: 45 809 QA from ViewSpatial visibility_data (ScanNet).
* Consistency pairs: 13 408 (camera ↔ person).
* ViewSpatial-Bench: 5 712 test (2 878 ScanNet + 2 834 COCO).
* MMSI-Bench: 1 000. Ego3D-Bench: 8 675.

### 10.4 Code inventory (2026-04-16)
Key scripts: `dryrun_full.py`, `dryrun_frame_gated.py`, `frame_token_controls.py`,
`text_instruction_controls.py`, `scene_leakage_audit.py`, `bootstrap_ci.py`,
`bootstrap_ci_scene.py`, `bootstrap_ci_domain.py`, `diagnostics_domain_split.py`,
`domain_confound_check.py`, `sanity_text_instr_labels.py`, `summarize_results.py`,
`fill_table1.sh`, `run_planB_chain.sh`, `run_refix_chain.sh`.

---

## 11. TODOs before ARR submission

High-priority:
* [ ] Re-run domain-split bootstrap with fixed Frame LoRA + Full Method predictions.
* [ ] Re-run wrong-frame 5-condition counterfactuals on Frame LoRA (fixed eval).
* [ ] Re-run FCA/CR/FG/PDR with fixed predictions.
* [ ] Re-run MMSI eval for Frame LoRA + Full Method with `--use_frame_token`.
* [ ] Run text-instruction 5-condition counterfactuals (never completed).
* [ ] Re-apply vision-encoder skip in `frame_lora.py`.
* [ ] Fix smoke custom_train_loop truncation bug.
* [ ] Decide paper direction based on re-run diagnostics.

Medium:
* [ ] Launch Frame-Gated LoRA training (after vision-skip + decision).
* [ ] Bootstrap with Holm/BH correction.
* [ ] Domain-stratified FCA/CR/FG table.
* [ ] Accuracy vs contradiction scatter figure.

Low:
* [ ] Scrambled-frame-label control (~27 h).
* [ ] Ego3D lightweight re-eval.
