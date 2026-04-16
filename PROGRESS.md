# ReFrame-VLM — Execution Progress & Paper Status

**Target venue**: EMNLP 2026 main conference (ARR 2026-05-25 cycle)
**Backbone**: Qwen2.5-VL-7B + LoRA
**Environment**: RunPod A100-SXM4-80GB, PyTorch 2.5.1, CUDA 12.8
**Latest update**: 2026-04-16 09:50 UTC (post-P0-d pivot 3)

---

## 1. Executive summary — where we are today

The paper has pivoted **three times** based on empirical findings. Current
state is a dual-track plan: floor is already a publishable empirical /
diagnostic paper (B); upper bound depends on a Frame-Gated LoRA experiment
now being implemented (C).

1. **Original plan** (pre-training): "Frame-Conditioned LoRA + Consistency Loss
   are the method; show they beat prompt engineering and naive SFT on 3 benchmarks."
2. **Pivot 1** (2026-04-14, after Full Method ep1): consistency loss regressed on
   both ViewSpatial (−1.03 pp) and MMSI (−1.10 pp). Retreat to "Frame token is
   the main contribution; consistency loss is a negative diagnostic finding."
3. **Pivot 2** (2026-04-15, scene-leakage + domain-split bootstrap): Frame LoRA
   gain was entirely OOD COCO (+2.86 pp, p < 0.001), not in-domain ScanNet
   (+0.45, not significant). Story moved to "frame tokens enable OOD transfer."
4. **Pivot 3** (2026-04-16, after LoRA + text-instruction SFT): text-instruction
   SFT clearly beats learned frame tokens on ViewSpatial-related axes, but the
   picture on cross-dataset MMSI is mixed and CR is *not* solved:
   - ViewSpatial overall: text-instr **50.82** vs Frame LoRA **49.12** (+1.70 pp).
   - COCO OOD: text-instr **+2.89 pp over Frame LoRA, CI [+1.48, +4.34], p < 0.001**.
   - FCA (paired-correct rate): text-instr **19.28** vs Frame LoRA **13.35**
     (+5.93 pp).
   - **MMSI is mixed**: Frame LoRA 26.60 *> text-instr 25.70* (~−0.9 pp). On
     1 000-sample MMSI the difference is not significant, but the directional
     reversal means we cannot claim text-instruction "dominates everywhere".
   - **Contradiction Rate is NOT solved**: text-instr CR = 62.55 vs zero-shot
     56.43 (+6.1 pp). Text-instruction lifts paired-correct rate (FCA) but
     does not reduce paired-disagreement (CR). Accuracy gain and contradiction
     avoidance are distinct axes.

   Net: Frame LoRA is no longer a viable headline method. Natural-language
   frame instructions become the new strongest supervised mechanism on
   ViewSpatial / FCA, with the caveats above.

**Plan B' (LOCKED 2026-04-16) — analysis paper, not method paper**:

> *"How Should VLMs Be Conditioned on Reference Frames? A Diagnostic Study
> of Spatial Reasoning."* Controlled comparison of four supervised
> conditioning mechanisms — input-space natural-language instructions,
> input-space learned tokens, parameter-space gated LoRA, and latent-space
> soft consistency — under matched LoRA training. Combined with a scene-leakage
> audit, ScanNet/COCO domain split, frame-switch diagnostic protocol
> (FCA/CR/FG), and counterfactual frame-token interventions. Headline finding:
> natural-language instructions are strongest on ViewSpatial OOD (+5.75 pp on
> COCO over naive LoRA, p < 0.001) and on paired-correct rate (FCA 19.28 vs
> 13.35), but contradiction rate rises under every fine-tuning recipe
> (56 → 62-64 %). Counterfactual controls show learned frame tokens are
> *vestigial at inference* — wrong tokens cost only −0.31 pp.

The Frame-Gated LoRA module is implemented and trained as the **fourth
mechanism baseline**, not a headline method. We do *not* claim a new SOTA
adapter; we claim a controlled empirical study of where reference-frame
conditioning lives in the network.

---

## 2. Runs currently live

*As of 2026-04-16 09:50 UTC — pipeline idle.*

Pipeline queue for Plan 2 (started 2026-04-16):

| Step | Role | GPU hours | Status |
|---|---|---:|---|
| 2a | `text_instruction_controls.py` — wrong / none / always-camera instruction on text_instruction_lora ckpt × ViewSpatial (parallel to Frame LoRA wrong-frame controls) | ~6 | queued |
| 2b | Implement Frame-Gated LoRA modules + configs + dry-run | 0 (CPU dev) | in progress |
| 2c | `token_gated` training (token + gate, 1 epoch) | ~25 | blocked on 2b |
| 2d | token_gated eval ViewSpatial + MMSI, refresh bootstrap / domain-split | ~2 | blocked on 2c |
| 2e | Decision: win → launch frame_gated-only ablation; else → freeze experiments, start writing | 0 | blocked on 2d |

Full Method / consistency-loss work is closed — see §6.

---

## 3. Tables ready to put in the paper (pre-P0-d snapshot)

### Table 1 — Main results (overall accuracy %)

| Method | ViewSpatial | MMSI | Ego3D |
|---|---:|---:|---:|
| Qwen2.5-VL zero-shot | 36.45 | — | — |
| + text prompt (inference only) | 38.27 | — | — |
| Naive LoRA | 47.48 | 25.10 | — |
| **LoRA + text-instr SFT** | **50.82** | 25.70 | — |
| Frame LoRA (learned frame tokens) | 49.12 | **26.60** | — |
| Full Method (frame + consist + perm) | 48.09 | 25.50 | 39.92 |
| *Frame-Gated LoRA (token_gated)* | *TBD 2026-04-19* | *TBD* | *skipped* |

### Table 2 — Per-domain bootstrap (ViewSpatial, 10k paired resamples)

**ScanNet subset** — scene-clustered (k=279, n=2878). Train shares 98.6 % of test scenes.

| Method | Acc | 95 % CI |
|---|---:|---|
| Zero-shot | 38.08 | [36.08, 40.09] |
| + text prompt | 39.54 | [37.47, 41.67] |
| Naive LoRA | 54.41 | [52.15, 56.58] |
| **LoRA + text-instr SFT** | **55.39** | [53.01, 57.79] |
| Frame LoRA | 54.86 | [52.59, 57.17] |
| Full Method | 54.97 | [53.00, 56.92] |

| Δ | 95 % CI | *P*(Δ > 0) |
|---|---|---:|
| Frame − text-instr (ScanNet) | −0.52 [−2.13, +1.05] | 0.253 |
| Frame − Naive (ScanNet) | +0.45 [−0.52, +1.44] | 0.808 |
| text-instr − Naive (ScanNet) | +0.97 [−0.64, +2.59] | 0.875 |
| Full − Frame (ScanNet) | +0.10 [−1.36, +1.58] | 0.549 |
| Naive − zero-shot (ScanNet) | +16.33 [+13.84, +18.82] | 1.000 |

**COCO subset** — sample-resampled (k=2834, OOD). No image overlap with training.

| Method | Acc | 95 % CI |
|---|---:|---|
| Zero-shot | 34.79 | [32.99, 36.59] |
| + text prompt | 36.98 | [35.22, 38.74] |
| Naive LoRA | 40.44 | [38.60, 42.20] |
| **LoRA + text-instr SFT** | **46.19** | **[44.39, 48.09]** |
| Frame LoRA | 43.30 | [41.43, 45.17] |
| Full Method | 41.11 | [39.31, 42.91] |

| Δ | 95 % CI | *P*(Δ > 0) |
|---|---|---:|
| **Frame − text-instr (COCO)** | **−2.89 [−4.34, −1.48]** | **0.000** |
| **text-instr − Naive (COCO)** | **+5.75 [+4.23, +7.23]** | **1.000** |
| Frame − Naive (COCO) | +2.86 [+1.80, +3.92] | 1.000 |
| Full − Frame (COCO) | −2.19 [−3.49, −0.88] | 0.000 |
| Naive − zero-shot (COCO) | +5.65 [+3.67, +7.69] | 1.000 |

### Table 3 — Confound check (domain × question-type)

Mapping is deterministic — each question type lives in exactly one domain:

| Domain | Question type | n | ZS | Naive | Frame | Δ Frame−Naive |
|---|---|---:|---:|---:|---:|---:|
| scannet | Camera - Relative Direction | 1773 | 45.57 | 58.60 | 59.84 | +1.24 |
| scannet | Person - Scene Simulation | 1105 | 26.06 | 47.69 | 46.88 | **−0.81** |
| coco | Camera - Object View Orientation | 996 | 28.41 | 19.78 | 21.99 | +2.21 |
| coco | Person - Object View Orientation | 996 | 40.46 | 64.56 | 68.37 | **+3.82** |
| coco | Person - Relative Direction | 842 | 35.63 | 36.34 | 38.84 | +2.49 |

Frame-token gain is positive on **all three COCO question types**, spanning both
camera and person frames. The one clean negative (−0.81) lives entirely in the
in-domain ScanNet subset.

### Table 4 — Frame-switch consistency (996 cross-frame pairs)

| Method | FCA ↑ | CR ↓ | Camera | Non-Cam | FG |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 6.22 | 56.43 | 28.41 | 40.46 | −12.05 |
| + text prompt | 9.24 | 57.03 | 29.82 | 45.68 | −15.86 |
| Naive LoRA | 11.24 | 61.85 | 19.78 | 64.56 | −44.78 |
| **LoRA + text-instr SFT** | **19.28** | 62.55 | 29.02 | **72.09** | −43.07 |
| Frame LoRA | 13.35 | 63.65 | 21.99 | 68.37 | −46.39 |
| Full Method | 10.34 | **62.85** | 21.59 | 61.95 | **−40.36** |

Key paper takeaways:

1. **All fine-tuning setups raise CR** (56 → 62–64), zero-shot has lowest CR. A
   reliable accuracy–consistency trade-off artefact across all mechanisms.
2. **text-instruction SFT is the only method that meaningfully lifts FCA**
   (11.24 → 19.28, a 72 % relative improvement over Naive LoRA). Every other
   method is essentially tied with Naive on FCA. **However, CR is *not*
   reduced** by text-instruction (62.55 vs zero-shot 56.43). Paired-correct
   rate (FCA) and paired-disagreement rate (CR) move on different axes — text
   instruction lifts the former, no method reliably lowers the latter.
3. Full Method has tightest |FG| but the gain comes from collapsing Person
   accuracy — we frame it as a controlled negative finding.

---

## 4. Scene leakage audit

Script: `scripts/scene_leakage_audit.py` · output: `results/scene_leakage_audit.json`.

| Metric | Value |
|---|---:|
| Train scenes (ScanNet) | 308 |
| Test scenes | 279 |
| **Scene overlap** | **275 / 279 = 98.6 %** |
| **Image overlap** | **0 / 5 250 = 0 %** |
| (question, images) overlap | 0 |

**Interpretation**: we do *not* have same-scene, same-view leakage — images are
disjoint. But the paper cannot claim scene-level generalisation on ViewSpatial
ScanNet subset. It can honestly claim:

* *Held-out-view* generalisation within seen ScanNet scenes.
* *Cross-domain* generalisation on the COCO subset of ViewSpatial.
* *Cross-dataset* generalisation on MMSI and Ego3D (those datasets have zero
  scene / image contact with our training data).

This limitation will go in the paper's Limitations section explicitly.

---

## 5. Paper-critical experiments queue

### Completed (in reverse chronological order)

| # | Experiment | Verdict |
|---|---|---|
| P0-c | Sample-level bootstrap CI (`scripts/bootstrap_ci.py`) | Frame − Naive on ViewSpatial +1.65 pp, 95 % CI [+0.93, +2.36], significant. Full − Frame on ViewSpatial −1.03, CI [−2.00, −0.05], **significant regression**. |
| P0-a | Scene leakage audit | 98.6 % scene overlap, 0 % image overlap; paper must split into ScanNet (in-domain) and COCO (OOD) subsets. |
| — | Domain-split bootstrap (`scripts/bootstrap_ci_domain.py`) | ScanNet: Frame − Naive +0.45 pp, not significant. COCO: Frame − Naive **+2.86 pp, *p* < 0.001**. Story pivots to "frame tokens help OOD". |
| — | Scene-level bootstrap (`scripts/bootstrap_ci_scene.py`) | Nearly identical to sample CIs (within-scene variance similar to between-scene). Can be kept in appendix. |
| — | Confound check (`scripts/domain_confound_check.py`) | Domain × question-type is 1-to-1 but Frame-LoRA gain on COCO is *consistent across all 3 COCO question types*, so the OOD claim is not reducible to a single question type. |

### Running

| # | Experiment | Why it matters |
|---|---|---|
| **P0-b** | Wrong-frame / no-token / always-camera inference controls on Frame LoRA | Answers "is the special token really doing work, or is it a decorative prefix?" If `wrong < correct` and `none < correct`, the token is mechanistically meaningful. |
| **P0-d** | LoRA + natural-language frame instruction SFT (same data, hyperparams) | **Fair baseline**: does the win survive when we compare learned tokens against a language prompt that is *also* fine-tuned? Three outcomes: Frame wins (strongest story), ≈ tie (moderate story — "frame-aware supervised tuning matters, token form is orthogonal"), Frame loses (rewrite to text-instr SFT as main method). |

### Will NOT run (per 2026-04-15 decision)

| # | Experiment | Reason |
|---|---|---|
| P2 | λ ∈ {0.03, 0.05} grid search | Consistency loss is confirmed to regress; chasing λ is low-EV. |
| P2 | `view_permutation_prob=0.0` ablation | Same reason. |
| P1 | Scrambled-frame-label control training | Would be compelling but ~27 h — deprioritised behind P0-d; may revisit if time allows after 2026-04-16. |

---

## 6. Full Method (consistency + permutation) — closed chapter

Training stats and the four bugs fixed along the way are preserved in
§10 below for historical accuracy. Decision for the paper:

> Soft latent consistency regularization reduces frame-gap magnitude but
> significantly hurts QA accuracy, and hurts most in the OOD visual domain
> (−2.19 pp on COCO, *p* < 0.001). We interpret this as evidence that
> unconstrained representation alignment collapses useful frame-specific
> distinctions rather than enforcing geometric consistency. Future work
> should consider explicitly geometry-supervised alignment.

This is a defensible negative finding, not a method failure — and it is more
interesting than "we found λ that works".

---

## 7. Paper decision tree — P0-d is in (2026-04-16) → activate Plan 2

### Resolved from P0-d

Δ = Frame LoRA − text-instr on COCO = **−2.89, CI [−4.34, −1.48], P(Δ>0) ≈ 0**.
This is the "Δ < −1" branch: learned frame tokens **lose** to text-instruction
SFT under identical training recipes.

### New decision (gated by token_gated result, ETA 2026-04-19)

Let Δ' = Frame-Gated LoRA (token_gated) − text-instruction SFT on COCO.

| Δ' sign & magnitude | Paper track | Implication |
|---|---|---|
| **Δ' > +1 pp, CI excludes 0** | **Paper C** — parameter-space frame conditioning is the new method | Headline: token_gated > text-instr on OOD; also run frame_gated-only ablation |
| **−1 < Δ' < +1, CI straddles 0** | **Paper B** — diagnostic comparison of 4 mechanisms | Honest tie; report as "parameter-space gating matches text instruction, neither beats the other alone" |
| **Δ' < −1, CI excludes 0** | **Paper B** — diagnostic with stronger text-instruction conclusion | text-instruction remains the winner across mechanisms; Frame-Gated becomes ablation row |

---

## 8. Full Results Summary — fixed headline tables in `results/`

* `results/summary.md` — Tables 1, 3, 4 auto-generated by
  `scripts/summarize_results.py`.
* `results/bootstrap_ci.md` — Sample-level CIs.
* `results/bootstrap_ci_scene.md` — Scene-clustered CIs (appendix).
* `results/bootstrap_ci_domain.md` — Per-domain (ScanNet vs COCO) CIs. **Primary
  source for the pivoted paper story.**
* `results/domain_confound.md` — Domain × question-type cross-tab.
* `results/scene_leakage_audit.json` — Scene / image / QA overlap.
* `results/frame_controls/viewspatial_{wrong,none,always_camera}.json` —
  Will appear when P0-b finishes (~14:48 UTC today).

---

## 9. 40-day calendar to ARR 2026-05-25

| Date | Task |
|---|---|
| 2026-04-15 | P0-b wrong-frame controls done. P0-d text-instruction SFT trained and evaluated. Scene + domain audit, all bootstrap tables current. |
| 2026-04-16 | **Pivot 3 applied**. Implement Frame-Gated LoRA (core module + ReFrameVLM integration + configs + dry-run). Launch text_instruction inference-time controls (~6 h GPU) overnight. |
| 2026-04-17 | Finish dry-run (include save/load round-trip). Launch `token_gated` training (~25 h GPU). Write method section draft for both Plan B (baseline) and Plan C (upside). |
| 2026-04-18 | token_gated training completes; eval ViewSpatial + MMSI + refresh domain-split bootstrap. |
| 2026-04-19 | **Decision point**: apply §7 decision tree. If C, launch `frame_gated`-only ablation (~25 h). Start figure 1 (motivation) and figure 2 (method). |
| 2026-04-20 → 04-25 | Related work, draft intro + results. Finalise all tables. |
| 2026-04-26 → 05-05 | Full first draft; polish diagnostic section; appendix. |
| 2026-05-06 → 05-15 | Internal review, limitations, ethics / checklist, anonymisation. |
| 2026-05-16 → 05-22 | Final polish, camera-ready formatting pass. |
| 2026-05-23 → 05-25 | ARR upload, preferred venue EMNLP 2026 main. |

---

## 10. Appendix — historical notes

### 10.1 Full Method training notes (ep1 only, stopped manually)

* Started: 2026-04-13 18:46 UTC (wandb offline-run-20260413\_184623).
* Stopped: 2026-04-14 22:20 UTC (after checkpoint-926 saved).
* Duration: 27 h 34 min wall clock, 926 optimizer steps.
* Config: λ\_consistency = 0.1, view\_permutation\_prob = 0.5, LoRA rank = 64,
  lr = 2e-5 cosine.
* Loss trajectory: qa\_loss 14.8 → 3.36; consistency\_loss 1.72 → 0.0235
  (effectively converged).
* Speed: ≈ 107 s / step vs 86 s baseline because each pair sample expands to
  two forwards.
* Checkpoint: `checkpoints/full/checkpoint-926/` (17.3 GB full-model save,
  handled by `run_benchmark.py`'s `is_reframe_full_save` branch).

### 10.2 Bugs fixed during Full Method development

1. **wandb auth** — crashed on "No API key"; relaunched with `WANDB_MODE=offline`.
2. **Consistency loss projections placed on the Trainer, not the model** —
   `nn.Linear`s never moved to GPU or included in the optimizer. Refactored
   `losses.py` to be stateless and take the model's `canonical_proj` as arg.
3. **dtype mismatch** — `relation_head` / `canonical_proj` stayed in fp32 under
   HF Accelerator despite `.to(bfloat16)` in `__init__`. Added runtime dtype
   alignment in `get_relation_logits` and `FrameCanonicalProjection.forward`.
4. **`BaselineTrainer` kwargs leak** — `pair_indices` / `frame_type_ids` leaked
   into the base-model forward for baseline / frame modes; fixed with try/except
   that strips unknown kwargs.

### 10.3 Data pipeline

* Training data: **45 809 QA** generated from ViewSpatial `visibility_data`
  (ScanNet), 308 distinct scenes, 3 544 distinct images.
* Consistency pairs: **13 408** (all camera ↔ person).
* Test benchmark: ViewSpatial-Bench 5 712 samples, 279 unique ScanNet scenes +
  2 834 COCO val2017 images.
* MMSI-Bench: 1 000 samples, 4 inline-byte images per sample, converted by
  `data/scripts/convert_mmsi_bench.py`.
* Ego3D-Bench: 8 675 samples, up to 6 outdoor nuScenes / Waymo / Argoverse
  views per sample, converted by `data/scripts/convert_ego3d_bench.py`.

### 10.4 Method implementation reality check

The file `src/model/frame_lora.py` containing the α\_f-modulated LoRA
B-matrix monkey-patch exists but is **never wired into training**. All current
"Frame LoRA" results use the simpler special-token approach
(`add_frame_tokens_to_tokenizer` + `resize_token_embeddings`). The paper
terminology should be **"Learned Frame Tokens"** or **"Frame-Aware LoRA
Tuning"**, *not* "Frame-Conditioned LoRA", to avoid a reviewer objection.

### 10.5 Code inventory (2026-04-15)

New files this cycle (beyond the original plan):

* `scripts/dryrun_full.py` — validated Full Method path end-to-end before
  the 27 h run.
* `scripts/stop_after_ep1.sh` — auto-watcher that would have killed training
  after ep1 ckpt (did not trigger because HF Trainer saves `model.safetensors`
  not `adapter_model.safetensors`; killed manually).
* `scripts/run_all_evals_overnight.sh`, `scripts/run_mmsi_only_overnight.sh`
  — orchestrators for Plan A / Plan B eval chains.
* `scripts/scene_leakage_audit.py`
* `scripts/bootstrap_ci.py`, `scripts/bootstrap_ci_scene.py`,
  `scripts/bootstrap_ci_domain.py`
* `scripts/domain_confound_check.py`
* `scripts/frame_token_controls.py` (wrong/none/always-camera inference)
* `scripts/run_text_instruction_after.sh` (P0-d orchestrator)
* `configs/train_text_instruction.yaml`
* `data/scripts/convert_mmsi_bench.py`, `convert_ego3d_bench.py`
* `scripts/summarize_results.py`

---

## 11. TODOs before ARR submission

High-priority (blocking for paper):

* [x] ~~Receive P0-d result → pick paper claim via §7 decision tree.~~ Done 2026-04-16.
* [ ] **Implement Frame-Gated LoRA (in progress, §2 step 2b).** Properly
  replace the vaporware `frame_lora.py` with identity-initialised per-layer
  gate `g_f = 1 + tanh(E_f)`, save/load round-trip verified.
* [ ] Run text-instruction inference-time controls (wrong / none / always-camera
  instruction variants) to match the Frame LoRA wrong-frame control set. Needed
  to claim whether text-instruction is *inference-time control* or (like Frame
  LoRA) mostly a *training signal*.
* [ ] Launch `token_gated` training → eval → decide Paper B vs C.
* [ ] Relabel "Frame-Conditioned LoRA" → "Learned Frame Tokens" throughout
  the codebase and all future paper drafts.
* [ ] Split every accuracy number in the paper into ScanNet (in-domain,
  same-scene-new-view) and COCO (OOD) columns.
* [ ] Write the Limitations paragraph covering scene overlap explicitly.

Medium:

* [ ] Run bootstrap with a Holm / BH multi-comparison correction for the final
  table (we do ~5 paired comparisons simultaneously).
* [ ] Produce a domain-stratified version of Table 4 (FCA/CR/FG on ScanNet vs
  COCO pairs).
* [ ] Mechanism figure: wrong-frame vs correct-frame accuracy by domain.

Low (post-submission backlog):

* [ ] Scrambled-frame-label control training (~27 h).
* [ ] λ scheduler / annealed consistency loss with explicit geometric signal.
