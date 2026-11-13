# Experimental setup: training and evaluation

How the models in [results.md](results.md) are trained and evaluated on CALVIN ABCD→D, MimicGen (26 `core` datasets) and LIBERO-10 / LIBERO-Plus / LIBERO-PRO.

**Legend.** **(FLOWER)** marks a value taken unchanged from the original FLOWER code (`intuitive-robots/flower_vla_calvin`, last upstream commit `acb32ee`). **(ours)** marks something we added or changed. **(benchmark)** marks a benchmark's own standard protocol. Unmarked rows in the per-dataset tables follow the row's source column.

## 1. Benchmarks at a glance

| | LIBERO-10 | CALVIN ABCD→D | MimicGen `core` |
|---|---|---|---|
| Tasks | 10 long-horizon tabletop tasks | 34 subtasks, chained 5 at a time | 26 datasets: 12 task families × difficulty d0–d2 |
| Scenes | several tasks share a scene (two share `LIVING_ROOM_SCENE2`) | one table, several possible tasks; the instruction picks one | one task per scene |
| Training demonstrations | 500: 50 per task (lengths 150–517 steps) | 22,966 language-annotated windows of 34–65 frames (389 distinct instructions), in environments A–D; 2,307,122 frames in total | 2,600: 100 per dataset (lengths 82–768 steps, median 270) **(ours)**: the MimicGen release has 1000 per dataset |
| Training samples | 138,090: one per frame, the action chunk padded at the episode end | 1,150,906 windows of 10 actions inside annotated windows | 818,353 windows of 10 actions (841,753 frames; no padding at the episode end) |
| Optimizer updates × global batch | 40,000 × 32 | 35,000 × 32 | 35,000 × 32 |
| Passes over the training samples | about 9.3 | about 1.0 | about 1.4 |
| Held-out data | none (validation loss is computed on training data) | environment D `validation/` split: 1,087 annotated windows, 53,923 open-loop samples | none (validation loss is computed on training data) |
| Rollouts per evaluation cell | 200 (20 per task) | 1000 chains of 5 subtasks | 2600 (100 per dataset) |
| Metric | success rate (%) | avg_seq_len (mean subtasks completed in a row, 0–5) and the share of chains completing all 5 | success rate (%) |
| 95% interval for one seed | about ±3–7 points (Wilson) | seed spread dominates (sd 0.03–0.29) | about ±1.9 points (Wilson) |

Robustness benchmarks, run only for LIBERO-10 models:

| | Tasks | Rollouts per configuration |
|---|---|---|
| LIBERO-Plus (libero_10 suite) | 2519 perturbed versions of the 10 tasks, in 7 categories and 5 difficulty levels | 2519 (one per task) |
| LIBERO-PRO (libero_10) | 4 suites (`lan`, `object`, `swap`, `task`) × 10 tasks | 200 per suite (20 per task) |

## 2. Model (shared by all benchmarks)

| Component | Setting | Source |
|---|---|---|
| Vision-language encoder | Florence-2-large (DaViT image encoder + BART-style encoder), fully fine-tuned (`freeze_florence: False`, `freeze_vision_tower: False`), about 947M parameters in total | (FLOWER) |
| Initialization | FLOWER's pretrained weights after 360k steps (`360000_model_weights.pt`, HF `mbreuss/flower_vla_pret`) | (FLOWER) |
| Image tokens | 1 pooled + h×w spatial tokens per image, one frame (`obs_seq_len: 1`): **17** per image at 112 px (LIBERO), **50** at 224 px (CALVIN, MimicGen) | (FLOWER) |
| Language | `Agent Type: 1-arm Franka Panda, Action Space: Delta End-Effector,  Task Instruction: <instruction>` (minimal prompt style): 25 tokens of template + the instruction | (FLOWER) |
| Readout | one learned `<Flow>` token, always present | (FLOWER) |
| Token dropout | 0.1 element-wise dropout on the encoder output, in every run | (FLOWER) |
| Action head | flow-matching DiT: 18 blocks, width 1024, 16 heads, cross-attention to the encoder tokens, causal attention, RoPE, AdaLN conditioning on the flow time | (FLOWER) |
| Pretrained action-head blocks | 12 blocks load the pretrained weights; 6 are randomly initialized and appended | (FLOWER) behaviour, made explicit (`pretrained_dit_layers: 12`) **(ours)** |
| RoPE θ | 1000 (upstream config: 32). The 12 pretrained blocks load 1000 from the checkpoint either way, so this affects only the 6 new blocks | **(ours)** |
| Flow matching | flow time sampled uniformly (stratified); 4 Euler steps at inference; actions clamped to [-1, 1] | (FLOWER) |
| Actions | 7-D: delta end-effector position and orientation + gripper, in [-1, 1], not further normalized | (FLOWER) / (benchmark) |
| Action chunk | 10 actions predicted per inference (`act_seq_len: 10`), all 10 executed (`multistep: 10`) | (FLOWER) |
| Proprioception (when `model.use_proprio=True`) | robot state → MLP (→1024, dropout 0.2, trained from scratch) → added to the AdaLN conditioning of every action-head block; it is not an encoder token | **(ours)**: upstream's `use_proprio` was inert (a zero encoder and a broken batch lookup) |

## 3. Training recipe (shared by all benchmarks)

| Setting | Value | Source |
|---|---|---|
| Optimizer | AdamW, learning rate 2e-5, betas (0.9, 0.95), weight decay 0.05 (0 for biases and norms), no gradient clipping | (FLOWER) |
| LR schedule | tri-stage over a nominal 50,000 steps: linear warmup from 2e-6 to 2e-5 over 2,500 steps, hold for 5,000, cosine decay towards 1e-5 over 42,500. Training stops before the decay ends: LR 1.28e-5 at 35k updates (CALVIN, MimicGen), 1.13e-5 at 40k (LIBERO) | (FLOWER) |
| Updates | 1000 optimizer updates per "epoch" (`limit_train_batches`): LIBERO 40 epochs, CALVIN and MimicGen 35 | (FLOWER) for LIBERO and CALVIN; MimicGen uses CALVIN's **(ours)** |
| Global batch | 32 in every reported run | (FLOWER): 8 per GPU × 4 GPUs |
| How the batch of 32 is formed | varies by run (below); with gradient accumulation the global batch and the update count stay the same | **(ours)** |
| Precision | bf16 mixed | (FLOWER) |
| EMA | an exponential moving average of the weights is kept and evaluated. Without NVIDIA apex (not installed in our images), the decay ramps up as 1 − (1 + step)^(−2/3): 0.99 by step 1000, 0.9991 at 35k. The configured `decay: 0.999` is only used with apex | (FLOWER) |
| Checkpoint | the last one (`last.ckpt`, EMA weights); no checkpoint selection | **(ours)**: upstream kept the best checkpoint by in-training rollouts |
| In-training rollouts | disabled; evaluation runs after training | **(ours)** |
| Seeds | 42, 5 and 100 for every model (101 replaces a degraded seed-100 run for one CALVIN model) | **(ours)** |

How each run formed its global batch of 32 (GPUs × micro-batch × accumulation):
- **LIBERO:** 1 × 32 × 1 for nearly all runs; the two 2026-11-02 runs used 2 × 16 × 1.
- **CALVIN:** the original seed-42 runs used 2 × 4 × 4; extra seeds and later models 2 × 8 × 2; a few runs 4 × 4 × 2 or 4 × 8 × 1.
- **MimicGen:** most runs 2 × 8 × 2; the first dropout runs 2 × 16 × 1; no dropout seed 100 used 4 × 8 × 1.

## 4. Per-dataset differences

| | LIBERO-10 | CALVIN ABCD→D | MimicGen `core` |
|---|---|---|---|
| Config | `conf/config_libero.yaml` | `conf/config_calvin.yaml` | `conf/config_mimicgen.yaml` **(ours)** |
| Launch | `./run.sh train[-dropout] libero_10` | `./run.sh train-calvin[-dropout]` | `./run.sh train-mimicgen[-dropout]` **(ours)** |
| Updates | 40,000 | 35,000 | 35,000 |
| Static camera | `agentview_rgb`, 128×128 → 112×112 | `rgb_static`, 200×200 → 224×224 | `agentview_image`, rendered at 200×200 to match CALVIN **(ours)** → 224×224 |
| Wrist camera | `eye_in_hand_rgb`, 128×128 → 112×112 | `rgb_gripper`, 84×84 → 224×224 | `robot0_eye_in_hand_image`, rendered at 84×84 **(ours)** → 224×224 |
| Tokens per image | 17 | 50 | 50 |
| Training augmentation | random shift (pad 10 static, 4 wrist), CLIP normalization | same, at 224 px | same as CALVIN |
| Language | the task's description, 1 per task | crowd-sourced annotation, 389 distinct instructions | one instruction per task family **(ours)** (`flower/datasets/mimicgen_tasks.py`) |
| Prompt length (tokens, incl. template) | 33–44 | 28–37 | 31–35 |
| Proprioception (proprio runs) | 9-D: joint positions (7) + gripper finger positions (2) | 8-D: end-effector position (3) + orientation (3) + gripper width (1) + gripper action (1) | 9-D: joint positions (7) + gripper finger positions (2) |
| Actions | `rel_actions`, delta EEF (OSC) | `rel_actions` (pre-extracted) | `actions`, OSC_POSE delta |
| Training sample | every frame, chunk padded at the episode end | every 10-action window inside an annotated window | every 10-action window inside a demo (no end padding) **(ours)** |
| Validation loss | on training data (`split_ratio: 0.0`) | on the env D validation split | on training data |

## 5. What we added for the experiments (all ours)

**Modality dropout in training** (`model.modality_dropout=True`, `flower/models/networks/modality_dropout.py`). Inputs are removed per training sample before the encoder. The remaining tokens keep their original positional embeddings, so the encoder sees the same positions as with every input present. Three strategies:
- **Dirichlet** (default): a fixed budget of n = round(f · min over the batch of the available tokens) tokens, split between static, wrist and language by proportions drawn from Dirichlet(1, 1, 1). Each modality keeps a random subset of its tokens, so it is usually thinned out and sometimes removed entirely. Keep fraction f = 0.75, 0.5 or 0.25.
- **Uniform:** n tokens drawn uniformly from all tokens, regardless of modality, so a whole modality is almost never removed.
- **Subset:** a non-empty subset S of {static, wrist, language, proprio} is kept whole, with P(S) ∝ 2^|S|. All inputs are kept 20% of the time and each input is present in 67.5% of samples.

**Proprio gate:** with Dirichlet or uniform, proprioception is kept with probability *proprio keep p*, independently of the tokens. Subset decides it as one of the four inputs.

`./run.sh simulate-sampling` (`scripts/simulate_modality_sampling.py`) writes out the exact algorithm as equations and simulates it with these token counts. Its outputs are histograms of kept tokens and the share of samples in which each modality is empty, partly kept or whole.

**Training without an input** (`model.modalities.<input>=False`): the input is never encoded, in training or evaluation. We used it to train the no dropout + proprio recipe without one input.

**Proprioception wiring:** upstream's `use_proprio` had no effect; we implemented it (Section 2).

**Trained configurations.** Every model has seeds 42, 5 and 100 unless noted; the per-run list is in [results.md, Reproduction](results.md#reproduction).

| Model | LIBERO-10 | CALVIN | MimicGen |
|---|---|---|---|
| no dropout | ✓ | ✓ | ✓ |
| no dropout + proprio | ✓ | ✓ | ✓ |
| Dirichlet keep 0.75 | | ✓ | ✓ (proprio keep 0.5; seeds 42, 5) |
| Dirichlet keep 0.5 | ✓ | ✓ | |
| Dirichlet keep 0.25 | | ✓ | |
| uniform keep 0.5 | | ✓ | ✓ (proprio keep 0.5) |
| Dirichlet keep 0.5 + proprio, proprio keep 1.0 / 0.75 / 0.5 / 0.25 | all four | 1.0 (seeds 42, 5, 101) and 0.5 | 1.0 and 0.5 |
| `subset` + proprio | ✓ | ✓ | ✓ |
| no dropout + proprio, trained without one input (static camera, wrist camera, language) | ✓ | ✓ (without language: seeds 42, 5) | without the static camera ✓; without the wrist camera seeds 42, 100; without language seeds 42, 100 |

The model trained without proprioception is the plain **no dropout** model. Missing seeds were being trained or evaluated as of 2026-11-12; results.md lists what each result is based on.

## 6. Evaluation protocol (shared)

| Setting | Value | Source |
|---|---|---|
| Weights | EMA weights of `last.ckpt` | (FLOWER) EMA; last checkpoint **(ours)** |
| Inference | 4 Euler steps; all 10 predicted actions executed before the next inference | (FLOWER) |
| Seeding | each episode's sampling noise comes from its own generator, seeded by (base seed, task, episode index). The environment reset is seeded the same way. Every model and input combination therefore sees the same episodes and the same noise, regardless of batching or GPU sharding | **(ours)** |
| Batching | 10 episodes in parallel in subprocess environments (LIBERO, MimicGen); CALVIN runs one environment per GPU | **(ours)**: upstream ran one environment at a time |
| Input combinations | models without dropout: their trained inputs, plus (for no dropout + proprio and the models trained without one input) each input withheld in turn. Dropout models: every non-empty subset of {static, wrist, language}, each with and without proprioception (7 or 14 combinations). A withheld input is removed before the encoder, as in training | **(ours)** |
| Records | one CSV row per episode, uploaded as a W&B `evaluation` artifact per run; [results.md](results.md) is computed from these | **(ours)** |

## 7. Per-benchmark evaluation

| | LIBERO-10 | CALVIN ABCD→D | MimicGen `core` |
|---|---|---|---|
| Rollouts per cell | 20 per task × 10 = 200. The LIBERO standard is 50 per task; upstream FLOWER's evaluation also used 20 | 1000 chains (benchmark). Upstream FLOWER's eval config had 100 | 100 per dataset × 26 = 2600 **(ours)** |
| Initial states | the first 20 of the 50 fixed initial states per task (benchmark files) | each chain's initial scene state (benchmark); the same 1000 chains in every evaluation | none exist: random reset, seeded per episode, then 5 zero-action warm-up steps |
| Horizon | 520 steps (FLOWER) | 360 steps per subtask; a chain stops at its first failure (benchmark, FLOWER) | per task family: MimicGen's training-rollout horizon + 25% **(ours)**: 500 (square, stack, stack_three, threading, coffee), 625 (three_piece_assembly, hammer_cleanup, mug_cleanup, nut_assembly), 800 (coffee_preparation, unchanged), 1000 (kitchen), 1250 (pick_place) |
| Rendering | 224×224, resized to 112 for the model | 200×200 static, 84×84 wrist, resized to 224 | 200×200 / 84×84, resized to 224 |
| Success | task predicate (benchmark) | subtask predicate per subtask (benchmark) | the environment's task success check (benchmark) |

**LIBERO-Plus** (`conf/eval_libero_plus.yaml`):
- **Tasks:** the full libero_10 suite, 2519 tasks, one episode each, the original task's initial state, 520 steps.
- **Categories:** Camera Viewpoints 419, Robot Initial States 393, Language Instructions 383, Light Conditions 274, Background Textures 289, Sensor Noise 449, Objects Layout 312.
- **Difficulty levels 1–5:** 357 / 450 / 484 / 523 / 705 tasks.
- **Passes:** run once with all inputs, then once with each input withheld (static camera, wrist camera, language, proprioception), giving up to 5 × 2519 episodes per model **(ours)**.
- **Instruction:** taken from the original task, except in the Language Instructions category, which uses its rewritten instruction **(ours)**.
- **Masked inputs (ours):** the no-dropout models are also run with one input masked instead of removed, on LIBERO-10 and LIBERO-Plus (and on CALVIN and MimicGen): a camera's frames replaced by their per-channel mean pixel, the instruction blanked with the prompt template kept, or proprioception replaced by its training-mean conditioning vector.

**LIBERO-PRO** (`conf/eval_libero_pro.yaml`):
- **Suites:** `lan` (reworded instruction), `task` (new goal), `swap` (swapped positions), `object` (new object meshes); 10 tasks × 20 episodes each.
- **`native` arm:** each suite's own initial states.
- **`matched` arm (ours):** `lan` and `task` are also run from the original LIBERO-10 initial states, paired episode by episode with the original evaluation, across input combinations.

**CALVIN open loop** (`scripts/open_loop_calvin.py`, **ours**):
- **Data:** the held-out environment D validation split, 1,087 annotated windows, 53,923 samples (one frame plus its 10-step action chunk).
- **Losses:** the predicted-action MSE of the 4-step sampled chunk, and the flow-matching loss at five fixed flow times.
- **Seeding:** noise is seeded per frame.
- **Input combinations:** all token-input combinations for dropout models (with proprio withheld too, for models trained with it). Models without dropout get their trained inputs; the no dropout + proprio models also get each input withheld in turn: cameras replaced by their mean pixel, the instruction blanked with the prompt template kept, proprio zeroed.

**Which results count.**
- LIBERO results use only runs trained with commit `ffba3d2` or later, and only evaluations from commit `ca969a8` on. Before `ca969a8`, environment resets weren't seeded.
- MimicGen evaluations were all rerun at 100 episodes per dataset.

## 8. Deviations from the standard protocols

- **LIBERO-10 evaluation:** 20 episodes per task instead of the standard 50, as in FLOWER.
- **LIBERO and MimicGen validation:** no held-out split; the validation loss is computed on training data and is not used for model selection.
- **MimicGen training data:** 100 demonstrations per dataset instead of the release's 1000, re-rendered at CALVIN's camera resolutions (200×200 static, 84×84 wrist).
- **MimicGen horizons:** 25% longer than MimicGen's own rollout horizons (coffee_preparation stays at 800).
- **MimicGen language:** MimicGen has no language annotations; we wrote one instruction per task family.
- **Model selection:** none. Every result uses the last checkpoint, where FLOWER selected the best checkpoint by rollouts.
- **Learning rate:** training stops at 35k / 40k updates of a 50k-step schedule, so the LR never reaches its final value (as in FLOWER).

## 9. Where to find it

- **Model and training:** `conf/model/flower.yaml`, `conf/config_{libero,calvin,mimicgen}.yaml`, `conf/datamodule/`, `flower/models/flower.py`, `flower/models/networks/modality_dropout.py`.
- **Evaluation:** `conf/eval_{calvin,libero,libero_plus,libero_pro,mimicgen}.yaml`, `flower/evaluation/`, `scripts/eval_pipeline.py`, `scripts/calvin_pipeline.py`, `scripts/mimicgen_pipeline.py`.
- **Commands:** [results.md, Reproduction](results.md#reproduction) and [README.md](../README.md).
