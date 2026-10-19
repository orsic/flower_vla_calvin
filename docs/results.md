# Modality dropout for FlowerVLA: CALVIN and MimicGen results

Results of fine-tuning FlowerVLA with **modality-token dropout**, which withholds part of the camera and language tokens during training. Models are compared with ordinary fine-tuning on two simulated benchmarks:

- **CALVIN ABCD→D:** language-conditioned, long-horizon tabletop manipulation, evaluated closed loop and open loop per timestep.
- **MimicGen `core`:** 26 single-task robosuite datasets, evaluated closed loop.

Every number below comes from the evaluation files of the runs listed in [Reproduction](#reproduction). Success rates are percentages of episodes.

## Summary

- **Light dropout costs nothing with all inputs and lets the model run without some of them.**
  - On CALVIN, Dirichlet dropout that removes 25% of the tokens matches ordinary fine-tuning with every input: over 3 seeds, an average chain of 4.44 ± 0.03 vs 4.44 ± 0.05 completed subtasks.
  - Without dropout, a model cannot be given only part of its inputs at all.
- **The amount of dropout is a trade-off** (3 seeds).
  - Removing 50% of the tokens lowers the all-inputs result (4.19 ± 0.13) but gives by far the best runs without a camera: 3.76 ± 0.19 with the static camera and language, 2.58 ± 0.27 with the wrist camera and language (keep 0.75: 1.72 and 1.62).
  - Removing 75% hurts every combination, including all inputs (1.72), which is lower than some subsets.
- **Removing whole modalities in training matters, not just removing tokens.** The *uniform* variant drops a random 50% of all tokens and rarely removes a whole camera. Over two seeds it does worse with all inputs (3.66 vs 4.19 for Dirichlet) and collapses without the wrist camera (0.30 vs 3.76).
- **Language is indispensable on CALVIN and nearly irrelevant on MimicGen.**
  - CALVIN puts several tasks in one scene, so the instruction picks the task. Without language, no CALVIN model completes more than about 0.2 subtasks per chain on average.
  - Each MimicGen scene has one task, so dropping language changes success by at most a few points.
- **Throughout a CALVIN subtask (open loop), language matters most at the start and vision matters most at the end.**
  - Removing language costs most in the first ~20 frames, when the target is chosen.
  - With language only, the prediction error grows steadily towards the end of the subtask, when positioning and release need visual feedback.
  - Release is the hardest phase for every model with all inputs.
- **Open-loop error does not predict closed-loop success with all inputs.** All five CALVIN models reach the same per-step action error on held-out demonstrations (0.062–0.065), although their chain lengths range from 1.7 to 4.4.
- **On MimicGen, dropout with proprioception costs 5–7 points with all inputs** (46.6% / 48.4% vs 53.2%). Proprioception itself is neutral: without dropout it gives 53.0%.
  - Both cameras matter. Without the wrist camera success falls to about 20%, without the static camera to about 10%.
  - Proprioception has to be dropped in training too: a model that always saw it falls from 48.4% to 18.0% without it.

## Setup

**Model and fine-tuning.**
- FlowerVLA, starting from the pretrained 360k-step weights, with a Florence-2 encoder over static-camera, wrist-camera and language tokens and an 18-block flow-matching action head.
- It predicts a 10-step action chunk from one observation.
- Each run: 35 epochs × 1000 optimizer updates, global batch 32, seed 42.
- CALVIN no dropout, Dirichlet keep 0.75 and Dirichlet keep 0.5 were also trained with seeds 5 and 100, using batch 8 × 2 accumulation instead of 4 × 4 (same global batch); uniform keep 0.5 with seed 100 the same way. No dropout with proprioception was trained with seeds 42 and 5.
- Evaluated with the EMA weights of the last checkpoint, 4 sampling steps, and all 10 predicted actions executed.

**Modality dropout** (`model.modality_dropout=True`).
- During training each sample keeps a fraction of the camera and language tokens: the *keep fraction* (0.75 / 0.5 / 0.25, i.e. 25 / 50 / 75% removed).
- How the kept tokens are split between modalities:
  - **Dirichlet** (default): split by a per-sample draw from Dirichlet(1, 1, 1). Whole modalities are often starved or dropped.
  - **uniform:** a uniformly random subset of all tokens, so each modality keeps about the same share and a whole camera is almost never removed.
- Proprioception, where used (MimicGen), is kept with probability *proprio keep p*.
- At evaluation, a modality combination withholds whole modalities from the model.

**Evaluation.**
- **CALVIN:** the standard 1000 chains of 5 subtasks in environment D.
  - *avg_seq_len* is the mean number of subtasks completed in a row (out of 5); 5/5 is the share of chains completing all five.
  - Every combination uses the same 1000 chains and noise seeds.
- **MimicGen:** each dataset's own success criterion and rollout horizon (500–1250 steps).
  - Every run: 100 episodes per dataset and input combination (2600 per cell over the 26 datasets), 95% Wilson intervals about ±1.9 points.
  - All MimicGen evals were rerun on 2026-10-17/19 with the same code and settings. Episodes are seeded by dataset and episode index, and the rerun reproduced 99.8–100% of the earlier episode outcomes.

## CALVIN ABCD→D

### Closed loop

#### Seed 42, all models

avg_seq_len, with the share of 5/5 chains in brackets, over 1000 chains. "n/a" marks models not evaluated on that combination: the no-dropout model is only meaningful with all inputs.

| inputs | no dropout | Dirichlet keep 0.75 | Dirichlet keep 0.5 | Dirichlet keep 0.25 | uniform keep 0.5 |
|---|---|---|---|---|---|
| static+wrist+lang | 4.403 (79.4) | **4.436 (79.4)** | 4.093 (67.0) | 1.717 (9.3) | 3.551 (46.1) |
| static+wrist | n/a | 0.185 (0.0) | 0.166 (0.0) | 0.141 (0.0) | 0.172 (0.0) |
| static+lang | n/a | 1.571 (12.0) | **3.885 (57.5)** | 2.196 (17.0) | 0.292 (0.0) |
| wrist+lang | n/a | 1.498 (5.2) | **2.716 (23.1)** | 1.208 (2.1) | 1.149 (2.1) |
| static | n/a | 0.060 (0.0) | 0.196 (0.1) | 0.135 (0.0) | 0.034 (0.0) |
| wrist | n/a | 0.081 (0.0) | 0.145 (0.0) | 0.117 (0.0) | 0.084 (0.0) |
| lang | n/a | 0.000 (0.0) | 0.000 (0.0) | 0.014 (0.0) | 0.000 (0.0) |

Chain success by position with all inputs (% of chains completing at least *k* subtasks):

| model | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | avg_seq_len |
|---|---|---|---|---|---|---|
| no dropout | 96.8 | 92.2 | 88.1 | 83.8 | 79.4 | 4.40 |
| Dirichlet keep 0.75 | 97.0 | 93.6 | 89.2 | 84.4 | 79.4 | 4.44 |
| Dirichlet keep 0.5 | 96.0 | 89.5 | 82.1 | 74.7 | 67.0 | 4.09 |
| Dirichlet keep 0.25 | 70.9 | 44.8 | 28.9 | 17.8 | 9.3 | 1.72 |
| uniform keep 0.5 | 96.2 | 84.4 | 70.1 | 58.3 | 46.1 | 3.55 |

Findings:
- **Language is required.** Without it, no model gets past about 0.2 subtasks per chain on average (static+wrist 0.14–0.19), against 1.7–4.4 when language is added.
- **The static camera is worth more than the wrist camera in the Dirichlet models.** Static+lang beats wrist+lang at every keep fraction.
- **Keep 0.75 is good with all inputs but weak with any input missing**, because it rarely trained without a camera.
- **Keep 0.25 trained on so few tokens that all inputs (1.72) is worse than static+lang (2.20).** The full token count is out of distribution for it.
  - Its failures with all inputs concentrate on a few subtasks: `close_drawer` 0%, `open_drawer` 3.5%, `turn_off_lightbulb` 23.5% (see the [appendix](#calvin-per-subtask-success-all-inputs)).
- **Uniform dropout fails without the wrist camera** (static+lang 0.29). It is the only model where wrist+lang beats static+lang.
  - With all inputs it is worse than Dirichlet at the same keep fraction (3.55 vs 4.09).
  - Most of that loss is one subtask: `place_in_slider` succeeds 7.4% of the time, against 76–95% for the other models.

#### Several seeds

No dropout, Dirichlet keep 0.75 and Dirichlet keep 0.5 were each trained with seeds 42, 5 and 100; uniform keep 0.5 with seeds 42 and 100 only (its seed-5 run does not exist). Mean ± standard deviation over the seeds of avg_seq_len, and of the share of 5/5 chains (%), each seed over the same 1000 chains:

| inputs | no dropout | Dirichlet keep 0.75 | Dirichlet keep 0.5 | uniform keep 0.5 (2 seeds) |
|---|---|---|---|---|
| static+wrist+lang | **4.444 ± 0.051** (81.0 ± 2.3) | **4.442 ± 0.029** (79.3 ± 0.4) | 4.191 ± 0.131 (71.2 ± 4.9) | 3.657 ± 0.151 (51.5 ± 7.6) |
| static+wrist | n/a | 0.174 ± 0.028 (0.0) | 0.187 ± 0.029 (0.0) | 0.172 ± 0.000 (0.0) |
| static+lang | n/a | 1.724 ± 0.159 (13.8 ± 2.0) | **3.759 ± 0.190** (54.6 ± 5.1) | 0.296 ± 0.005 (0.1 ± 0.1) |
| wrist+lang | n/a | 1.617 ± 0.126 (5.5 ± 0.7) | **2.581 ± 0.272** (21.0 ± 5.5) | 1.126 ± 0.032 (1.6 ± 0.7) |
| static | n/a | 0.050 ± 0.010 (0.0) | 0.173 ± 0.020 (0.0) | 0.036 ± 0.002 (0.0) |
| wrist | n/a | 0.088 ± 0.017 (0.0) | 0.149 ± 0.009 (0.0) | 0.090 ± 0.008 (0.0) |
| lang | n/a | 0.000 (0.0) | 0.000 (0.0) | 0.000 (0.0) |

avg_seq_len of each seed with all inputs:

| model | seed 42 | seed 5 | seed 100 |
|---|---|---|---|
| no dropout | 4.403 | 4.501 | 4.428 |
| Dirichlet keep 0.75 | 4.436 | 4.417 | 4.474 |
| Dirichlet keep 0.5 | 4.093 | 4.340 | 4.141 |
| uniform keep 0.5 | 3.551 | n/a | 3.764 |

Chain success by position with all inputs, mean over the seeds (%):

| model | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | avg_seq_len |
|---|---|---|---|---|---|---|
| no dropout | 96.7 | 92.6 | 89.1 | 85.0 | 81.0 | 4.44 |
| Dirichlet keep 0.75 | 97.4 | 93.6 | 89.3 | 84.6 | 79.3 | 4.44 |
| Dirichlet keep 0.5 | 95.8 | 90.2 | 83.9 | 78.1 | 71.2 | 4.19 |
| uniform keep 0.5 (2 seeds) | 95.4 | 84.9 | 72.1 | 61.9 | 51.5 | 3.66 |

Findings:
- **Keep 0.75 and no dropout are equal with all inputs** (4.44 each). The small seed-42 lead of keep 0.75 (4.436 vs 4.403) is seed noise.
- **Keep 0.5 costs about 0.25 with all inputs, consistently.** All three of its seeds score below all six seeds of the other two models. Seed 42 was its worst seed, so the seed-42 table overstates the cost (0.31).
- **Keep 0.5's robustness without a camera is far larger than the seed spread.** Its lead over keep 0.75 is 2.0 with static+lang and 1.0 with wrist+lang, against standard deviations of 0.13–0.27.
- **Keep 0.5 varies most across seeds**, both with all inputs (± 0.13) and with wrist+lang (± 0.27). With three seeds the standard deviations are themselves rough.
- **Uniform's failure without the wrist camera is reproducible.** Static+lang is 0.292 and 0.299 for its two seeds, against 3.76 for Dirichlet at the same keep fraction.
- **Uniform is also worse with all inputs** (3.66 vs 4.19 for Dirichlet keep 0.5). Seed 100 (3.764) is better than seed 42 (3.551), mostly through `place_in_slider`: 42.6% vs 7.4% success, against 76–96% for every other model. That subtask is uniform's weak point in both seeds, but how weak varies with the seed.

#### No dropout with proprioception

The no-dropout recipe with the robot state added as an input (`model.use_proprio=True proprio_dims=8`: gripper position, orientation, width and open/close state), seeds 42 and 5, all inputs, 1000 chains each. These two runs trained on one A100 40 GB each, 35k optimizer steps like every other run; their batch split is not recorded in W&B.

| model | seed 42 | seed 5 | mean ± sd | 5/5 (%) |
|---|---|---|---|---|
| no dropout | 4.403 | 4.501 | 4.452 ± 0.069 | 81.5 ± 3.0 |
| no dropout + proprio | 4.467 | 4.353 | 4.410 ± 0.081 | 79.9 ± 2.0 |

Chain success by position, mean over seeds 42 and 5 (%):

| model | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | avg_seq_len |
|---|---|---|---|---|---|---|
| no dropout | 96.8 | 92.6 | 89.1 | 85.3 | 81.5 | 4.45 |
| no dropout + proprio | 96.2 | 91.9 | 88.3 | 84.7 | 79.9 | 4.41 |

- **Proprioception does not measurably change CALVIN performance.** The per-seed differences go in opposite directions (+0.06 at seed 42, −0.15 at seed 5), and both are within the seed-to-seed spread of the model without it (4.40–4.50).
- These runs cannot show how much the model relies on proprioception: without dropout it is only evaluated with it. On MimicGen, withholding proprioception from a model that always saw it cost over 30 points.

### Open loop along the demonstrations

**Method.**
- **Data:** the CALVIN `validation/` split, environment D demonstrations that no run trains on.
  - It has 1,087 language-annotated subtask windows of 34–65 frames (77% are 65 frames), covering 34 subtask types.
  - Every observation frame whose 10-step action chunk stays inside its window is one sample: 53,923 in total.
- **Losses:** for each model and input combination, two per-sample losses against the demonstrated actions:
  - **predicted-action MSE:** the error of the action chunk the policy samples, as in a rollout;
  - **training loss:** the flow-matching objective at five fixed noise levels.
- **Determinism:** noise is seeded per frame, so every model and combination sees the same draws.
- **Batching:** batches never mix subtask windows. Language is padded to the batch's longest instruction, and the action head attends to the padded positions, so in early test runs that mixed instructions the padding changed the loss by up to 0.18.
  - Within one window every sample has the same instruction, so no padding occurs, the same as in a rollout. Combinations without language have no text tokens at all.
  - Check: 202 samples (one per task × 10-frame bin) were re-scored per model and combination with batch size 1. No pad tokens appeared. The re-scored losses match the reported ones to within 5e-5, far below the 0.01–0.04 effects reported.

**Annotations per sample.** Each sample records:
- its subtask;
- its timestep: frames since the start of the annotated window;
- a heuristic gripper phase from the gripper command over the window:
  - *approach*: before the first close;
  - *grasp*: within ±2 frames of closing;
  - *hold*: closed;
  - *release*: within ±2 frames of opening;
  - *retreat*: after reopening.

The gripper closes at least once in every window, so pushes and button presses also get approach/hold labels.

Mean predicted-action MSE over all samples (lower is better):

| inputs | no dropout | Dirichlet keep 0.75 | Dirichlet keep 0.5 | Dirichlet keep 0.25 | uniform keep 0.5 |
|---|---|---|---|---|---|
| static+wrist+lang | 0.062 | 0.063 | 0.065 | 0.065 | 0.064 |
| static+wrist | n/a | 0.091 | 0.092 | 0.088 | 0.092 |
| static+lang | n/a | 0.112 | 0.073 | 0.075 | 0.173 |
| wrist+lang | n/a | 0.076 | 0.072 | 0.076 | 0.085 |
| static | n/a | 0.163 | 0.107 | 0.109 | 0.224 |
| wrist | n/a | 0.116 | 0.105 | 0.103 | 0.127 |
| lang | n/a | 0.296 | 0.302 | 0.260 | 0.310 |

**With missing inputs, the ordering of the models follows the closed-loop results:**
- uniform is worst without the wrist camera;
- keep 0.75 is worse than keep 0.5 without the wrist camera.

**With all inputs, every model scores the same (0.062–0.065), unlike in closed loop.** Predicting the next actions from a demonstration's states is not where the models differ. The difference shows up once a model's own actions take it off the demonstrated states.

Predicted-action MSE along the subtask, one panel per model:

![Predicted-action MSE along the subtask per model](results/loss_vs_timestep_mse_mean.png)

The same, as the increase over each model's own all-inputs error. All inputs is the dashed zero line.

![Increase in MSE from withholding inputs along the subtask](results/delta_vs_timestep_mse_mean.png)

One panel per input combination, models overlaid (the no-dropout model only has all inputs):

![MSE along the subtask per input combination](results/combo_vs_timestep_mse_mean.png)

**Cost of each modality over time.** The tables give the increase in predicted-action MSE over all inputs from removing one modality, or from keeping language only, in 10-frame bins. The all-inputs error itself is about 0.06–0.07.

Dirichlet keep 0.5:

| frames | no static | no wrist | no lang | lang only |
|---|---|---|---|---|
| 0–9 | 0.009 | 0.008 | 0.034 | 0.214 |
| 10–19 | 0.008 | 0.010 | 0.034 | 0.205 |
| 20–29 | 0.006 | 0.007 | 0.029 | 0.209 |
| 30–39 | 0.004 | 0.007 | 0.017 | 0.237 |
| 40–49 | 0.006 | 0.005 | 0.018 | 0.297 |
| 50–59 | 0.010 | 0.013 | 0.028 | 0.338 |

Uniform keep 0.5:

| frames | no static | no wrist | no lang | lang only |
|---|---|---|---|---|
| 0–9 | 0.019 | 0.094 | 0.032 | 0.182 |
| 10–19 | 0.014 | 0.082 | 0.033 | 0.178 |
| 20–29 | 0.019 | 0.098 | 0.030 | 0.206 |
| 30–39 | 0.020 | 0.125 | 0.019 | 0.258 |
| 40–49 | 0.024 | 0.137 | 0.020 | 0.369 |
| 50–59 | 0.048 | 0.150 | 0.031 | 0.409 |

Dirichlet keep 0.75 and 0.25 follow keep 0.5's pattern. One difference: keep 0.75 also loses 0.04–0.06 throughout without the wrist camera.

**How each modality matters through a subtask:**
1. **Language matters most early, while the target is chosen.** Removing it costs about 0.03 in the first 20 frames and about half that in frames 30–49, then rises slightly at the very end.
2. **Vision matters more as the subtask proceeds.** With language only, the gap grows from about 0.2 in the first frames to 0.34–0.41 in the last.
3. **Whether the two cameras can stand in for each other depends on training.**
   - In Dirichlet keep 0.5 and 0.25, removing either camera costs ≤ 0.01 for most of the subtask; both matter a little more in the last ~10 frames.
   - The uniform model never learned to do without the wrist camera. Its cost grows from 0.08 to 0.15 over the subtask, i.e. it leans on the wrist view most near contact.

Predicted-action MSE by gripper phase, one heatmap per model:

![MSE by gripper phase](results/phase_mse_mean.png)

**With all inputs, release is the hardest phase for every model**, even though it is only 2.8% of the samples. It is also the hardest for most reduced combinations; the exception is the uniform model without the wrist camera, which does worst while grasping and holding. All-inputs MSE by phase is 0.15–0.16 at release vs 0.04 during hold. Approach is 34% of the samples, grasp 8%, hold 48% and retreat 7%.

**Caveats.**
- CALVIN's annotated windows are cut around the moment a subtask is completed, so frame 0 is an approximate start.
- Only the 65-frame windows reach the last timesteps; bins with fewer than 200 samples are not drawn.
- The 95% intervals in the figures treat overlapping frames of one window as independent, so they are too narrow.

## MimicGen

All 26 MimicGen `core` datasets (12 task families, difficulty variants d0–d2), 100 demonstrations each, are trained jointly with the CALVIN fine-tuning recipe. Unlike CALVIN, each scene contains a single task.

### Modality combinations

Success rate (%) per input combination, on 100 episodes per dataset × 26 datasets = 2600 per cell (95% Wilson interval about ±1.9 points; ±1.4 for the two pooled proprio seeds).
- The dropout runs use Dirichlet keep 0.5, with proprioception, and are evaluated on every combination.
- The two no-dropout runs are evaluated with all inputs only: without proprioception (seed 42), and with it (seeds 42 and 5, pooled: 53.1% and 52.9%).

| inputs | no dropout | no dropout + proprio (2 seeds) | dropout, proprio keep 0.5 | dropout, proprio keep 1.0 |
|---|---|---|---|---|
| static+wrist+lang+proprio | n/a | 53.0 | 46.6 | 48.4 |
| static+wrist+lang | **53.2** | n/a | 45.5 | 18.0 |
| static+wrist+proprio | n/a | n/a | 46.3 | 46.5 |
| static+wrist | n/a | n/a | 45.5 | 18.0 |
| static+lang+proprio | n/a | n/a | 20.1 | 17.7 |
| static+lang | n/a | n/a | 18.6 | 6.6 |
| wrist+lang+proprio | n/a | n/a | 10.0 | 18.3 |
| wrist+lang | n/a | n/a | 6.2 | 2.0 |
| static+proprio | n/a | n/a | 19.1 | 17.4 |
| static | n/a | n/a | 19.4 | 6.5 |
| wrist+proprio | n/a | n/a | 9.2 | 15.3 |
| wrist | n/a | n/a | 5.6 | 2.0 |
| lang+proprio | n/a | n/a | 0.0 | 0.0 |
| lang | n/a | n/a | 0.0 | 0.0 |

Findings:
- **Language barely matters.** Without it, success is 46.3% (proprio keep 0.5) and 46.5% (keep 1.0), against 46.6% and 48.4% with it. The scene already determines the task.
- **Both cameras are needed, and they complement each other.**
  - Removing the wrist camera drops success from 46.6% to 20.1%; removing the static camera drops it to 10.0% (proprio keep 0.5).
  - Either camera alone gives 6–19% (proprio keep 0.5).
  - This differs from CALVIN, where the Dirichlet models lose little from one missing camera. MimicGen's precise insertion and assembly tasks need both views at once.
- **Proprioception has to be dropped in training too.**
  - With keep 1.0 (proprioception always present in training), removing it collapses success: 48.4% → 18.0% with every other input.
  - With keep 0.5 the model is indifferent to it (46.6% vs 45.5%).
  - With all inputs the two keep values are not distinguishable (48.4% vs 46.6%).
- **Proprioception alone does not help.** Without dropout, adding it gives 53.0% against 53.2% without.
- **Dropout with all inputs is 5–7 points below no dropout** (46.6% / 48.4% vs 53.2%). The gap is largest on `threading_d0` (28% / 23% vs 62%), `pick_place_d0` (35% / 24% vs 47%) and `coffee_preparation_d1` (14% / 18% vs 35%; see the [appendix](#mimicgen-per-dataset-success-all-inputs)).
  - On `threading_d0` the no-dropout runs with proprioception are low too (37% / 42%), so that gap comes partly from proprioception, not from dropout.

### Task family × difficulty, all inputs (%)

Each model is evaluated with all of its inputs; "+ proprio" is no dropout with proprioception, pooled over seeds 42 and 5.

| family | no dropout d0 | d1 | d2 | + proprio d0 | d1 | d2 | proprio keep 0.5 d0 | d1 | d2 | proprio keep 1.0 d0 | d1 | d2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coffee | 95 | 65 | 14 | 86 | 74 | 27 | 88 | 50 | 10 | 88 | 73 | 14 |
| coffee_preparation | 61 | 35 | – | 81 | 34 | – | 39 | 14 | – | 64 | 18 | – |
| hammer_cleanup | 98 | 65 | – | 98 | 58 | – | 89 | 48 | – | 94 | 53 | – |
| kitchen | 100 | 68 | – | 100 | 62 | – | 100 | 68 | – | 100 | 51 | – |
| mug_cleanup | 67 | 46 | – | 62 | 46 | – | 50 | 39 | – | 56 | 36 | – |
| nut_assembly | 15 | – | – | 21 | – | – | 17 | – | – | 11 | – | – |
| pick_place | 47 | – | – | 44 | – | – | 35 | – | – | 24 | – | – |
| square | 71 | 38 | 31 | 79 | 40 | 22 | 69 | 41 | 26 | 67 | 47 | 36 |
| stack | 99 | 99 | – | 100 | 99 | – | 99 | 100 | – | 100 | 100 | – |
| stack_three | 82 | 71 | – | 86 | 70 | – | 89 | 73 | – | 81 | 74 | – |
| threading | 62 | 10 | 3 | 40 | 4 | 4 | 28 | 6 | 3 | 23 | 6 | 2 |
| three_piece_assembly | 21 | 15 | 5 | 22 | 18 | 2 | 18 | 10 | 2 | 21 | 15 | 4 |

Success falls steeply with difficulty everywhere. The d2 variants, which randomize object poses widely, stay at 2–36% for every model.

### Ablations (no dropout, all inputs)

| run | change from the reference | SR % | episodes |
|---|---|---|---|
| 2026-10-01_19-26-58 | reference: 26 datasets, 35 epochs | 53.2 | 2600 |
| 2026-10-01_19-26-58 | execute 5 of the 10 predicted actions before replanning | 51.1 | 2600 |
| 2026-10-01_19-26-58 | 20 sampling steps instead of 4 | 47.6 | 2600 |
| 2026-10-01_19-26-58 | three-piece assembly, rollout horizon ×1.25 | 15.7 | 300 |
| 2026-10-02_14-43-02 | trained on 6 datasets only, evaluated on all 26 | 15.3 | 2600 |
| 2026-10-02_14-43-02 | the same, three-piece assembly, horizon ×1.25 | 7.0 | 300 |
| 2026-10-04_13-09-51_6ds_2x | 6 datasets, 70 epochs, evaluated on those 6 | 35.5 | 600 |
| 2026-10-04_13-09-54_6ds_2x_dit12 | 6 datasets, 70 epochs, 12-block action head | 37.0 | 600 |

The 6 datasets are: `coffee_preparation_d1`, `pick_place_d0`, `square_d2`, `stack_three_d1`, `threading_d2`, `three_piece_assembly_d2`.

An earlier 40-epoch run (`2026-09-30_17-05-35`, 52.0%) is left out: it was trained on an older rendering of the datasets (224 px cameras, 112 px model input), so it is not comparable with these runs.

- **Replanning more often or sampling more carefully does not help:**
  - executing only 5 of the 10 predicted actions gives 51.1%;
  - 20 sampling steps gives 47.6%, worse than the default 4.
- **Three-piece assembly is not limited by its time budget.** With 25% more steps it reaches 15.7% vs 13.7% for the reference on the same three datasets.
- **Specializing on the 6 hardest datasets gains little.**
  - The 26-dataset model scores 32.0% on those 6.
  - A 6-dataset model scores 35.3% at 35 epochs, 35.5% at 70 epochs, and 37.0% with a 12-block action head.
  - It scores 9.3% on the 20 datasets it never saw, so there is little transfer between task families.

## Across the two benchmarks

- **Language matters only when the scene is ambiguous.** CALVIN needs it to know which of several tasks to do; MimicGen does not.
- **The static camera is the stronger single view on both benchmarks.** How well the wrist camera substitutes for it depends on the task: well on CALVIN's coarse tabletop tasks, poorly on MimicGen's precise assembly and insertion tasks.
- **How dropout is applied matters as much as how much is applied.**
  - Dropout makes a model work without some inputs only if training removed *whole* modalities often enough: Dirichlet rather than uniform, and proprioception keep below 1.
  - Heavy dropout (keep 0.25) moves the training inputs so far from the full input that the full input itself becomes out of distribution.

## Reproduction

Training runs live under `$SAVES_DIR/train_logs/`, and closed-loop results in `<run>/eval_logs/last/<suite>/result.csv`.

| model | run |
|---|---|
| CALVIN no dropout | `calvin_abcd_d/2026-10-07_17-51-44` |
| CALVIN Dirichlet keep 0.75 | `calvin_abcd_d_dropout/2026-10-08_20-42-04` |
| CALVIN Dirichlet keep 0.5 | `calvin_abcd_d_dropout/2026-10-07_17-31-38` |
| CALVIN Dirichlet keep 0.25 | `calvin_abcd_d_dropout/2026-10-08_20-42-09` |
| CALVIN uniform keep 0.5 | `calvin_abcd_d_dropout/2026-10-10_10-59-33` |
| CALVIN no dropout, seeds 5 / 100 | `calvin_abcd_d/2026-10-12_22-31-31` / `2026-10-12_22-32-26` |
| CALVIN Dirichlet keep 0.75, seeds 5 / 100 | `calvin_abcd_d_dropout/2026-10-13_11-14-02` / `2026-10-13_11-14-03` |
| CALVIN Dirichlet keep 0.5, seeds 5 / 100 | `calvin_abcd_d_dropout/2026-10-12_22-47-23` / `2026-10-12_22-47-28` |
| CALVIN uniform keep 0.5, seed 100 | `calvin_abcd_d_dropout/2026-10-14_09-58-42` |
| CALVIN no dropout + proprio, seeds 42 / 5 | `calvin_abcd_d/2026-10-14_13-37-34` / `2026-10-14_13-37-42` (W&B runs `calvin_abcd_d_2026-10-14_13-37-34` / `_13-37-42`; trained on another host) |
| MimicGen no dropout | `mimicgen/2026-10-01_19-26-58` |
| MimicGen no dropout + proprio, seeds 42 / 5 | `mimicgen/2026-10-16_16-38-11` / `2026-10-16_16-38-21` |
| MimicGen dropout, proprio keep 0.5 / 1.0 | `mimicgen_dropout/2026-10-05_09-31-11_pkeep0.5` / `_pkeep1.0` |

```bash
# CALVIN: train (auto-runs the closed-loop pipeline afterwards), open loop, figures
CUDA_VISIBLE_DEVICES=0,1 ./run.sh train-calvin batch_size=4 accumulate_grad_batches=4
CUDA_VISIBLE_DEVICES=2,3 ./run.sh train-calvin-dropout batch_size=4 accumulate_grad_batches=4 \
    model.modality_dropout_keep_fraction=0.75          # or 0.5, 0.25
./run.sh train-calvin-dropout batch_size=4 accumulate_grad_batches=2 model.modality_dropout_strategy=uniform
# extra seeds (5, 100)
CUDA_VISIBLE_DEVICES=0,1 ./run.sh train-calvin-dropout batch_size=8 accumulate_grad_batches=2 seed=5 \
    model.modality_dropout_keep_fraction=0.75          # or train-calvin / keep 0.5
# no dropout with proprioception (CALVIN's robot state has 8 dims after keep_indices)
./run.sh train-calvin model.use_proprio=True proprio_dims=8 seed=42
CUDA_VISIBLE_DEVICES=0 ./run.sh open-loop-calvin /saves/train_logs/calvin_abcd_d_dropout/<ts>
./run.sh analyze-open-loop

# MimicGen
./run.sh train-mimicgen
./run.sh train-mimicgen batch_size=8 accumulate_grad_batches=2 model.use_proprio=True seed=5
# re-evaluate an existing run at 100 episodes per dataset (and combination)
PIPELINE_REEVAL=1 ./run.sh pipeline-mimicgen /saves/train_logs/mimicgen/<run> n_eval=100
./run.sh train-mimicgen-dropout model.use_proprio=True model.modality_dropout_proprio_keep_p=0.5
```

## Appendix

### CALVIN per-subtask success, all inputs

Success rate (%) of each attempted subtask in the 1000 closed-loop chains.

| subtask | no dropout | Dirichlet keep 0.75 | Dirichlet keep 0.5 | Dirichlet keep 0.25 | uniform keep 0.5 |
|---|---|---|---|---|---|
| close_drawer | 100.0 | 100.0 | 100.0 | 0.0 | 100.0 |
| lift_blue_block_drawer | 100.0 | 95.0 | 100.0 | 100.0 | 93.8 |
| lift_blue_block_slider | 93.1 | 93.3 | 89.1 | 93.2 | 99.2 |
| lift_blue_block_table | 100.0 | 99.0 | 99.4 | 94.4 | 100.0 |
| lift_pink_block_drawer | 100.0 | 92.3 | 92.9 | 75.0 | 92.9 |
| lift_pink_block_slider | 96.4 | 98.6 | 96.3 | 89.3 | 97.7 |
| lift_pink_block_table | 100.0 | 98.3 | 99.4 | 48.7 | 96.5 |
| lift_red_block_drawer | 94.1 | 94.4 | 100.0 | 87.5 | 93.3 |
| lift_red_block_slider | 97.8 | 96.1 | 95.4 | 87.7 | 95.3 |
| lift_red_block_table | 99.5 | 97.9 | 99.4 | 75.9 | 99.4 |
| move_slider_left | 100.0 | 99.2 | 99.6 | 100.0 | 100.0 |
| move_slider_right | 100.0 | 99.6 | 100.0 | 99.4 | 100.0 |
| open_drawer | 100.0 | 98.9 | 100.0 | 3.5 | 100.0 |
| place_in_drawer | 100.0 | 98.4 | 97.7 | 91.1 | 99.5 |
| place_in_slider | 95.4 | 92.0 | 76.0 | 50.3 | 7.4 |
| push_blue_block_left | 94.2 | 97.1 | 85.5 | 94.0 | 91.9 |
| push_blue_block_right | 69.4 | 77.8 | 69.1 | 46.8 | 40.3 |
| push_into_drawer | 79.2 | 78.4 | 78.3 | 74.1 | 79.8 |
| push_pink_block_left | 85.7 | 90.8 | 86.7 | 66.0 | 90.0 |
| push_pink_block_right | 70.8 | 84.8 | 77.3 | 56.1 | 69.5 |
| push_red_block_left | 94.8 | 96.1 | 93.2 | 73.2 | 91.0 |
| push_red_block_right | 66.2 | 72.2 | 64.8 | 42.6 | 48.5 |
| rotate_blue_block_left | 100.0 | 100.0 | 95.4 | 97.3 | 100.0 |
| rotate_blue_block_right | 92.3 | 90.9 | 92.1 | 97.8 | 94.0 |
| rotate_pink_block_left | 93.0 | 100.0 | 90.7 | 79.4 | 95.9 |
| rotate_pink_block_right | 76.4 | 83.3 | 97.2 | 76.5 | 98.4 |
| rotate_red_block_left | 100.0 | 98.4 | 100.0 | 83.8 | 98.3 |
| rotate_red_block_right | 94.7 | 94.6 | 95.8 | 82.2 | 98.5 |
| stack_block | 88.3 | 88.9 | 83.3 | 63.7 | 81.9 |
| turn_off_led | 98.8 | 97.7 | 98.8 | 98.1 | 97.9 |
| turn_off_lightbulb | 100.0 | 100.0 | 71.8 | 23.5 | 100.0 |
| turn_on_led | 98.4 | 98.9 | 99.4 | 100.0 | 98.7 |
| turn_on_lightbulb | 100.0 | 100.0 | 99.4 | 62.2 | 100.0 |
| unstack_block | 100.0 | 100.0 | 100.0 | 95.8 | 98.4 |

### MimicGen per-dataset success, all inputs

Success rate (%), 100 episodes per dataset. The dropout runs and the "+ proprio" runs are evaluated with all four inputs.

| dataset | no dropout | + proprio, seed 42 | + proprio, seed 5 | dropout, proprio keep 0.5 | dropout, proprio keep 1.0 |
|---|---|---|---|---|---|
| coffee_d0 | 95 | 93 | 78 | 88 | 88 |
| coffee_d1 | 65 | 77 | 71 | 50 | 73 |
| coffee_d2 | 14 | 21 | 33 | 10 | 14 |
| coffee_preparation_d0 | 61 | 71 | 91 | 39 | 64 |
| coffee_preparation_d1 | 35 | 31 | 38 | 14 | 18 |
| hammer_cleanup_d0 | 98 | 95 | 100 | 89 | 94 |
| hammer_cleanup_d1 | 65 | 64 | 52 | 48 | 53 |
| kitchen_d0 | 100 | 100 | 100 | 100 | 100 |
| kitchen_d1 | 68 | 65 | 58 | 68 | 51 |
| mug_cleanup_d0 | 67 | 66 | 58 | 50 | 56 |
| mug_cleanup_d1 | 46 | 48 | 45 | 39 | 36 |
| nut_assembly_d0 | 15 | 24 | 18 | 17 | 11 |
| pick_place_d0 | 47 | 42 | 45 | 35 | 24 |
| square_d0 | 71 | 87 | 71 | 69 | 67 |
| square_d1 | 38 | 35 | 45 | 41 | 47 |
| square_d2 | 31 | 19 | 25 | 26 | 36 |
| stack_d0 | 99 | 100 | 99 | 99 | 100 |
| stack_d1 | 99 | 99 | 99 | 100 | 100 |
| stack_three_d0 | 82 | 87 | 84 | 89 | 81 |
| stack_three_d1 | 71 | 74 | 65 | 73 | 74 |
| threading_d0 | 62 | 37 | 42 | 28 | 23 |
| threading_d1 | 10 | 5 | 4 | 6 | 6 |
| threading_d2 | 3 | 3 | 5 | 3 | 2 |
| three_piece_assembly_d0 | 21 | 21 | 24 | 18 | 21 |
| three_piece_assembly_d1 | 15 | 14 | 23 | 10 | 15 |
| three_piece_assembly_d2 | 5 | 2 | 2 | 2 | 4 |
