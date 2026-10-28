# Modality dropout for FlowerVLA: CALVIN, MimicGen and LIBERO results

Results of fine-tuning FlowerVLA with **modality-token dropout**, which withholds part of the camera and language tokens during training. Models are compared with ordinary fine-tuning on three simulated benchmarks:

- **CALVIN ABCD→D:** language-conditioned, long-horizon tabletop manipulation, evaluated closed loop and open loop per timestep.
- **MimicGen `core`:** 26 single-task robosuite datasets, evaluated closed loop.
- **LIBERO-10:** ten long-horizon tabletop tasks. It is evaluated closed loop on the original tasks, on 2519 perturbed versions of them (LIBERO-Plus), and on modified tasks (LIBERO-PRO).

Every number below comes from the evaluation files of the runs listed in [Reproduction](#reproduction) and, for LIBERO, in the [LIBERO section](#libero). Success rates are percentages of episodes.

## Summary

- **On CALVIN, light dropout costs nothing with all inputs and lets the model run without some of them.**
  - On CALVIN, Dirichlet dropout that removes 25% of the tokens matches ordinary fine-tuning with every input: over 3 seeds, an average chain of 4.44 ± 0.03 vs 4.44 ± 0.05 completed subtasks.
  - Without dropout, a model cannot be given only part of its inputs at all.
  - On MimicGen even light dropout (keep 0.75) costs about 4 points with all inputs.
- **The amount of dropout is a trade-off** (3 seeds).
  - Removing 50% of the tokens lowers the all-inputs result (4.19 ± 0.13) but gives by far the best runs without a camera: 3.76 ± 0.19 with the static camera and language, 2.58 ± 0.27 with the wrist camera and language (keep 0.75: 1.72 and 1.62).
  - Removing 75% hurts every combination, including all inputs (1.64 ± 0.07).
- **Removing whole modalities in training matters, not just removing tokens.** The *uniform* variant drops a random 50% of all tokens and rarely removes a whole camera. Over three seeds it does worse with all inputs (3.63 vs 4.19 for Dirichlet) and collapses without the wrist camera (0.28 vs 3.76). On MimicGen (1 seed) it matches Dirichlet with all inputs but gets 0–3% with either camera missing.
- **On CALVIN, Dirichlet keep 0.5 trained with proprioception (1 seed) does not depend on it**, and is far better without the wrist camera than without proprioception training (4.17–4.24 vs 3.76 with static camera and language). It needs more seeds to confirm.
- **Language is indispensable on CALVIN and nearly irrelevant on MimicGen.**
  - CALVIN puts several tasks in one scene, so the instruction picks the task. Without language, no CALVIN model completes more than about 0.2 subtasks per chain on average.
  - Each MimicGen scene has one task, so dropping language changes success by at most a few points, except for the uniform model (46.9% → 34.7%).
- **Throughout a CALVIN subtask (open loop), language matters most at the start and vision matters most at the end.**
  - Removing language costs most in the first ~20 frames, when the target is chosen.
  - With language only, the prediction error grows steadily towards the end of the subtask, when positioning and release need visual feedback.
  - Release is the hardest phase for every model with all inputs.
- **Open-loop error does not predict closed-loop success with all inputs.** All five CALVIN models reach the same per-step action error on held-out demonstrations (0.061–0.067 over all three seeds), although their chain lengths range from 1.7 to 4.4.
- **On MimicGen, dropout with proprioception costs 4–6 points with all inputs** (45.1% and 47.0% for proprio keep 0.5 and 1.0, vs 50.8% without dropout; 2 seeds each). Seeds of the same model differ by up to about 6 points, so smaller MimicGen differences between single runs are noise. Proprioception without dropout does not clearly help (53.0% vs 50.8%).
  - Removing less (Dirichlet keep 0.75) does not close the gap (46.4%), and it gives up almost all robustness: 2–4% without either camera.
  - Both cameras matter. Without the wrist camera success falls to about 18%, without the static camera to about 10%.
  - Proprioception has to be dropped in training too: a model that always saw it falls from 47.0% to 15.2% without it (2 seeds).
- **On LIBERO-10, dropout costs nothing on the original tasks and makes the model much more robust to perturbations** (2 valid seeds per model).
  - With all inputs every model scores 91.5–95.2% on LIBERO-10.
  - On the 2519 perturbed LIBERO-Plus tasks every dropout seed (75.7–82.3%) beats every no-dropout seed (64.6–72.4%).
  - The largest gains are for camera viewpoints, reworded instructions and robot initial pose.
  - Proprio keep 1.0 is the most robust (80.9%) but, as on MimicGen, collapses without proprioception.
  - Without dropout, withholding language or the wrist camera gives 0%. The dropout models keep 62–73% on LIBERO-Plus without language or the static camera.
  - No model survives a changed task on LIBERO-PRO (0–12%).
  - Several LIBERO runs still need re-evaluation or are missing; see [missing or unusable results](#missing-or-unusable-libero-results).

## Setup

**Model and fine-tuning.**
- FlowerVLA, starting from the pretrained 360k-step weights, with a Florence-2 encoder over static-camera, wrist-camera and language tokens and an 18-block flow-matching action head.
- It predicts a 10-step action chunk from one observation.
- Each run: 35 epochs × 1000 optimizer updates, global batch 32, seed 42.
- Extra seeds use batch 8 × 2 accumulation instead of 4 × 4 (same global batch):
  - CALVIN: every model with seeds 5 and 100.
  - MimicGen: no dropout and Dirichlet keep 0.5 with proprio keep 0.5 and 1.0, seed 5; no dropout with proprioception, seeds 42 and 5.
- New configurations, seed 42, also batch 8 × 2: CALVIN Dirichlet keep 0.5 with proprioception; MimicGen Dirichlet keep 0.75 and uniform keep 0.5, both with proprio keep 0.5.
- CALVIN no dropout with proprioception: seeds 42 and 5 trained on another host (batch split not recorded), seed 100 with batch 8 × 2.
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
- **Keep 0.25 trained on so few tokens that all inputs (1.72) is worse than static+lang (2.20).** The full token count looks out of distribution for it. Over three seeds the two are equal, though (1.64 each; see [Several seeds](#several-seeds)).
  - Its failures with all inputs concentrate on a few subtasks: `close_drawer` 0%, `open_drawer` 3.5%, `turn_off_lightbulb` 23.5%. Seed 100 fails on the same three (28%, 20.5% and 5.1%), seed 5 much less (73.6%, 92.1% and 59.7%; the [appendix](#calvin-per-subtask-success-all-inputs) gives the means over seeds).
- **Uniform dropout fails without the wrist camera** (static+lang 0.29). It is the only model where wrist+lang beats static+lang.
  - With all inputs it is worse than Dirichlet at the same keep fraction (3.55 vs 4.09).
  - Most of that loss is one subtask: `place_in_slider` succeeds 7.4% of the time, against 76–95% for the other models.

#### Several seeds

Every model was trained with seeds 42, 5 and 100. Mean ± standard deviation over the seeds of avg_seq_len, and of the share of 5/5 chains (%), each seed over the same 1000 chains:

| inputs | no dropout | Dirichlet keep 0.75 | Dirichlet keep 0.5 | Dirichlet keep 0.25 | uniform keep 0.5 |
|---|---|---|---|---|---|
| static+wrist+lang | **4.444 ± 0.051** (81.0 ± 2.3) | **4.442 ± 0.029** (79.3 ± 0.4) | 4.191 ± 0.131 (71.2 ± 4.9) | 1.636 ± 0.071 (8.8 ± 0.5) | 3.630 ± 0.117 (52.6 ± 5.7) |
| static+wrist | n/a | 0.174 ± 0.028 (0.0) | 0.187 ± 0.029 (0.0) | 0.157 ± 0.016 (0.0) | 0.156 ± 0.027 (0.0) |
| static+lang | n/a | 1.724 ± 0.159 (13.8 ± 2.0) | **3.759 ± 0.190** (54.6 ± 5.1) | 1.638 ± 0.551 (9.1 ± 7.3) | 0.276 ± 0.034 (0.0) |
| wrist+lang | n/a | 1.617 ± 0.126 (5.5 ± 0.7) | **2.581 ± 0.272** (21.0 ± 5.5) | 0.979 ± 0.266 (1.1 ± 0.9) | 1.072 ± 0.097 (1.5 ± 0.6) |
| static | n/a | 0.050 ± 0.010 (0.0) | 0.173 ± 0.020 (0.0) | 0.152 ± 0.018 (0.0) | 0.034 ± 0.003 (0.0) |
| wrist | n/a | 0.088 ± 0.017 (0.0) | 0.149 ± 0.009 (0.0) | 0.126 ± 0.024 (0.0) | 0.074 ± 0.028 (0.0) |
| lang | n/a | 0.000 (0.0) | 0.000 (0.0) | 0.018 ± 0.008 (0.0) | 0.000 (0.0) |

avg_seq_len of each seed with all inputs, and with one camera missing:

| model | all, seed 42 | all, seed 5 | all, seed 100 | static+lang, 42 / 5 / 100 | wrist+lang, 42 / 5 / 100 |
|---|---|---|---|---|---|
| no dropout | 4.403 | 4.501 | 4.428 | n/a | n/a |
| Dirichlet keep 0.75 | 4.436 | 4.417 | 4.474 | 1.571 / 1.713 / 1.889 | 1.498 / 1.605 / 1.749 |
| Dirichlet keep 0.5 | 4.093 | 4.340 | 4.141 | 3.885 / 3.852 / 3.540 | 2.716 / 2.268 / 2.760 |
| Dirichlet keep 0.25 | 1.717 | 1.587 | 1.604 | 2.196 / 1.094 / 1.624 | 1.208 / 0.687 / 1.042 |
| uniform keep 0.5 | 3.551 | 3.574 | 3.764 | 0.292 / 0.237 / 0.299 | 1.149 / 0.963 / 1.104 |

Chain success by position with all inputs, mean over the seeds (%):

| model | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | avg_seq_len |
|---|---|---|---|---|---|---|
| no dropout | 96.7 | 92.6 | 89.1 | 85.0 | 81.0 | 4.44 |
| Dirichlet keep 0.75 | 97.4 | 93.6 | 89.3 | 84.6 | 79.3 | 4.44 |
| Dirichlet keep 0.5 | 95.8 | 90.2 | 83.9 | 78.1 | 71.2 | 4.19 |
| Dirichlet keep 0.25 | 69.4 | 43.5 | 26.3 | 15.6 | 8.8 | 1.64 |
| uniform keep 0.5 | 93.6 | 83.2 | 71.5 | 62.1 | 52.6 | 3.63 |

Findings:
- **Keep 0.75 and no dropout are equal with all inputs** (4.44 each). The small seed-42 lead of keep 0.75 (4.436 vs 4.403) is seed noise.
- **Keep 0.5 costs about 0.25 with all inputs, consistently.** All three of its seeds score below all six seeds of the other two models. Seed 42 was its worst seed, so the seed-42 table overstates the cost (0.31).
- **Keep 0.5's robustness without a camera is far larger than the seed spread.** Its lead over keep 0.75 is 2.0 with static+lang and 1.0 with wrist+lang, against standard deviations of 0.13–0.27.
- **Keep 0.5 varies most across seeds among the usable models**, both with all inputs (± 0.13) and with wrist+lang (± 0.27). With three seeds the standard deviations are themselves rough.
- **Keep 0.25 is poor with all inputs in every seed** (1.59–1.72). Its seed-42 result that static+lang beats all inputs does not hold up: over the three seeds the two are equal (1.64 each), and static+lang swings from 1.09 to 2.20 between seeds (± 0.55, the largest spread in the table).
- **Uniform's failure without the wrist camera is reproducible.** Static+lang is 0.29, 0.24 and 0.30 for its three seeds, against 3.76 for Dirichlet at the same keep fraction.
- **Uniform is also worse with all inputs** (3.63 vs 4.19 for Dirichlet keep 0.5), in all three seeds. Its weakest subtask is `place_in_slider`, at 7.4%, 70.9% and 42.6% success for seeds 42, 5 and 100. Every seed of no dropout and of Dirichlet keep 0.75 and 0.5 gets 76–96% there. Seed 5's much better `place_in_slider` does not raise its total (3.57 vs 3.55 for seed 42), because it loses as much elsewhere: the three `push_*_block_left` subtasks are 23–40 points lower and `rotate_pink_block_right` 46 points lower than for seed 42 (see the [appendix](#calvin-per-subtask-success-all-inputs)).

#### No dropout with proprioception

The no-dropout recipe with the robot state added as an input (`model.use_proprio=True proprio_dims=8`: gripper position, orientation, width and open/close state), seeds 42, 5 and 100, all inputs, 1000 chains each. Seeds 42 and 5 trained on one A100 40 GB each on another host (35k optimizer steps like every other run; their batch split is not recorded in W&B), seed 100 with batch 8 × 2 like the other extra seeds.

| model | seed 42 | seed 5 | seed 100 | mean ± sd | 5/5 (%) |
|---|---|---|---|---|---|
| no dropout | 4.403 | 4.501 | 4.428 | 4.444 ± 0.051 | 81.0 ± 2.3 |
| no dropout + proprio | 4.467 | 4.353 | 4.545 | 4.455 ± 0.097 | 81.1 ± 2.5 |

Chain success by position, mean over the three seeds (%):

| model | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | avg_seq_len |
|---|---|---|---|---|---|---|
| no dropout | 96.7 | 92.6 | 89.1 | 85.0 | 81.0 | 4.44 |
| no dropout + proprio | 96.7 | 92.8 | 89.3 | 85.6 | 81.1 | 4.46 |

- **Proprioception does not measurably change CALVIN performance.** The means differ by 0.01, the per-seed differences go both ways (+0.06, −0.15, +0.12), and the two models' seed ranges overlap (4.40–4.50 vs 4.35–4.55).
- These runs cannot show how much the model relies on proprioception: without dropout it is only evaluated with it. On MimicGen, withholding proprioception from a model that always saw it cost about 30 points. The dropout run below answers it for CALVIN.

#### Dirichlet dropout with proprioception

Dirichlet keep 0.5 with the robot state as a fourth input (`model.use_proprio=True proprio_dims=8`), kept with probability 0.5 in training like on MimicGen. One seed (42), evaluated on all 14 combinations. avg_seq_len (5/5 %), against the same recipe without proprioception (3 seeds):

| inputs | with proprio | proprio withheld | trained without proprio (3 seeds) |
|---|---|---|---|
| static+wrist+lang | **4.301** (73.9) | 4.238 (73.3) | 4.191 ± 0.131 |
| static+wrist | 0.149 (0.0) | 0.178 (0.0) | 0.187 ± 0.029 |
| static+lang | 4.168 (67.0) | **4.236** (69.8) | 3.759 ± 0.190 |
| wrist+lang | 2.543 (20.4) | 2.380 (17.0) | 2.581 ± 0.272 |
| static | 0.224 (0.0) | 0.199 (0.0) | 0.173 ± 0.020 |
| wrist | 0.129 (0.0) | 0.139 (0.0) | 0.149 ± 0.009 |
| lang | 0.000 (0.0) | 0.000 (0.0) | 0.000 |

- **The model hardly relies on proprioception.** Withholding it changes avg_seq_len by at most 0.16 in any combination, and not always downwards. On MimicGen the same recipe (proprio keep 0.5) is equally indifferent to it, while proprio keep 1.0 collapses without it.
- **Without the wrist camera this run is much better than any run trained without proprioception:** 4.17–4.24 with static+lang, against 3.54–3.89 for the three seeds of keep 0.5 without it, and almost as good as with all inputs. That holds whether or not proprioception is given at evaluation, so it is not proprioception standing in for the wrist camera. With one seed this may be seed luck; it needs more seeds before it is a finding.
- With all inputs it is at the top of the keep 0.5 range (4.30 vs 4.09–4.34), still below no dropout (4.44).

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

**Other seeds.** The open-loop evaluation was repeated for the seed-5 and seed-100 runs of every model. Mean predicted-action MSE, mean ± sd over the three seeds:

| inputs | no dropout | Dirichlet keep 0.75 | Dirichlet keep 0.5 | Dirichlet keep 0.25 | uniform keep 0.5 |
|---|---|---|---|---|---|
| static+wrist+lang | 0.062 ± 0.001 | 0.063 ± 0.000 | 0.065 ± 0.001 | 0.066 ± 0.001 | 0.063 ± 0.001 |
| static+wrist | n/a | 0.089 ± 0.002 | 0.091 ± 0.001 | 0.088 ± 0.000 | 0.092 ± 0.003 |
| static+lang | n/a | 0.114 ± 0.002 | 0.072 ± 0.002 | 0.077 ± 0.002 | 0.162 ± 0.020 |
| wrist+lang | n/a | 0.077 ± 0.001 | 0.075 ± 0.003 | 0.074 ± 0.002 | 0.083 ± 0.004 |
| static | n/a | 0.166 ± 0.003 | 0.106 ± 0.002 | 0.109 ± 0.002 | 0.223 ± 0.034 |
| wrist | n/a | 0.118 ± 0.002 | 0.110 ± 0.004 | 0.100 ± 0.002 | 0.126 ± 0.002 |
| lang | n/a | 0.298 ± 0.006 | 0.300 ± 0.003 | 0.262 ± 0.005 | 0.309 ± 0.035 |

- **The open-loop results do not depend on the seed.** With all inputs every seed of every model is within 0.061–0.067, and the seed spread of the four Dirichlet and no-dropout models is at most ± 0.004 in every combination except language only (up to ± 0.006).
- **Keep 0.25's open-loop error stays as good as keep 0.5's** with a camera missing (0.077 vs 0.072 with static+lang), although in closed loop it is far worse (1.64 vs 3.76). As with all inputs, per-step error on demonstrations does not show what goes wrong in a rollout.
- **Uniform varies more across seeds when the wrist camera is missing** (± 0.020–0.035 with static+lang, static only and language only), but every seed is still the worst model with static+lang and static only.

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

Success rate (%) per input combination, on 100 episodes per dataset × 26 datasets = 2600 per cell and seed (95% Wilson interval about ±1.9 points for one seed, ±1.4 for two pooled seeds).
- The dropout runs use proprioception and are evaluated on every combination: Dirichlet keep 0.5 with proprio keep 0.5 and with proprio keep 1.0 (seeds 42 and 5 each, pooled), uniform keep 0.5 and Dirichlet keep 0.75, both with proprio keep 0.5 (seed 42).
- The no-dropout runs are evaluated with all inputs only: without proprioception (seeds 42 and 5: 53.2% and 48.4%) and with it (seeds 42 and 5: 53.1% and 52.9%).

| inputs | no dropout (2 seeds) | no dropout + proprio (2 seeds) | keep 0.5, proprio keep 0.5 (2 seeds) | keep 0.5, proprio keep 1.0 (2 seeds) | keep 0.75, proprio keep 0.5 | uniform keep 0.5, proprio keep 0.5 |
|---|---|---|---|---|---|---|
| static+wrist+lang+proprio | n/a | **53.0** | 45.1 | 47.0 | 46.4 | 46.9 |
| static+wrist+lang | 50.8 | n/a | 44.4 | 15.2 | 45.5 | 46.3 |
| static+wrist+proprio | n/a | n/a | 45.6 | 46.7 | 47.3 | 34.7 |
| static+wrist | n/a | n/a | 45.2 | 16.3 | 45.3 | 34.2 |
| static+lang+proprio | n/a | n/a | 17.5 | 16.5 | 3.7 | 0.1 |
| static+lang | n/a | n/a | 17.8 | 5.1 | 4.0 | 0.1 |
| wrist+lang+proprio | n/a | n/a | 9.9 | 18.1 | 2.2 | 3.4 |
| wrist+lang | n/a | n/a | 6.4 | 2.1 | 1.7 | 2.1 |
| static+proprio | n/a | n/a | 17.3 | 16.2 | 2.6 | 0.3 |
| static | n/a | n/a | 17.7 | 5.2 | 2.6 | 0.2 |
| wrist+proprio | n/a | n/a | 8.9 | 16.1 | 4.3 | 0.2 |
| wrist | n/a | n/a | 5.2 | 2.6 | 2.7 | 0.1 |
| lang+proprio | n/a | n/a | 0.0 | 0.0 | 0.0 | 0.0 |
| lang | n/a | n/a | 0.0 | 0.0 | 0.0 | 0.0 |

**Seed spread.** The two seeds of a model differ by up to about 6 points in a cell (keep 0.5 with proprio keep 1.0 without proprio: 18.0% vs 12.3%), more than the ±1.9-point sampling interval: no dropout 53.2% vs 48.4%, keep 0.5 with proprio keep 0.5 46.6% vs 43.5%, keep 0.5 with proprio keep 1.0 48.4% vs 45.7% (all inputs). Only no dropout with proprioception gave nearly the same result twice (53.1% vs 52.9%). Differences of a few points between single runs are therefore not meaningful.

Findings:
- **Language barely matters for the Dirichlet models.** Without it, success is 45.6% (proprio keep 0.5), 46.7% (keep 1.0) and 47.3% (keep 0.75), against 45.1%, 47.0% and 46.4% with it. The scene already determines the task.
- **Both cameras are needed, and they complement each other.**
  - Removing the wrist camera drops success from 45.1% to 17.5%; removing the static camera drops it to 9.9% (proprio keep 0.5).
  - Either camera alone gives 5–18% (proprio keep 0.5).
  - This differs from CALVIN, where the Dirichlet keep 0.5 models lose little from one missing camera. MimicGen's precise insertion and assembly tasks need both views at once.
- **Removing less does not close the gap to no dropout, and gives up robustness.** Keep 0.75 reaches 46.4% with all inputs, the same as keep 0.5 (45.1%), but only 2–4% with either camera missing, against 5–18% for keep 0.5. On CALVIN keep 0.75 also lost most of its robustness without a camera, but not as completely.
- **Uniform dropout gives no robustness on MimicGen either.** With all inputs it matches the Dirichlet runs (46.9%), but with either camera missing it gets 0–3%. It is also the only model that needs language: 34.7% without it. Uniform dropout never removes a whole modality in training, so any input with one missing is new to it.
- **Proprioception has to be dropped in training too.**
  - With keep 1.0 (proprioception always present in training), removing it collapses success: 47.0% → 15.2% with every other input (18.0% and 12.3% for the two seeds).
  - With keep 0.5 the model is indifferent to it (45.1% vs 44.4%).
  - With all inputs keep 1.0 is 2 points above keep 0.5 (47.0% vs 45.1%), less than the spread between seeds.
- **Proprioception alone does not clearly help.** Without dropout, adding it gives 53.0% against 50.8% without; the two seeds without it (53.2%, 48.4%) bracket that.
- **Dropout with all inputs is 4–6 points below no dropout** (45.1–47.0% vs 50.8%). Every dropout seed is at or below the worse no-dropout seed (48.4%). The gap is largest on `threading_d0` (17–43% vs 57–62%) and `coffee_preparation_d1` (11–20% vs 25–35%; see the [appendix](#mimicgen-per-dataset-success-all-inputs)).
  - On `threading_d0` the no-dropout runs with proprioception are low too (37% / 42%), so that gap comes partly from proprioception, not from dropout.

### Task family × difficulty, all inputs (%)

Each model is evaluated with all of its inputs and pooled over seeds 42 and 5: no dropout, no dropout with proprioception ("+ proprio"), and Dirichlet keep 0.5 with proprio keep 0.5 and 1.0. Keep 0.75 and uniform are in the [appendix](#mimicgen-per-dataset-success-all-inputs).

| family | no dropout d0 | d1 | d2 | + proprio d0 | d1 | d2 | proprio keep 0.5 d0 | d1 | d2 | proprio keep 1.0 d0 | d1 | d2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coffee | 81 | 60 | 14 | 86 | 74 | 27 | 90 | 48 | 10 | 88 | 62 | 12 |
| coffee_preparation | 60 | 30 | – | 81 | 34 | – | 58 | 13 | – | 54 | 14 | – |
| hammer_cleanup | 93 | 60 | – | 98 | 58 | – | 90 | 45 | – | 94 | 52 | – |
| kitchen | 100 | 66 | – | 100 | 62 | – | 100 | 64 | – | 100 | 55 | – |
| mug_cleanup | 60 | 43 | – | 62 | 46 | – | 49 | 31 | – | 57 | 38 | – |
| nut_assembly | 10 | – | – | 21 | – | – | 12 | – | – | 8 | – | – |
| pick_place | 46 | – | – | 44 | – | – | 36 | – | – | 24 | – | – |
| square | 77 | 36 | 23 | 79 | 40 | 22 | 62 | 34 | 22 | 58 | 43 | 32 |
| stack | 98 | 99 | – | 100 | 99 | – | 98 | 100 | – | 100 | 100 | – |
| stack_three | 82 | 73 | – | 86 | 70 | – | 84 | 69 | – | 78 | 76 | – |
| threading | 60 | 5 | 3 | 40 | 4 | 4 | 22 | 4 | 2 | 33 | 4 | 2 |
| three_piece_assembly | 21 | 16 | 4 | 22 | 18 | 2 | 14 | 12 | 2 | 18 | 14 | 4 |

Success falls steeply with difficulty everywhere. The d2 variants, which randomize object poses widely, stay at 1–32% for every model.

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
- All ablations are seed 42. The evaluation-setting ones (replanning, sampling steps, horizon) reuse the reference checkpoint on the same seeded episodes, so they compare like for like. The 6-dataset runs are separate trainings: their 3–5 point gain on the 6 datasets is within the 5-point spread between training seeds seen above.

## LIBERO

**Training.**
- LIBERO-10: ten long-horizon tabletop tasks.
- Fine-tuned from the same pretrained weights for 40 epochs × 1000 optimizer updates, batch 32.
- Proprioception, where used, is the 9-dim joint and gripper state.
- Dropout runs use Dirichlet keep 0.5 with a proprio keep probability, as on MimicGen.

**Evaluation.**
All evaluations use the last checkpoint, 4 sampling steps, all 10 predicted actions executed, and episodes seeded by task and episode.
- **LIBERO-10:** 20 episodes per task and input combination (200 per cell, 95% Wilson interval about ±3–7 points).
- **LIBERO-Plus:** 2519 perturbed versions of the ten tasks, one episode each, all inputs.
  - Seven perturbation categories: Camera Viewpoints 419, Robot Initial States 393, Language Instructions 383, Light Conditions 274, Background Textures 289, Sensor Noise 449, Objects Layout 312.
  - Difficulty levels 1–5.
- **LIBERO-Plus with one modality withheld:** the same 2519 tasks, with one input removed at evaluation.
- **LIBERO-PRO:**
  - Modified tasks: language (`lan`), object (`object`), swapped positions (`swap`) and task (`task`), 200 episodes each.
  - The `lan` and `task` perturbations are also paired episode by episode with the original LIBERO-10 episodes.

**Which evaluations are used.** Results are taken from the runs' W&B evaluation artifacts.
- Only evaluations made after commit `ca969a8` (2026-09-23) are used. Before it, the environment reset was not seeded, so five of the ten LIBERO-10 tasks got a random fixture layout in every episode.
- Earlier evaluations are listed under [missing or unusable results](#missing-or-unusable-libero-results).
- This leaves two valid seeds per model:

| model | valid seeds | runs (group `libero_10…`) |
|---|---|---|
| no dropout | 42, 5 | `libero_10/2026-09-18_16-47-38`, `16-47-48` |
| no dropout + proprio | 42, 5 | `libero_10_static+wrist+lang+proprio/2026-09-18_16-47-17`, `16-47-22` |
| Dirichlet keep 0.5, no proprio | 42, 5 | `libero_10_dropout/2026-09-14_17-14-47`, `17-15-19` |
| Dirichlet keep 0.5, proprio keep 1.0 | 5, 100 | `libero_10_dropout/2026-09-14_17-07-42`, `17-08-28` |
| Dirichlet keep 0.5, proprio keep 0.75 | 42, 100 | `libero_10_dropout/2026-09-14_17-12-09`, `17-12-40` |
| Dirichlet keep 0.5, proprio keep 0.5 | 5, 100 | `libero_10_dropout/2026-09-14_17-07-23`, `17-08-24` |

Values below are mean ± sd over the two seeds, in %.

### All inputs

| model | LIBERO-10, per seed | LIBERO-10 | LIBERO-Plus, per seed | LIBERO-Plus |
|---|---|---|---|---|
| no dropout | 94.5 / 92.5 | 93.5 ± 1.4 | 70.9 / 64.6 | 67.8 ± 4.5 |
| no dropout + proprio | 94.5 / 88.5 | 91.5 ± 4.2 | 68.6 / 72.4 | 70.5 ± 2.7 |
| dropout, no proprio | 94.5 / 96.0 | 95.2 ± 1.1 | 75.7 / 77.3 | 76.5 ± 1.1 |
| dropout, proprio keep 1.0 | 93.5 / 93.5 | 93.5 ± 0.0 | 79.5 / 82.3 | **80.9 ± 1.9** |
| dropout, proprio keep 0.75 | 92.5 / 92.5 | 92.5 ± 0.0 | 80.2 / 77.1 | 78.6 ± 2.2 |
| dropout, proprio keep 0.5 | 93.0 / 96.5 | 94.8 ± 2.5 | 76.5 / 78.2 | 77.4 ± 1.2 |

- **On LIBERO-10 itself all models are equal** (91.5–95.2%), within the seed spread and the ±3–7-point sampling interval.
- **Dropout makes the model more robust to perturbations.** On LIBERO-Plus every dropout seed (75.7–82.3%) beats every no-dropout seed (64.6–72.4%), about 9 points on average.
- **Proprio keep 1.0 is the most robust** (80.9%), mainly through camera viewpoints (below).

### Modality combinations on LIBERO-10

| inputs | no dropout | no dropout + proprio | dropout, no proprio | dropout, proprio keep 1.0 | dropout, proprio keep 0.75 | dropout, proprio keep 0.5 |
|---|---|---|---|---|---|---|
| static+wrist+lang+proprio | n/a | 91.5 ± 4.2 | n/a | 93.5 ± 0.0 | 92.5 ± 0.0 | 94.8 ± 2.5 |
| static+wrist+lang | 93.5 ± 1.4 | n/a | 95.2 ± 1.1 | 38.5 ± 17.7 | 95.0 ± 0.7 | 95.2 ± 2.5 |
| static+wrist+proprio | n/a | n/a | n/a | 82.2 ± 1.8 | 85.0 ± 1.4 | 84.2 ± 2.5 |
| static+wrist | n/a | n/a | 83.2 ± 2.5 | 35.0 ± 13.4 | 84.2 ± 0.4 | 85.5 ± 2.1 |
| static+lang+proprio | n/a | n/a | n/a | 45.8 ± 10.3 | 40.2 ± 2.5 | 45.2 ± 7.4 |
| static+lang | n/a | n/a | 44.8 ± 6.0 | 4.0 ± 0.7 | 38.8 ± 4.6 | 43.0 ± 7.8 |
| wrist+lang+proprio | n/a | n/a | n/a | 87.8 ± 3.9 | 85.0 ± 2.1 | 82.8 ± 2.5 |
| wrist+lang | n/a | n/a | 83.2 ± 4.6 | 19.8 ± 9.5 | 80.8 ± 4.6 | 83.8 ± 2.5 |
| static+proprio | n/a | n/a | n/a | 41.2 ± 17.3 | 35.8 ± 5.3 | 36.8 ± 11.0 |
| static | n/a | n/a | 39.0 ± 4.9 | 3.0 ± 1.4 | 32.5 ± 1.4 | 32.8 ± 11.7 |
| wrist+proprio | n/a | n/a | n/a | 75.5 ± 2.8 | 71.0 ± 0.7 | 70.2 ± 3.2 |
| wrist | n/a | n/a | 65.0 ± 0.7 | 15.2 ± 8.8 | 66.5 ± 2.8 | 65.5 ± 2.1 |
| lang (± proprio) | n/a | n/a | 0.0 | 0.0 | 0.0 | 0.0 |

- **The wrist camera is the important view on LIBERO**, unlike CALVIN.
  - Without the static camera the dropout models keep 81–88% (with proprioception, where they were trained with it).
  - Without the wrist camera they fall to 39–46%.
  - The wrist camera alone gives 65–76%; the static camera alone gives 33–41%.
- **Language matters much less than on CALVIN.** Without it the dropout models still reach 82–86% (static+wrist, with or without proprio). With all inputs they reach 93–95%.
- **Proprioception must be dropped in training, as on MimicGen.** With proprio keep 1.0, every combination without proprioception collapses: 38.5% with both cameras and language, 3–35% otherwise. With keep 0.75 and 0.5 the model is indifferent to it.

### LIBERO-Plus by perturbation category, all inputs

| category | no dropout | no dropout + proprio | dropout, no proprio | dropout, proprio keep 1.0 | dropout, proprio keep 0.75 | dropout, proprio keep 0.5 |
|---|---|---|---|---|---|---|
| Camera Viewpoints | 37.6 ± 8.9 | 38.8 ± 6.9 | 51.6 ± 7.4 | **68.5 ± 4.7** | 55.7 ± 4.9 | 48.9 ± 6.4 |
| Robot Initial States | 63.5 ± 3.4 | 64.1 ± 2.5 | 73.2 ± 1.3 | 71.2 ± 2.2 | 71.0 ± 4.0 | 72.9 ± 1.6 |
| Language Instructions | 84.5 ± 0.6 | 85.8 ± 2.0 | 94.0 ± 0.7 | 94.0 ± 2.6 | 94.8 ± 0.7 | 97.1 ± 0.4 |
| Light Conditions | 69.7 ± 14.5 | 73.7 ± 4.6 | 80.8 ± 3.9 | 72.3 ± 3.1 | 83.0 ± 3.9 | 80.3 ± 0.5 |
| Background Textures | 80.3 ± 7.8 | 78.9 ± 0.0 | 77.3 ± 2.2 | 85.1 ± 2.4 | 74.7 ± 13.2 | 81.3 ± 4.9 |
| Sensor Noise | 70.5 ± 4.6 | 77.8 ± 3.9 | 80.4 ± 2.2 | 86.3 ± 2.4 | 87.4 ± 1.1 | 81.8 ± 6.1 |
| Objects Layout | 76.0 ± 1.4 | 81.4 ± 3.6 | 82.2 ± 1.6 | 89.4 ± 0.5 | 86.1 ± 2.9 | 84.5 ± 1.1 |
| all | 67.8 ± 4.5 | 70.5 ± 2.7 | 76.5 ± 1.1 | 80.9 ± 1.9 | 78.6 ± 2.2 | 77.4 ± 1.2 |

By difficulty level (number of tasks: 357 / 450 / 484 / 523 / 705):

| model | level 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| no dropout | 85.0 ± 4.2 | 82.9 ± 3.1 | 77.6 ± 3.4 | 66.0 ± 6.8 | 44.0 ± 4.6 |
| no dropout + proprio | 85.0 ± 2.6 | 84.3 ± 0.2 | 82.5 ± 1.6 | 70.3 ± 4.5 | 46.3 ± 3.9 |
| dropout, no proprio | 88.8 ± 1.2 | 88.9 ± 2.5 | 85.5 ± 0.9 | 78.9 ± 2.8 | 54.3 ± 2.3 |
| dropout, proprio keep 1.0 | 92.2 ± 0.8 | 92.2 ± 2.2 | 88.8 ± 1.5 | 82.0 ± 3.2 | 61.6 ± 1.7 |
| dropout, proprio keep 0.75 | 89.8 ± 1.4 | 90.7 ± 0.3 | 87.4 ± 2.6 | 80.7 ± 5.1 | 57.7 ± 1.3 |
| dropout, proprio keep 0.5 | 90.2 ± 2.0 | 90.3 ± 1.4 | 87.1 ± 1.3 | 79.9 ± 2.4 | 54.1 ± 3.5 |

- **Dropout helps most against camera viewpoints** (+10 to +31 points over the no-dropout models), language instructions (+8 to +13), robot initial states (+7 to +10), sensor noise (+3 to +17) and object layout (+1 to +13). Light conditions and background textures show no consistent gain.
- **Dropout's gain grows with difficulty:** 4–7 points at level 1, 8–18 at level 5.
- **Dropout models ignore language perturbations.** They score 94–97% on reworded instructions, about the same as with no instruction at all (89–95%, see the [appendix](#libero-plus-per-category-with-one-modality-withheld)). The no-dropout models lose 6–9 points to rewording, compared with LIBERO-10.
- **Camera viewpoints are the hardest category** and the one where proprio keep 1.0 stands out (68.5% vs 49–56% for the other dropout models). Pooled over its two seeds, it still succeeds 39–58% of the time at camera rotations of 45–90°, where every other model gets 0–27%.
- **Perturbation size matters most for the robot's initial pose.** Success falls from 90–97% at 0.1 rad to 39–58% at 0.5 rad, and every dropout model is 13–19 points above the no-dropout models at 0.5 rad.

### LIBERO-Plus with one modality withheld

Success on all 2519 LIBERO-Plus tasks with one input removed at evaluation. In brackets: the same combination on unperturbed LIBERO-10.

| model | all inputs | no language | no static camera | no wrist camera | no proprio |
|---|---|---|---|---|---|
| no dropout | 67.8 ± 4.5 | 0.0 | 22.5 ± 9.4 | 0.0 | n/a |
| no dropout + proprio | 70.5 ± 2.7 | 1.2 ± 1.7 | 30.2 ± 4.3 | 0.0 | 41.8 ± 0.6 |
| dropout, no proprio | 76.5 ± 1.1 | 68.2 ± 0.1 (83.2) | 61.6 ± 8.9 (83.2) | 21.0 ± 2.1 (44.8) | n/a |
| dropout, proprio keep 1.0 | 80.9 ± 1.9 | 73.2 ± 0.7 (82.2) | 72.7 ± 4.2 (87.8) | 22.6 ± 7.4 (45.8) | 31.9 ± 8.7 (38.5) |
| dropout, proprio keep 0.75 | 78.6 ± 2.2 | 69.7 ± 6.0 (85.0) | 69.7 ± 5.0 (85.0) | 19.8 ± 0.2 (40.2) | 76.3 ± 0.8 (95.0) |
| dropout, proprio keep 0.5 | 77.4 ± 1.2 | 70.5 ± 1.2 (84.2) | 66.0 ± 1.5 (82.8) | 23.3 ± 8.0 (45.2) | 76.1 ± 1.9 (95.2) |

- **Without dropout, a missing input is fatal.** The no-dropout models fail completely without language or the wrist camera, and keep 22–30% without the static camera.
- **The dropout models lose 7–15 points without language or the static camera**, and 54–59 points without the wrist camera.
- **Removing the static camera helps under camera-viewpoint perturbations.** The dropout models score 74–89% there without it, against 49–69% with it, because the perturbation moves the static camera (see the [appendix](#libero-plus-per-category-with-one-modality-withheld)).

### LIBERO-PRO (seed 5 only)

Only the seed-5 runs of four models were evaluated on LIBERO-PRO. All inputs, 200 episodes per perturbation:

| model | lan | object | swap | task |
|---|---|---|---|---|
| no dropout | 92.0 | 57.0 | 0.0 | 9.0 |
| no dropout + proprio | 90.0 | 50.0 | 0.5 | 7.5 |
| dropout, proprio keep 1.0 | 92.0 | 48.5 | 0.0 | 11.5 |
| dropout, proprio keep 0.5 | 93.0 | 38.5 | 0.0 | 10.0 |

- **Reworded instructions (`lan`) cost nothing.** Paired with the same LIBERO-10 episodes, every model is within 3 points of its LIBERO-10 result (exact McNemar p ≥ 0.18).
- **A changed task (`task`) or swapped object positions (`swap`) defeats every model** (0–12%).
  - The paired `task` drop is 80–85 points (p < 1e-37).
  - The dropout models drop as much with any subset of inputs.
  - The policies execute the trained motions rather than follow a new instruction.
- **Replaced objects (`object`)** cost the dropout models more (38.5–48.5%) than the no-dropout models (50–57%), but this is one seed.

### Earlier experiments (old evaluation code)

These runs (W&B project `multimodal_policies`, 2026-09-03 to 09-12) used batch 32 like the main study. Their evaluations predate every eval fix listed above, so they are only roughly comparable. The batch-size-8 runs before them are not reported (see [below](#missing-or-unusable-libero-results)).

**Proprio-keep sweep and action-head variants.** Dirichlet keep 0.5 with proprioception unless noted. The 12-block head uses the first 12 pretrained blocks; the 24-block head adds 12 zero-initialized blocks, interleaved. Success (%) on LIBERO-10 with all inputs and without proprioception, and on LIBERO-Plus with all inputs:

| run | model | seed | LIBERO-10 | LIBERO-10, no proprio | LIBERO-Plus |
|---|---|---|---|---|---|
| `libero_10_dropout/2026-09-09_13-41-45` | proprio keep 1.0 | 5 | 96.0 | 36.0 | 82.3 |
| `libero_10_dropout/2026-09-09_13-53-25` | proprio keep 1.0 | 0 | 93.5 | 59.0 | 76.2 |
| `libero_10_dropout/2026-09-11_15-19-28` | proprio keep 1.0 | 100 | 94.5 | 37.0 | 81.4 |
| `libero_10_dropout/2026-09-11_14-58-49` | proprio keep 0.75 | 42 | 94.5 | 91.5 | 75.2 |
| `libero_10_dropout/2026-09-11_15-04-42` | proprio keep 0.75 | 5 | 96.5 | 95.0 | 78.2 |
| `libero_10_dropout/2026-09-12_10-38-53` | proprio keep 0.75 | 100 | 93.0 | 96.0 | 76.3 |
| `libero_10_dropout/2026-09-07_17-26-01` | proprio keep 0.5 | 42 | 93.5 | 95.0 | 75.1 |
| `libero_10_dropout/2026-09-09_13-55-28` | proprio keep 0.5 | 42 | 94.0 | 94.5 | 75.3 |
| `libero_10_dropout/2026-09-09_13-55-47` | proprio keep 0.5 | 5 | 98.0 | 95.5 | 76.0 |
| `libero_10_dropout/2026-09-11_15-11-37` | proprio keep 0.5 | 100 | 94.5 | 94.0 | 73.1 |
| `libero_10_dropout/2026-09-11_15-06-55` | proprio keep 0.25 | 42 | 95.0 | 94.0 | 74.0 |
| `libero_10_dropout/2026-09-11_15-07-27` | proprio keep 0.25 | 5 | 97.0 | 98.0 | 76.2 |
| `libero_10_dropout/2026-09-11_15-08-07` | proprio keep 0 | 42 | 0.5 | 93.5 | 0.1* |
| `libero_10_dropout/2026-09-11_15-10-27` | proprio keep 0 | 5 | 0.0 | 98.0 | 0.0* |
| `libero_10/2026-09-04_17-36-21` | no dropout + proprio | 42 | 95.0 | n/a | 74.2 |
| `libero_10_static+wrist+lang+proprio/2026-09-09_13-56-20` | no dropout + proprio | 42 | 93.0 | n/a | 74.8 |
| `libero_10_static+wrist+lang+proprio/2026-09-09_13-56-15` | no dropout + proprio | 5 | 93.0 | n/a | 72.2 |
| `libero_10_dropout/2026-09-11_16-03-00` | 12-block head, proprio keep 0.5 | 42 | 96.5 | 96.0 | 75.8 |
| `libero_10_dropout/2026-09-12_10-42-30` | 12-block head, proprio keep 0.5 | 5 | 95.5 | 96.5 | 74.4 |
| `libero_10_dropout/2026-09-12_11-05-01` | 12-block head, proprio keep 1.0 | 42 | 93.5 | 48.5 | 80.0 |
| `libero_10_dropout/2026-09-12_11-05-37` | 12-block head, proprio keep 1.0 | 5 | 95.0 | 59.5 | 79.5 |
| `libero_10_dropout/2026-09-11_16-06-56` | 24-block head, proprio keep 0.5 | 42 | 95.0 | 94.5 | 72.8 |
| `libero_10_dropout/2026-09-12_11-02-05` | 24-block head, proprio keep 0.5 | 5 | 96.5 | 95.5 | 75.3 |
| `libero_10_dropout/2026-09-12_11-03-33` | 24-block head, proprio keep 1.0 | 42 | 93.5 | 36.0 | 79.9 |
| `libero_10_dropout/2026-09-12_11-04-37` | 24-block head, proprio keep 1.0 | 5 | 95.0 | 50.5 | 78.5 |
| `libero_10_dropout/2026-09-12_10-56-45` | 18-block head (12 pretrained + 6 appended), proprio keep 0.25 | 100 | 96.5 | 95.0 | 75.7 |

\* LIBERO-Plus was run with proprioception on, which the proprio keep 0 models never saw.

- **These earlier runs show the same pattern as the main study.**
  - Proprio keep 1.0 gives the best LIBERO-Plus results (76–82%) but collapses without proprioception (36–60%).
  - Every keep below 1 is indifferent to proprioception.
  - Keep 0 never learned to use it: proprio on at evaluation gives 0%.
- **Shrinking or deepening the action head changes little** (72.8–80.0% on LIBERO-Plus).

**Fixed-modality training and other suites** (seed 42, no evaluation artifact). These show only the success rate of the in-training rollout on the unperturbed suite, at the end of training:

| run | inputs during training and rollout | in-training success |
|---|---|---|
| `libero_10_static/2026-09-03_13-35-06` | wrist + language | 87.0 |
| `libero_10_wrist/2026-09-03_13-35-05` | static + language | 87.2 |
| `libero_10_lang/2026-09-03_13-35-02` | static + wrist | 81.4 |
| `libero_10_static+lang/2026-09-03_13-33-48` | wrist | 81.6 |
| `libero_10_wrist+lang/2026-09-03_13-33-47` | static | 76.8 |
| `libero_10_static+wrist/2026-09-03_13-35-04` | language | 0.0 |
| `libero_10_dropout/2026-09-03_10-51-36` | Dirichlet keep 0.5, no proprio, LIBERO-10 | 92.4 |
| `libero_goal_dropout/2026-09-03_11-44-38` | the same, LIBERO-Goal | 96.0 |
| `libero_spatial_dropout/2026-09-03_11-48-13` | the same, LIBERO-Spatial | 97.2 |
| `libero_10_dropout/2026-09-04_17-20-02` | Dirichlet keep 0.5, proprio always kept | 95.6 |

The group names of the fixed-modality runs list the inputs that were *removed*.

### Missing or unusable LIBERO results

Run folders are under `train_logs/<group>/<run>`. "W&B host" is where the run trained. Everything that is on disk here lives on goldberry (`/mnt/goldberry_sdb1/morsic/flower_vla_calvin/train_logs`).

**Main study, evaluated before the fix: re-evaluate with `PIPELINE_REEVAL=1 ./run.sh pipeline /saves/train_logs/<group>/<run>`:**

| run | model, seed | W&B host | checkpoint here |
|---|---|---|---|
| `libero_10_dropout/2026-09-14_17-00-43` | proprio keep 1.0, seed 42 | glaurung | yes (goldberry) |
| `libero_10_dropout/2026-09-14_17-00-58` | proprio keep 0.5, seed 42 | glaurung | yes (goldberry) |
| `libero_10_dropout/2026-09-14_17-03-21` | proprio keep 0.75, seed 5 | glaurung | no |
| `libero_10_dropout/2026-09-15_16-11-38` | proprio keep 0.25, seed 5 | tom | no |
| `libero_10/2026-09-18_16-48-01` | no dropout, seed 100 | tom | no |
| `libero_10_static+wrist+lang+proprio/2026-09-18_16-47-49` | no dropout + proprio, seed 100 | tom | no |

**Main study, never evaluated or never trained:**
- `libero_10_dropout/2026-09-15_16-05-08`: proprio keep 0.25, seed 42 (tom). Trained, but there is no evaluation on W&B.
- Never trained: proprio keep 0.25 with seed 100, and Dirichlet dropout without proprio with seed 100.
- With the six re-evaluations above, every model would have three seeds. Proprio keep 0.25 currently has no valid evaluation at all.
- LIBERO-PRO exists only for the seed-5 runs of no dropout, no dropout + proprio, and proprio keep 1.0 and 0.5. It was never run for proprio keep 0.75, dropout without proprio, or any seed 42 or 100 run.

**Earlier phase:**
- **No evaluation artifact:** the six fixed-modality runs above (checkpoints on goldberry), the three 2026-09-03 dropout runs (`libero_10_dropout/2026-09-03_10-51-36`, `libero_goal_dropout/2026-09-03_11-44-38`, `libero_spatial_dropout/2026-09-03_11-48-13`; not on disk here) and `libero_10_dropout/2026-09-04_17-20-02`.
- **Unusable LIBERO-Plus result:** proprio keep 0 (`libero_10_dropout/2026-09-11_15-08-07`, daisy; `2026-09-11_15-10-27`, primrose). It was evaluated with proprioception on.
- **Debug or smoke-test runs, not reported:** `libero_10/2026-09-09_12-16-14` (1 epoch, 2 batches).
- **Invalid, batch size 8:** `libero_10_frozen/2026-08-27_12-15-55` (frozen VLM; crashed, no checkpoint), `train_logs_libero_10`, `libero_10_dropout/2026-08-28_15-45-43`, `2026-08-30_09-26-46`, `2026-08-30_13-10-49`, `2026-09-02_11-37-06`, `libero_goal_dropout/2026-09-02_10-58-48`, `libero_object_dropout/2026-09-02_10-58-43`, `libero_spatial_dropout/2026-09-02_10-58-38`.
- **LIBERO-Object** was only trained in that invalid batch-8 phase. No valid LIBERO-Goal, -Spatial or -Object result has been evaluated.

## Across the benchmarks

- **Language matters only when the scene is ambiguous.** CALVIN needs it to know which of several tasks to do. MimicGen does not. On LIBERO-10, dropout models lose about 10 points without it (82–86% vs 93–95%).
- **Which camera matters most depends on the benchmark.**
  - The static camera is the stronger single view on CALVIN and MimicGen.
  - On LIBERO-10 it is the wrist camera (65–76% alone vs 33–41% for the static camera).
  - How well one camera substitutes for the other also depends on the task: well on CALVIN's coarse tabletop tasks and LIBERO, poorly on MimicGen's precise assembly and insertion tasks.
- **How dropout is applied matters as much as how much is applied.**
  - Dropout makes a model work without some inputs only if training removed *whole* modalities often enough: Dirichlet rather than uniform (on CALVIN and MimicGen), a keep fraction of 0.5 rather than 0.75, and proprioception keep below 1 (on MimicGen and LIBERO).
  - Heavy dropout (keep 0.25, CALVIN) moves the training inputs so far from the full input that the full input is no better than static camera + language.
- **Dropout costs nothing on unperturbed tasks on CALVIN (keep 0.75) and LIBERO, but 4–6 points on MimicGen.** Only LIBERO tested perturbed scenes. There it raised success by about 9 points (LIBERO-Plus 77–81% vs 68–71%).

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
| CALVIN uniform keep 0.5, seeds 5 / 100 | `calvin_abcd_d_dropout/2026-10-21_03-14-30` / `2026-10-14_09-58-42` |
| CALVIN Dirichlet keep 0.25, seeds 5 / 100 | `calvin_abcd_d_dropout/2026-10-23_16-48-27` / `2026-10-27_02-46-45` |
| CALVIN Dirichlet keep 0.5 + proprio | `calvin_abcd_d_dropout/2026-10-21_03-14-19` |
| CALVIN no dropout + proprio, seeds 42 / 5 | `calvin_abcd_d/2026-10-14_13-37-34` / `2026-10-14_13-37-42` (W&B runs `calvin_abcd_d_2026-10-14_13-37-34` / `_13-37-42`; trained on another host) |
| CALVIN no dropout + proprio, seed 100 | `calvin_abcd_d/2026-10-27_22-04-48` |
| MimicGen no dropout, seeds 42 / 5 | `mimicgen/2026-10-01_19-26-58` / `2026-10-27_02-50-24` |
| MimicGen no dropout + proprio, seeds 42 / 5 | `mimicgen/2026-10-16_16-38-11` / `2026-10-16_16-38-21` |
| MimicGen dropout, proprio keep 0.5 / 1.0 | `mimicgen_dropout/2026-10-05_09-31-11_pkeep0.5` / `_pkeep1.0` |
| MimicGen dropout, proprio keep 0.5 / 1.0, seed 5 | `mimicgen_dropout/2026-10-19_11-46-58` / `2026-10-25_11-13-31` |
| MimicGen uniform keep 0.5, proprio keep 0.5 | `mimicgen_dropout/2026-10-25_11-13-39` |
| MimicGen Dirichlet keep 0.75, proprio keep 0.5 | `mimicgen_dropout/2026-10-19_11-47-04` |

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
# Dirichlet dropout with proprioception
./run.sh train-calvin-dropout batch_size=8 accumulate_grad_batches=2 model.use_proprio=True proprio_dims=8
CUDA_VISIBLE_DEVICES=0 ./run.sh open-loop-calvin /saves/train_logs/calvin_abcd_d_dropout/<ts>
./run.sh analyze-open-loop

# MimicGen
./run.sh train-mimicgen
./run.sh train-mimicgen batch_size=8 accumulate_grad_batches=2 model.use_proprio=True seed=5
# re-evaluate an existing run at 100 episodes per dataset (and combination)
PIPELINE_REEVAL=1 ./run.sh pipeline-mimicgen /saves/train_logs/mimicgen/<run> n_eval=100
./run.sh train-mimicgen-dropout model.use_proprio=True model.modality_dropout_proprio_keep_p=0.5
# Dirichlet keep 0.75; SKIP_PIPELINE=1 then pipeline-mimicgen n_eval=100, since the chained eval runs 20 episodes
SKIP_PIPELINE=1 ./run.sh train-mimicgen-dropout batch_size=8 accumulate_grad_batches=2 model.use_proprio=True \
    model.modality_dropout_keep_fraction=0.75

# LIBERO-10 (batch 32 by default; training chains ./run.sh pipeline: LIBERO-10 per combo, LIBERO-Plus, modality-off, LIBERO-PRO)
./run.sh train libero_10                                   # no dropout; add model.use_proprio=True for + proprio
./run.sh train-dropout libero_10 model.use_proprio=True model.modality_dropout_proprio_keep_p=0.5 seed=5
PIPELINE_REEVAL=1 ./run.sh pipeline /saves/train_logs/libero_10_dropout/<run>   # re-evaluate with the current eval code
./run.sh analyze filter --filters '{"config.modality_dropout": true}'           # tables from the W&B eval artifacts
```

## Appendix

### CALVIN per-subtask success, all inputs

Success rate (%) of each attempted subtask in the 1000 closed-loop chains, averaged over seeds 42, 5 and 100.

| subtask | no dropout | Dirichlet keep 0.75 | Dirichlet keep 0.5 | Dirichlet keep 0.25 | uniform keep 0.5 |
|---|---|---|---|---|---|
| close_drawer | 100.0 | 100.0 | 100.0 | 33.9 | 100.0 |
| lift_blue_block_drawer | 98.4 | 96.4 | 96.6 | 81.8 | 97.9 |
| lift_blue_block_slider | 94.5 | 93.3 | 91.0 | 81.4 | 92.9 |
| lift_blue_block_table | 99.8 | 99.5 | 99.4 | 74.8 | 99.2 |
| lift_pink_block_drawer | 94.9 | 95.2 | 91.0 | 66.7 | 88.6 |
| lift_pink_block_slider | 97.9 | 97.9 | 96.6 | 87.0 | 98.2 |
| lift_pink_block_table | 98.9 | 97.2 | 97.9 | 40.0 | 92.9 |
| lift_red_block_drawer | 94.1 | 98.1 | 98.1 | 78.6 | 93.9 |
| lift_red_block_slider | 96.1 | 96.3 | 96.1 | 80.8 | 95.3 |
| lift_red_block_table | 99.3 | 98.8 | 98.5 | 51.4 | 98.6 |
| move_slider_left | 100.0 | 99.6 | 99.2 | 99.5 | 99.4 |
| move_slider_right | 100.0 | 99.8 | 100.0 | 99.8 | 99.7 |
| open_drawer | 99.9 | 99.6 | 99.9 | 38.7 | 99.6 |
| place_in_drawer | 99.8 | 98.6 | 98.7 | 90.2 | 99.2 |
| place_in_slider | 95.4 | 91.7 | 83.4 | 29.6 | 40.3 |
| push_blue_block_left | 92.8 | 93.3 | 86.1 | 61.0 | 76.2 |
| push_blue_block_right | 72.2 | 74.8 | 60.6 | 27.8 | 37.3 |
| push_into_drawer | 81.3 | 79.0 | 79.6 | 77.1 | 81.0 |
| push_pink_block_left | 88.7 | 91.7 | 84.7 | 55.3 | 74.0 |
| push_pink_block_right | 71.6 | 81.5 | 73.6 | 43.9 | 65.9 |
| push_red_block_left | 90.7 | 94.8 | 91.3 | 53.9 | 80.5 |
| push_red_block_right | 68.8 | 71.2 | 59.4 | 27.6 | 47.6 |
| rotate_blue_block_left | 100.0 | 99.0 | 97.5 | 83.3 | 99.5 |
| rotate_blue_block_right | 94.9 | 93.1 | 92.2 | 84.5 | 94.4 |
| rotate_pink_block_left | 95.9 | 98.2 | 93.9 | 65.4 | 93.0 |
| rotate_pink_block_right | 76.9 | 84.3 | 95.3 | 60.0 | 79.2 |
| rotate_red_block_left | 99.5 | 99.5 | 100.0 | 71.8 | 98.9 |
| rotate_red_block_right | 95.1 | 94.6 | 95.0 | 73.3 | 97.6 |
| stack_block | 93.2 | 88.8 | 84.0 | 60.0 | 85.1 |
| turn_off_led | 99.2 | 98.8 | 99.6 | 99.4 | 99.1 |
| turn_off_lightbulb | 100.0 | 99.6 | 90.4 | 29.4 | 98.1 |
| turn_on_led | 98.4 | 98.7 | 99.0 | 100.0 | 99.1 |
| turn_on_lightbulb | 99.8 | 100.0 | 99.8 | 68.8 | 100.0 |
| unstack_block | 99.5 | 99.0 | 99.5 | 98.6 | 99.5 |

### MimicGen per-dataset success, all inputs

Success rate (%), 100 episodes per dataset; columns give the seed. The dropout runs and the "+ proprio" runs are evaluated with all four inputs. Uniform and keep 0.75 use proprio keep 0.5.

| dataset | no dropout, 42 | no dropout, 5 | + proprio, 42 | + proprio, 5 | keep 0.5, proprio keep 0.5, 42 | same, 5 | keep 0.5, proprio keep 1.0, 42 | same, 5 | uniform keep 0.5 | keep 0.75 |
|---|---|---|---|---|---|---|---|---|---|---|
| coffee_d0 | 95 | 67 | 93 | 78 | 88 | 93 | 88 | 89 | 74 | 74 |
| coffee_d1 | 65 | 55 | 77 | 71 | 50 | 46 | 73 | 52 | 52 | 52 |
| coffee_d2 | 14 | 14 | 21 | 33 | 10 | 9 | 14 | 9 | 7 | 10 |
| coffee_preparation_d0 | 61 | 59 | 71 | 91 | 39 | 78 | 64 | 43 | 70 | 66 |
| coffee_preparation_d1 | 35 | 25 | 31 | 38 | 14 | 12 | 18 | 11 | 20 | 11 |
| hammer_cleanup_d0 | 98 | 88 | 95 | 100 | 89 | 91 | 94 | 95 | 98 | 98 |
| hammer_cleanup_d1 | 65 | 55 | 64 | 52 | 48 | 42 | 53 | 50 | 66 | 51 |
| kitchen_d0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 99 |
| kitchen_d1 | 68 | 65 | 65 | 58 | 68 | 61 | 51 | 58 | 62 | 62 |
| mug_cleanup_d0 | 67 | 52 | 66 | 58 | 50 | 48 | 56 | 58 | 53 | 53 |
| mug_cleanup_d1 | 46 | 40 | 48 | 45 | 39 | 23 | 36 | 41 | 41 | 50 |
| nut_assembly_d0 | 15 | 6 | 24 | 18 | 17 | 7 | 11 | 5 | 6 | 6 |
| pick_place_d0 | 47 | 46 | 42 | 45 | 35 | 36 | 24 | 24 | 42 | 46 |
| square_d0 | 71 | 83 | 87 | 71 | 69 | 54 | 67 | 50 | 57 | 69 |
| square_d1 | 38 | 35 | 35 | 45 | 41 | 28 | 47 | 39 | 35 | 31 |
| square_d2 | 31 | 15 | 19 | 25 | 26 | 18 | 36 | 29 | 26 | 22 |
| stack_d0 | 99 | 97 | 100 | 99 | 99 | 98 | 100 | 100 | 97 | 98 |
| stack_d1 | 99 | 99 | 99 | 99 | 100 | 99 | 100 | 100 | 99 | 100 |
| stack_three_d0 | 82 | 82 | 87 | 84 | 89 | 78 | 81 | 75 | 76 | 82 |
| stack_three_d1 | 71 | 75 | 74 | 65 | 73 | 65 | 74 | 79 | 72 | 70 |
| threading_d0 | 62 | 57 | 37 | 42 | 28 | 17 | 23 | 43 | 17 | 26 |
| threading_d1 | 10 | 0 | 5 | 4 | 6 | 2 | 6 | 2 | 3 | 4 |
| threading_d2 | 3 | 3 | 3 | 5 | 3 | 1 | 2 | 1 | 4 | 1 |
| three_piece_assembly_d0 | 21 | 21 | 21 | 24 | 18 | 11 | 21 | 16 | 19 | 13 |
| three_piece_assembly_d1 | 15 | 17 | 14 | 23 | 10 | 13 | 15 | 14 | 21 | 9 |
| three_piece_assembly_d2 | 5 | 2 | 2 | 2 | 2 | 2 | 4 | 4 | 2 | 4 |

### LIBERO-Plus per category with one modality withheld

Success rate (%) per LIBERO-Plus perturbation category with one input removed at evaluation, mean ± sd over the two valid seeds.

**No language**

| category | no dropout | no dropout + proprio | dropout, no proprio | dropout, proprio keep 1.0 | dropout, proprio keep 0.75 | dropout, proprio keep 0.5 |
|---|---|---|---|---|---|---|
| Camera Viewpoints | 0.0 ± 0.0 | 0.1 ± 0.2 | 41.3 ± 5.1 | 57.0 ± 4.1 | 50.0 ± 3.5 | 42.0 ± 2.0 |
| Robot Initial States | 0.0 ± 0.0 | 0.5 ± 0.7 | 68.2 ± 1.8 | 67.2 ± 0.7 | 67.0 ± 5.2 | 68.3 ± 0.5 |
| Language Instructions | 0.0 ± 0.0 | 3.7 ± 5.2 | 94.5 ± 0.0 | 94.5 ± 0.0 | 88.6 ± 8.3 | 94.5 ± 0.0 |
| Light Conditions | 0.0 ± 0.0 | 0.9 ± 1.3 | 68.4 ± 2.3 | 71.0 ± 4.4 | 74.3 ± 6.5 | 71.0 ± 6.5 |
| Background Textures | 0.0 ± 0.0 | 0.7 ± 1.0 | 66.4 ± 1.5 | 71.1 ± 1.7 | 63.8 ± 8.6 | 67.0 ± 1.2 |
| Sensor Noise | 0.0 ± 0.0 | 0.3 ± 0.5 | 70.3 ± 2.4 | 76.3 ± 0.2 | 74.9 ± 5.8 | 78.6 ± 2.8 |
| Objects Layout | 0.2 ± 0.2 | 2.2 ± 3.2 | 70.7 ± 1.1 | 76.1 ± 1.1 | 70.2 ± 4.5 | 73.6 ± 4.8 |

**No static camera**

| category | no dropout | no dropout + proprio | dropout, no proprio | dropout, proprio keep 1.0 | dropout, proprio keep 0.75 | dropout, proprio keep 0.5 |
|---|---|---|---|---|---|---|
| Camera Viewpoints | 19.9 ± 19.1 | 35.1 ± 2.0 | 73.5 ± 23.0 | 88.7 ± 15.0 | 88.9 ± 15.7 | 84.2 ± 6.8 |
| Robot Initial States | 19.1 ± 4.0 | 20.5 ± 0.9 | 50.9 ± 1.8 | 57.9 ± 2.0 | 49.7 ± 2.3 | 49.9 ± 1.4 |
| Language Instructions | 35.6 ± 3.9 | 37.1 ± 1.8 | 77.5 ± 5.9 | 78.2 ± 1.3 | 78.6 ± 7.8 | 77.5 ± 7.4 |
| Light Conditions | 25.0 ± 12.1 | 33.9 ± 13.9 | 47.6 ± 3.4 | 52.0 ± 5.4 | 65.3 ± 9.3 | 46.4 ± 9.8 |
| Background Textures | 15.9 ± 4.4 | 23.5 ± 8.8 | 46.4 ± 1.0 | 58.8 ± 6.4 | 32.9 ± 2.9 | 43.6 ± 8.8 |
| Sensor Noise | 22.6 ± 14.6 | 35.6 ± 5.7 | 75.1 ± 20.8 | 89.4 ± 13.1 | 89.8 ± 14.5 | 85.7 ± 6.3 |
| Objects Layout | 17.6 ± 4.5 | 22.6 ± 4.8 | 46.3 ± 0.7 | 69.9 ± 1.4 | 66.8 ± 2.5 | 56.9 ± 2.5 |

**No wrist camera**

| category | no dropout | no dropout + proprio | dropout, no proprio | dropout, proprio keep 1.0 | dropout, proprio keep 0.75 | dropout, proprio keep 0.5 |
|---|---|---|---|---|---|---|
| Camera Viewpoints | 0.0 ± 0.0 | 0.0 ± 0.0 | 4.4 ± 0.8 | 5.4 ± 1.5 | 3.6 ± 0.3 | 3.1 ± 1.0 |
| Robot Initial States | 0.0 ± 0.0 | 0.0 ± 0.0 | 24.9 ± 4.7 | 25.7 ± 6.8 | 22.1 ± 2.5 | 24.6 ± 8.5 |
| Language Instructions | 0.0 ± 0.0 | 0.0 ± 0.0 | 43.1 ± 1.1 | 37.7 ± 22.7 | 37.2 ± 5.0 | 46.3 ± 16.1 |
| Light Conditions | 0.0 ± 0.0 | 0.0 ± 0.0 | 17.0 ± 1.3 | 21.5 ± 9.8 | 15.5 ± 1.8 | 21.2 ± 12.4 |
| Background Textures | 0.0 ± 0.0 | 0.0 ± 0.0 | 23.7 ± 2.7 | 23.4 ± 8.1 | 22.3 ± 1.7 | 29.4 ± 9.3 |
| Sensor Noise | 0.0 ± 0.0 | 0.0 ± 0.0 | 8.4 ± 1.1 | 14.1 ± 5.8 | 12.0 ± 0.3 | 12.8 ± 6.8 |
| Objects Layout | 0.0 ± 0.0 | 0.0 ± 0.0 | 30.4 ± 6.3 | 36.1 ± 2.9 | 29.6 ± 2.0 | 31.6 ± 3.4 |

**No proprioception**

| category | no dropout + proprio | dropout, proprio keep 1.0 | dropout, proprio keep 0.75 | dropout, proprio keep 0.5 |
|---|---|---|---|---|
| Camera Viewpoints | 19.5 ± 1.9 | 19.9 ± 5.6 | 48.9 ± 4.4 | 46.7 ± 6.2 |
| Robot Initial States | 44.8 ± 2.9 | 36.1 ± 8.3 | 71.4 ± 1.6 | 72.0 ± 0.4 |
| Language Instructions | 61.1 ± 5.2 | 52.3 ± 12.0 | 96.3 ± 1.5 | 96.1 ± 1.8 |
| Light Conditions | 39.2 ± 11.6 | 25.9 ± 11.9 | 79.9 ± 3.1 | 74.8 ± 2.1 |
| Background Textures | 49.3 ± 2.7 | 24.2 ± 1.0 | 73.4 ± 11.3 | 81.1 ± 1.2 |
| Sensor Noise | 37.8 ± 0.2 | 27.7 ± 10.2 | 83.6 ± 0.8 | 80.8 ± 7.9 |
| Objects Layout | 45.8 ± 2.7 | 35.9 ± 13.6 | 83.7 ± 0.5 | 85.7 ± 2.9 |
