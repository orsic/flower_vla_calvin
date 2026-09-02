# FlowerVLA

[Paper](https://www.arxiv.org/pdf/2509.04996), [Project Page](https://intuitive-robots.github.io/flower_vla/), [Pretraining Code](https://github.com/intuitive-robots/flower_vla_pret)

[Moritz Reuss](https://mbreuss.github.io/)<sup>1</sup>,
[Hongyi Zhou](https://hongyizhoucn.github.io/)<sup>1</sup>,
[Marcel Ruehle]()<sup>1</sup>,
[Ömer Erdinç Yağmurlu](https://scholar.google.com/citations?user=I_Mxp5cAAAAJ&hl=en)<sup>1</sup>,
[Fabian Otto](https://ottofabian.github.io/)<sup>2</sup>,
[Rudolf Lioutikov](http://rudolf.intuitive-robots.net/)<sup>1</sup>

<sup>1</sup>Intuitive Robots Lab (IRL), Karlsruhe Institute of Technology (KIT)
<sup>2</sup>Microsoft Research


##  FLOWER: An efficient & versatile Flow-VLA

FLOWER VLA is a lightweight, efficient Vision-Language-Action (VLA) policy for robotic manipulation tasks that achieves state-of-the-art performance on multiple benchmarks. Built on a rectified flow architecture with several key architecture features:

- **Efficient Architecture**: At ~1B parameters, FLOWER is significantly smaller than most VLA models
- **Low Training Cost**: Only requires ~200 GPU hours of pretraining
- **Low Memory Footprint**: Uses <3GB of GPU memory for inference with single image setting
- **SOTA Performance**: Achieves sota results on CALVIN and LIBERO benchmarks 

For the pretraining of FLOWER and finetuning for Aloha check out our other codebase.


## Model Overview

FLOWER VLA uses a Florence-2-large-based VLM combined with a rectified flow architecture to predict robot actions from visual observations and language instructions. 
The model efficiently handles multi-step action sequences through a chunking mechanism.

[Insert model architecture diagram here]

Key architectural components:
- Florence-2 DaVit Image Encoder [350M params]
- Language conditioning through cross-attention 
- Half of the Florence-2 LLM layers for vision and language fusion [205M]
- Rectified Flow Predition for fast action generation (all results in CALVIN and LIBERO are achieved using just 4 denoising steps) 
- Global AdaLN for parameter efficient conditioning
- Action chunking for multi-step action generation


## Installation
To begin, clone this repository locally
```bash
git clone --recurse-submodules git@github.com:intuitive-robots/flower_vla_calvin.git
export flower_calvin_ROOT=$(pwd)/flower_vla_calvin

```
Install requirements
(Note we provided a changed verison of pyhash, given numerous problems we encountered when installing it manually on our slurm cluster)
You can also try to install setup tools using pip. 
 
```bash
cd $flower_calvin_ROOT
conda create -n flower_cal python=3.9
conda activate flower_cal
cd calvin_env/tacto
pip install -e .
cd ..
pip install -e .
cd ..
cd LIBERO
pip install -r requirements.txt
pip install -e .
pip install numpy~=1.23
cd ..
pip install setuptools==57.5.0
cd pyhash-0.9.3
python setup.py build
python setup.py install
cd ..
```
Next we can install the rest of the missing packages

```
pip install -r requirements.txt
```

---

## Container Environment

A self-contained Podman environment covers building, training, evaluation, and interactive development — no conda setup required.

**Prerequisites:** Podman with CDI GPU support, `podman-compose`.

### Setup

```bash
cp vars.env.example vars.env   # fill in DATA_DIR, SAVES_DIR, HF_HOME
./run.sh build                 # build flower-vla-eval:latest (~11 GB, cached layers)
./run.sh download-pret         # pretrained FlowerVLA checkpoint → /saves/checkpoints/flower_vla_pret/
./run.sh download-data         # LIBERO-10 HDF5 demos → DATA_DIR/libero_hdf5/ (see below for other benchmarks)
```

`vars.env` (gitignored) sets host-specific paths:

| Variable | Mounted at | Purpose |
|---|---|---|
| `DATA_DIR` | — | Parent of `libero_hdf5/`; sets `LIBERO_HDF5_DIR` |
| `HF_HOME` | `/root/.cache/huggingface` | HuggingFace model cache |
| `SAVES_DIR` | `/saves` | Checkpoints, train logs, eval logs |

### Commands

```bash
./run.sh train [bench] [...]          # Fine-tune (default: libero_10, all modalities, GPU count = CUDA_VISIBLE_DEVICES length)
./run.sh train-frozen                 # Ablation: frozen Florence VLM, action expert from random init
./run.sh train-dropout [bench]        # Fine-tune with Dirichlet modality-token dropout (default: libero_10)
./run.sh train-dropout-resume [bench] # Resume train-dropout from CKPT_PATH env var
./run.sh eval                         # LIBERO-10 evaluation (~94.5% target)
./run.sh shell                        # Interactive bash inside the container
./run.sh smoke                        # Quick sanity check (CUDA + imports + 1 env step)
./run.sh devenv                       # Regenerate .devcontainer/.env after editing vars.env
```

**Benchmark selection** — `train` (and `train-dropout` / `train-dropout-resume`) accept an
optional benchmark name as the first positional arg, followed by any Hydra overrides:
```bash
./run.sh train libero_90
./run.sh train-dropout libero_spatial model.modality_dropout_keep_fraction=0.7
CKPT_PATH=/saves/.../last.ckpt ./run.sh train-dropout-resume libero_90
```
Valid benchmarks: `libero_10`, `libero_90`, `libero_spatial`, `libero_object`, `libero_goal`.

**Downloading hdf5 data for other benchmarks** — `download-data` takes the same benchmark arg,
or `all` to fetch every suite:
```bash
./run.sh download-data libero_90
./run.sh download-data all   # fetch all suites (~several GB)
```
Run `download-data <benchmark>` before training on a suite whose hdf5 files aren't on disk yet.

**GPU selection** — all services use `CUDA_VISIBLE_DEVICES` from `vars.env` (default `0` for eval,
`0,1,2,3` for training). Override per-run or set in `vars.env`:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh train   # 4-GPU training
CUDA_VISIBLE_DEVICES=2,3     ./run.sh eval    # eval on GPUs 2 & 3, leaving 0 & 1 free
```

**Batched eval** — `./run.sh eval` runs `eval_batch_size` episodes in parallel per task using
`SubprocVectorEnv` (one MuJoCo subprocess per episode, EGL offscreen rendering). Model inference
is batched across all parallel episodes. When `CUDA_VISIBLE_DEVICES` exposes multiple GPUs the
10 tasks are partitioned across them, each GPU running its own batched eval independently.
Tune `eval_batch_size` in `conf/eval_libero.yaml` (default 10) to trade RAM vs. parallelism.

### Modality-token dropout (`train-dropout`)

`train-dropout` trains with per-sample, pre-encoder token removal across three modality groups:
static view, wrist view, and language. For each sample in a batch, a Dirichlet distribution
(α=1 per group) samples proportions that determine how many tokens from each group to keep;
the remaining tokens are physically removed from the sequence before it enters the Florence-2
encoder, providing a real compute saving (not masking).

The position-correction step ensures every retained token's net positional embedding equals
its absolute position in the original sequence — removal is analytically equivalent to
attention-masking with positions preserved (verified by a unit test against the masking oracle).

The `<Flow>` prompt token is always kept. Rollout evaluation runs only at the final training
epoch (rollout_lh_skip_epochs=39). Hyperparameters (model.yaml defaults):

| Flag | Default | Meaning |
|---|---|---|
| `modality_dropout` | `False` | Enable pre-encoder token removal |
| `modality_dropout_keep_fraction` | `0.5` | Fraction of total tokens to keep per sample |
| `modality_dropout_alphas` | `[1.0, 1.0, 1.0]` | Dirichlet α for [static, wrist, language] |

### Train-time modality ablation (`train`)

Unlike `train-dropout` (one model exposed to *randomly varying* modality proportions every
step), a fixed modality-ablation model trains without dropout but only ever observes a fixed
subset of `{rgb_static, rgb_gripper, language}` — the withheld modalities' tokens are physically
removed pre-encoder in every forward, exactly like `eval_modalities` masking, but applied at
training time too. This is set with `model.modalities.*` Hydra overrides on `./run.sh train`:

```bash
./run.sh train                                    # default: all three modalities
./run.sh train libero_10 model.modalities.language=False
./run.sh train libero_10 model.modalities.rgb_static=False model.modalities.rgb_gripper=False
```

| Key | Default | Meaning |
|---|---|---|
| `modalities.rgb_static` | `True` | Static (scene) view enabled |
| `modalities.rgb_gripper` | `True` | Wrist view enabled |
| `modalities.language` | `True` | Language instruction enabled |

At least one modality must stay on (an all-`False` combo raises `ValueError` before any data
loads), and `model.modalities` is mutually exclusive with `model.modality_dropout=True`.

The run's Hydra directory (and hence its wandb `group`/`name`/`id`, both derived from it — see
`setup_logger` in `flower/training_libero.py`) is suffixed with the combo's *dropped* modalities,
joined with `+`, using the same short names (`static`/`wrist`/`lang`) as
`scripts/compare_eval_csvs.py`:

| Command | Run dir | wandb name |
|---|---|---|
| `./run.sh train` | `.../libero_10/<ts>` | `libero_10/<ts>` (unchanged) |
| `./run.sh train libero_10 model.modalities.language=False` | `.../libero_10_lang/<ts>` | `libero_10_lang/<ts>` |
| `./run.sh train libero_10 model.modalities.rgb_static=False model.modalities.rgb_gripper=False` | `.../libero_10_static+wrist/<ts>` | `libero_10_static+wrist/<ts>` |

The trained combo is part of the model's saved hyperparameters, so it's restored automatically
by `load_mode_from_safetensor`/checkpoint loading — evaluating one of these checkpoints with no
`eval_modalities` override re-applies the same subset it was trained on (see the precedence note
in "Modality-ablation eval" below).

### VS Code Devcontainer

The devcontainer runs the same `flower-vla-eval` image with GPU, all three mounts, and the LIBERO path config wired up automatically.

**One-time setup on the remote machine:**

```bash
./run.sh devenv    # generates .devcontainer/devcontainer.json (gitignored) from vars.env
```

**VS Code user settings** (Settings → Remote [SSH: hostname] → open JSON):
```json
{
    "dev.containers.dockerPath": "podman"
}
```

Then open the repo in VS Code and choose **Reopen in Container**. The container starts with all mounts active, `PYTHONPATH` set, and LIBERO paths configured. Re-run `./run.sh devenv` whenever `vars.env` changes.

---

## Download
### CALVIN Dataset

If you want to train on the [CALVIN](https://github.com/mees/calvin) dataset, choose a split with:
```bash
cd $flower_calvin_ROOT/dataset
sh download_data.sh D | ABCD
```

## Training
To train the FLOWER FLOWERl with the 4 GPUS, run:
```
python flower/training.py 
```

You can use the pretrained FLOWER checkpoint from [hf-link](https://huggingface.co/mbreuss/flower_vla_pret) to train your own model on any of the datasets. 

Note that during training the full CALVIN eval or LIBERO rollouts will be called after _rollout_lh_skip_epochs_ and then every _callbacks.rollout_lh.rollout_freq_*1k training steps. Check out the training config for adopting the parameters.

For replication of the orginial training results I recommend to use 4 GPUs with a batch_size of 8 and train them for 40k steps for ABC (ABCD) and evaluating after 19 epochs to get the best possible results.
See training configs for details.

#### Preprocessing with CALVIN
Since FLOWER uses action chunking, it needs to load multiple (~10) `episode_{}.npz` files for each inference. In combination with batching, this results in a large disk bandwidth needed for each iteration (usually ~2000MB/iteration).
This has the potential of significantly reducing your GPU utilization rate during training depending on your hardware.
Therefore, you can use the script `extract_by_key.py` to extract the data into a single file, avoiding opening too many episode files when using the CALVIN dataset.

##### Usage example:
```shell
python preprocess/extract_by_key.py -i /YOUR/PATH/TO/CALVIN/ \
    --in_task all
```

```
python preprocess/extract_by_key.py -i /hkfs/work/workspace/scratch/ft4740-play3/data --in_task all
```

##### Params:
Run this command to see more detailed information:
```shell
python preprocess/extract_by_key.py -h
```

Important params:
* `--in_root`: `/YOUR/PATH/TO/CALVIN/`, e.g `/data3/geyuan/datasets/CALVIN/`
* `--extract_key`: A key of `dict(episode_xxx.npz)`, default is **'rel_actions'**, the saved file name depends on this (i.e `ep_{extract_key}.npy`)
Optional params:
* `--in_task`: default is **'all'**, meaning all task folders (e.g `task_ABCD_D/`) of CALVIN
* `--in_split`: default is **'all'**, meaning both `training/` and `validation/`
* `--out_dir`: optional, default is **'None'**, and will be converted to `{in_root}/{in_task}/{in_split}/extracted/`
* `--force`: whether to overwrite existing extracted data
Thanks to @ygtxr1997 for debugging the GPU utilization and providing a merge request.


## Evaluation

Download the pretrained FLOWER from Hugging Face: 
You can find all checkpoints under:

- [FLOWER Collection](https://huggingface.co/collections/mbreuss/flower-vla-67d60e95bf2990699fcef81f)

We provide pretrained checkpoints for all CALVIN and LIBERO challenges.

## Performance Comparison on CALVIN Challenges (1000 chains)

Below is the average performance of FLOWER on CALVIN. It currenty is SOTA for all variants of CALVIN:

| Train→Test | Method | PrT | 1 | 2 | 3 | 4 | 5 | **Avg. Len.** |
|------------|---------|-----|---|---|---|---|---|---------------|
| ABCD→D | Diff-P-CNN | × | 86.3% | 72.7% | 60.1% | 51.2% | 41.7% | 3.16±0.06 |
| | Diff-P-T | × | 78.3% | 53.9% | 33.8% | 20.4% | 11.3% | 1.98±0.09 |
| | RoboFlamingo | ✓ | 96.4% | 89.6% | 82.4% | 74.0% | 66.0% | 4.09±0.00 |
| | GR-1 | ✓ | 94.9% | 89.6% | 84.4% | 78.9% | 73.1% | 4.21±0.00 |
| | MDT | × | 98.6% | 95.8% | 91.6% | 86.2% | 80.1% | 4.52±0.02 |
| | MoDE | ✓ | 97.1% | 92.5% | 87.9% | 83.5% | 77.9% | 4.39±0.04 |
| | KomosVLA | ✓ | 98.0% | 93.6% | 85.4% | 77.8% | 70.4% | 4.49 |
| | **FLOWER (ours)** | × | **99.1%** | **97.8%** | **95.2%** | **92.4%** | **87.8%** | **4.67±0.04** |
| ABC→D | Diff-P-CNN | × | 63.5% | 35.3% | 19.4% | 10.7% | 6.4% | 1.35±0.05 |
| | Diff-P-T | × | 62.2% | 30.9% | 13.2% | 5.0% | 1.6% | 1.13±0.02 |
| | RoboFlamingo | ✓ | 82.4% | 61.9% | 46.6% | 33.1% | 23.5% | 2.47±0.00 |
| | GR-1 | ✓ | 85.4% | 71.2% | 59.6% | 49.7% | 40.1% | 3.06±0.00 |
| | 3DDA | × | 93.8% | 80.3% | 66.2% | 53.3% | 41.2% | 3.35 |
| | MoDE | ✓ | 96.2% | 88.9% | 81.1% | 71.8% | 63.5% | 4.01±0.04 |
| | KomosVLA | ✓ | 98.0% | 93.6% | 85.4% | 77.8% | 70.4% | 4.25 |
| | VPP | ✓ | 95.7% | 91.2% | 86.3% | 81.0% | 75.0% | 4.29 |
| | Seer | ✓ | 96.3% | 91.6% | 86.1% | 80.3% | 74.0% | 4.28 |
| | **FLOWER (ours)** | × | **99.3%** | **95.9%** | **90.5%** | **84.8%** | **77.5%** | **4.54±0.02** |
| D→D| **FLOWER (ours)** | × | **98.4%** | **94.0%** | **87.9%** | **81.7%** | **74.1%** | **4.36±0.04** |

## Performance on LIBERO Benchmarks

FLOWER achieves strong performance across all LIBERO benchmarks:

| Benchmark | FLOWER Success Rate |
|-----------|---------------------|
| LIBERO-10 | 94.5% |
| LIBERO-90 | 93.4% | 
| LIBERO-SPATIAL | 97.2% |
| LIBERO-OBJECT | 99.3% | 
| LIBERO-GOAL | 96.9% | 


## LIBERO-Plus Robustness Evaluation

[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) ([paper](https://huggingface.co/papers/2510.13626))
extends the LIBERO benchmark with **10,030 perturbed task instances** across 7 categories to measure
VLA robustness. Its key finding: VLA models collapse from ~95% to <30% under modest perturbations
and largely ignore language instructions.

We integrate LIBERO-Plus as a side-by-side submodule, enabling direct comparison of the baseline
and modality-dropout models under controlled perturbations.

### Setup

```bash
# 1. Download the baseline checkpoint (if not already done)
./run.sh download

# 2. Download LIBERO-Plus simulation assets (~several GB, one-time)
./run.sh download-plus

# 3. Build the image (if not already built)
./run.sh build
```

### Running the two-model comparison

Let `CKPT_BASE=/saves/checkpoints/libero_10` (baseline) and
`CKPT_DROP=/saves/train_logs/libero_10_dropout/<run>/saved_models/<best.ckpt>` (dropout model).

**A. Baseline LIBERO-10 (original 10 tasks, n_eval=20):**
```bash
./run.sh eval train_folder=$CKPT_BASE checkpoint=$CKPT_BASE
./run.sh eval train_folder=$CKPT_DROP checkpoint=$CKPT_DROP
```

**B. LIBERO-Plus — per perturbation category (n_eval=1 per task):**
```bash
# Run a single category (faster, useful for spot-checking):
./run.sh eval-plus task_category="Camera Viewpoints" train_folder=$CKPT_BASE checkpoint=$CKPT_BASE
./run.sh eval-plus task_category="Camera Viewpoints" train_folder=$CKPT_DROP checkpoint=$CKPT_DROP

# Run all categories at once (multi-GPU recommended; set CUDA_VISIBLE_DEVICES in vars.env):
./run.sh eval-plus train_folder=$CKPT_BASE checkpoint=$CKPT_BASE
./run.sh eval-plus train_folder=$CKPT_DROP checkpoint=$CKPT_DROP
```
`train_folder` also fixes where results are written (see [Evaluation results (CSV)](#evaluation-results-csv)
below) — set it to each model's own directory so the two models' CSVs don't collide.

> **Memory note:** always keep `n_eval=1` for LIBERO-Plus. Each Plus task is one
> deterministic perturbed instance, so higher values add no new perturbations.
> More importantly, MuJoCo/robosuite offscreen render contexts don't fully free
> native EGL/GL memory on `env.close()`, so peak RSS grows monotonically with
> the total env create/destroy cycle count:
> `n_tasks × ceil(n_eval / eval_batch_size)`.
> With `n_eval=50` and 419 Camera-Viewpoint tasks that is ~20,950 instantiations,
> which OOMs under the 64 GB container cap (`MEM_LIMIT` in `vars.env`).
> If a full 2519-task run approaches the cap, split it by category (as above)
> or use multi-GPU (`CUDA_VISIBLE_DEVICES=2,3`) so each spawned worker process
> releases its address space on exit.

### Evaluation results (CSV)

Every `./run.sh eval` / `./run.sh eval-plus` run writes one row per episode to:
```
<train_folder>/eval_logs/<checkpoint_name>/<libero_variant>_<benchmark_name>/result.csv
```
e.g. `$CKPT_DROP/eval_logs/last/plus_libero_10/result.csv`. `checkpoint_name` is the
checkpoint file's stem (`last.ckpt` → `last`) or, for a HuggingFace-layout checkpoint
directory, the directory name. Override the whole path with `csv_dir=<path>` when
`train_folder` isn't writable (e.g. a shared read-only checkpoint). Re-running (e.g. one `task_category` at a time) **merges**
into the existing file — a rerun of the same task/episode/checkpoint replaces that row in
place, everything else is kept — so running all 7 Plus categories separately accumulates
into one complete `result.csv`. On multi-GPU, each worker writes `result_rank<i>.csv`
into the same directory; the master merges them into `result.csv` and deletes the shards.

Columns identify every variable that determines the episode: `suite`, `task_idx`,
`task_name` (LIBERO-Plus encodes the full perturbation — camera angle, robot-init id,
noise id, etc. — in this name; no separate columns), `task_category` /
`difficulty_level` (LIBERO-Plus only), `language` (the exact instruction fed to the
model), `init_state_idx`, `rollout_seed`, `num_sampling_steps`, `multistep`,
`eval_batch_size`, and the three modality flags `use_rgb_static`, `use_rgb_gripper`,
`use_language`. These three are part of the row's merge key (see below), so evaluating
the same task/episode/checkpoint under a different modality combo adds a new row
instead of overwriting the previous combo's result. Result columns are `success`
(0/1) and `steps_taken`.

**Reproducibility:** the model draws one shared flow-matching noise tensor for an entire
batch on each replan, so a rollout is only reproducible at batch granularity, not
per-episode — `rollout_seed` is derived from `(seed, task_idx, batch_start_episode)` and
recorded per row along with `eval_batch_size`; reproducing a row requires matching both.
**Known exception:** LIBERO-Plus Sensor-Noise tasks using motion_blur, fog, or
glass_blur corruption (noise ids 1-10, 31-40, 41-50) call unseeded `np.random` on every
rendered frame inside the vendored env, so those specific episodes are not bit-exact
reproducible even with a matching seed; gaussian_blur/zoom_blur (ids 11-30) are unaffected.

**Modality-ablation eval.** `eval_modalities` in `conf/eval_libero.yaml` /
`conf/eval_libero_plus.yaml` is functional: setting any of `rgb_static`,
`rgb_gripper`, `language` to `False` physically removes that modality's tokens
pre-encoder (mirroring how `train-dropout` trains, but with a fixed on/off group
instead of Dirichlet-sampled proportions — see `FLOWERVLA.eval_modality_mask` in
`flower/models/flower.py`, `deterministic_keep_counts` in
`flower/models/networks/modality_dropout.py`). At least one modality must stay on.
This works on any checkpoint, not just `train-dropout` ones — it's how you'd test
whether *any* model degrades gracefully when a modality is missing.

**Precedence for a fixed-modality-ablation checkpoint** (see "Train-time modality ablation"
above): the trained combo is restored from the checkpoint and applied automatically, so
evaluating with no `eval_modalities` override reproduces the same subset it trained on. An
explicit `eval_modalities=` override still wins — e.g. to probe a `static,wrist`-trained
checkpoint with language re-added, or with static also dropped.

Example, dropping language on the dropout checkpoint:
```bash
./run.sh eval train_folder=$CKPT_DROP checkpoint=$CKPT_DROP/seed_42/saved_models/last.ckpt \
  eval_modalities.language=False
```
Compare modality combos — either within one checkpoint's own `result.csv` or across
two different checkpoints' files — with `scripts/compare_eval_csvs.py`, which takes
`--modalities-a`/`--modalities-b` (each a comma-separated subset of
`static,wrist,lang`, default all three):
```bash
python scripts/compare_eval_csvs.py $CKPT_DROP/eval_logs/last/orig_libero_10/result.csv \
                                     $CKPT_DROP/eval_logs/last/orig_libero_10/result.csv \
  --modalities-a static,wrist,lang --modalities-b lang
```

**Partial information decomposition (PID).** Success rates say how much each
modality's presence is worth; PID says *how* that information is shared between two
modalities — redundant (either alone would do), unique (only that one carries it), or
synergistic (only the pair together does). `scripts/pid_modality.py` runs this over a
`result.csv` that has all 7 modality combos evaluated (`static,wrist,lang` full through
each single modality): for each of the three modality pairs it treats availability of
the two as sources `X1, X2` and `success` as target `Y`, holding the third modality on
so all four `(x1, x2)` cells are populated, and decomposes `I(X1, X2 ; Y)` into
redundancy `R`, unique `U1`/`U2`, and synergy `S`.

Uses [`dit`](https://github.com/dit/dit)'s `PID_CCS` — Ince (2017)'s common-change-in-
surprisal measure, the same one implemented in MATLAB by
[`robince/partial-info-decomp`](https://github.com/robince/partial-info-decomp)
(`Iccs.m`); `dit` is a pure-Python reimplementation, avoiding a MATLAB/Octave
dependency (Octave lacks the Statistics Toolbox's `changem`, which `Iccs.m` calls).
Pinned to `dit==1.2.3` — the last release supporting Python 3.9 (this image's
interpreter); `scripts/pid_modality.py` shims two API changes that library predates
(networkx's `Graph.node` → `.nodes` rename, scipy's stricter `minimize(x0=...)` shape
check) without altering what gets computed. `--measure broja` swaps in the
non-negative Bertschinger et al. BROJA measure as a sanity cross-check, since `I_ccs`
can go slightly negative.

```bash
python scripts/pid_modality.py $CKPT_DROP/eval_logs/last/orig_libero_10/result.csv
```

**Perturbation categories** (pass as `task_category="<name>"`):

| Category | Description |
|---|---|
| `Background Textures` | Table/wall surface variations |
| `Camera Viewpoints` | Camera angle and position shifts |
| `Robot Initial States` | Different robot arm start configurations |
| `Language Instructions` | Paraphrased task descriptions |
| `Light Conditions` | Lighting intensity and direction changes |
| `Objects Layout` | Additional distractor objects in the scene |
| `Sensor Noise` | Simulated image noise |

Each category reports a per-category average success rate (`eval_lh/cat_<category>`).
The hypothesis is that the modality-dropout model degrades less, especially on `Language Instructions`
(where LIBERO-Plus shows standard VLAs regress to pure visuomotor control).

#### Common Issues

Sometimes this causes problems for the python env so just delete it:

```python
log.info(f"Using calvin_env with commit {get_git_commit_hash(Path(calvin_env.__file__))}.")
```
The path for this line is in the CALVIN env repo: https://github.com/mees/calvin_env/blob/797142c588c21e76717268b7b430958dbd13bf48/calvin_env/envs/play_table_env.py#L72

---

## Acknowledgements

This work is only possible because of the code from the following open-source projects and datasets. We thank all authors for their work:

#### CALVIN
Original:  [https://github.com/mees/calvin](https://github.com/mees/calvin)

License: [MIT](https://github.com/mees/calvin/blob/main/LICENSE)

#### LIBERO

Original: [https://github.com/Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

License: [https://github.com/Lifelong-Robot-Learning/LIBERO?tab=MIT-1-ov-file](https://github.com/Lifelong-Robot-Learning/LIBERO?tab=MIT-1-ov-file)

#### Mimictest 

Original: [mimictest](https://github.com/EDiRobotics/mimictest)
License: [license](https://github.com/EDiRobotics/mimictest?tab=Apache-2.0-1-ov-file)

#### HULC
Original: [https://github.com/lukashermann/hulc](https://github.com/lukashermann/hulc)

License: [MIT](https://github.com/lukashermann/hulc/blob/main/LICENSE)

#### FLOWER Pretraining Codebase

Original: [https://github.com/intuitive-robots/FLOWER_Diffusion_Policy](https://github.com/intuitive-robots/FLOWER_Diffusion_Policy)

License: [https://github.com/intuitive-robots/FLOWER_Diffusion_Policy/blob/main/LICENSE](https://github.com/intuitive-robots/FLOWER_Diffusion_Policy/blob/main/LICENSE) 


## Citation

If you found the code usefull, please cite our work: (arxiv coming very soon)

```bibtex
@inproceedings{
reuss2025flower,
title={{FLOWER}: Democratizing Generalist Robot Policies with Efficient Vision-Language-Flow Models},
author={Moritz Reuss and Hongyi Zhou and Marcel R{\"u}hle and {\"O}mer Erdin{\c{c}} Ya{\u{g}}murlu and Fabian Otto and Rudolf Lioutikov},
booktitle={9th Annual Conference on Robot Learning},
year={2025},
url={https://openreview.net/forum?id=JeppaebLRD}
}
```
