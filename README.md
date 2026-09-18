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
                                       # then auto-runs ./run.sh pipeline on the same GPUs (SKIP_PIPELINE=1 to skip)
./run.sh train-frozen                 # Ablation: frozen Florence VLM, action expert from random init (no auto-pipeline)
./run.sh train-dropout [bench]        # Fine-tune with Dirichlet modality-token dropout (default: libero_10); auto-pipeline too
./run.sh train-dropout-resume [bench] # Resume train-dropout from CKPT_PATH env var; auto-pipeline too
./run.sh eval                         # LIBERO-10 evaluation (~94.5% target)
./run.sh pipeline <train_run_dir> [...] # Full post-training eval (LIBERO[-Plus], modality sweep if dropout) + W&B upload
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

**Hostname** — every container runs with the host machine's own hostname (not a random podman
one), so W&B runs and log lines identify which machine they came from. Override with
`HOST_HOSTNAME` in `vars.env` if needed.

**Training batch size** — LIBERO training (`conf/config_libero.yaml`) defaults `batch_size`
(per-GPU dataloader batch) to `32`. Override with `./run.sh train libero_10 batch_size=16` if
that doesn't fit your GPUs.

**Batched eval** — `./run.sh eval` runs `eval_batch_size` episodes in parallel per task, each in
its own spawned MuJoCo subprocess (EGL offscreen rendering); model inference is batched across
all parallel episodes, and an episode stops being simulated (though its slot keeps drawing
inference, so the batch stays full width — see `env_start_method` below) the moment it finishes.
When `CUDA_VISIBLE_DEVICES` exposes multiple GPUs the 10 tasks are partitioned across them, each
GPU running its own batched eval independently. Tune `eval_batch_size` in `conf/eval_libero.yaml`
(default 10) to trade RAM vs. parallelism.

`env_start_method` (`conf/eval_libero.yaml`, default `spawn`) selects how those subprocesses are
created: `spawn` runs episodes truly in parallel; `dummy` steps them sequentially in the parent
process instead (useful as a slow-but-simple fallback, or for debugging). Fork-based subprocess
creation isn't offered — MuJoCo's EGL context creation fails once the parent process has touched
GL/EGL (as importing `robosuite` does), and even a GL-clean fork races when several children call
into the driver at nearly the same instant; `spawn`'s naturally staggered process startup avoids
both. See `flower/evaluation/libero_venv.py` for the full writeup, credit, and benchmarks.

`cross_task_batching` (`conf/eval_libero.yaml`, default `false`; `true` in `conf/eval_libero_plus.yaml`)
fills each batch with episodes drawn from different tasks instead of one task at a time. It's
required whenever `n_eval` is smaller than `eval_batch_size` — LIBERO-Plus's `n_eval=1` means the
per-task path can never fill a batch. Results still carry a `batching_mode` column (`per_task` /
`cross_task`), because batch width selects Florence-2/DiT GEMM kernels and reduction order — but
each episode's own flow-matching noise no longer depends on which mode produced it (see
Reproducibility below).

### Modality-token dropout (`train-dropout`)

`train-dropout` trains with per-sample, pre-encoder token removal across three modality groups:
static view, wrist view, and language. For each sample in a batch, a Dirichlet distribution
(α=1 per group) samples proportions that determine how many tokens from each group to keep;
the remaining tokens are physically removed from the sequence before it enters the Florence-2
encoder, providing a real compute saving (not masking). Because the kept proportion is sampled
independently per sample, every group is still run through its encoder (DaViT / the tokenizer)
for the whole batch — there's no fixed subset to skip at input time the way a `model.modalities`
or `eval_modalities` combo can (see below).

The position-correction step ensures every retained token's net positional embedding equals
its absolute position in the original sequence — removal is analytically equivalent to
attention-masking with positions preserved (verified by a unit test against the masking oracle).

The `<Flow>` prompt token is always kept. Rollout evaluation during training is disabled
(`rollout_lh_skip_epochs` defaults to `max_epochs` in `conf/config_libero.yaml`); evaluate a
finished run with `./run.sh pipeline` instead (see [Evaluation](#evaluation)). Hyperparameters
(model.yaml defaults):

| Flag | Default | Meaning |
|---|---|---|
| `modality_dropout` | `False` | Enable pre-encoder token removal |
| `modality_dropout_keep_fraction` | `0.5` | Fraction of total tokens to keep per sample |
| `modality_dropout_alphas` | `[1.0, 1.0, 1.0]` | Dirichlet α for [static, wrist, language] |
| `modality_dropout_proprio_keep_p` | `0.5` | Proprio keep probability (Bernoulli, not part of the Dirichlet budget above — see below) |

Proprioception isn't a token — see [Training with proprioception](#training-with-proprioception-modeluse_proprio)
— so it can't share the three groups' Dirichlet token budget: at typical batch sizes it would
be 1 "token" against ~500-1000 image/language tokens, and the Dirichlet allocation floors to 0
or 1 almost regardless of its sampled proportion, making it an effectively inert ablation. It
instead gets its own per-sample Bernoulli gate, resolved once per batch (not re-drawn per
sampling step) and applied by zeroing its conditioning vector — exactly reproducing the
`use_proprio=False` state when dropped.

### Train-time modality ablation (`train`)

Unlike `train-dropout` (one model exposed to *randomly varying* modality proportions every
step), a fixed modality-ablation model trains without dropout but only ever observes a fixed
subset of `{rgb_static, rgb_gripper, language, proprio}` — a withheld token modality (vision or
language) is skipped at the very input in every forward: its encoder (DaViT for a view, the
tokenizer for language) is never run, not just masked out afterwards. Proprio, not being a
token, is gated to zero instead. This is set with `model.modalities.*`
Hydra overrides on `./run.sh train`:

```bash
./run.sh train                                    # default: static+wrist+lang, no proprio
./run.sh train libero_10 model.modalities.language=False
./run.sh train libero_10 model.modalities.rgb_static=False model.modalities.rgb_gripper=False
./run.sh train libero_10 model.use_proprio=True    # add proprio to the default combo
```

| Key | Default | Meaning |
|---|---|---|
| `modalities.rgb_static` | `True` | Static (scene) view enabled |
| `modalities.rgb_gripper` | `True` | Wrist view enabled |
| `modalities.language` | `True` | Language instruction enabled |
| `modalities.proprio` | `True` | Proprioception enabled (requires `model.use_proprio=True`; see below) |

At least one of the three vision/language modalities must stay on (an all-`False` combo raises
`ValueError` before any data loads); `modalities.proprio=False` requires `use_proprio=True`
(nothing to withhold otherwise); and `model.modalities` is mutually exclusive with
`model.modality_dropout=True`.

The run's Hydra directory (and hence its wandb `group`/`name`/`id`, both derived from it — see
`setup_logger` in `flower/training_libero.py`) is suffixed with the combo's *present* modalities,
joined with `+`, using the same short names (`static`/`wrist`/`lang`/`proprio`) as
`scripts/compare_eval_csvs.py`. The default combo (static+wrist+lang, no proprio) keeps an empty
suffix so existing run directories are unaffected:

| Command | Run dir | wandb name |
|---|---|---|
| `./run.sh train` | `.../libero_10/<ts>` | `libero_10/<ts>` (unchanged) |
| `./run.sh train libero_10 model.modalities.language=False` | `.../libero_10_static+wrist/<ts>` | `libero_10_static+wrist/<ts>` |
| `./run.sh train libero_10 model.modalities.rgb_static=False model.modalities.rgb_gripper=False` | `.../libero_10_lang/<ts>` | `libero_10_lang/<ts>` |
| `./run.sh train libero_10 model.use_proprio=True` | `.../libero_10_static+wrist+lang+proprio/<ts>` | `libero_10_static+wrist+lang+proprio/<ts>` |

The trained combo is part of the model's saved hyperparameters, so it's restored automatically
by `load_mode_from_safetensor`/checkpoint loading — evaluating one of these checkpoints with no
`eval_modalities` override re-applies the same subset it was trained on (see the precedence note
in "Modality-ablation eval" below).

### Training with proprioception (`model.use_proprio`)

By default FLOWER conditions only on images and language. Setting `model.use_proprio=True`
additionally feeds the robot's proprioceptive state — joint positions concatenated with
gripper state (`robot_obs`, dim `proprio_dims`: 9 for LIBERO, 7 for CALVIN) — through a
dedicated MLP encoder and sums it into the DiT's global conditioning. Works with both plain
training and modality-token dropout:

```bash
./run.sh train libero_10 model.use_proprio=True
./run.sh train-dropout libero_10 model.use_proprio=True
```

The same `robot_obs` signal is fed during rollout evaluation (`RolloutLibero`, run periodically
during training, and `flower_eval_libero.py`), so a proprio-trained checkpoint is evaluated
under the same conditioning it was trained with.

This MLP is trained from scratch for LIBERO/CALVIN's single-arm action space. The pretrained
`flower_vla_pret` checkpoint only ships real proprio weights for its bimanual action space
(trained on bimanual ALOHA data with that dataset's own normalization statistics), which
LIBERO/CALVIN don't use — there's no pretrained single-arm proprio signal to warm-start from.

Once `use_proprio=True`, proprioception can also be ablated like the other three modalities —
`model.modalities.proprio=False` (fixed, train+eval), `model.modality_dropout_proprio_keep_p`
(training-time Bernoulli dropout), or `eval_modalities.proprio=False` (eval-only) — see the
sections above and "Modality-ablation eval" below.

### Action-expert capacity (`model.n_layers`, `extra_layer_*`, `dit_learning_rate`)

The pretrained `flower_vla_pret` checkpoint only covers **12** DiT blocks
(`model.pretrained_dit_layers`), but the default config uses `model.n_layers=18` — the
remaining 6 blocks (113M params, +18.9M params/layer) are extra and don't come from the
checkpoint. How they're placed, initialized, and trained is controlled by:

| Knob | Values | Effect |
|---|---|---|
| `model.n_layers` | int ≥ `pretrained_dit_layers` | Total DiT depth. Anything beyond `pretrained_dit_layers` is "extra". |
| `model.extra_layer_placement` | `append` (default), `interleave` | `append`: extras go after all pretrained blocks. `interleave`: extras are spread evenly among them (depth up-scaling style). |
| `model.extra_layer_init` | `random` (default), `zero`, `copy` | `random`: default PyTorch init. `zero`: block is the exact identity at step 0 (self-attn/cross-attn/MLP output projections zeroed), so it can't scramble the pretrained blocks' output before training nudges it away from identity. `copy`: duplicates a pretrained block's weights instead of an extra slot (LLaMA-Pro/SOLAR-style depth up-scaling) — requires `extra_layer_lora_dim`/`extra_layer_mlp_hidden_dim` to match the base width. |
| `model.extra_layer_lora_dim`, `model.extra_layer_mlp_hidden_dim` | int or `null` | Give the extra blocks their own AdaLN/MLP width instead of the pretrained blocks' (`model.lora_dim`/`model.mlp_hidden_dim`). `null` (default) matches the base width. |
| `model.optimizer.dit_learning_rate`, `model.optimizer.new_layer_learning_rate` | float or `null` | Separate learning rates for the pretrained action expert and the extra blocks, vs. `model.optimizer.learning_rate` for the VLM. `null` (default) falls back to `learning_rate`/`dit_learning_rate` respectively, i.e. today's single-LR behavior. Pre-training itself used `1e-4` for the whole DiT vs `2e-5` for the VLM. |

The 12 pretrained blocks' RoPE `cos`/`sin` tables are persistent buffers, so they always
load the checkpoint's own `rope_theta` (1000.0) regardless of `model.rope_theta` — that
config value only reaches the extra blocks. `model.rope_theta` defaults to `1000.0` so
extras match the pretrained blocks instead of silently diverging from them.

`model.dit_dim`/`model.n_heads` must stay at `1024`/`16` — changing either discards all
226.6M params of pretrained DiT weight (the load log always reports how many blocks
loaded vs. stayed fresh, so this is never silent).

The same loader also reloads a fully-trained/finetuned checkpoint (e.g. at eval time),
which already has `model.n_layers` blocks rather than the pretrained base's 12 — it tells
the two apart by the checkpoint's DiT block count, so `model.n_layers` can differ freely
between runs without breaking eval of older checkpoints, and a trained checkpoint's extra
blocks are loaded as-is rather than remapped or zero-inited again.

```bash
# Depth up-scaling: duplicate the last 6 pretrained blocks instead of random-initializing
# 6 new ones, spread evenly through the stack.
./run.sh train libero_10 model.extra_layer_init=copy model.extra_layer_placement=interleave

# Give the action expert its own (higher) learning rate, matching pre-training's DiT LR.
./run.sh train libero_10 model.optimizer.dit_learning_rate=1e-4

# Add real capacity (6 extra blocks) that starts as a no-op and trains faster than the
# pretrained ones.
./run.sh train libero_10 model.n_layers=24 model.extra_layer_init=zero \
    model.optimizer.dit_learning_rate=1e-4 model.optimizer.new_layer_learning_rate=2e-4
```

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

Note that during CALVIN training the full CALVIN eval will be called after _rollout_lh_skip_epochs_ and then every _callbacks.rollout_lh.rollout_freq_*1k training steps. LIBERO training no longer runs rollout eval at all (`rollout_lh_skip_epochs` is set to `max_epochs`) — evaluate a finished LIBERO run with `./run.sh pipeline` instead (see [Pipeline: automated post-training evaluation](#pipeline-automated-post-training-evaluation)). Check out the training config for adopting the parameters.

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
`CKPT_DROP=/saves/train_logs/libero_10_dropout/<run>/seed_<seed>/saved_models/last.ckpt`
(dropout model). `./run.sh pipeline` (below) runs both of these evaluations for you, plus
the full modality sweep for a dropout run — use the manual commands here for one-off spot
checks.

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
> the total env create/destroy cycle count. With `cross_task_batching: true`
> (the `eval_libero_plus.yaml` default) that count is
> `ceil(n_tasks × n_eval / eval_batch_size)` — for 419 Camera-Viewpoint tasks at
> `n_eval=1` that's ~42 instantiations, not one per task, since batches now span
> tasks instead of collapsing to size 1. (Falling back to `cross_task_batching: false`
> reintroduces the old `n_tasks × ceil(n_eval / eval_batch_size)` count — ~419
> instantiations here, and ~20,950 at `n_eval=50` — which can OOM under the
> container's `MEM_LIMIT` cap in `vars.env`.)
> If a full 2519-task run still approaches the cap, split it by category (as above)
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
`train_folder` isn't writable (e.g. a shared read-only checkpoint) — this is exactly how
`./run.sh pipeline`'s modality-withheld LIBERO-Plus evals (see
[Pipeline](#pipeline-automated-post-training-evaluation) below) each land in their own
`plus_<benchmark_name>_no_<modality>/result.csv` instead of the default
`<variant>_<benchmark_name>` layout. Re-running (e.g. one `task_category` at a time) **merges**
into the existing file — a rerun of the same task/episode/checkpoint replaces that row in
place, everything else is kept — so running all 7 Plus categories separately accumulates
into one complete `result.csv`. On multi-GPU, each worker writes `result_rank<i>.csv`
into the same directory; the master merges them into `result.csv` and deletes the shards.

Columns identify every variable that determines the episode: `suite`, `task_idx`,
`task_name` (LIBERO-Plus encodes the full perturbation — camera angle, robot-init id,
noise id, etc. — in this name; no separate columns), `task_category` /
`difficulty_level` (LIBERO-Plus only), `language` (the exact instruction fed to the
model), `init_state_idx`, `rollout_seed`, `num_sampling_steps`, `multistep`,
`eval_batch_size`, `batching_mode` (`per_task` or `cross_task`, see `cross_task_batching`
above), and the four modality flags `use_rgb_static`, `use_rgb_gripper`,
`use_language`, `use_proprio`. These four are part of the row's merge key (see below), so
evaluating the same task/episode/checkpoint under a different modality combo adds a new row
instead of overwriting the previous combo's result. Result columns are `success`
(0/1) and `steps_taken`. (`use_proprio` postdates the other three; rows in a `result.csv`
written before it existed simply lack the column — `eval_records.py`/`compare_eval_csvs.py`
read that as proprio-absent, which is factually correct for those older runs.)

**`language` is the instruction the model was fine-tuned on, not LIBERO-Plus's
perturbation-suffixed one.** Upstream LIBERO derives a task's prompt from its BDDL
*filename* (`libero.libero.benchmark.grab_language_from_filename`), and that is what
training uses — the LIBERO datamodule prompts with `benchmark.get_task(i).language`,
under `LIBERO_VARIANT=orig`. LIBERO-Plus inherits the same derivation, so its
`task.language` leaks the perturbation id into the prompt for every category except
Language Instructions — e.g. `"turn on the stove and put the moka pot on it table 1"`
or `"... view 0 0 100 2 6 initstate 0"` (~85% of Plus tasks). `flower_eval_libero.py`'s
`task_language()` instead maps the Plus task back to the original LIBERO task it was
generated from — longest-prefix match against the suite's original task names, read
from `<init_states>/<suite>/*.pruned_init` (`flower/evaluation/libero_tasks.py`; 0
unmatched out of 2402/2518/2591/2519 tasks) — and re-derives the instruction from
*that* name. Language Instructions is exempt: its reworded instruction is the
independent variable, and upstream already returns the LLM rewrite from the task's own
BDDL for a `_language_` name, so it is passed through untouched. (`"_language_"` in a
task name matches the `Language Instructions` category exactly across every suite, so
no `task_category` lookup is involved and a missing or empty `task_category` cannot
affect the prompt.) `LIBERO_VARIANT=orig` is a pure passthrough.

Reading the BDDL's own `(:language ...)` field — what this eval did previously — is
*not* equivalent: upstream LIBERO's filename-derived text and its BDDL text disagree
for 10/10 `libero_spatial`, 10/10 `libero_object` and 5/10 `libero_goal` tasks (`"pick
up the black bowl between the plate and the ramekin …"` vs `"pick the akita black bowl
between the plate and the ramekin …"`), so the BDDL text is off the model's training
distribution on those suites. They agree byte-for-byte on all 10 `libero_10` tasks,
which is why **this is an exact no-op for `libero_10`** (0 of 2519 prompts change, so
no `libero_10` result needs re-evaluating); it changes prompts on
`libero_spatial`/`libero_object`/`libero_goal` instead. One consequence: LIBERO-Plus
numbers produced here are not comparable to the upstream leaderboard, which evaluates
with the suffix-contaminated prompts.

`scripts/perturbation_sr.py <result.csv>` prints the success rate per `task_category`
(plus an overall line) straight from this file — no `task_classification.json` lookup
needed, since the category is already a column.

`scripts/severity_sr.py <result.csv>` prints success rate by *physical* perturbation
severity within each category (e.g. camera rotation in degrees, robot joint-space
offset in radians, corruption type + severity 1-10), recovered by parsing `task_name`
(and, for Light Conditions, the corresponding scene XML under
`LIBERO-plus/libero/libero/assets/scenes/`; pass `--libero-plus-root` if the submodule
isn't at the default location) — see `scripts/perturbation_severity.py`. This is printed
alongside `difficulty_level` rather than instead of it: `difficulty_level` is an
upstream per-task annotation, not a measurement, and disagrees with the physical
magnitude roughly two-thirds of the time for Robot Initial States, and is non-monotone
for Objects Layout (it pools two physically different sub-mechanisms — distractor count
and target displacement — into one label). Language Instructions and Background
Textures have no severity axis (an unordered rewrite id and a texture identity,
respectively) and get a categorical sub-type breakdown instead.

Pass `--csv OUT.csv` to also write the full breakdown as a machine-readable CSV, one row
per `(category, axis, bin)`: `axis` is `difficulty_level`, `severity`, `subtype`
(Background Textures only), or `total` (one `bin=ALL` row per category, the category-level
counterpart of the paired baseline below); `successes`/`n`/`success_rate`/`ci_low`/`ci_high`
are the same numbers the printed table shows; `note` carries the "no severity axis" caveat
on every row of a category that has one (Language Instructions, Background Textures,
unclassified rows); and a category's rows that didn't match the expected `task_name`
pattern get their own explicit `<unparsed>` bin (under `axis=severity`) instead of being
silently dropped, so a category's `n` reconciles across both axes. `./run.sh pipeline`
writes this next to every LIBERO-Plus `result.csv` as `severity_sr.csv` and uploads it
alongside `result.csv`/`pid_modality.txt` (see
[Pipeline](#pipeline-automated-post-training-evaluation) below).

**Known limitation: Robot Initial States currently has no physical effect.** LIBERO-Plus
implements this one category's perturbation by substituting the robot's Python class
(`Panda` → `MountedPanda{N}`/`OnTheGroundPanda{N}`, `N` in 1–500 — see
`LIBERO-plus/libero/libero/envs/robots/new_init.py`), which only changes `init_qpos`,
applied by `robosuite`'s `robot.reset()`. `flower_eval_libero.py` then calls
`env.set_init_state()` with the array `get_task_init_states()` returns for this
category — the *unperturbed* base task's own recorded state — which overwrites that
qpos with a full MuJoCo-state restore. Verified against a real checkpoint (not just by
reading code): re-running the exact same checkpoint on the exact same task gives a
*different* per-severity-band success pattern than the original eval, with no
reproducible trend, consistent with every episode running from the same unperturbed
state rather than a real physical perturbation. This is a LIBERO-Plus upstream gap, not
something specific to this repo's usage — LIBERO-Plus's own reference script
(`LIBERO-plus/benchmark_scripts/render_single_task.py`) calls `set_init_state()` the
same unconditional way. It is left unfixed here deliberately, to keep results
comparable with other work evaluating on this benchmark as shipped;
`scripts/severity_sr.py`'s `ROBOT_INITSTATE_NOTE` documents it in code, and its printed
report and `--csv` output both carry the same warning on every Robot Initial States row.

**Paired original baseline.** Pass `--orig-csv <orig_result.csv>` to add an init-state-
matched LIBERO original baseline to every row (printed as an `orig_sr`/`orig 95% CI`/
`orig_n` column, or `orig_successes`/`orig_n`/`orig_success_rate`/`orig_ci_low`/
`orig_ci_high` in the CSV). LIBERO-Plus's own `get_task_init_states`
(`LIBERO-plus/libero/libero/benchmark/__init__.py`) strips the perturbation suffix off a
task_name and loads the *original* task's init file, so every LIBERO-Plus episode
(Objects Layout excepted) runs original init state 0 of its base task, under the same
modality combo. The baseline for a group is that set of (modality combo, base task) orig
episodes, deduplicated by task — `orig_n` is the number of distinct base tasks the group
covers, **not** the number of LIBERO-Plus rows in it, so it is small by construction (a
severity bin can cover as few as 2 of the suite's 10 tasks) and the Wilson CI
correspondingly wide; that's expected, not a bug. Objects Layout's init states are
generated fresh into `libero_newobj/`, not derived from the original task's own init
file, so its baseline is the task's *nominal* layout rather than the same init state —
still the meaningful contrast for a layout perturbation, but its rows carry an explicit
note saying so. Rows without a matching original episode (no orig CSV given, or no base
task/init-state-0/modality-combo match) leave the baseline columns empty rather than 0.
`./run.sh pipeline`'s `severity_sr.csv` always includes these columns, populated whenever
the run's `libero_orig.csv` exists.

A LIBERO-Plus row's own `init_state_idx` reads `0` for every row, always — expected, not
a bug. The trailing number in its `task_name`/`init_states_file` (e.g. `_table_5`,
`_initstate_50`) is the perturbation's own parameter id (view sample, robot qpos-offset
sample, noise instance, texture/light id), applied by `env_wrapper.py` at
env-construction time from the bddl filename — it is not an index into the init-states
array, and `n_eval=1` means array index 0 is all that's ever loaded regardless.

**Reproducibility:** each episode's flow-matching noise is drawn from its own
generator, seeded by `rollout_seed(seed, base_task_name, episode_idx)` and recorded per
row as `rollout_seed` — so an episode's noise is independent of `eval_batch_size`, of
its slot in the batch, of which other episodes shared that batch, of `batching_mode`,
of the `task_category` filter, and of GPU shard assignment.
`flower.evaluation.eval_records.base_task_name` strips LIBERO-Plus's perturbation
suffix, so every Plus variant of a task and that task's original-LIBERO episode 0 — the
row `scripts/severity_sr.py` pairs it against — share one noise stream (common random
numbers: a paired comparison, which is the point). Generators are CPU-side, so the draw
is identical across GPU models and counts. This does **not** make the two batching
modes bit-identical: batch width still selects Florence-2/DiT GEMM kernels and
reduction order, so only `eval_batch_size=1` reproduces exactly across modes. Results
collected before this change used a per-batch seed with no task-identity component and
are not reproducible against the current code.
**Known exception:** LIBERO-Plus Sensor-Noise tasks using motion_blur, fog, or
glass_blur corruption (noise ids 1-10, 31-40, 41-50) call unseeded `np.random` on every
rendered frame inside the vendored env, so those specific episodes are not bit-exact
reproducible even with a matching seed; gaussian_blur/zoom_blur (ids 11-30) are unaffected.

**Modality-ablation eval.** `eval_modalities` in `conf/eval_libero.yaml` /
`conf/eval_libero_plus.yaml` is functional: setting any of `rgb_static`,
`rgb_gripper`, `language` to `False` skips that modality at the input — its
encoder (DaViT for a view, the tokenizer for language) is never run for the withheld
group, not just masked out afterwards (a fixed on/off group per `FLOWERVLA.eval_modality_mask`
in `flower/models/flower.py`, vs. `train-dropout`'s Dirichlet-sampled proportions, which
still runs every group's encoder — see `compact_layout` /
`deterministic_keep_counts` in `flower/models/networks/modality_dropout.py`). Retained
tokens still land at exactly the absolute position they'd carry in a full, non-skipped
sequence, so results are unaffected — only the withheld modality's compute is saved.
`proprio` works the same way but isn't
a token — it zeros the proprio conditioning vector instead (`FLOWERVLA.eval_proprio_mask`)
and requires a `use_proprio=True` checkpoint. At least one of the three vision/language
modalities must stay on. This works on any checkpoint, not just `train-dropout` ones —
it's how you'd test whether *any* model degrades gracefully when a modality is missing.

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
`static,wrist,lang,proprio`, default `static,wrist,lang`):
```bash
python scripts/compare_eval_csvs.py $CKPT_DROP/eval_logs/last/orig_libero_10/result.csv \
                                     $CKPT_DROP/eval_logs/last/orig_libero_10/result.csv \
  --modalities-a static,wrist,lang --modalities-b lang
```

**Partial information decomposition (PID).** Success rates say how much each
modality's presence is worth; PID says *how* that information is shared between two
modalities — redundant (either alone would do), unique (only that one carries it), or
synergistic (only the pair together does). `scripts/pid_modality.py` runs this over a
`result.csv` that has all 15 modality combos evaluated (`static,wrist,lang,proprio` full
through each single modality — a full sweep on a `use_proprio=True` checkpoint): for each
of the six modality pairs it treats availability of the two as sources `X1, X2` and
`success` as target `Y`, holding the other two modalities on so all four `(x1, x2)` cells
are populated, and decomposes `I(X1, X2 ; Y)` into redundancy `R`, unique `U1`/`U2`, and
synergy `S`.

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

The script first inspects the CSV to see which modalities it actually exercises —
a column that's absent (e.g. a `model.use_proprio=False` run's `result.csv` has no
`use_proprio` column) or present but never `1` is dropped from the report entirely,
rather than producing pairs that can only ever read "not evaluated". A header line
states what was detected and what was dropped, and why:

```
modalities: static, wrist, lang (varying) | dropped: proprio (column absent)
```

so a `use_proprio=False` run yields a 3-pair report instead of six empty ones.

`--marginalize` adds, for each pair, a second row that pools episodes across every
held-modality configuration present in the file instead of filtering to held-all-on —
marginalizing the held modalities out of the `(x1, x2, y)` joint. Both rows print
side by side, labelled `held ...` vs. `marginalized`, each with the episode count `n`
backing it:

```
pair             held               I(X1,X2;Y)        R       U1       U2        S      n
static+wrist     lang+proprio           0.1007   0.0225   0.0346   0.0155   0.0281     80
static+wrist     marginalized           0.0771   0.0159   0.0394   0.0033   0.0186    320
```

It also prints a second table: success rate per modality-presence combination found in the
CSV, restricted to the detected modalities (`static wrist lang [proprio] -> success_rate,
n`), rows sorted by the presence flags read as a binary number, descending — the all-on
combo first, all-off last.

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

### Pipeline: automated post-training evaluation

`./run.sh pipeline <train_run_dir> [hydra_overrides...]` automates everything above for one
completed training run — deciding which evaluations it needs, running them, and uploading
the results — instead of invoking `eval`/`eval-plus`/`compare_eval_csvs.py`/`pid_modality.py`
by hand:

```bash
./run.sh pipeline /saves/train_logs/libero_10_dropout/2026-09-08_10-00-00
```

**`train`, `train-dropout`, and `train-dropout-resume` run this automatically** once training
finishes, on the same GPUs training used (`CUDA_VISIBLE_DEVICES`, exported by `run.sh` before
the training container starts so it's available to the pipeline call chained after it) — no
manual invocation needed for the common case. It only runs after a successful training exit
(a crashed/OOM'd run doesn't trigger it). Set `SKIP_PIPELINE=1` to opt out, e.g. when queuing
several training runs back to back without waiting on each one's eval:

```bash
SKIP_PIPELINE=1 ./run.sh train libero_90
./run.sh pipeline /saves/train_logs/libero_90/<run>   # run it yourself, later
```

`train-frozen` is the one exception — it doesn't auto-run the pipeline (its run directory is
resolved inside the container, not known to `run.sh`) — invoke `./run.sh pipeline <dir>`
manually for it.

It reads `<train_run_dir>/.hydra/config.yaml` to branch:

- **Regular training** (`model.modality_dropout=False`): one full-modality LIBERO eval, then
  one full-modality LIBERO-Plus eval, then one LIBERO-Plus eval per withheld modality (below).
- **`train-dropout` runs** (`model.modality_dropout=True`): every non-empty combination of
  `{rgb_static, rgb_gripper, language}` on LIBERO — 7 combos, or 14 if the run also used
  `model.use_proprio=True` (each token combo crossed with proprio on/off; "proprio only" is
  never evaluated, since the model requires at least one vision/language modality) — then one
  full-modality LIBERO-Plus eval, then one LIBERO-Plus eval per withheld modality (below).

**LIBERO-Plus with one modality withheld at inference.** In addition to the full-modality
LIBERO-Plus eval, the pipeline plans 4 more, in this order — each with exactly one of
`rgb_gripper` (3rd-person/wrist), `rgb_static` (1st-person/agentview), `proprio`, or
`language` off and the other three on — using the same `eval_modalities.*` mechanism as
[Modality-ablation eval](#evaluation-results-csv) above. Each writes to its own
`plus_<benchmark>_no_<modality>/result.csv` (`no_wrist`/`no_static`/`no_proprio`/`no_lang`) via
an explicit `csv_dir=` override — never merged into `plus_<benchmark>/result.csv`, since
`scripts/perturbation_sr.py`/`scripts/severity_sr.py` both assume a LIBERO-Plus `result.csv`
holds exactly one modality combo. `benchmark_name=` on these lines stays the real LIBERO
benchmark-registry key (e.g. `libero_10`); the modality suffix can only travel via `csv_dir=`.
`no_proprio` is never planned for a `model.use_proprio=False` checkpoint — `EvaluateLibero`
raises on `eval_modalities.proprio=False` there (there is no proprioception to withhold), so
planning it would crash the eval container, not just waste a duplicate run.

Since this makes the LIBERO-Plus portion of the pipeline take roughly 5x as long (each of the
4 extra evals covers the same ~2519 LIBERO-Plus tasks as the full-modality one),
`PIPELINE_SKIP_MODALITY_OFF=1` skips planning all 4 — e.g. to keep `train`/`train-dropout`'s
auto-chained pipeline call from ballooning while queuing several trainings back to back:

```bash
PIPELINE_SKIP_MODALITY_OFF=1 ./run.sh pipeline /saves/train_logs/libero_10_dropout/2026-09-08_10-00-00
```

Catch the skipped evals up later the same way as any other suite, via `PIPELINE_REEVAL_SUITES`
below (e.g. `PIPELINE_REEVAL_SUITES=plus_libero_10_no_static,plus_libero_10_no_lang`).

Every combo appends into the same `result.csv` (see [Evaluation results (CSV)](#evaluation-results-csv)
above), so a dropout run's file ends up exactly the "full sweep" `scripts/pid_modality.py`
needs. Trailing Hydra overrides are appended, last, to every eval it launches — e.g. to
resize the eval batch or shrink `n_eval` for a quick check:

```bash
./run.sh pipeline /saves/train_logs/libero_10/2026-09-08_10-00-00 eval_batch_size=32 n_eval=5
```

`PIPELINE_RESUME=1` skips any combo whose modality flags already have rows in the target
`result.csv`, so a sweep interrupted partway through restarts cheaply instead of
re-evaluating everything:

```bash
PIPELINE_RESUME=1 ./run.sh pipeline /saves/train_logs/libero_10_dropout/2026-09-08_10-00-00
```

On a run whose `result.csv` files already cover every combo (including all 4
modality-off suites — resume covers those exactly like `orig_<bench>`/`plus_<bench>`),
`PIPELINE_RESUME=1` plans nothing and goes straight to the upload step below — the way to
regenerate and re-upload `pid_modality.txt`/`severity_sr.csv` (e.g. after a fix to either
script) with no GPU work at all.

`PIPELINE_REEVAL=1` does the opposite: instead of merging into the existing `result.csv`
(which keeps every row the new run doesn't overwrite — stale rows from a category or
modality combo no longer evaluated), it backs the old file up to `results_<mtime>.csv`
(named for the old file's own modification time) and re-plans that suite from empty.
`PIPELINE_REEVAL_SUITES` narrows *which* suites — a comma-separated list of result
directory names, one of `orig_<bench>`, `plus_<bench>`, or `plus_<bench>_no_<modality>`
(`no_wrist`/`no_static`/`no_proprio`/`no_lang`) — leaving the rest untouched and
unevaluated; omitted, `PIPELINE_REEVAL=1` re-evaluates every suite for the run's
benchmark. An invalid suite name (wrong benchmark, typo) aborts before anything runs; a
suite name that's merely inapplicable to this run (`plus_<bench>_no_proprio` on a
`model.use_proprio=False` checkpoint) is skipped with a note on stderr instead — a real,
just-not-here suite name is a milder case than a typo.

```bash
# Re-evaluate everything, keeping the old numbers as a timestamped backup
PIPELINE_REEVAL=1 ./run.sh pipeline /saves/train_logs/libero_10_dropout/2026-09-08_10-00-00

# Re-evaluate only LIBERO-Plus, e.g. after a prompt/scoring fix that only affects it --
# orig_libero_10/result.csv is left alone
PIPELINE_REEVAL=1 PIPELINE_REEVAL_SUITES=plus_libero_10 \
    ./run.sh pipeline /saves/train_logs/libero_10_dropout/2026-09-08_10-00-00

# Re-evaluate only the language-withheld LIBERO-Plus eval
PIPELINE_REEVAL=1 PIPELINE_REEVAL_SUITES=plus_libero_10_no_lang \
    ./run.sh pipeline /saves/train_logs/libero_10_dropout/2026-09-08_10-00-00
```

Set these on the `pipeline` invocation itself, not in `vars.env` — `train`/`train-dropout`/
`train-dropout-resume`'s auto-chained pipeline call explicitly clears both first, since
re-evaluating a run with no prior results is meaningless and a stale suite name for a
different benchmark would otherwise abort a freshly finished training run. `PIPELINE_RESUME`
has no effect on a suite selected for re-evaluation (its `result.csv` is already gone by the
time resume would check it) but still applies normally to any suite *not* selected. The
`results_<mtime>.csv` backups stay on disk next to the fresh file — they are never uploaded —
and on W&B, `upload` simply logs a new `evaluation` artifact version; `./run.sh analyze`
already reads the newest one, and older versions remain as W&B-side history.

Once every eval finishes, it runs `scripts/pid_modality.py` over the LIBERO `result.csv` and
`scripts/severity_sr.py` over the LIBERO-Plus `result.csv` and over each present
modality-off `result.csv` (writing a `severity_sr.csv` next to each), and uploads all of
that as a W&B artifact (`eval-<run_id>`, type `evaluation`) attached to the *training*
run — same project/entity/id `setup_logger` gave it in `flower/training_libero.py`,
reconstructed from the run directory name, so the artifact lands next to the training
curves without a separate W&B run being created. Every member is optional and simply
omitted when its source `result.csv` doesn't exist yet (e.g. a suite not evaluated, or
`no_proprio` on a non-proprio run): `libero_orig.csv`, `libero_plus.csv`,
`pid_modality.txt`, `severity_sr.csv`, plus, per present modality-off suite,
`libero_plus_no_<modality>.csv` and `severity_sr_no_<modality>.csv` — up to 12 members
total. A missing/broken LIBERO-Plus assets checkout only drops the affected
`severity_sr*.csv` member(s) (with a warning) rather than failing the whole upload.
Prerequisite: `./run.sh download-plus` (once, for LIBERO-Plus assets, and specifically for
`severity_sr.csv`'s Light Conditions rows).

### Cross-run analysis from W&B

`scripts/analyze_wandb.py` (`./run.sh analyze`) reads that artifact back and reports the
PID decomposition (`scripts/pid_modality.py`), the per-perturbation-category success rate
(`scripts/perturbation_sr.py`), the success rate by physical perturbation severity and
by upstream `difficulty_level` (`scripts/severity_sr.py`, in two separate tables, same
split as that script's own printed report), and the same per-category success rate for
each present modality-withheld LIBERO-Plus eval — either for one or more named runs, or
averaged with min/max across every run matching a W&B config filter:

```bash
./run.sh analyze run libero_10_dropout_2026-09-09_13-56-20

./run.sh analyze filter --filters \
    '{"config.modality_dropout": true, "config.modality_dropout_proprio_keep_p": 0.5}'

./run.sh analyze filter --modalities rgb_static,rgb_gripper,language,proprio

# Also show the severity / difficulty_level breakdown for each withheld-modality eval,
# not just its per-category success rate
./run.sh analyze run libero_10_dropout_2026-09-09_13-56-20 --modality-off-detail full
```

Requires `WANDB_API_KEY` (vars.env or shell env). Filter keys are matched against the
run's **W&B config**, which is populated only by `FLOWERVLA.save_hyperparameters()`
(`flower/models/flower.py`) — its `__init__` argument names, flat (`modality_dropout`,
not `model.modality_dropout`); top-level Hydra keys like `seed` or `libero_benchmark`
aren't in it. `--entity`/`--project` default to `conf/config_libero.yaml`'s
`logger.entity`/`logger.project`, currently `multimodal_florence`. New training runs log
there; `./run.sh pipeline`'s eval-artifact upload targets whatever project the *training
run's own* saved hydra config recorded, so evals of runs from before this project was
renamed still land in the old `multimodal_policies` project next to their training
curves — pass `--project multimodal_policies` to `./run.sh analyze` to see those.

The severity/difficulty_level tables are **recomputed from the artifact's
`libero_plus.csv`**, not read from its `severity_sr.csv` member — same as the PID table is
recomputed from `libero_orig.csv` rather than read from `pid_modality.txt` — so every
artifact already uploaded works immediately, including ones logged before
`severity_sr.csv` existed. Recomputing needs the LIBERO-Plus assets locally for Light
Conditions rows (`./run.sh download-plus`); pass `--libero-plus-root` if the submodule
isn't at the default location. A run whose severity computation fails (e.g. assets not
downloaded) is reported via the `WARNING:` block rather than aborting the whole report,
same as a run missing a required CSV member.

Whenever at least one matched run's severity breakdown carries an init-state-matched
LIBERO original baseline (see `scripts/severity_sr.py`'s "Paired original baseline"
above — the artifact's own `libero_orig.csv` is used automatically, no extra flag),
three more tables print alongside the per-perturbation, physical-severity, and
difficulty_level ones: the same three splits, each with a `plus`/`orig` pair of
mean-min-max columns instead of one. Because `orig_n` (the number of distinct base tasks
a group covers) can differ across runs whose LIBERO-Plus rows sampled different tasks per
bin, the `plus_n`/`orig_n` columns render as a single number when every run agrees, or
`lo-hi` when they don't — rather than an average of counts, which wouldn't mean anything.
A run whose artifact lacks `libero_orig.csv` simply contributes nothing to these three
tables; if none do, they're omitted entirely and only the unpaired tables print.

**Modality-absent LIBERO-Plus reports.** For each of the (up to 4) modality-withheld
LIBERO-Plus evals present in a run's artifact, single-run mode prints a
`=== <run_id>: LIBERO-Plus (no_<modality>) ===` block with the same per-category success
rate table as the full-modality report; pass `--modality-off-detail full` to also add its
severity/difficulty_level breakdown. A variant that's applicable to the run but missing
from the artifact prints a `(no no_<modality> eval)` placeholder; `no_proprio` on a
`model.use_proprio=False` checkpoint prints nothing at all for that run — it's not
applicable, not incomplete. Aggregate (`filter`) mode adds one combined table, "LIBERO-Plus
with one modality withheld", with rows labelled `<category> [no_<modality>]` (`OVERALL`
rows last) so the same category's four ablations sit next to each other — plus the same
two severity-axis tables under `--modality-off-detail full` — omitted entirely when no
matched run has any modality-off data. Two caveats worth knowing when reading these
numbers: withholding `language` makes the Language Instructions perturbation category
degenerate (every LLM rewrite of an instruction becomes the same no-language input, so
that category stops measuring rewrite robustness); and the paired plus/orig baseline
(above) is deliberately *not* shown for modality-off evals — a non-dropout run's
`libero_orig.csv` never has a modality-off combo, so every such baseline cell would be
`orig_n=0`/`nan`.

`--filters` can't select runs by *trained* modality set: W&B stores `modalities` as a
Python repr string (a `DictConfig` passed through `str()` on its way into the config), not
a nested value, so `config.modalities.rgb_static` can't be matched, and the key set itself
drifted (older runs' `modalities` has 3 keys, not 4, with no `proprio` entry at all). Use
`--modalities <comma-separated names>` instead — it selects runs client-side by the
*effective* set of modalities the model was trained to observe (every named modality on,
every other one off), correctly accounting for both of the above and for `use_proprio`
gating `modalities.proprio` (a model with `use_proprio=False` never receives proprio
regardless of what `modalities` says).

Before printing results, it always prints the run ID(s) or filter used, and a `WARNING:`
block for anything that would otherwise silently skew the report: a matched run with no
evaluation artifact, an artifact missing `libero_orig.csv`/`libero_plus.csv` or an
applicable-but-absent `libero_plus_no_<modality>.csv` (never `no_proprio` on a
`model.use_proprio=False` run — that one simply isn't applicable), a run whose severity
breakdown (full-modality or modality-off) couldn't be computed, or runs that don't share
the same evaluated modality combos, episodes, LIBERO-Plus categories, or severity bins.
Zero matched runs is a hard error; everything else is reported and still aggregated, with
a `runs` column on every table so a cell backed by fewer runs than the header claims is
visible rather than hidden.

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
