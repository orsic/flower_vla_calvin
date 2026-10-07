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


## MimicGen

[MimicGen](https://mimicgen.github.io/) generates large demonstration datasets for
robosuite manipulation tasks from a handful of human demos. We train a single FLOWER
model across all 26 datasets in MimicGen's `core` release (12 task families, difficulty
variants `d0`/`d1`/`d2`), then evaluate it with every modality on.

MimicGen's `core` datasets are state-only -- no camera images are recorded, only MuJoCo
simulator states -- so a render step turns them into the same robomimic hdf5 layout
LIBERO's own demos use (`agentview_image`/`robot0_eye_in_hand_image`/
`robot0_joint_pos`/`robot0_gripper_qpos`, `actions`), read by the exact same
`SequenceDataset` (`flower/datasets/robomimic_dataset.py`). MimicGen's action space is
already robosuite's `OSC_POSE` delta convention -- `[dx, dy, dz, drx, dry, drz, gripper]`
in `[-1, 1]`, axis-angle rotation -- identical to LIBERO's, so **no action conversion
happens anywhere on this path**: `action_dim`, `act_window_size`, and the flow-matching
head's `clamp(-1, 1)` output are unchanged from the LIBERO recipe, and the pretrained
`eef_delta` action encoder/decoder load as-is.

### Why a separate container image

MimicGen needs a newer robosuite/robomimic than LIBERO's pinned `robosuite==1.4.0` /
`robomimic==0.2.0`. Rather than risk LIBERO/LIBERO-Plus reproducibility, MimicGen support
ships as `flower-vla-mimicgen:latest`
(`scripts/podman/Containerfile.mimicgen`), built `FROM flower-vla-eval:latest` and
layering MimicGen's pinned sim stack on top -- every large shared layer (CUDA, torch,
the flower requirements, pyhash) is reused; only the sim stack differs.

### Setup and training

```bash
# 1. Build the MimicGen image (one-time)
./run.sh build-mimicgen

# 2. Download the general pretrained checkpoint (if not already done)
./run.sh download-pret

# 3. Download + render all 26 MimicGen `core` datasets, 100 demos each, N datasets at a
#    time. Each ~1-13 GB source hdf5 (~95 GB total) is deleted right after its render, so
#    peak extra disk stays ~N datasets' worth. Outputs already rendered at the current
#    camera sizes are skipped; older renders (e.g. the earlier 128x128 ones) are
#    re-rendered in place. Pass KEEP_MIMICGEN_SOURCE=1 to keep sources.
./run.sh rerender-mimicgen -j 4
# Or per dataset: ./run.sh download-mimicgen square_d0 && ./run.sh prepare-mimicgen square_d0

# 4. Fine-tune on every dataset with the CALVIN training recipe (see below). On success,
#    auto-runs ./run.sh pipeline-mimicgen on the same GPUs (SKIP_PIPELINE=1 to skip).
CUDA_VISIBLE_DEVICES=0 ./run.sh train-mimicgen
```

### Training recipe: CALVIN's

MimicGen fine-tuning follows `conf/config_calvin.yaml`, not LIBERO's recipe:

| | Value | Where |
|---|---|---|
| Optimizer updates | 35 epochs x `updates_per_epoch` 1000 = 35k | `conf/config_mimicgen.yaml` |
| Batch size | `batch_size` 16 x `accumulate_grad_batches` 2 = CALVIN's global 32 (8/GPU x 4 GPUs) on one GPU; N GPUs give 32*N | `conf/config_mimicgen.yaml` |
| Rendered cameras | 200x200 static, 84x84 wrist (CALVIN's native sizes) | `scripts/prepare_mimicgen.py` |
| Model input | 224x224, RandomShifts (pad 10/4), CLIP normalization | `calvin_transforms.yaml` |
| Action windows | only fully inside an episode (CALVIN's `pad: false`) | `conf/datamodule/mimicgen.yaml` |

`trainer.limit_train_batches` counts micro-batches, so it is derived as
`updates_per_epoch x accumulate_grad_batches`: any batch/accumulation override keeps 35k
optimizer updates (and with them the LR schedule and EMA, which count optimizer steps) --
e.g. `batch_size=32 accumulate_grad_batches=1` on a GPU with room for it (a 32-sample
micro-batch at 224x224 needs ~57 GB; 16 needs ~38 GB). Model, optimizer and LR schedule are
shared with every benchmark (`conf/model/flower.yaml`).
Rollouts render at the per-camera sizes recorded in each dataset's `env_args` -- the sizes
the training data was rendered at -- so rollout frames are pixel-identical to training
frames before the shared transform chain
(`tests/test_mimicgen_env_real.py::test_eval_env_reproduces_training_frames_pixel_exact`);
the only train/rollout difference is train-only RandomShifts augmentation.

### Evaluation

```bash
# One-off eval against a specific checkpoint, all modalities on:
./run.sh eval-mimicgen checkpoint=/saves/train_logs/mimicgen/<run>/seed_42/saved_models/last.ckpt \
                       train_folder=/saves/train_logs/mimicgen/<run>

# Post-training pipeline (what train-mimicgen auto-runs): evaluates every dataset in
# flower.datasets.mimicgen_tasks.CORE_DATASETS and uploads result.csv to the training
# run's W&B artifact (same artifact scheme as ./run.sh pipeline, member "mimicgen.csv"):
./run.sh pipeline-mimicgen /saves/train_logs/mimicgen/<run>
PIPELINE_REEVAL=1 ./run.sh pipeline-mimicgen /saves/train_logs/mimicgen/<run>   # discard + redo

# One-off longer-horizon re-evaluation: max_steps_scale multiplies every dataset's
# registered rollout limit (the CSV's max_steps column records the scaled value). Write it
# to its own csv_dir so the run's main result.csv keeps the standard-limit rows:
./run.sh eval-mimicgen checkpoint=... train_folder=... max_steps_scale=1.25 \
    'datasets=[three_piece_assembly_d0,three_piece_assembly_d1,three_piece_assembly_d2]' \
    +csv_dir=/saves/train_logs/mimicgen/<run>/eval_logs/last/mimicgen_tpa_steps_x1.25
```

Evaluation results land in `<train_folder>/eval_logs/<checkpoint>/mimicgen_core/result.csv`,
using the same CSV schema as LIBERO's `result.csv` (`flower/evaluation/eval_records.py`,
unchanged) with `libero_variant=mimicgen`, `suite=core`.

To break a run's results down by task family and difficulty variant, read them back from
W&B (needs `WANDB_API_KEY`):

```bash
# One table per run: success rate + 95% Wilson CI per family x d0/d1/d2, a pooled
# per-family ALL row (families with >1 variant), and OVERALL:
./run.sh analyze-mimicgen run <wandb_run_id> [<wandb_run_id>...]

# Every run matching a W&B config filter: mean/min/max of the per-run success rate per cell
# (runs without a mimicgen.csv artifact are listed and skipped):
./run.sh analyze-mimicgen filter --filters '{"config.use_proprio": true}'
```

### Modality dropout

Training with modality dropout and evaluating every modality combination works as on LIBERO
(`./run.sh train-dropout`, see [Pipeline](#pipeline-automated-post-training-evaluation)):

```bash
# train-mimicgen plus train-dropout's settings: model.modality_dropout=True, keep_fraction
# 0.5, Dirichlet alphas [1,1,1], proprio keep probability 0.5. Runs land in
# /saves/train_logs/mimicgen_dropout/<ts> (W&B run id mimicgen_dropout_<ts>).
CUDA_VISIBLE_DEVICES=0 ./run.sh train-mimicgen-dropout
CUDA_VISIBLE_DEVICES=0 ./run.sh train-mimicgen-dropout model.use_proprio=True   # proprio is dropped too

# A subset of datasets: train and evaluate on the same list
SKIP_PIPELINE=1 ./run.sh train-mimicgen-dropout 'datamodule.dataset_names=[square_d0,stack_d0]'
./run.sh pipeline-mimicgen /saves/train_logs/mimicgen_dropout/<ts> 'datasets=[square_d0,stack_d0]'
```

For a `model.modality_dropout=True` run, `pipeline-mimicgen` plans one eval per modality
combination instead of one all-modalities eval:
- **Which combinations:** every non-empty subset of (static RGB, wrist RGB, language), 7 in
  all, or 14 with `model.use_proprio=True`, each crossed with proprio on/off. These are the
  same combinations as LIBERO (`eval_pipeline.modality_combos`).
- **Where results go:** all combinations merge into the same `result.csv`, since the `use_*`
  columns are part of the merge key.
- **Cost:** about 7× (or 14×) the plain eval at the same `n_eval`.
- **Re-runs:** `PIPELINE_REEVAL=1` rotates the old CSV aside once, before the first
  combination. There is no resume: a re-run evaluates every combination again.
- **Upload:** adds a `pid_modality.txt` member (`scripts/pid_modality.py`) next to `mimicgen.csv`.
- **Analysis:** `analyze-mimicgen` (both `run` and `filter`) prints one family × d0/d1/d2
  table per combination, labeled e.g. `-- static+wrist+lang --`, all-modalities first.

### The 26 `core` datasets

| Task family | Variants | Instruction | Max steps |
|---|---|---|---|
| coffee | coffee_d0, coffee_d1, coffee_d2 | make coffee using the coffee machine and a pod | 500 |
| coffee_preparation | coffee_preparation_d0, coffee_preparation_d1 | make coffee using the coffee machine and a pod | 800 |
| hammer_cleanup | hammer_cleanup_d0, hammer_cleanup_d1 | put the hammer in the drawer and close it | 625 |
| kitchen | kitchen_d0, kitchen_d1 | cook the food on the stove and serve it | 1000 |
| mug_cleanup | mug_cleanup_d0, mug_cleanup_d1 | store the mug inside the drawer | 625 |
| nut_assembly | nut_assembly_d0 | assemble both square and round nuts onto their pegs | 625 |
| pick_place | pick_place_d0 | collect all objects and place them into the container | 1250 |
| square | square_d0, square_d1, square_d2 | insert the square nut onto the square peg | 500 |
| stack | stack_d0, stack_d1 | stack the blocks into a tower | 500 |
| stack_three | stack_three_d0, stack_three_d1 | stack three blocks into a vertical tower | 500 |
| threading | threading_d0, threading_d1, threading_d2 | thread the needle through the eye | 500 |
| three_piece_assembly | three_piece_assembly_d0, three_piece_assembly_d1, three_piece_assembly_d2 | assemble the three toy pieces together | 625 |

Every difficulty variant (`d0`/`d1`/`d2`) of a family shares one instruction -- they are
the same task with progressively wider initial-state distributions. `nut_assembly_d0`
and `pick_place_d0` use a Sawyer arm rather than Franka Panda; FLOWER's language prompt
still declares "Franka Panda" for these (a fixed string, same as LIBERO's own prompt
construction -- see `flower/models/flower.py`'s `format_instruction`).

### Budget

~115 GB of rendered training data (26 datasets x 100 demos x 200x200 + 84x84 RGB, no
`next_obs` -- training never reads it), plus a ~95 GB source download streamed through
`rerender-mimicgen`. Training is CALVIN's budget (35k optimizer updates, global batch 32). Evaluation is 26
datasets x 20 episodes, up to 500-1250
steps each -- longer than LIBERO-10's 10 x 20; `n_eval` is the knob to shrink it.


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
> The `eval_batch_size` MuJoCo worker processes are now reused across batches
> (rebuilt in place — `env.close()` + reconstruct in the same process — rather
> than respawned; see `flower.evaluation.libero_venv._CtxSubprocVectorEnv.rebuild`),
> which removed most of the per-batch env-creation cost. Measured over 25
> in-process rebuild cycles, RSS held flat at ~1.6–2.2 GB per worker with no
> upward trend, so plan for **~2 GB RSS per worker** against the container's
> `MEM_LIMIT` cap in `vars.env` (independent of run length, since workers are no
> longer torn down between batches). With `cross_task_batching: true` (the
> `eval_libero_plus.yaml` default), worker count is `eval_batch_size`
> regardless of `n_tasks`.
>
> `eval_batch_size` sizing: pick it from cores available ÷ concurrently-running
> evals, not "as large as possible" — MuJoCo workers are single-threaded but the
> host still context-switches all of them. With several eval containers sharing
> the box, keep the default (`10`); running one eval alone, raising it toward
> `~24` amortizes the (now small) per-batch setup over more rollouts. If a full
> 2519-task run still approaches the memory cap, split it by category (as above)
> or use multi-GPU (`CUDA_VISIBLE_DEVICES=2,3`) to spread workers across
> processes.

### Evaluation results (CSV)

Every `./run.sh eval` / `./run.sh eval-plus` / `./run.sh eval-pro` run writes one row per
episode to:
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
are the same numbers the printed table shows; `plus_tasks` is the number of distinct base
tasks the row's Plus episodes cover (via `eval_records.base_task_name`) — alongside `n`
and `orig_n`, this is what makes the task-weighting mismatch below legible without
reading a comment; `note` carries the "no severity axis" caveat
on every row of a category that has one (Language Instructions, Background Textures,
unclassified rows); and a category's rows that didn't match the expected `task_name`
pattern get their own explicit `<unparsed>` bin (under `axis=severity`) instead of being
silently dropped, so a category's `n` reconciles across both axes. `./run.sh pipeline`
writes this next to every LIBERO-Plus `result.csv` as `severity_sr.csv` and uploads it
alongside `result.csv`/`pid_modality.txt` (see
[Pipeline](#pipeline-automated-post-training-evaluation) below).

**Robot Initial States perturbs the controller, not the start pose.** LIBERO-Plus
implements this one category's perturbation by substituting the robot's Python class
(`Panda` → `MountedPanda{N}`/`OnTheGroundPanda{N}`, `N` in 1–500 — see
`LIBERO-plus/libero/libero/envs/robots/new_init.py`), which only changes `init_qpos`,
applied by `robosuite`'s `Robot.reset()`. `flower_eval_libero.py` then calls
`env.set_init_state()` with the array `get_task_init_states()` returns for this
category — the *unperturbed* base task's own recorded state — which overwrites that
qpos with a full MuJoCo-state restore (`scripts/debug_robot_initstate_qpos.py` confirms:
identical post-restore qpos across every `N`). The robot's base/mount pose isn't the
mechanism either — LIBERO-Plus picks `Mounted`/`OnTheGround` per *scene*, not per `N`
(`scripts/debug_robot_initstate_controller.py` confirms `base_pos` is identical across
`N` too).

The perturbation survives anyway, through a different path: `Robot.reset()` writes the
perturbed `init_qpos` into `sim.data.qpos` and *then* calls `_load_controller()`, which
builds a fresh OSC controller whose `__init__` captures `self.initial_joint` from that
just-written (perturbed) qpos (`robosuite/controllers/base_controller.py`). OSC uses
`initial_joint` as its nullspace torque reference on every control step
(`robosuite/controllers/osc.py`) — and `set_init_state()`'s state restore never rebuilds
the controller, so `initial_joint` keeps the perturbed value even after qpos itself is
back to nominal. Verified against a real checkpoint (not just by reading code):
`scripts/debug_robot_initstate_controller.py` shows identical qpos/base_pos across `N`
after `set_init_state()`, but `controller.initial_joint` still differs by `N`'s own
perturbation magnitude, and a 30-step zero-action rollout from that restored state
diverges by an amount that grows with `N`'s severity band (0.1 rad → 0.2 rad → 0.5 rad
gave qpos-norm drifts of 0.075 → 0.236 → 0.277 over 30 steps). So this category *is*
physically perturbing the rollout, as a persistent controller bias rather than a
start-pose offset — a different mechanism than its own severity labeling ("robot
joint-space offset in radians") suggests, but a real, reproducible one once the
episode's env reset is itself seeded (see Reproducibility above).
`scripts/severity_sr.py`'s `ROBOT_INITSTATE_NOTE` documents this in code, and its
printed report and `--csv` output both carry the same note on every Robot Initial
States row — the severity bins still index the right physical magnitude, just not the
mechanism the category's own name implies.

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
of the `task_category` filter, and of GPU shard assignment. The same key also seeds the
episode's environment reset (`flower.evaluation.flower_eval_libero.seed_and_reset`,
folded to `np.random.seed`'s range by `eval_records.env_seed`) — see the fixture-
placement paragraph below for why that reset needs its own seed at all.
`flower.evaluation.eval_records.base_task_name` strips LIBERO-Plus's perturbation
suffix, so every Plus variant of a task and that task's original-LIBERO episode 0 — the
row `scripts/severity_sr.py` pairs it against — share one noise stream *and* one env
seed (common random numbers: a paired comparison, which is the point). Generators are
CPU-side, so the draw is identical across GPU models and counts. This does **not** make
the two batching modes bit-identical: batch width still selects Florence-2/DiT GEMM
kernels and reduction order, so only `eval_batch_size=1` reproduces exactly across
modes. Results collected before this change used a per-batch seed with no task-identity
component and are not reproducible against the current code; results collected before
the env-reset seed was added (below) additionally have a silently-random fixture layout
on the five `libero_10` tasks that sample one (a stove, a cabinet+rack, a microwave, a
caddy — see below) and are not reproducible against the current code either.
**Known exception:** LIBERO-Plus Sensor-Noise tasks using motion_blur, fog, or
glass_blur corruption (noise ids 1-10, 31-40, 41-50) call unseeded `np.random` on every
rendered frame inside the vendored env, so those specific episodes are not bit-exact
reproducible even with a matching seed; gaussian_blur/zoom_blur (ids 11-30) are unaffected.

**Comparing a LIBERO-Plus category against original LIBERO is not a same-episode
comparison, even at `init_state_idx=0` and even with the perturbed modality withheld.**
Three things separate them:

1. **Different coverage.** LIBERO-Plus's `Language Instructions` category for
   `libero_10`, for instance, is 383 rows over the same 10 base tasks, all at init-state
   index 0, with an unequal 21–47 episodes per task. Original `libero_10` is 10 tasks ×
   `n_eval` (20 by default) init states. Only the original run's `init_state_idx==0`
   rows are directly comparable, task-weighted — pooling all 200 original episodes
   against 383 Plus episodes conflates the modality/perturbation effect with both the
   init-state-0-vs-all-20 coverage difference and the per-task weighting difference.
   `scripts/compare_eval_csvs.py --base-task --init-state-idx 0` (see below) does this
   pairing correctly; a raw diff of two `result.csv` files' overall success rates does
   not.
2. **A small constant env mismatch.** LIBERO-Plus rewrote `_setup_camera` in
   `LIBERO-plus/libero/libero/envs/problems/*.py` to unconditionally route the
   `agentview` camera pose through `rotate_around_z(..., degrees=0)` then `round(x, 4)`,
   even for the identity (zero) rotation. This shifts the camera by roughly 1.3e-5 m /
   5e-5 in a quaternion component relative to original LIBERO's hard-coded pose — far
   sub-pixel at typical eval resolutions, but renders are not bit-identical to `orig`
   even for an otherwise-unperturbed task. Constant across every Plus category, so it
   cancels within LIBERO-Plus comparisons and only shows up against `orig`.
3. **Fixture placement used to be silently random, independent of any seed.**
   `bddl_base_domain.BddlBaseDomain._reset_internal` samples every fixture's placement
   on each `reset()`, writing it into the MuJoCo *model* (`sim.model.body_pos`/
   `body_quat`) — unlike movable objects, whose placement lands in `qpos`/`qvel`, part
   of the *state*. `env.set_init_state()` restores only `time`/`qpos`/`qvel`
   (`sim.set_state_from_flattened`), so a fixture's placement was never restored by it,
   and the process RNG that placement drew from was never seeded (`libero_venv.py`'s
   worker explicitly entropy-reseeds `np.random` on every batch rebuild). Concretely,
   for `libero_10`: `KITCHEN_SCENE3`/`KITCHEN_SCENE8` (`flat_stove_1`), `KITCHEN_SCENE4`
   (`white_cabinet_1`, `wine_rack_1`), `KITCHEN_SCENE6` (`microwave_1`), and
   `STUDY_SCENE1` (`desk_caddy_1`) each land their fixture at a random point in a
   roughly 2 cm × 2 cm box on every episode — the other five `libero_10` tasks have no
   sampled fixture and were never affected. Verified against a real checkpoint: with
   the fixture's own instruction physically withheld (so every one of a base task's
   Language Instructions variants sees an identical model input), the five
   fixture-bearing tasks showed up to 21 distinct `steps_taken` values across
   nominally-identical episodes, and two of them (`KITCHEN_SCENE8`, `STUDY_SCENE1`)
   showed `success` itself flip between episodes — while the five fixture-free tasks
   were perfectly reproducible. **Fixed**: `seed_and_reset` now seeds the env from the
   same per-episode key that already seeds the noise, immediately before `reset()` (the
   sampling this needs to affect happens *inside* `reset()`, so seeding afterward would
   be a no-op) — a LIBERO-Plus episode and its original-LIBERO counterpart therefore
   draw the same fixture layout, the same way they already draw the same noise. No
   seeding choice recovers the layout the original demonstrations were collected under
   — that was never recorded — so this buys reproducibility and orig/Plus pairing, not
   a return to some ground truth. **Any `result.csv` written before this fix is not
   reproducible and should be re-evaluated** (`PIPELINE_REEVAL=1`, see
   [Pipeline](#pipeline-automated-post-training-evaluation) below) before drawing
   conclusions about these five tasks specifically; the other five `libero_10` tasks'
   pre-fix rows are unaffected by this particular issue.

**Inert-perturbation equivalence check (orig vs. Plus).** `Camera Viewpoints` and
`Sensor Noise` perturb only the static/agentview camera
(`LIBERO-plus/libero/libero/envs/env_wrapper.py`'s `motion_blur`/.../`glass_blur` calls
and the `_view_h_v_s_er_ev` pose all target `agentview_image`/the agentview camera only
— the wrist view and physics are untouched). So on a modality-dropout checkpoint
evaluated with `rgb_static` withheld (`eval_modalities.rgb_static=False`, e.g.
`plus_libero_10_no_static`), both categories are physically inert: the model's input is
identical to the unperturbed base task's, and with `rollout_seed` keyed on base task
name and `seed_and_reset` seeding the env before `reset()` (both above), every Plus
variant of a base task is expected to reproduce that task's original-LIBERO
`init_state_idx=0` episode's `success` *exactly* — the sharpest available end-to-end
check that the `orig` and `plus` harnesses agree, since a residual delta here can't be
attributed to the perturbation.

Checked across four `libero_10_dropout` checkpoints
(`scripts/compare_eval_csvs.py --base-task --init-state-idx 0 --task-category-a
"Camera Viewpoints"`/`"Sensor Noise"` against `wrist,lang,proprio`, i.e. `rgb_static`
withheld): 7-9 of 10 base tasks match exactly on every checkpoint, but 1-3 per
checkpoint don't — always the *same* base task set for both categories within one
checkpoint, and always a full success/failure flip (e.g. 100%→0%), never a partial
shift. Every divergent task but one is a fixture-sampling task (`KITCHEN_SCENE3`,
`KITCHEN_SCENE8`, `STUDY_SCENE1` — see the fixture-placement note above); the exception,
`LIVING_ROOM_SCENE6`, has no sampled fixture at all.

Root-caused to two distinct, already-documented mechanisms, not a new bug:
1. **Not fixture-placement desync.** Re-verified directly:
   `scripts/debug_fixture_reset_randomization.py`, seeded with each task's actual
   production `rollout_seed`, prints a bit-identical fixture pose (position, quaternion,
   and a cross-process-stable crc32 fingerprint) in both the `eval` and `eval-plus`
   containers for `KITCHEN_SCENE3`/`KITCHEN_SCENE8`/`STUDY_SCENE1` — the September fix
   (above) works correctly; this isn't why they diverge.
2. **Batch-shape GEMM/kernel numerics.** `conf/eval_libero_plus.yaml` sets
   `cross_task_batching: true` (Plus batches episodes across many tasks; orig batches
   within one task), so an otherwise-identical episode's trajectory runs through a
   different-shaped/composed batch on each side — non-associative matmul reduction
   order, not a different computation (see the Reproducibility paragraph above, and
   `scripts/debug_language_variant_determinism.py`'s docstring, which localizes this
   same residual to policy-side batch-shape numerics for a different task set). A
   handful of `libero_10` episodes apparently sit close enough to a success/failure
   decision boundary that this is enough to flip the outcome. This also explains why
   the flip isn't always the same direction or task across checkpoints/seeds — it's
   checkpoint-specific proximity to a boundary, not a systematic bias. One checkpoint's
   `KITCHEN_SCENE3` sits so close to that boundary it's even non-uniform *within* the
   Plus category itself (its own ~45-47 cross-task-batched variants don't all land on
   the same batch composition either) — consistent with, not contradicted by, this
   explanation.

`Sensor Noise` noise ids 48-50 (`glass_blur`, severities 8-10) are the one documented
exception to a shared seed guaranteeing reproducibility (unseeded `np.random` inside the
vendored env — see the "Known exception" paragraph above); on the one checkpoint where
`KITCHEN_SCENE3` was internally non-uniform, its Sensor-Noise minority-outcome episodes
were exactly noise ids 48-50 — plausibly compounding the batch-numerics effect on that
task, though its Camera-Viewpoints counterpart (no noise involved) was non-uniform too,
so batch-shape numerics alone already accounts for the bulk of it.

Bottom line: **the orig and Plus harnesses agree** wherever a checkpoint's true behavior
isn't already balanced on a knife's edge; the residual few-percent of base tasks that
disagree do so for a known, order-of-non-associative-floating-point-ops reason common to
any batched-inference eval, not a LIBERO-vs-LIBERO-Plus harness bug.

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
Comparing a LIBERO-Plus file against an original-LIBERO file (see the comparison
caveats under Reproducibility above) needs `--base-task` (keys both sides on
`eval_records.base_task_name`, so a Plus file's
per-variant task names collapse onto their shared base task) and `--init-state-idx 0`
(restricts the original-LIBERO side to the one init state LIBERO-Plus itself draws
from); the printed table also gains `n_a`/`n_b` columns so the unequal per-task episode
counts a LIBERO-Plus category can have are visible instead of silently averaged away:
```bash
python scripts/compare_eval_csvs.py $CKPT_DROP/eval_logs/last/plus_libero_10_no_lang/result.csv \
                                     $CKPT_DROP/eval_logs/last/orig_libero_10/result.csv \
  --base-task --init-state-idx 0 \
  --modalities-a static,wrist,proprio --modalities-b static,wrist,proprio
```
Restricting to one LIBERO-Plus perturbation category — e.g. the inert-perturbation
equivalence check above — needs `--task-category-a` (per-side, not shared with
`--task-category-b`, since an original-LIBERO file's rows all carry an empty
`task_category`):
```bash
python scripts/compare_eval_csvs.py $CKPT_DROP/eval_logs/last/plus_libero_10_no_static/result.csv \
                                     $CKPT_DROP/eval_logs/last/orig_libero_10/result.csv \
  --base-task --init-state-idx 0 --task-category-a "Camera Viewpoints" \
  --modalities-a wrist,lang,proprio --modalities-b wrist,lang,proprio
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

### Rollout recordings (Rerun)

`result.csv` says whether an episode succeeded; it doesn't let you *watch* one. Setting
`rerun.enabled=true` on any `./run.sh eval` / `eval-plus` / `eval-pro` additionally writes a
[Rerun](https://rerun.io) `.rrd` recording of the first `rerun.max_episodes` episodes
(`flower/evaluation/rerun_recorder.py`):

```
/<modality combo>/
    input/rgb_static     3rd-person camera, JPEG at rerun.jpeg_quality
    input/rgb_gripper    wrist camera, JPEG
    input/proprio        9-D robot0_joint_pos + robot0_gripper_qpos, named series
    input/present        0/1 per modality -- which of static/wrist/lang/proprio
                         actually reached the model
    output/action        the 7-D action the model emitted (x y z rx ry rz gripper)
    output/success       the outcome, one point per episode
    info                 task name, instruction, combo, rollout_seed, steps_taken,
                         success -- structured fields, for programmatic reading
    summary              the same, as markdown for a viewer text panel
```

**The episode is a position on the timeline, not a level in the entity path.** Every
episode logs to the same entity paths, on one `frame` timeline where episode *i* occupies
the fixed slot `[i * frame_stride, (i+1) * frame_stride)`. One time slider therefore
scrolls continuously through every recorded rollout in a single set of views. Rerun
resolves data as "latest value at time T on the active timeline", so an episode can live
either in the path or on the timeline — not both; putting it on the timeline is what makes
a single scrubbing view possible.

Fixed-width slots (rather than packing episodes end to end) keep two recordings
frame-aligned: a full-modality arm and a withheld-modality arm run the same episodes in the
same order, so frame F lands on the same episode of both even though their episodes end at
different steps. `info`/`summary` are written at each episode's *first* frame, so scrubbing
anywhere inside an episode tells you which rollout you are in and how it ended.

```yaml
rerun:
  enabled: false       # off by default — nothing changes, and rerun-sdk is never imported
  max_episodes: 10     # first N episodes in work order
  jpeg_quality: 40     # 1-100, both camera views
  path: null           # null -> <log_dir>/rollouts.rrd
  combo_name: null     # null -> the '+'-joined present-modality combo; the entity root
  frame_stride: null   # null -> max_steps; width of each episode's slot on `frame`
  recording_id: null   # null -> "<libero_variant>_<benchmark_name>"
```

**Withheld modalities are still recorded.** The camera frames and proprio come from the
environment observation, so `eval_modalities.rgb_gripper=False` does *not* blank out
`input/rgb_gripper` — the rollout stays watchable, and the withholding shows up as a zero
in `input/present` (and in `info`'s `present_*` fields) instead. Those flags come from the
same place `result.csv`'s `use_*` columns do, so a recording and the CSV never disagree.

**Camera frames are vertically flipped, for display only.** robosuite renders bottom-up, so
the raw `agentview_image`/`robot0_eye_in_hand_image` arrays — the ones the model is fed, and
the ones the training data was rendered in — are upside down to a human. The recording flips
them so the viewer is legible. The consequence worth knowing: a recorded frame is **not**
byte-identical to the model's input, it is that input mirrored vertically. Nothing on the
inference path sees the flip.

**Episode selection** is the first N episodes *in work order*, applied identically in both
batching modes — deliberately unlike `num_videos`, which caps per task under
`cross_task_batching: false` and globally under `true`. Two evals of the same suite that
differ only in `eval_modalities` therefore record the identical episode set into the
identical frame slots, which is what makes them comparable. Combine with `task_category=`
to choose which slice of LIBERO-Plus gets recorded. On multi-GPU, only rank 0 records (all
ranks share one `rerun.path`).

**`rerun.max_episodes` caps recording, not evaluation** — the eval still rolls out every
task in the suite (or category) to produce its `result.csv`. Recording 10 episodes of a
LIBERO-Plus category costs the whole category's GPU time.

**Size**: roughly 2–4 MB per `libero_10` episode at `jpeg_quality: 40` and 224x224 — so the
default cap of 10 is ~20-40 MB, while an uncapped 2519-task LIBERO-Plus run would be several
GB. Raise `max_episodes` deliberately.

**Dependency**: `rerun-sdk==0.26.2`, its own trailing layer in
`scripts/podman/Containerfile` (rebuild with `./run.sh build`). 0.26 is the last release
supporting this image's python3.9; 0.27+ needs >=3.10. Its `numpy>=1.23` constraint would
otherwise resolve to numpy 2.x, which LIBERO and robomimic can't use, so the same layer
re-pins `numpy~=1.23`.

#### The viewer layout (`scripts/rerun_blueprint.py`)

`scripts/rerun_blueprint.py` writes a Rerun blueprint (`.rbl`) giving one row per modality
combo found in the recordings — two camera views side by side, then the action chart above
the proprio chart — with a thin strip on top naming the rollout under the cursor:

```
┌───────────── rollout: task, instruction, outcome ─────────────┐
├─ 3rd person ─┬─ 1st person ─┬──────── action ────────────────┤  static+wrist+lang+proprio
│              │              ├──────── proprio ───────────────┤
├─ 3rd person ─┬─ 1st person ─┬──────── action ────────────────┤  static+lang+proprio
│              │              ├──────── proprio ───────────────┤
└───────────────────────────────────────────────────────────────┘
 ◀──────────────────────── frame ──────────────────────────────▶
   ep0        ep1        ep2        ...                     ep9
```

```bash
python scripts/rerun_blueprint.py /saves/rerun/layout.rbl \
    /saves/rerun/plus_libero_10_all.rrd /saves/rerun/plus_libero_10_no_wrist.rrd

rerun /saves/rerun/layout.rbl \
    /saves/rerun/plus_libero_10_all.rrd /saves/rerun/plus_libero_10_no_wrist.rrd
```

The combos are read back out of the recordings, so the blueprint can never name a row with
no data behind it; `--combo static+lang+proprio` (repeatable) names them explicitly instead.
Dragging the one time slider scrolls through all recorded rollouts, in both arms at once.

#### Comparing a modality combo against the full one

Two evals of the same episodes, one with every modality and one with the wrist view
withheld. They share a `recording_id`, so opening both `.rrd` files together merges them
into a single recording whose two combos become the two rows above.

**A few recordings, cheaply.** `n_eval=1` on plain LIBERO-10 is 10 episodes — one per task,
all 10 tasks — which is minutes of GPU rather than the better part of an hour.
`cross_task_batching=true` is what makes them run as one parallel batch instead of ten
batches of one (it's required whenever `n_eval < eval_batch_size`):

```bash
RUN=/saves/train_logs/libero_10_dropout/2026-09-14_17-00-58   # a train-dropout run, use_proprio=true
CKPT=$RUN/seed_42/saved_models/last.ckpt
COMMON="n_eval=1 cross_task_batching=true eval_batch_size=10 num_videos=0 log_wandb=False"

# A. every modality present
./run.sh eval checkpoint=$CKPT train_folder=$RUN $COMMON \
    csv_dir=/saves/rerun/csv_orig_all \
    rerun.enabled=true rerun.max_episodes=10 \
    rerun.path=/saves/rerun/orig_libero_10_all.rrd

# B. wrist view withheld at the model input
./run.sh eval checkpoint=$CKPT train_folder=$RUN $COMMON \
    eval_modalities.rgb_gripper=False csv_dir=/saves/rerun/csv_orig_no_wrist \
    rerun.enabled=true rerun.max_episodes=10 \
    rerun.path=/saves/rerun/orig_libero_10_no_wrist.rrd
```

Note `n_eval` is the knob for *plain* LIBERO (default 20 per task). **LIBERO-Plus is already
1 episode per task** — its cost is the number of tasks (~2519, or ~419 in a single
`task_category`), which `base_task`/`max_tasks` below are for. Swap `./run.sh eval` for
`./run.sh eval-plus task_category="Camera Viewpoints"` (dropping `$COMMON`, whose defaults
Plus already sets) when the perturbed scenes are what you want to look at.

#### One episode, many perturbations (`base_task`, `max_tasks`)

`base_task` keeps only the tasks whose `eval_records.base_task_name` matches — on
LIBERO-Plus that is every perturbation variant of one original task. Since all of a base
task's variants share its init state and, via `rollout_seed` (keyed on the base task name),
its flow-matching noise and env reset, the result is **the same episode under a sweep of
perturbations** — e.g. one rollout seen from ten camera angles. `max_tasks` then caps how
many of those variants run.

Both narrow the *evaluation*, unlike `rerun.max_episodes` which only caps recording, so
both mark `result.csv`'s `task_category_filter` non-empty and such a run can never be
mistaken for a finished suite by `./run.sh pipeline --resume`
(`eval_pipeline.already_done`).

```bash
BT=KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it

./run.sh eval-plus checkpoint=$CKPT train_folder=$RUN \
    task_category="Camera Viewpoints" base_task=$BT max_tasks=10 \
    csv_dir=/saves/rerun/csv_view_all \
    rerun.enabled=true rerun.max_episodes=10 \
    rerun.path=/saves/rerun/views_all.rrd
```

Each frame slot is then one camera pose of the same rollout, and the `summary` panel's task
name carries that pose's own id (`_view_<h>_<v>_<s>_<er>_<ev>_initstate_0`). A base task's
name is any of its variants' names with the perturbation suffix stripped; the per-base-task
breakdown in `scripts/severity_sr.py` and `scripts/compare_eval_csvs.py --base-task` uses
the same derivation.

Both arms draw the same episodes, in the same order, into the same frame slots, and use the
same per-episode seeds (`rollout_seed` is keyed on the base task name, not on the modality
combo — see [Reproducibility](#evaluation-results-csv) above), so this is a paired
comparison. `csv_dir=` keeps each arm's rows in its own file.

To read a recording back programmatically, note that entity paths are escaped per Rerun's
path grammar — the `+` in a combo name is written `\+` on the wire. Build paths with
`flower.evaluation.rerun_recorder.combo_entity_path` rather than by hand:

```python
import rerun as rr
from flower.evaluation.rerun_recorder import TIMELINE, combo_entity_path

root = combo_entity_path("static+wrist+lang+proprio")
rec = rr.dataframe.load_recording("/saves/rerun/orig_libero_10_all.rrd")
rec.view(index=TIMELINE, contents={f"{root}/output/action": ["Scalars:scalars"]}).select().read_all()
```

### Pipeline: automated post-training evaluation

`./run.sh pipeline <train_run_dir> [hydra_overrides...]` automates everything above for one
completed training run — deciding which evaluations it needs, running them, and uploading
the results — instead of invoking `eval`/`eval-plus`/`eval-pro`/`compare_eval_csvs.py`/
`pid_modality.py` by hand:

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

Both branches then run the **LIBERO-PRO** lines (see
[LIBERO-PRO Generalization Evaluation](#libero-pro-generalization-evaluation)): the 4
published perturbation suites on the native init-state arm at full modality, plus
`libero_10_lan` and `libero_10_task` on the matched arm swept across every modality combo
— 32 lines for a `use_proprio=true` checkpoint, 18 otherwise. This is unaffected by
`PIPELINE_SKIP_MODALITY_OFF` (which targets the four `plus_no_*` suites); use
`PIPELINE_SKIP_PRO_COMBOS=1` to collapse it to 6 full-modality lines. Their reeval-suite
tokens are the result-directory names, e.g.
`PIPELINE_REEVAL_SUITES=pro_libero_10_swap,pro_matched_libero_10_lan` — selecting a
matched suite replans all of its combo lines, and only the first carries `reeval=true`
since they share one `result.csv`.

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

`PIPELINE_RESUME=1` skips any combo whose modality flags already have a *complete* row in
the target `result.csv` (one from a full, unfiltered run — a row merged in by, say, a
manual `./run.sh eval-plus task_category="..."` targeting the same suite doesn't count,
so a partial suite like that still gets finished rather than permanently mistaken for
done), so a sweep interrupted partway through restarts cheaply instead of re-evaluating
everything:

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
modality combo no longer evaluated), each selected suite's own eval invocation backs its
old file up to `results_<mtime>.csv` (named for the old file's own modification time)
immediately before writing its fresh results, and starts that suite from empty. This
happens lazily, suite by suite as the pipeline actually gets to it — not all at once
before anything runs — so an interruption partway through a multi-suite re-evaluation
never leaves a suite the run hasn't reached yet without any `result.csv` at all.
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
has no effect on a suite selected for re-evaluation — it's replanned unconditionally,
regardless of what's currently in its `result.csv` — but still applies normally to any
suite *not* selected. The
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

### Plotting evaluation results

`scripts/plot_eval.py` (`./run.sh plot`) turns the same W&B evaluation artifacts
`analyze_wandb.py` reads into paper figures (PDF). It's split into two units:

- **`scripts/plot_data.py`** — gather / filter / prepare. Reuses `analyze_wandb.py`'s
  fetching and analysis (`default_entity_project`, `run_modalities`,
  `parse_modality_spec`, `applicable_modality_off`, `analyze`) rather than re-reading
  W&B, but caches downloaded artifacts under a persistent `--cache-dir` (default
  `/saves/plot_cache`, skipped when already complete; `--refresh` forces a re-download)
  instead of `analyze_wandb.py`'s per-invocation temp directory. No matplotlib import.
- **`scripts/plot_eval.py`** — shared Physical-Intelligence-paper-style rcParams (white
  background, no top/right/left spines, light horizontal gridlines, frameless legend,
  font scale 2.5x default), the figures, and a `tyro` subcommand CLI.

One `--filters` (same mongo-style JSON as `analyze_wandb.py`'s) defines the run pool for
every subcommand:

```bash
./run.sh plot presence --filters '{"config.modality_dropout": true}'
./run.sh plot perturbation --filters '{"config.modality_dropout": true}'
./run.sh plot severity --filters '{"config.modality_dropout": true}'
./run.sh plot clear --filters '{"config.modality_dropout": true, "config.modality_dropout_proprio_keep_p": 1}'
./run.sh plot all --filters '{"config.modality_dropout": true}'   # one fetch, all 6 PDFs + whatever of the clear/ set applies
```

| Subcommand | Output | Shows |
|---|---|---|
| `presence` | `presence_libero10.pdf` | Clean LIBERO-10, x = training config (`modality_dropout_proprio_keep_p`, plus the two `modality_dropout=False` baselines below), one bar per inference-time modality config (all-4, and each of the 4 withheld in turn) |
| `perturbation` | `perturbation_libero10plus.pdf` | LIBERO-10-Plus, x = the 7 perturbation categories, one filled bar per training config |
| `severity` | 4 PDFs (below) | All-modality vs. 1-left-out, each bar paired with its init-state-matched LIBERO original baseline |
| `clear` | 7 PDFs under `clear/` (below) | Presentation-grade versions of `perturbation`/`severity`'s figures, with the McNemar apparatus stripped and the series restricted to a single comparison — see below for why 4 of the 7 need a single-training-config `--filters` |

A run's *training config* series is its `keep_p` when `modality_dropout=True` (the
cividis ramp below), or one of two fixed-color baselines when `modality_dropout=False`
(`plot_data._series_key`) — a non-dropout run has no `keep_p` to place it on that ramp
with, so it gets its own slot instead, split on whether it was trained with
`use_proprio`:

| Series | Color |
|---|---|
| `no dropout (+proprio)` | crimson `#c1121f` |
| `no dropout (no proprio)` | teal `#0e9594` |

Both sort after the (descending) `keep_p` ramp in every legend/x-axis.

`perturbation` and every `severity` figure pair each filled bar with a hollow,
45-degree-hatched bar (full solid outline) in the same color: the init-state-matched
LIBERO original baseline for **the exact initial states that filled bar's episodes were
drawn from** — for a withheld-modality bar, that means clean LIBERO performance with that
SAME modality also withheld, not full-modality clean LIBERO (severity_sr's per-row
modality-combo matching makes this automatic; see "Paired original baseline" above).

`severity`'s 4 PDFs cover both axes (physical severity, upstream `difficulty_level`) in
both a per-category breakdown and a pooled ("totals") view:

| File | x-axis | Bars |
|---|---|---|
| `severity_modality_off_percategory.pdf` | faceted per category, each category's own severity bins | modality config, paired with orig |
| `severity_modality_off_pooled.pdf` | the 7 perturbation categories | modality config, pooled over each category's own severity bins, paired with orig |
| `difficulty_modality_off_percategory.pdf` | faceted per category, `difficulty_level` (1-5) | modality config, paired with orig |
| `difficulty_modality_off_pooled.pdf` | `difficulty_level` (1-5) | modality config, pooled across every category, paired with orig |

All bar heights are success rates with 95% Wilson intervals (`severity_sr.wilson_interval`)
computed over successes/n **pooled across every run in `--filters`** that covers that bar
— not an average of per-run rates. Whenever a bar pools more than one run, `plot_data.py`
prints a `WARNING:` line to stderr naming them, so cross-run/cross-`keep_p` pooling (e.g.
every `severity` figure deliberately pools across every matched `keep_p`) never happens
silently. Colors are consistent across figures for the same thing: one categorical hue
per modality config (`presence`/`severity`/`difficulty`), one step of matplotlib's
`cividis` colormap per `keep_p` plus the two fixed baseline colors above
(`presence`/`perturbation`). A modality config with no data anywhere in a figure is
simply not drawn (no legend entry, no zero-height bar standing in for "no data").

Every bar prints its own success-rate value just inside its top edge (white on a filled
bar, the bar's own series color on a hollow hatched orig-LIBERO bar). Every filled bar
with a paired orig-LIBERO baseline also prints `p=.../n=...` below the axis: a **task-level
exact McNemar test** (`severity_sr.mcnemar_exact_p`) against that baseline — each
`(modality combo, base task)` group's LIBERO-Plus episodes collapse to one binary outcome
by strict majority vote (an exact tie is dropped, not broken either way), paired 1:1 against
that base task's own orig-LIBERO outcome. `n` is the number of base tasks the test is built
from, printed alongside `p` because it's the test's own power, not just its result.

Four data caveats worth knowing when reading these figures:

- **The hatched bars carry wide CIs by construction.** `orig_n` (the init-state-matched
  LIBERO original baseline) is deduplicated by `(modality combo, base task)`, so it's ≤10
  for LIBERO-10 — noted directly on `perturbation`'s figure.
- **The McNemar test's own power is capped the same way.** `mcnemar_n` is also ≤10 base
  tasks, so the smallest reachable two-sided p-value is `2 * 0.5**10 ≈ 0.002` — most bars
  will show p near 1 regardless of the true effect size, not because there's no effect.
- **The two pooled figures' (`severity_modality_off_pooled.pdf`,
  `difficulty_modality_off_pooled.pdf`) McNemar test is anti-conservative.** They sum
  `mcnemar_b`/`mcnemar_c`/`mcnemar_n` across a category's own bins or across categories
  (`plot_data.merge_bars`), which reuses the same base tasks' pairs several times over —
  `n` there overstates independent evidence. Read the un-pooled per-category/per-bin figure
  for the honest test; the pooled figures carry a footnote saying so.
- **The Robot Initial States facet is annotated with a known LIBERO-Plus upstream
  quirk**: that category's perturbed robot `init_qpos` is restored to nominal before
  rollout, but survives as a persistent controller bias instead (see
  `severity_sr.ROBOT_INITSTATE_NOTE`) — its severity bins reflect the right physical
  magnitude through a different mechanism than a start-pose offset, not a spurious
  gradient.

`clear`'s 7 PDFs, written into `outdir/clear/`, are presentation-grade versions of
`perturbation`/`severity`'s figures: **no `p=.../n=...` McNemar label, no statistical
caveat footnote** (those stay on the audit-grade figures above), and each restricted to
a single comparison so the reader sees one story per figure. The per-category severity/
difficulty breakdowns are one standalone PDF per category instead of one PDF with N
stacked facets:

| File | x-axis | Bars |
|---|---|---|
| `perturbation_libero10plus_dropout_vs_nodropout.pdf` | the 7 perturbation categories | only `keep_p=1` vs. `no dropout (+proprio)`, paired with orig |
| `severity_all_pooled.pdf` | the 7 perturbation categories | only the all-modality series, pooled over each category's own severity bins, paired with orig |
| `severity_all_percategory_<category>.pdf` (one per category) | that category's own severity bins | only the all-modality series, paired with orig |
| `difficulty_all_pooled.pdf` | `difficulty_level` (1-5) | only the all-modality series, pooled across every category, paired with orig |
| `difficulty_all_percategory_<category>.pdf` (one per category) | `difficulty_level` (1-5) | only the all-modality series, paired with orig |

The two data caveats about wide hatched-bar CIs and the Robot Initial States upstream
quirk still apply and are still noted on the relevant `clear` figures — only the McNemar
apparatus is stripped.

**The 4 severity/difficulty files need `--filters` to select exactly one training
config.** They come from `plot_data.prepare_severity`, which pools every run in the
filtered pool into one bar regardless of `modality_dropout_proprio_keep_p` — fine for
the audit-grade `severity` figures (their whole point is comparing configs on the same
axes), but silently misleading for a `clear` figure meant to show a single trained
model. `clear` therefore **raises** if `--filters` matches more than one
`modality_dropout_proprio_keep_p` value, or both `modality_dropout=false` baselines
(`BASELINE_PROPRIO`/`BASELINE_NO_PROPRIO` count as two configs — pin `use_proprio` too
to pick one):

```bash
./run.sh plot clear --filters '{"config.modality_dropout": true, "config.modality_dropout_proprio_keep_p": 1}'
./run.sh plot clear --filters '{"config.modality_dropout": false, "config.use_proprio": true}'
```

`all` doesn't fail this way on a broader filter: it still writes every audit-grade PDF
and the clear perturbation figure (which is always a 2-config comparison — see below —
and is exempt from this check), and just skips the 4 clear severity/difficulty PDFs with
a stderr `WARNING`.

Since a `clear` severity/difficulty bar now represents exactly one training config, it's
colored/labelled by that config instead of the neutral "all modalities" gray — the same
`keep_p`/baseline color and legend text (`series_color`/`series_label`) `perturbation`'s
own figure already uses, so the same config reads as the same color everywhere.
`perturbation_libero10plus_dropout_vs_nodropout.pdf` is the one deliberate exception: it
always compares exactly `keep_p=1` against `no dropout (+proprio)`, two configs, by
design — not affected by the single-config requirement above.

Requires `WANDB_API_KEY`, same as `./run.sh analyze`.

#### Common Issues

Sometimes this causes problems for the python env so just delete it:

```python
log.info(f"Using calvin_env with commit {get_git_commit_hash(Path(calvin_env.__file__))}.")
```
The path for this line is in the CALVIN env repo: https://github.com/mees/calvin_env/blob/797142c588c21e76717268b7b430958dbd13bf48/calvin_env/envs/play_table_env.py#L72

---

## LIBERO-PRO Generalization Evaluation

[LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) ([paper](https://arxiv.org/abs/2510.03827))
perturbs *what a task is* — which object, where it starts, how the instruction is worded,
what counts as success — rather than how the scene is sensed (LIBERO-Plus's axis). Its
finding is that VLA success on LIBERO comes largely from memorising training scenarios:
models above 0.9 on the originals collapse toward 0 under task and position perturbations.

It is integrated as a third side-by-side submodule (`LIBERO-pro/`), selected by
`LIBERO_VARIANT=pro` exactly like LIBERO-Plus.

### Why this benchmark pairs so cleanly with our baseline

A LIBERO-PRO suite keeps the original libero_10 **task names, bddl filenames and init
filenames verbatim** — only the file contents change. So a PRO task maps to its orig
baseline task by identity (no prefix matching, no suffix grammar), and because
`eval_records.base_task_name` is the identity on those names, a PRO episode draws the
**same `rollout_seed`** — hence the same fixture placement and the same flow-matching
noise — as the orig episode with the same task and episode index.

Diffing the four published libero_10 suites against `LIBERO/libero/libero/bddl_files/libero_10/`:

| suite (`benchmark_name`) | perturbation | bddl delta vs original | scene model |
|---|---|---|---|
| `libero_10_lan` | semantic | `(:language)` | identical |
| `libero_10_task` | task redefinition | `(:language)` `(:goal)` `(:obj_of_interest)` | identical |
| `libero_10_swap` | position | `(:init (On …))` placements | identical |
| `libero_10_object` | object | object/fixture classes (`moka_pot` → `yellow_moka_pot`) | different meshes |

Confirmed in-simulator by `scripts/debug_pro_scene_identity.py` over all 10 tasks:
`_lan`/`_task`/`_swap` reproduce the original's sim-state width **and** its joint and
body name ordering exactly; `_object` keeps the width (its replacements are same-DOF
bodies) but renames joints and bodies — which is why matchability is a property of the
suite, not something a shape check could decide.

### The two init-state arms

`pro_init_states` selects which initial states the rollouts start from, and the arm is
recorded in the result CSV's `libero_variant` column (`pro` / `pro_matched`), so the two
never merge together:

- **`native`** — the suite's own `.pruned_init`. Upstream-faithful and comparable to the
  paper's leaderboard. Pairs with the orig baseline at task level only.
- **`matched`** — the **original** libero_10 `.pruned_init`, so each PRO episode starts
  from the exact state the corresponding `libero_orig.csv` episode did and the pair
  differs *only* by the perturbation. Legal for `libero_10_lan` and `libero_10_task`;
  requesting it for `_swap` (where the init state *is* the perturbation) or `_object`
  (different meshes) is a config error, not a silent fallback.

### The matched-arm modality sweep

The matched arm is evaluated at **every modality combination** the checkpoint supports —
14 for a `use_proprio=true` model (7 token combos × proprio on/off), 7 otherwise — while
the native arm stays full-modality.

This is what makes the modality-dropout hypothesis testable on PRO. Only the matched arm
pairs one-to-one with an orig episode, and `pro_sr.py` keys that pairing on the modality
combo, so a swept record answers exactly *"how much does withholding this modality cost on
this perturbation"* against an episode it differs from in nothing else. The native arm can
only ever pair at task level, so sweeping it would add lines without adding the
measurement.

The sweep keys off `use_proprio`, **not** off `modality_dropout`: a non-dropout model
evaluated with a modality withheld is the control the dropout model is compared against,
so it has to exist for both — the same reason the modality-off LIBERO-Plus suites are
already planned for every run. On a non-dropout run the orig side is missing for the
withheld combos (its `orig_libero_10/result.csv` holds only full modality), and `pro_sr.py`
reports those records with empty paired columns rather than inventing a baseline; the
cross-model success-rate comparison still works.

All of a matched suite's combo lines merge into **one** `result.csv` — the modality flags
are `eval_records.KEY_COLUMNS`, so they key apart rather than collide, exactly as orig's
combos do. (LIBERO-Plus needs its `csv_dir=` split only because `perturbation_sr.py` and
`severity_sr.py` assume one combo per file; `pro_sr.py` groups by combo natively.)

| run | combos | PRO lines | PRO episodes | approx. cost (1 GPU) |
|---|---|---|---|---|
| `use_proprio=true` | 14 | 32 | 6400 | ~6.5 h |
| `use_proprio=false` | 7 | 18 | 3600 | ~3.3 h |
| any, with `PIPELINE_SKIP_PRO_COMBOS=1` | 1 | 6 | 1200 | ~1.4 h |

Measured at ~14 min per 200-episode suite on one GPU. `PIPELINE_SKIP_PRO_COMBOS=1` drops
the sweep back to full modality only — use it to keep a chained
`train-dropout` → `pipeline` run short, then catch the rest up later with
`PIPELINE_RESUME=1`, which back-fills only the combos still missing. It is independent of
`PIPELINE_SKIP_MODALITY_OFF` (that one gates the LIBERO-Plus axis).

`pro_sr.py` prints a `mod` column — a 4-slot mask over (static, wrist, lang, proprio):
`SWLP` is full modality, `SW-P` is language withheld. `./run.sh analyze` shows the
full-modality row per suite by default and every combo under `--modality-off-detail=full`.

### Instruction handling (a deliberate deviation from upstream)

LIBERO derives `task.language` from the *filename*, and PRO filenames are the original
ones — so prompting from the filename would hand the model the **unperturbed**
instruction and make the semantic and task suites inert. `task_language()` therefore
reads `(:language …)` out of the task's own bddl for `_lan` and `_task`, and passes
`task_i.language` through for `_object` and `_swap`, where the original instruction is
the correct control. Our `_lan` numbers will not match the published leaderboard if
upstream prompts from the filename there; the alternative is a perturbation that is a
no-op by construction.

### Scope

Only the four suites published on [HuggingFace](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro)
are wired up, for `libero_10`. `libero_10_env` (environment replacement) is not published
and upstream documents an object-drift bug in it; multi-flag `_temp` combinations require
running `perturbation.create_env()` locally, whose init states are freshly sampled and so
not reproducible against the published set. `scripts/pro_sr.py`'s `perturbation_vector`
column is already the full 5-flag tuple, so either would slot in without a schema change.

### Setup

```bash
git submodule update --init LIBERO-pro   # ~900 MB; the 3D assets ship inside the repo
./run.sh build                           # entrypoint.sh is baked into the image
./run.sh download-pro                    # bddl + init files for the 4 libero_10_* suites
```

### Running

```bash
# One suite per invocation.
./run.sh eval-pro benchmark_name=libero_10_object train_folder=$CKPT_BASE checkpoint=$CKPT_BASE
./run.sh eval-pro benchmark_name=libero_10_task pro_init_states=matched train_folder=$CKPT_BASE checkpoint=$CKPT_BASE

# Scene-identity pre-flight (re-run after a LIBERO-PRO submodule bump):
podman-compose -f scripts/podman/compose.yml run --rm -T eval-pro \
    python scripts/debug_pro_scene_identity.py
```

`./run.sh pipeline` runs all six lines automatically (4 native suites + the 2 matched
arms, full modality only) and uploads `libero_pro.csv` and `pro_sr.csv` to the run's W&B
artifact alongside the LIBERO/LIBERO-Plus members.

### Back-filling PRO onto runs evaluated before it existed

`PIPELINE_RESUME=1` skips every suite whose `result.csv` already covers the combo, so on
an older run the PRO lines are the only ones left to plan — no re-evaluation of LIBERO or
LIBERO-Plus. The check is per *combo*, not per suite, so a run evaluated before the
matched-arm sweep existed back-fills only the combos it is missing (13 of 14 per matched
suite, the full-modality one already being on disk):

```bash
# PRO only (use when the run's modality-off evals are already done, or you don't want them):
PIPELINE_RESUME=1 PIPELINE_SKIP_MODALITY_OFF=1 \
    ./run.sh pipeline /saves/train_logs/libero_10_dropout/<run>

# PRO plus anything else still missing:
PIPELINE_RESUME=1 ./run.sh pipeline /saves/train_logs/libero_10_dropout/<run>
```

Check what it would do first — the planner prints one tab-separated line per eval and runs
nothing:

```bash
podman-compose -f scripts/podman/compose.yml run --rm -T pipeline-artifacts \
    python scripts/eval_pipeline.py plan --train-folder /saves/train_logs/.../<run> \
    --resume --skip-modality-off
```

The pipeline's upload step runs even when the plan is empty, so the same command
regenerates `libero_pro.csv`/`pro_sr.csv` and re-uploads the artifact once the evals exist.

**Episode pairing across old and new runs.** `pro_sr.py`'s matched arm pairs on
(modality combo, task, episode index), so the old `libero_orig.csv` has to have been
written with the same `seed` and a compatible `n_eval` — `seed: 0` and `n_eval: 20` are
the defaults in both `eval_libero.yaml` and `eval_libero_pro.yaml`, which is what runs on
this box used. If an older run used a smaller `n_eval`, nothing breaks: the equated subset
is the intersection of the two key sets, and `mcnemar_n` reports how many pairs actually
contributed. Check with:

```bash
python - <<'EOF'
import csv
rows = list(csv.DictReader(open("<run>/eval_logs/last/orig_libero_10/result.csv")))
print({r["base_seed"] for r in rows}, len({r["episode_idx"] for r in rows}))
EOF
```

**Checkpoints with no training run** (the HuggingFace baseline at
`/saves/checkpoints/libero_10` has no `.hydra/config.yaml`, which `pipeline` reads to
decide what to plan) — invoke the four suites directly and analyse with `pro_sr.py`:

```bash
CKPT=/saves/checkpoints/libero_10
for s in lan object swap task; do
    ./run.sh eval-pro benchmark_name=libero_10_$s train_folder=$CKPT checkpoint=$CKPT
done
for s in lan task; do
    ./run.sh eval-pro benchmark_name=libero_10_$s pro_init_states=matched \
        train_folder=$CKPT checkpoint=$CKPT
done
```

### Analysis — the equated-subset measurement

```bash
python scripts/pro_sr.py <pro result.csv>... --orig-csv libero_orig.csv --out pro_sr.csv
```

A PRO suite's raw success rate says little on its own; what says something is how it
moves relative to the **same** episodes run unperturbed. So each record's baseline is not
the whole `libero_orig.csv` — it is exactly the subset of orig rows whose
(modality combo, task, episode) keys appear on the PRO side. Pairing granularity follows
the arm: `pro_matched` pairs one-to-one per episode (same task, same init state, same
seed); `pro` collapses to one majority-vote pair per task, ties dropped, since its init
states were sampled independently. `delta` is read off those pairs on both sides, so it
compares like with like (and equals `(mcnemar_b − mcnemar_c) / mcnemar_n`), while
`success_rate` stays the raw per-episode rate.

Baseline checkpoint (`mbreuss/flower_libero_10`), spot check at small `n_eval`:

| suite | perturbation | success rate |
|---|---|---|
| `libero_10` (orig) | — | 0.80 |
| `libero_10_lan` | semantic | 0.85 |
| `libero_10_object` | object | 0.50 |
| `libero_10_task` | task redefinition | 0.10 |
| `libero_10_swap` | position | 0.00 |

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
