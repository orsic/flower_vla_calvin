"""
Smoke test for the FlowerVLA LIBERO container environment.

Run inside the container after ./run.sh download:
    ./run.sh smoke

Tests (in order):
  1. CUDA is available and on GPU 0
  2. pyhash, libero, and transformers import correctly
  3. LIBERO-10 checkpoint loads from /saves/checkpoints/libero_10
  4. One MuJoCo OffScreenRenderEnv can be created and reset (tests GL/EGL stack)

On success prints a summary and exits 0. Any failure raises an AssertionError.
"""
import sys
from pathlib import Path

# ---- 1. CUDA ----------------------------------------------------------------
import torch

assert torch.cuda.is_available(), (
    "CUDA not available. Check that the container was started with "
    "--device nvidia.com/gpu=0 and that the CDI device file exists."
)
print(f"[1/4] CUDA OK  — {torch.cuda.get_device_name(0)}")

# ---- 2. Key imports ---------------------------------------------------------
import pyhash  # noqa: F401

print("[2/4] pyhash OK")

import libero  # noqa: F401  (also writes ~/.libero/config.yaml if missing)

print("      libero OK")

import transformers

print(f"      transformers OK  (v{transformers.__version__})")

# ---- 3. Checkpoint load -----------------------------------------------------
CKPT_DIR = Path("/saves/checkpoints/libero_10")
assert CKPT_DIR.exists(), (
    f"Checkpoint not found at {CKPT_DIR}. Run: ./run.sh download"
)
assert (CKPT_DIR / "model.safetensors").exists(), (
    f"model.safetensors missing in {CKPT_DIR}"
)
assert (CKPT_DIR / "config.yaml").exists(), (
    f"config.yaml missing in {CKPT_DIR}"
)

sys.path.insert(0, "/workspace")
from flower.evaluation.utils import get_default_mode_and_env

# eval_libero.yaml supplies this override so that the LIBERO checkpoint config
# (which lacks lang_folder under lang_dataset) satisfies get_default_mode_and_env.
eval_cfg_overwrite = {
    "datamodule": {"datasets": {"lang_dataset": {"lang_folder": "lang_annotations"}}},
    "model": {"num_sampling_steps": 4},
}
model, _, dm, _ = get_default_mode_and_env(
    str(CKPT_DIR),
    "/workspace",
    str(CKPT_DIR),
    env=42,
    lang_embeddings=None,
    device_id=0,
    prep_dm_and_deps=False,
    eval_cfg_overwrite=eval_cfg_overwrite,
)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"[3/4] Model load OK  — {n_params:.0f}M parameters")

# ---- 4. MuJoCo environment --------------------------------------------------
from libero.libero import get_libero_path
from libero.libero import benchmark as libero_benchmark
from libero.libero.envs import OffScreenRenderEnv

bddl_folder = get_libero_path("bddl_files")
bench = libero_benchmark.get_benchmark_dict()["libero_10"]()
task_0 = bench.get_task(0)
bddl_path = f"{bddl_folder}/{task_0.problem_folder}/{task_0.bddl_file}"

env = OffScreenRenderEnv(
    bddl_file_name=bddl_path,
    camera_heights=224,
    camera_widths=224,
)
obs = env.reset()
assert "agentview_image" in obs, f"Unexpected obs keys: {list(obs.keys())}"
env.close()
print(f"[4/4] MuJoCo env OK  — task: {task_0.name}")

print("\nAll smoke tests passed. Ready to run: ./run.sh eval")
