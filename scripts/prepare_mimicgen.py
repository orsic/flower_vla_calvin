#!/usr/bin/env python3
"""Render MimicGen's state-only `core` demos into robomimic-format hdf5s with image
observations, matching LIBERO's hdf5 layout closely enough that the repo's vendored
robomimic SequenceDataset (flower/datasets/robomimic_dataset.py, already used for
LIBERO) reads either unchanged -- see flower.datasets.mimicgen_data_module.

The actual rendering is robomimic's own robomimic.scripts.dataset_states_to_obs; this
script is just a per-dataset driver: locate the source hdf5, run the converter, then
delete the source (26 raw core datasets are ~95 GB -- this keeps peak extra disk to
about one dataset's worth at a time). Pass KEEP_MIMICGEN_SOURCE=1 to keep sources.

dataset_states_to_obs.py itself never imports `mimicgen`, so it can't construct a
MimicGen env on its own (the env classes only register with robosuite on import) --
this script imports mimicgen itself, then runs the converter's __main__ in the same
process via runpy, rather than a plain `python -m robomimic.scripts.dataset_states_to_obs`.

Cameras render at CALVIN's native sizes (CAMERA_SIZES: 200x200 static, 84x84 wrist), part
of the CALVIN training recipe MimicGen follows. dataset_states_to_obs only takes one
size for all cameras, so the embedded script patches robomimic to pass per-camera sizes
(robosuite accepts a list per camera). next_obs is not written: training never reads it.
An existing output is re-rendered unless its recorded camera sizes match CAMERA_SIZES.

Usage:
  python scripts/prepare_mimicgen.py [dataset|all] [-j N] [--n-demo N]
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py

sys.path.insert(0, str(Path(__file__).parents[1]))
from flower.datasets import mimicgen_tasks  # noqa: E402

DEFAULT_DATA_DIR = "/mimicgen_hdf5"
DEFAULT_N_DEMO = 100
# CALVIN's native camera sizes (static 200x200, gripper 84x84) -- see module docstring.
CAMERA_SIZES = {"agentview": 200, "robot0_eye_in_hand": 84}

# Runs before dataset_states_to_obs's __main__ (same process, see module docstring):
# swap its single --camera_height/--camera_width for a per-camera list when creating the
# env, and in the per-demo camera_info intrinsics it records alongside.
_PER_CAMERA_PATCH = """
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_robosuite import EnvRobosuite
_create_env = EnvUtils.create_env_for_data_processing
def _create_env_per_camera(*args, **kwargs):
    kwargs["camera_height"] = [CAMERA_SIZES[c] for c in kwargs["camera_names"]]
    kwargs["camera_width"] = [CAMERA_SIZES[c] for c in kwargs["camera_names"]]
    return _create_env(*args, **kwargs)
EnvUtils.create_env_for_data_processing = _create_env_per_camera
_intrinsics = EnvRobosuite.get_camera_intrinsic_matrix
def _intrinsics_per_camera(self, camera_name, camera_height, camera_width):
    return _intrinsics(self, camera_name, CAMERA_SIZES[camera_name], CAMERA_SIZES[camera_name])
EnvRobosuite.get_camera_intrinsic_matrix = _intrinsics_per_camera
"""


def rendered_camera_sizes(path: str) -> Optional[Dict[str, Tuple[int, int]]]:
    """camera -> (height, width) recorded in a rendered hdf5's env_args, or None if the
    file isn't a readable rendered dataset."""
    try:
        with h5py.File(path, "r") as f:
            env_kwargs = json.loads(f["data"].attrs["env_args"])["env_kwargs"]
    except (OSError, KeyError):
        return None
    names = env_kwargs["camera_names"]
    heights, widths = env_kwargs["camera_heights"], env_kwargs["camera_widths"]
    if isinstance(heights, int):
        heights = [heights] * len(names)
    if isinstance(widths, int):
        widths = [widths] * len(names)
    return {name: (h, w) for name, h, w in zip(names, heights, widths)}


def render_one(dataset: str, data_dir: str, n_demo: int) -> None:
    output_path = os.path.join(data_dir, f"{dataset}.hdf5")
    if os.path.exists(output_path):
        wanted = {name: (size, size) for name, size in CAMERA_SIZES.items()}
        if rendered_camera_sizes(output_path) == wanted:
            print(f"[prepare-mimicgen] {dataset}: {output_path} already up to date, skipping")
            return
        print(f"[prepare-mimicgen] {dataset}: {output_path} rendered at other camera sizes, re-rendering")

    # `./run.sh download-mimicgen` mirrors amandlek/mimicgen_datasets' own layout
    # (--include "core/*"), so huggingface-cli preserves the "core/" prefix under
    # source/ instead of flattening it.
    source_path = os.path.join(data_dir, "source", "core", f"{dataset}.hdf5")
    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"{source_path} not found -- run ./run.sh download-mimicgen {dataset} first"
        )

    # Render to a temp name and rename on success: a render killed midway would otherwise
    # leave a truncated file at output_path, which the exists-check above skips forever.
    tmp_path = f"{output_path}.tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print(f"[prepare-mimicgen] {dataset}: rendering {n_demo} demos -> {output_path}")
    argv = [
        "dataset_states_to_obs.py",
        "--dataset", source_path,
        "--output_name", tmp_path,
        "--n", str(n_demo),
        "--done_mode", "2",
        "--camera_names", *CAMERA_SIZES,
        # Overridden per camera by _PER_CAMERA_PATCH; argparse still wants a value.
        "--camera_height", str(CAMERA_SIZES["agentview"]),
        "--camera_width", str(CAMERA_SIZES["agentview"]),
        "--exclude-next-obs",
    ]
    script = (
        "import sys, runpy\n"
        "import mimicgen  # noqa: F401 -- registers MimicGen's robosuite env classes\n"
        f"CAMERA_SIZES = {json.dumps(CAMERA_SIZES)}\n"
        + _PER_CAMERA_PATCH
        + f"sys.argv = {json.dumps(argv)}\n"
        "runpy.run_module('robomimic.scripts.dataset_states_to_obs', run_name='__main__')\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
    os.replace(tmp_path, output_path)

    if not os.environ.get("KEEP_MIMICGEN_SOURCE"):
        os.remove(source_path)
        print(f"[prepare-mimicgen] {dataset}: removed source {source_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", default="all", help="dataset name, or 'all' (default)")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="datasets to render concurrently")
    parser.add_argument("--n-demo", type=int, default=DEFAULT_N_DEMO)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = list(mimicgen_tasks.CORE_DATASETS)
    else:
        if args.dataset not in mimicgen_tasks.CORE_DATASETS:
            raise ValueError(
                f"unknown dataset {args.dataset!r} -- expected one of "
                f"{mimicgen_tasks.CORE_DATASETS} or 'all'"
            )
        datasets = [args.dataset]

    if args.jobs <= 1:
        for dataset in datasets:
            render_one(dataset, args.data_dir, args.n_demo)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(render_one, d, args.data_dir, args.n_demo): d for d in datasets}
        for future in concurrent.futures.as_completed(futures):
            dataset = futures[future]
            future.result()  # re-raise any exception (tagged with which dataset by the traceback)
            print(f"[prepare-mimicgen] {dataset}: done")


if __name__ == "__main__":
    main()
