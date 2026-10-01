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

sys.path.insert(0, str(Path(__file__).parents[1]))
from flower.datasets import mimicgen_tasks  # noqa: E402

DEFAULT_DATA_DIR = "/mimicgen_hdf5"
DEFAULT_N_DEMO = 100
CAMERA_NAMES = ["agentview", "robot0_eye_in_hand"]
CAMERA_SIZE = 128


def render_one(dataset: str, data_dir: str, n_demo: int) -> None:
    output_path = os.path.join(data_dir, f"{dataset}.hdf5")
    if os.path.exists(output_path):
        print(f"[prepare-mimicgen] {dataset}: {output_path} already exists, skipping")
        return

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
        "--camera_names", *CAMERA_NAMES,
        "--camera_height", str(CAMERA_SIZE),
        "--camera_width", str(CAMERA_SIZE),
    ]
    script = (
        "import sys, runpy\n"
        "import mimicgen  # noqa: F401 -- registers MimicGen's robosuite env classes\n"
        f"sys.argv = {json.dumps(argv)}\n"
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
