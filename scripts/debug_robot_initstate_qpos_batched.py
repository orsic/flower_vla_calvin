#!/usr/bin/env python3
"""Same diagnostic as debug_robot_initstate_qpos.py, but through the REAL batched
SubprocVectorEnv path (flower/evaluation/libero_venv.py's make_libero_venv, spawn
start method) that flower_eval_libero.py's evaluate_work_list actually uses -- to rule
out the isolated single-env test missing something process/batch-specific.

Replicates evaluate_work_list's exact sequence for two Robot Initial States task
variants of the SAME base task (different N/severity band), packed into one 2-slot
cross-task batch:
  1. build both envs via make_libero_venv
  2. env.reset()                      -> dump full sim state per slot
  3. env.set_init_state(states, ids)  -> dump full sim state per slot again

Run inside the eval-plus container:
  podman-compose -f scripts/podman/compose.yml run --rm -T eval-plus \
      python scripts/debug_robot_initstate_qpos_batched.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from flower.evaluation.libero_venv import make_libero_venv

BENCHMARK_NAME = "libero_10"
N_VALUES = [1, 141]
BASE_TASK_SUBSTR = "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce"


def find_task_indices(benchmark_instance, base_task_substr: str, n_values):
    found = {}
    for i in range(benchmark_instance.n_tasks):
        name = benchmark_instance.get_task(i).name
        if base_task_substr not in name or "_view_0_0_100_0_0_initstate_" not in name:
            continue
        n = int(name.split("_initstate_")[1].split("_")[0].split(".")[0])
        if n in n_values and n not in found:
            found[n] = i
        if len(found) == len(n_values):
            break
    return found


def main():
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[BENCHMARK_NAME]()

    task_indices = find_task_indices(benchmark_instance, BASE_TASK_SUBSTR, N_VALUES)
    print(f"task indices found for N in {N_VALUES}: {task_indices}")
    idxs = [task_indices[n] for n in N_VALUES]

    metas = []
    for idx in idxs:
        task = benchmark_instance.get_task(idx)
        bddl_path = benchmark_instance.get_task_bddl_file_path(idx)
        initial_states = benchmark_instance.get_task_init_states(idx)
        metas.append({"task": task, "bddl_path": bddl_path, "initial_states": initial_states})
        print(f"N via task_name: {task.name} -> bddl {bddl_path}")

    env = make_libero_venv(
        [
            (lambda args={"bddl_file_name": m["bddl_path"], "camera_heights": 128, "camera_widths": 128}: OffScreenRenderEnv(**args))
            for m in metas
        ],
        start_method="spawn",
    )
    env.reset()
    post_reset = np.stack(env.get_sim_state())
    print(f"\npost_reset full-state shapes: {[s.shape for s in post_reset]}")

    state_idxs = [0, 0]  # ep=0 for both, matching n_eval=1 -> ep%n_states
    env.set_init_state([m["initial_states"][state_idxs[k]] for k, m in enumerate(metas)], id=[0, 1])
    post_set = np.stack(env.get_sim_state())

    print(f"\npost_reset delta norm (N={N_VALUES[0]} vs N={N_VALUES[1]}): "
          f"{np.linalg.norm(post_reset[0] - post_reset[1]):.6f}")
    print(f"post_set   delta norm (N={N_VALUES[0]} vs N={N_VALUES[1]}): "
          f"{np.linalg.norm(post_set[0] - post_set[1]):.6f}")

    env.close()


if __name__ == "__main__":
    main()
