#!/usr/bin/env python3
"""Diagnostic (not part of the pipeline): does set_init_state() clobber the perturbed
robot qpos for a LIBERO-Plus "Robot Initial States" episode?

Theory under test: LIBERO-plus/libero/libero/envs/env_wrapper.py substitutes the robot
class to "Panda{N}" (init_qpos = original qpos + a fixed offset, see
LIBERO-plus/libero/libero/envs/robots/new_init.py), applied by robosuite's
robot.reset() -- but flower_eval_libero.py then calls env.set_init_state(initial_states[0])
where initial_states comes from get_task_init_states(), which for this category loads the
*original* (unperturbed) task's own init file. If set_init_state's full-state restore
overwrites the arm qpos, two different N's should end up with IDENTICAL post-set_init_state
arm qpos; if it doesn't (or only partially does), they should differ.

Run inside the eval-plus container (needs LIBERO_VARIANT=plus + the LIBERO-Plus assets):
  podman-compose -f scripts/podman/compose.yml run --rm -T eval-plus \
      python scripts/debug_robot_initstate_qpos.py
"""
import numpy as np

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

BENCHMARK_NAME = "libero_10"
TASK_CATEGORY = "Robot Initial States"


def find_task_indices(benchmark_instance, base_task_substr: str, n_values):
    """Task indices whose task_name matches base_task_substr and carries
    "_initstate_{N}" for one of the given N values (view params held nominal)."""
    found = {}
    for i in range(benchmark_instance.n_tasks):
        task = benchmark_instance.get_task(i)
        name = task.name
        if base_task_substr not in name or "_view_0_0_100_0_0_initstate_" not in name:
            continue
        suffix = name.split("_initstate_")[1]
        n = int(suffix.split("_")[0].split(".")[0])
        if n in n_values and n not in found:
            found[n] = i
        if len(found) == len(n_values):
            break
    return found


def arm_qpos(env) -> np.ndarray:
    robot = env.robots[0]
    return np.array(env.sim.data.qpos[robot._ref_joint_pos_indexes])


def main():
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[BENCHMARK_NAME]()

    # Two different robot-offset severity bands (N=1 -> band 1 "0.1rad", N=141 -> band 2
    # "0.2rad") on the same base task, so any qpos difference is attributable to N alone.
    n_values = [1, 141]
    task_indices = find_task_indices(benchmark_instance, "LIVING_ROOM_SCENE2", n_values)
    print(f"task indices found for N in {n_values}: {task_indices}")

    results = {}
    for n, idx in task_indices.items():
        task = benchmark_instance.get_task(idx)
        bddl_path = benchmark_instance.get_task_bddl_file_path(idx)
        print(f"\n=== N={n}: {task.name} ===")
        print(f"bddl: {bddl_path}")

        env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=128, camera_widths=128)
        env.seed(0)
        env.reset()
        post_reset = arm_qpos(env)
        print(f"arm qpos after reset() (expect Panda{n}'s perturbed init_qpos): {post_reset}")

        initial_states = benchmark_instance.get_task_init_states(idx)
        print(f"get_task_init_states() array shape: {np.asarray(initial_states).shape}")
        state0 = initial_states[0]
        env.set_init_state(state0)
        post_set = arm_qpos(env)
        print(f"arm qpos after set_init_state(row 0) (expect nominal if clobbered): {post_set}")

        results[n] = {"post_reset": post_reset, "post_set": post_set}
        env.close()

    n1, n2 = n_values
    print("\n=== Comparison ===")
    print(f"post_reset delta norm (N={n1} vs N={n2}): "
          f"{np.linalg.norm(results[n1]['post_reset'] - results[n2]['post_reset']):.6f}")
    print(f"post_set   delta norm (N={n1} vs N={n2}): "
          f"{np.linalg.norm(results[n1]['post_set'] - results[n2]['post_set']):.6f}")
    print(
        "\nIf post_set delta norm ~= 0 (while post_reset delta norm is not), set_init_state()"
        " clobbers the perturbation -- the two N's converge to the same arm qpos.\n"
        "If post_set delta norm stays close to post_reset's, the perturbation survives."
    )


if __name__ == "__main__":
    main()
