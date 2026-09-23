#!/usr/bin/env python3
"""Diagnostic (not part of the pipeline): if set_init_state() clobbers a LIBERO-Plus
"Robot Initial States" episode's perturbed qpos (confirmed by
debug_robot_initstate_qpos.py), why does that category still change success rate on an
otherwise-deterministic task?

Theory under test (see the plan this lands with): robosuite's Robot.reset() writes the
robot class's (possibly perturbed) init_qpos into sim.data.qpos, THEN calls
_load_controller(), which builds a fresh OSC whose __init__ captures
self.initial_joint = self.joint_pos (base_controller.py) from that just-written qpos.
env.set_init_state() -> regenerate_obs_from_state() restores qpos/qvel via
set_state_from_flattened() but never rebuilds the controller -- so initial_joint, used
as the OSC's nullspace torque reference on every subsequent control step
(controllers/osc.py), keeps the perturbed value even though qpos itself is back to
nominal. The perturbation survives as a persistent control bias, not a start-pose
offset -- so it is not "no physical effect", it's a different mechanism than the one
LIBERO-Plus's own naming implies.

Also rules out the base/mount pose as the mechanism: LIBERO-Plus selects Mounted vs.
OnTheGround per *scene*, not per N, so it cannot explain an N-dependent effect.

Run inside the eval-plus container (needs LIBERO_VARIANT=plus + the LIBERO-Plus
assets):
  podman-compose -f scripts/podman/compose.yml run --rm -T eval-plus \
      python scripts/debug_robot_initstate_controller.py
"""
import numpy as np

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

BENCHMARK_NAME = "libero_10"
N_VALUES = [0, 1, 141, 341]  # 0 = unperturbed Panda; 1/141/341 sample the 0.1/0.2/0.5 rad bands
N_ZERO_ACTION_STEPS = 30


def find_task_indices(benchmark_instance, base_task_substr: str, n_values):
    """Task indices whose task_name matches base_task_substr and carries
    "_initstate_{N}" for one of the given N values (view params held nominal). N=0 is
    the un-suffixed original task itself (no perturbation applied)."""
    found = {}
    for i in range(benchmark_instance.n_tasks):
        task = benchmark_instance.get_task(i)
        name = task.name
        if 0 in n_values and name == base_task_substr:
            found[0] = i
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

    task_indices = find_task_indices(benchmark_instance, "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket", N_VALUES)
    print(f"task indices found for N in {N_VALUES}: {task_indices}")

    trajectories = {}
    for n, idx in task_indices.items():
        task = benchmark_instance.get_task(idx)
        bddl_path = benchmark_instance.get_task_bddl_file_path(idx)
        print(f"\n=== N={n}: {task.name} ===")

        env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=128, camera_widths=128)
        env.seed(0)
        env.reset()

        robot = env.robots[0]
        controller = robot.controller
        print(f"robot_model class: {robot.robot_model.__class__.__name__}")
        print(f"base_pos: {robot.base_pos}")
        base_body_id = env.sim.model.body_name2id("robot0_base")
        print(f"sim.model.body_pos[robot0_base]: {env.sim.model.body_pos[base_body_id]}")
        print(f"arm qpos after reset(): {arm_qpos(env)}")
        print(f"controller.initial_joint (post-reset): {controller.initial_joint}")
        print(f"controller.initial_ee_pos (post-reset): {controller.initial_ee_pos}")

        initial_states = benchmark_instance.get_task_init_states(idx)
        env.set_init_state(initial_states[0])

        print(f"arm qpos after set_init_state(row 0): {arm_qpos(env)}")
        print(f"controller.initial_joint (post-set_init_state): {controller.initial_joint}")
        print(f"controller.initial_ee_pos (post-set_init_state): {controller.initial_ee_pos}")

        traj = [arm_qpos(env).copy()]
        zero_action = np.zeros(env.action_dim if hasattr(env, "action_dim") else 7)
        for _ in range(N_ZERO_ACTION_STEPS):
            env.step(zero_action)
            traj.append(arm_qpos(env).copy())
        trajectories[n] = np.stack(traj)

        env.close()

    ns = sorted(trajectories)
    baseline_n = ns[0]
    print(f"\n=== Zero-action trajectory divergence from N={baseline_n} ===")
    for n in ns[1:]:
        delta = np.linalg.norm(trajectories[n] - trajectories[baseline_n], axis=-1)
        print(f"N={n}: final-step qpos delta norm = {delta[-1]:.6f} (step-0 = {delta[0]:.6f})")

    print(
        "\nIf controller.initial_joint differs across N (post-set_init_state) and the "
        "zero-action trajectories diverge despite identical qpos/base_pos, the "
        "perturbation survives as a controller bias rather than a start-pose offset."
    )


if __name__ == "__main__":
    main()
