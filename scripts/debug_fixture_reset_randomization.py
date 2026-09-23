#!/usr/bin/env python3
"""Diagnostic (not part of the pipeline): does set_init_state() restore fixture
placement, or does an episode's fixture layout stay silently random across resets?

Theory under test (see the plan this lands with): bddl_base_domain.BddlBaseDomain
._reset_internal samples every fixture's placement on each reset() (from the
process-global numpy RNG) and writes it into the MuJoCo *model*
(sim.model.body_pos/body_quat). env.set_init_state() -> set_state() ->
sim.set_state_from_flattened() restores only `time`, `qpos`, `qvel` -- MuJoCo *data*,
never touching that model-level placement. The eval harness never seeds this RNG (see
flower.evaluation.libero_venv's worker, which explicitly entropy-reseeds numpy on every
rebuild), so a fixture's placement is silently random every episode, even though
set_init_state() makes qpos itself look identical.

For libero_10, five of the ten tasks sample a fixture (the stove/cabinet/rack/
microwave/caddy); the four LIVING_ROOM_* tasks have none. Comparing the two groups
isolates the effect: only the fixture-bearing tasks should show unseeded variation.

Also dumps the "agentview" camera pose, to quantify LIBERO-Plus's _setup_camera
rewrite (rotate_around_z(..., degrees=0) then round(x, 4)), which shifts it by a small
constant amount relative to original LIBERO's hard-coded pose.

Run inside the eval-plus container (needs LIBERO_VARIANT=plus + the LIBERO-Plus
assets; set LIBERO_VARIANT=orig / run inside the eval container for the original-
LIBERO side of the comparison):
  podman-compose -f scripts/podman/compose.yml run --rm -T eval-plus \
      python scripts/debug_fixture_reset_randomization.py
  podman-compose -f scripts/podman/compose.yml run --rm -T eval \
      python scripts/debug_fixture_reset_randomization.py
"""
import os

import numpy as np
import torch

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

BENCHMARK_NAME = "libero_10"
N_RESETS = 20
SEED = 1234

# A representative task from each group: fixture-bearing (samples a movable fixture
# on every reset) vs. fixture-free (only movable *objects*, which set_init_state's
# qpos restore does cover).
FIXTURE_TASKS = [
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
]
NO_FIXTURE_TASKS = [
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
]


def fixture_poses(env) -> dict:
    """{fixture_name: (body_pos copy, body_quat copy)} for every sampled fixture."""
    poses = {}
    for name, obj in env.env.fixtures_dict.items():
        body_id = env.sim.model.body_name2id(obj.root_body)
        poses[name] = (
            np.array(env.sim.model.body_pos[body_id]),
            np.array(env.sim.model.body_quat[body_id]),
        )
    return poses


def camera_pose(env, cam_name: str = "agentview") -> tuple:
    cam_id = env.sim.model.camera_name2id(cam_name)
    return (
        np.array(env.sim.model.cam_pos[cam_id]),
        np.array(env.sim.model.cam_quat[cam_id]),
    )


def run_task(task_name: str, seeded: bool) -> None:
    # Built directly from the task name rather than through a benchmark instance's
    # task index: under LIBERO_VARIANT=plus, every task name carries a perturbation
    # suffix (there is no bare "KITCHEN_SCENE3_..." task in that registry), while the
    # bddl/init files these paths point at are byte-identical between LIBERO/ and
    # LIBERO-plus/ for original libero_10 tasks -- see the plan this script lands
    # with. This also lets one run diff orig-vs-plus without needing two lookups.
    bddl_path = os.path.join(get_libero_path("bddl_files"), BENCHMARK_NAME, f"{task_name}.bddl")
    init_states_path = os.path.join(
        get_libero_path("init_states"), BENCHMARK_NAME, f"{task_name}.pruned_init"
    )
    initial_states = torch.load(init_states_path)
    state0 = initial_states[0]

    fixture_names = None
    per_fixture_poses = {}
    qpos_hashes = set()
    cam_pose = None

    for _ in range(N_RESETS):
        # A fresh env per reset, not one env reset() N times: this matches how
        # evaluate_task/evaluate_work_list actually use an env -- each worker's env is
        # freshly constructed (env_fn(), a clean MJCF parse) exactly once per (task,
        # episode) and reset() exactly once before its one episode, never reset() twice
        # on the same instance. Reusing one instance across resets does not reproduce
        # that: LIBERO uses hard_reset=False (reset() doesn't reload the MJCF model,
        # only MuJoCo *data*), so a fixture's randomized body_pos from a PRIOR reset
        # persists as read-back "reference" state and can perturb a movable object's
        # rejection-sampling retry count on the NEXT reset, adding a second few-RNG
        # -draws-of-drift on top of an unseeded run's real desync. Confirmed on
        # KITCHEN_SCENE4 specifically (2 fixtures + 2 movable objects, more crowded
        # regions): reused-instance resets kept desyncing even seeded; fresh-per-reset
        # instances did not.
        env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=128, camera_widths=128)
        if fixture_names is None:
            fixture_names = sorted(env.env.fixtures_dict.keys())
            per_fixture_poses = {name: [] for name in fixture_names}
        if seeded:
            env.seed(SEED)
        env.reset()
        env.set_init_state(state0)
        for name, (pos, quat) in fixture_poses(env).items():
            per_fixture_poses[name].append((tuple(np.round(pos, 6)), tuple(np.round(quat, 6))))
        qpos_hashes.add(hash(np.round(env.sim.data.qpos, 6).tobytes()))
        cam_pose = camera_pose(env)
        env.close()

    print(f"\n=== {task_name} (seeded={seeded}) ===")
    print(f"fixtures: {fixture_names or '(none)'}")
    for name, poses in per_fixture_poses.items():
        distinct = set(poses)
        spans = [p[0] for p in distinct]
        max_span = 0.0
        if len(spans) > 1:
            arr = np.array(spans)
            max_span = float(np.max(np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)))
        print(f"  fixture {name!r}: {len(distinct)} distinct pose(s) over {N_RESETS} resets"
              f", max pairwise pos span={max_span:.6f} m")
    print(f"  post-set_init_state qpos hash: {len(qpos_hashes)} distinct value(s) over {N_RESETS} resets"
          " (expected 1 in both regimes -- set_init_state does restore qpos)")
    print(f"  agentview cam_pos={cam_pose[0]}, cam_quat={cam_pose[1]}")


def main():
    for seeded in (False, True):
        print(f"\n{'=' * 70}\nSEEDED = {seeded}\n{'=' * 70}")
        for task_name in FIXTURE_TASKS + NO_FIXTURE_TASKS:
            run_task(task_name, seeded)


if __name__ == "__main__":
    main()
