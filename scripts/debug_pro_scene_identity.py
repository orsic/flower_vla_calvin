#!/usr/bin/env python3
"""Diagnostic (not part of the pipeline): does a LIBERO-PRO suite compile to the same
MuJoCo scene as the original libero_10 task it was derived from?

This is the empirical backing for conf/eval_libero_pro.yaml's `pro_init_states=matched`
arm. Diffing the published bddl files says the semantic (_lan) and task (_task)
perturbations touch only (:language), (:goal) and (:obj_of_interest) -- declarations the
simulator never sees -- so the original task's .pruned_init states should load into the
perturbed env unchanged, making every PRO episode an exact counterfactual of its orig
baseline episode. "Should" is the part this script checks: it builds the original env and
each perturbed env for every libero_10 task and compares the flattened sim-state width
(1 + nq + nv, what set_init_state consumes) plus the joint and body name ordering that
width is laid out in.

Expected: _lan, _task and _swap identical to the original (swap moves objects, it does
not change which objects exist); _object differs in body/joint names (yellow_moka_pot_1
for moka_pot_1) and is excluded from matching regardless -- a pose transplanted onto a
differently-sized mesh can penetrate the table.

A mismatch on _lan or _task means EvaluateLibero._task_init_states' shape guard would
fire at eval time, and matched mode must be switched off for that suite before any
numbers are produced.

Run inside the eval-pro container (needs LIBERO_VARIANT=pro + ./run.sh download-pro):
  podman-compose -f scripts/podman/compose.yml run --rm -T eval-pro \
      python scripts/debug_pro_scene_identity.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from flower.evaluation.libero_tasks import PRO_SUITE_TAGS

BASE_SUITE = "libero_10"


def scene_signature(bddl_path: str):
    """(state width, joint names, body names) for one bddl -- everything set_init_state's
    flattened array is laid out by."""
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=84, camera_widths=84)
    try:
        sim = env.env.sim
        return (
            len(sim.get_state().flatten()),
            tuple(sim.model.joint_names),
            tuple(sim.model.body_names),
        )
    finally:
        env.close()


def main() -> None:
    if os.environ.get("LIBERO_VARIANT") != "pro":
        print(f"LIBERO_VARIANT={os.environ.get('LIBERO_VARIANT')!r}, expected 'pro'", file=sys.stderr)
        return

    bddl_folder = get_libero_path("bddl_files")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_names = benchmark_dict[BASE_SUITE]().get_task_names()

    mismatches = 0
    for task in task_names:
        base = scene_signature(os.path.join(bddl_folder, BASE_SUITE, task + ".bddl"))
        print(f"\n{task}\n  {BASE_SUITE:<24} width={base[0]}")
        for tag in PRO_SUITE_TAGS:
            suite = f"{BASE_SUITE}_{tag}"
            path = os.path.join(bddl_folder, suite, task + ".bddl")
            if not os.path.exists(path):
                print(f"  {suite:<24} MISSING ({path})")
                continue
            sig = scene_signature(path)
            same = "identical" if sig == base else "DIFFERS"
            detail = ""
            if sig != base:
                mismatches += 1
                detail = (
                    f" (width {sig[0]} vs {base[0]};"
                    f" joints {'same' if sig[1] == base[1] else 'differ'};"
                    f" bodies {'same' if sig[2] == base[2] else 'differ'})"
                )
            print(f"  {suite:<24} width={sig[0]} {same}{detail}")

    print(f"\n{mismatches} suite/task pairs differ from the original scene.")


if __name__ == "__main__":
    main()
