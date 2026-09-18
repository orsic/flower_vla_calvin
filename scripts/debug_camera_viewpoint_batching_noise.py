#!/usr/bin/env python3
"""Real-checkpoint rollout check (not part of the pipeline): now that rollout_seed is
keyed on base task name instead of batch position (see flower/evaluation/eval_records.py
and flower/models/flower.py's inference_noise/set_eval_noise_seeds), do every
Camera-Viewpoint variant of ONE base task -- confirmed physically inert to a model with
the static camera withheld, and here run WITH the static camera present so the model
can't even see the (irrelevant) view change -- actually draw the same noise and land on
the same success/steps_taken, regardless of which cross-task batch they land in?

Before this fix: rollout_seed was batch_seed(base_seed, batch_index) -- a function of
ordinal batch position only -- so different variants of the same base task, landing in
different batches, drew uncorrelated noise and were expected to disagree (see the
severity_sr.py investigation this script settles). After the fix: every variant shares
one rollout_seed (base_seed, base_task_name, episode_idx=0), so their trajectories
should be UNIFORM -- either every one succeeds or every one fails, batch position and
batch composition notwithstanding.

Monkeypatches flower_eval_libero.select_task_indices to restrict the run to every
Camera-Viewpoint variant of one base task (ignoring the task_category override), then
calls flower_eval_libero.main() directly with a manually composed cfg (cfg_passthrough
-- same rationale as scripts/debug_robot_initstate_severity_rollout.py) -- same model
loading, same evaluate_policy/evaluate_work_list code path production evals use.

Usage:
  podman-compose -f scripts/podman/compose.yml run --rm -T eval-plus \
      python scripts/debug_camera_viewpoint_batching_noise.py \
      <train_folder> <checkpoint> [<csv_dir>]
"""
import csv
import re
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parents[1]))
import flower.evaluation.flower_eval_libero as fel  # noqa: E402

BASE_TASK_SUBSTR = "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce"

# "_view_0_0_100_0_0_initstate_0" is the nominal (unperturbed) view -- excluded, it
# isn't its own category. Robot Initial States keeps this nominal view but varies
# initstate ("_initstate_51"); Sensor Noise keeps nominal view + initstate 0 but adds a
# "_noise_N" suffix. Genuine Camera Viewpoints varies h/v/s/er/ev, keeps initstate 0,
# and carries no further suffix -- anchored at the end of the string to exclude both.
_CAMERA_VIEWPOINT_RE = re.compile(r"_view_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_initstate_0$")
_NOMINAL_VIEW = ("0", "0", "100", "0", "0")


def _select_camera_viewpoint_variants(benchmark_instance, task_category):
    task_names = benchmark_instance.get_task_names()
    indices = []
    for i, name in enumerate(task_names):
        if BASE_TASK_SUBSTR not in name:
            continue
        match = _CAMERA_VIEWPOINT_RE.search(name)
        if match and match.groups() != _NOMINAL_VIEW:
            indices.append(i)
    print(
        f"[debug] restricting eval to {len(indices)} Camera Viewpoints variants of "
        f"{BASE_TASK_SUBSTR!r} (task_category override {task_category!r} ignored)"
    )
    return indices


fel.select_task_indices = _select_camera_viewpoint_variants


def main():
    train_folder, checkpoint = sys.argv[1], sys.argv[2]
    csv_dir = sys.argv[3] if len(sys.argv) > 3 else "/saves/tmp/debug_camera_viewpoint_batching_noise"

    cfg = OmegaConf.load(str(Path(__file__).parents[1] / "conf" / "eval_libero_plus.yaml"))
    cfg.train_folder = train_folder
    cfg.checkpoint = checkpoint
    cfg.csv_dir = csv_dir
    cfg.task_category = "Camera Viewpoints"  # ignored by the monkeypatch above
    # The actual scenario under investigation: static camera withheld, so a
    # physically-inert perturbation to the static view can't be observed at all.
    cfg.eval_modalities.rgb_static = False

    fel.main(cfg_passthrough=cfg)

    result_csv = Path(csv_dir) / "result.csv"
    with open(result_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    seeds = {row["rollout_seed"] for row in rows}
    successes = {row["success"] for row in rows}
    steps = {row["steps_taken"] for row in rows}
    batches = {row["eval_batch_size"] for row in rows}

    print(f"\n[debug] {len(rows)} episodes across {len(batches)} distinct batch width(s) {sorted(batches)}")
    print(f"[debug] distinct rollout_seed values: {sorted(seeds)}")
    print(f"[debug] distinct success values: {sorted(successes)}")
    print(f"[debug] distinct steps_taken values: {sorted(steps)}")

    if len(seeds) == 1:
        print("[debug] PASS: every variant shares one rollout_seed, as intended.")
    else:
        print("[debug] FAIL: rollout_seed still varies across variants of the same base task.")

    # success is the real pass/fail gate: the perturbation is confirmed inert (§ module
    # docstring), so a shared seed must produce a uniform outcome. steps_taken is
    # allowed to jitter a little -- batch width still selects Florence-2/DiT GEMM
    # kernels/reduction order (matmul non-associativity), so a shared noise draw isn't
    # a bit-identical trajectory guarantee (see README's Reproducibility paragraph).
    if len(successes) == 1:
        print(
            "[debug] PASS: every variant landed on the same success outcome -- the "
            "Camera Viewpoints perturbation is confirmed inert once noise is held fixed."
        )
        if len(steps) > 1:
            spread = max(int(s) for s in steps) - min(int(s) for s in steps)
            print(
                f"[debug]   steps_taken still spans {sorted(steps)} (range {spread}) -- "
                "expected from residual batch-shape numerics, not a concern on its own."
            )
    else:
        print(
            "[debug] FAIL: success itself differs across variants of the same base task "
            "despite a shared seed -- this should not happen if the perturbation is truly "
            "inert. Inspect which batch (eval_batch_size, position) each row came from."
        )


if __name__ == "__main__":
    main()
