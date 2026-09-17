#!/usr/bin/env python3
"""Real-checkpoint rollout check (not part of the pipeline): with a loaded model, does
success rate actually vary across "Robot Initial States" severity bands for ONE base
task, whose every N-variant we've confirmed (debug_robot_initstate_qpos*.py) starts from
a bit-identical physical sim state after set_init_state()?

Monkeypatches flower_eval_libero.select_task_indices to restrict the run to every
N-variant of one base task (ignoring the task_category override), then calls
flower_eval_libero.main() directly with a manually composed cfg (cfg_passthrough --
see hydra.main's decorator: this bypasses hydra's CLI/argv config resolution, which
misbehaves for a script other than flower_eval_libero.py itself since config_path is
resolved relative to sys.modules['__main__'] in this hydra version, not the declaring
module) -- same model loading, same evaluate_policy/evaluate_work_list code path
production evals use.

Usage:
  podman-compose -f scripts/podman/compose.yml run --rm -T eval-plus \
      python scripts/debug_robot_initstate_severity_rollout.py \
      <train_folder> <checkpoint> [<csv_dir>]
"""
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parents[1]))
import flower.evaluation.flower_eval_libero as fel  # noqa: E402

BASE_TASK_SUBSTR = "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce"


def _select_one_base_task(benchmark_instance, task_category):
    task_names = benchmark_instance.get_task_names()
    indices = [
        i for i, name in enumerate(task_names)
        if BASE_TASK_SUBSTR in name and "_view_0_0_100_0_0_initstate_" in name
    ]
    print(
        f"[debug] restricting eval to {len(indices)} Robot Initial States N-variants "
        f"of {BASE_TASK_SUBSTR!r} (task_category override {task_category!r} ignored)"
    )
    return indices


fel.select_task_indices = _select_one_base_task


def main():
    train_folder, checkpoint = sys.argv[1], sys.argv[2]
    csv_dir = sys.argv[3] if len(sys.argv) > 3 else "/saves/tmp/debug_robot_initstate"

    cfg = OmegaConf.load(str(Path(__file__).parents[1] / "conf" / "eval_libero_plus.yaml"))
    cfg.train_folder = train_folder
    cfg.checkpoint = checkpoint
    cfg.csv_dir = csv_dir
    cfg.task_category = "Robot Initial States"  # ignored by the monkeypatch above

    fel.main(cfg_passthrough=cfg)


if __name__ == "__main__":
    main()
