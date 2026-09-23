#!/usr/bin/env python3
"""Real-checkpoint rollout check (not part of the pipeline): among LIBERO-Plus's
Language Instructions variants of ONE base task, evaluated with language withheld
(eval_modalities.language=False, so the model input is identical across variants too),
a few base tasks still show steps_taken/success varying across nominally-identical
episodes even though they have no sampled fixture (see the plan's finding F4 -- the
fixture-placement bug this repo's other diagnostics/fix cover does not apply to these
tasks: their reset() only randomizes movable *objects*, whose placement is written into
qpos/qvel, which env.set_init_state() *does* restore).

This script localizes that residual to either the environment or the policy, using a
discriminator that's free: evaluate_work_list runs 5 dummy warm-up steps before the
model is ever consulted, so the very first real observation of each batch is already
post-warmup. If that first observation already differs across two nominally-identical
episodes, the divergence is environment-side and pre-dates any model involvement. If
observations agree but the first action differs, it's policy-side (batch-shape GEMM/
kernel numerics -- see README's Reproducibility paragraph). If both agree and the
divergence appears only later, it's chaotic amplification over the rollout rather than
either single point.

Monkeypatches (mirroring debug_camera_viewpoint_batching_noise.py's approach):
  - flower_eval_libero.select_task_indices -> restrict to the base task's Language
    Instructions variants (ignoring the task_category override).
  - EvaluateLibero.process_env_obs_batch -> hash each batch's first-call obs
    (agentview_image) per slot, keyed by which (task_idx, episode) it belongs to.
  - FLOWERVLA.step_batch -> hash each batch's first-call output action per slot.

Usage:
  podman-compose -f scripts/podman/compose.yml run --rm -T eval-plus \
      python scripts/debug_language_variant_determinism.py \
      <train_folder> <checkpoint> [<csv_dir>] [<eval_batch_size>]
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parents[1]))
import flower.evaluation.flower_eval_libero as fel  # noqa: E402
from flower.models.flower import FLOWERVLA  # noqa: E402

BASE_TASK_SUBSTR = "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"


def _select_language_variants(benchmark_instance, task_category):
    task_names = benchmark_instance.get_task_names()
    indices = [
        i for i, name in enumerate(task_names)
        if BASE_TASK_SUBSTR in name and "_language_" in name
    ]
    print(
        f"[debug] restricting eval to {len(indices)} Language Instructions variants of "
        f"{BASE_TASK_SUBSTR!r} (task_category override {task_category!r} ignored)"
    )
    return indices


fel.select_task_indices = _select_language_variants

# batch_items for the batch currently in flight -- set right before seed_and_reset
# in evaluate_work_list runs, by wrapping EvaluateLibero.evaluate_work_list's own
# rollout_seed calls would require patching too much internal state; instead we key
# purely on slot position within a batch and rely on process_env_obs_batch/step_batch
# firing once per batch (guarded by _pending, cleared on model.reset()).
_pending = {"obs": False, "action": False}
_obs_hashes = []   # list of list[str], one inner list per batch
_action_hashes = []  # list of list[str], one inner list per batch

_orig_model_reset = FLOWERVLA.reset


def _reset_and_arm(self):
    _pending["obs"] = True
    _pending["action"] = True
    return _orig_model_reset(self)


FLOWERVLA.reset = _reset_and_arm

_orig_process_env_obs_batch = fel.EvaluateLibero.process_env_obs_batch


def _hashing_process_env_obs_batch(self, obs_list, lang_embed, lang_text=None):
    if _pending["obs"]:
        hashes = [
            hashlib.sha256(np.ascontiguousarray(obs["agentview_image"]).tobytes()).hexdigest()[:12]
            for obs in obs_list
        ]
        _obs_hashes.append(hashes)
        _pending["obs"] = False
        print(f"[debug] batch {len(_obs_hashes)}: first-call obs hashes = {hashes}")
    return _orig_process_env_obs_batch(self, obs_list, lang_embed, lang_text)


fel.EvaluateLibero.process_env_obs_batch = _hashing_process_env_obs_batch

_orig_step_batch = FLOWERVLA.step_batch


def _hashing_step_batch(self, obs, goal):
    action = _orig_step_batch(self, obs, goal)
    if _pending["action"]:
        hashes = [
            hashlib.sha256(np.ascontiguousarray(action[k].detach().cpu().numpy()).tobytes()).hexdigest()[:12]
            for k in range(action.shape[0])
        ]
        _action_hashes.append(hashes)
        _pending["action"] = False
        print(f"[debug] batch {len(_action_hashes)}: first-call action hashes = {hashes}")
    return action


FLOWERVLA.step_batch = _hashing_step_batch


def main():
    train_folder, checkpoint = sys.argv[1], sys.argv[2]
    csv_dir = sys.argv[3] if len(sys.argv) > 3 else "/saves/tmp/debug_language_variant_determinism"
    eval_batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else None

    cfg = OmegaConf.load(str(Path(__file__).parents[1] / "conf" / "eval_libero_plus.yaml"))
    cfg.train_folder = train_folder
    cfg.checkpoint = checkpoint
    cfg.csv_dir = csv_dir
    cfg.task_category = "Language Instructions"  # ignored by the monkeypatch above
    cfg.eval_modalities.language = False
    if eval_batch_size is not None:
        cfg.eval_batch_size = eval_batch_size

    fel.main(cfg_passthrough=cfg)

    all_obs_hashes = [h for batch in _obs_hashes for h in batch]
    all_action_hashes = [h for batch in _action_hashes for h in batch]
    print(f"\n[debug] {len(all_obs_hashes)} first-batch obs hashes across "
          f"{len(_obs_hashes)} batches: {len(set(all_obs_hashes))} distinct")
    print(f"[debug] {len(all_action_hashes)} first-batch action hashes across "
          f"{len(_action_hashes)} batches: {len(set(all_action_hashes))} distinct")

    if len(set(all_obs_hashes)) > 1:
        print(
            "[debug] VERDICT: env-side. The very first observation (post-warmup, "
            "pre-inference) already differs across nominally-identical episodes."
        )
    elif len(set(all_action_hashes)) > 1:
        print(
            "[debug] VERDICT: policy-side. Observations agree but the first action "
            "does not -- consistent with batch-shape GEMM/kernel numerics."
        )
    else:
        print(
            "[debug] VERDICT: neither the first observation nor the first action "
            "differs. Any residual steps_taken/success spread in result.csv comes "
            "from divergence later in the rollout (chaotic amplification), not from "
            "either single point checked here."
        )


if __name__ == "__main__":
    main()
