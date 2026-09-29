"""Tests for flower.evaluation.flower_eval_mimicgen.

No MuJoCo/robomimic/mimicgen involved: _create_mimicgen_env is monkeypatched out with a
picklable fake env (mirrors tests/test_batched_eval.py's _FakeEnv), run through the real
BaseVectorEnv machinery (env_start_method="dummy", sequential/in-process) so the actual
eval loop in EvaluateMimicgen.evaluate_dataset runs unmodified.
"""

import math

import numpy as np
import pytest
import torch

from flower.datasets import mimicgen_tasks
from flower.evaluation import flower_eval_mimicgen as fem
from flower.evaluation.eval_records import ALL_COLUMNS, merge_result_csv, read_csv


def test_module_imports_without_mimicgen_or_newer_robosuite():
    """Importing this module (done implicitly by every other test here) must not
    require mimicgen/robomimic-with-MimicGen's-robosuite-pin to be installed -- those
    are only imported lazily inside _create_mimicgen_env. This test documents that
    contract explicitly; if it ever needs a real import guard to pass, the module
    broke the contract."""
    assert hasattr(fem, "EvaluateMimicgen")
    assert hasattr(fem, "_create_mimicgen_env")


def test_partition_datasets_round_robin():
    datasets = [f"d{i}" for i in range(7)]
    splits = fem.partition_datasets(datasets, n_gpus=3)
    assert splits == [["d0", "d3", "d6"], ["d1", "d4"], ["d2", "d5"]]
    # every dataset appears exactly once across the splits
    assert sorted(sum(splits, [])) == sorted(datasets)


class _FakeModel:
    """Minimal FLOWERVLA stand-in -- see tests/test_batched_eval.py's _MinimalFlower,
    extended with the attributes EvaluateMimicgen.__init__/evaluate_dataset touch."""

    multistep = 10
    num_sampling_steps = 4

    def __init__(self, B_max=10, use_proprio=False, success_at_step=None):
        self.use_proprio = use_proprio
        self.eval_modality_mask = None
        self.eval_proprio_mask = None
        self.rollout_step_counter = 0
        self.pred_action_seq = None
        self._noise_seeds = None
        self._success_at_step = success_at_step
        self._global_step = 0

    def set_eval_noise_seeds(self, seeds):
        self._noise_seeds = list(seeds)

    def reset(self):
        self.rollout_step_counter = 0
        self.pred_action_seq = None
        self._global_step = 0

    def step_batch(self, obs, goal):
        if self.rollout_step_counter % self.multistep == 0:
            B = len(self._noise_seeds)
            self.pred_action_seq = torch.zeros(B, self.multistep, 7)
        action = self.pred_action_seq[:, self.rollout_step_counter]
        self.rollout_step_counter = (self.rollout_step_counter + 1) % self.multistep
        self._global_step += 1
        return action


class _FakeMimicgenEnv:
    """Picklable stand-in for the robomimic-backed env _create_mimicgen_env normally
    builds: reset/step/seed/close, success flips on step >= success_at_step."""

    def __init__(self, label, success_at_step=None):
        self.label = label
        self.success_at_step = success_at_step
        self._step = 0
        self.seeded = None
        self.closed = False

    def reset(self):
        self._step = 0
        return self._obs()

    def _obs(self):
        return {
            "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_joint_pos": np.zeros(7, dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        }

    def step(self, action):
        self._step += 1
        done = self.success_at_step is not None and self._step >= self.success_at_step
        return self._obs(), 0.0, bool(done), {}

    def seed(self, seed):
        self.seeded = seed

    def close(self):
        self.closed = True


def _make_fake_transforms():
    def identity(x):
        return x.float()

    return {"val": {"rgb_static": [identity], "rgb_gripper": [identity]}}


def _build_evaluator(monkeypatch, success_at_step, n_eval, eval_batch_size, model=None):
    def fake_create_env(dataset_path, img_h, img_w):
        return _FakeMimicgenEnv(dataset_path, success_at_step=success_at_step)

    monkeypatch.setattr(fem, "_create_mimicgen_env", fake_create_env)

    evaluator = fem.EvaluateMimicgen(
        model=model or _FakeModel(),
        transforms=_make_fake_transforms(),
        log_dir="/tmp",
        data_dir="/fake/mimicgen_hdf5",
        datasets=["square_d0"],
        n_eval=n_eval,
        eval_batch_size=eval_batch_size,
        checkpoint="/fake/last.ckpt",
        base_seed=0,
        env_start_method="dummy",
    )
    return evaluator


def test_evaluate_dataset_all_episodes_succeed_early(monkeypatch):
    # evaluate_dataset steps 5 dummy warmup actions before the real rollout loop (see
    # its `for _ in range(5): env.step(dummy, ...)`), which count toward _FakeMimicgenEnv's
    # step counter -- so success_at_step=8 fires on the rollout loop's 3rd real step.
    model = _FakeModel()
    evaluator = _build_evaluator(monkeypatch, success_at_step=8, n_eval=4, eval_batch_size=2, model=model)

    rows = evaluator.evaluate_dataset(model, "square_d0", idx=0, store_video=0)

    assert len(rows) == 4
    for row in rows:
        assert row["success"] == 1
        assert row["steps_taken"] == 3
        assert row["task_name"] == "square_d0"
        assert row["task_category"] == "square"
        assert row["libero_variant"] == "mimicgen"
        assert row["suite"] == "core"
        assert row["language"] == mimicgen_tasks.language("square_d0")
        assert row["init_state_idx"] == -1
        assert row["use_rgb_static"] == 1
        assert row["use_proprio"] == 0  # model.use_proprio=False


def test_evaluate_dataset_runs_to_max_steps_when_never_successful(monkeypatch):
    model = _FakeModel()
    evaluator = _build_evaluator(monkeypatch, success_at_step=None, n_eval=1, eval_batch_size=1, model=model)

    rows = evaluator.evaluate_dataset(model, "square_d0", idx=0, store_video=0)

    assert len(rows) == 1
    assert rows[0]["success"] == 0
    assert rows[0]["steps_taken"] == mimicgen_tasks.max_steps("square_d0")
    assert rows[0]["max_steps"] == mimicgen_tasks.max_steps("square_d0")


def test_evaluate_dataset_episode_idx_and_rollout_seed_unique_per_episode(monkeypatch):
    model = _FakeModel()
    evaluator = _build_evaluator(monkeypatch, success_at_step=2, n_eval=5, eval_batch_size=2, model=model)

    rows = evaluator.evaluate_dataset(model, "square_d0", idx=0, store_video=0)

    assert sorted(row["episode_idx"] for row in rows) == [0, 1, 2, 3, 4]
    assert len(set(row["rollout_seed"] for row in rows)) == 5  # every episode got its own seed


def test_evaluate_policy_aggregates_across_datasets(monkeypatch):
    model = _FakeModel()

    def fake_create_env(dataset_path, img_h, img_w):
        return _FakeMimicgenEnv(dataset_path, success_at_step=1)

    monkeypatch.setattr(fem, "_create_mimicgen_env", fake_create_env)

    evaluator = fem.EvaluateMimicgen(
        model=model,
        transforms=_make_fake_transforms(),
        log_dir="/tmp",
        data_dir="/fake/mimicgen_hdf5",
        datasets=["square_d0", "threading_d1"],
        n_eval=2,
        eval_batch_size=2,
        checkpoint="/fake/last.ckpt",
        base_seed=0,
        env_start_method="dummy",
    )
    rows = evaluator.evaluate_policy(model, store_video=0)
    assert len(rows) == 4
    task_names = {row["task_name"] for row in rows}
    assert task_names == {"square_d0", "threading_d1"}
    task_idxs = {row["task_idx"] for row in rows}
    assert task_idxs == {0, 1}


# ---------------------------------------------------------------------------
# Modality-mask validation, mirroring EvaluateLibero's __init__ (already covered for
# LIBERO by tests/test_input_token_drop.py etc.) -- pinning the same behavior here since
# EvaluateMimicgen duplicates that logic rather than sharing a base class.
# ---------------------------------------------------------------------------

def _minimal_ctor_kwargs(model):
    return dict(
        model=model,
        transforms=_make_fake_transforms(),
        log_dir="/tmp",
        data_dir="/fake",
        datasets=["square_d0"],
    )


def test_all_modalities_off_raises():
    model = _FakeModel()
    with pytest.raises(ValueError):
        fem.EvaluateMimicgen(
            **_minimal_ctor_kwargs(model),
            eval_modalities={"rgb_static": False, "rgb_gripper": False, "language": False, "proprio": True},
        )


def test_proprio_withheld_without_use_proprio_raises():
    model = _FakeModel(use_proprio=False)
    with pytest.raises(ValueError):
        fem.EvaluateMimicgen(
            **_minimal_ctor_kwargs(model),
            eval_modalities={"rgb_static": True, "rgb_gripper": True, "language": True, "proprio": False},
        )


def test_default_modalities_all_true_no_mask_set():
    model = _FakeModel()
    fem.EvaluateMimicgen(**_minimal_ctor_kwargs(model))
    assert model.eval_modality_mask is None
    assert model.eval_proprio_mask is None


# ---------------------------------------------------------------------------
# CSV round-trip: rows built by evaluate_dataset must be writable/readable through the
# shared eval_records schema unchanged (plan requires zero changes to eval_records.py).
# ---------------------------------------------------------------------------

def test_rows_round_trip_through_eval_records_csv(tmp_path, monkeypatch):
    model = _FakeModel()
    evaluator = _build_evaluator(monkeypatch, success_at_step=2, n_eval=2, eval_batch_size=2, model=model)
    rows = evaluator.evaluate_dataset(model, "square_d0", idx=0, store_video=0)

    csv_path = tmp_path / "result.csv"
    merge_result_csv(csv_path, rows)
    read_back = read_csv(csv_path)

    assert len(read_back) == 2
    for col in ("libero_variant", "suite", "task_name", "success", "eval_timestamp"):
        assert col in ALL_COLUMNS
        assert read_back[0][col] != "" or col == "eval_timestamp"
    assert read_back[0]["libero_variant"] == "mimicgen"
    assert read_back[0]["suite"] == "core"
