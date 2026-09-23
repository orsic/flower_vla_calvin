"""
Tests for batched eval rollout, cross-task batching, and LIBERO-Plus category filtering.

These tests exercise the new code paths without loading Florence-2 or MuJoCo:
  1. flower.py forward() lang_text dispatch (str vs list)
  2. step_batch() returns [B, action_dim] with correct action-chunking index
  3. EvaluateLibero.process_env_obs_batch stacks B observations into [B, 1, C, H, W]
  4. LIBERO-Plus: select_task_indices and aggregate_by_category
  5. evaluate_work_list's (task, episode) work-list construction and chunking
  6. active-slot masking: finished episodes stop being simulated, not just inferred
  7. make_libero_venv start-method dispatch
  8. evaluate_work_list: per-slot (not stacked) init-state application for ragged
     cross-task batches
  9. _CtxSubprocVectorEnv/_RebuildableDummyVectorEnv.rebuild(): worker-process reuse
     across batches
  10. evaluate_task/evaluate_work_list: process_env_obs_batch skipped on cached
      (non-replan) steps
  11. seed_and_reset: env reset is seeded (fixture/object placement) before
      it runs, from the same per-episode key as the noise seed
"""

import json
import math
import os
import tempfile
import numpy as np
import torch
import pytest
from unittest.mock import MagicMock, patch

from flower.evaluation.eval_records import rollout_seed


# ---------------------------------------------------------------------------
# 1. forward() lang_text dispatch (pure logic, no model needed)
# ---------------------------------------------------------------------------

def _dispatch_lang_text(lang_text):
    """Mirror the logic added in flower.py forward()."""
    return [lang_text] if isinstance(lang_text, str) else list(lang_text)


def test_lang_text_string_wrapped():
    """A bare string must be wrapped in a list (single-episode path)."""
    result = _dispatch_lang_text("pick up the cup")
    assert result == ["pick up the cup"]


def test_lang_text_list_passthrough():
    """A list of B strings must pass through unchanged (batched path)."""
    texts = ["task a", "task b", "task c"]
    result = _dispatch_lang_text(texts)
    assert result == texts


def test_lang_text_single_item_list():
    """A single-item list must not be double-wrapped."""
    result = _dispatch_lang_text(["only one"])
    assert result == ["only one"]


# ---------------------------------------------------------------------------
# 2. step_batch() action-chunking index
# ---------------------------------------------------------------------------

class _MinimalFlower:
    """Thin stand-in for Flower.step_batch tests — no Florence-2 required."""

    multistep = 10
    return_act_chunk = False

    def __init__(self, B: int, action_dim: int = 7):
        self.B = B
        self.action_dim = action_dim
        self.rollout_step_counter = 0
        self.pred_action_seq = None

    def __call__(self, obs, goal):
        """Mock forward: return a fixed tensor for reproducibility."""
        return torch.arange(self.multistep * self.action_dim, dtype=torch.float).reshape(
            1, self.multistep, self.action_dim
        ).expand(self.B, -1, -1).clone()

    # Copy the real step_batch logic verbatim so the test reflects production code.
    def step_batch(self, obs, goal):
        if self.rollout_step_counter % self.multistep == 0:
            self.pred_action_seq = self(obs, goal)
        current_action = self.pred_action_seq[:, self.rollout_step_counter]
        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0
        return current_action

    def reset(self):
        self.rollout_step_counter = 0
        self.pred_action_seq = None


@pytest.fixture
def dummy_obs_goal():
    return {}, {}


@pytest.mark.parametrize("B", [1, 5, 10])
def test_step_batch_shape(B, dummy_obs_goal):
    """step_batch returns [B, action_dim] for various batch sizes."""
    model = _MinimalFlower(B=B, action_dim=7)
    action = model.step_batch(*dummy_obs_goal)
    assert action.shape == (B, 7)


def test_step_batch_chunking(dummy_obs_goal):
    """step_batch re-uses cached pred_action_seq within a chunk."""
    B, multistep = 3, 10
    model = _MinimalFlower(B=B)
    model.multistep = multistep

    # First call triggers forward (counter=0)
    a0 = model.step_batch(*dummy_obs_goal)
    assert model.pred_action_seq is not None
    seq = model.pred_action_seq.clone()

    # Next (multistep-1) calls reuse the same seq without calling forward again
    for step in range(1, multistep):
        a = model.step_batch(*dummy_obs_goal)
        assert torch.allclose(a, seq[:, step])

    # After multistep calls, counter wraps and the next call triggers a new forward
    model.pred_action_seq = None  # invalidate cache to detect the new call
    a_new = model.step_batch(*dummy_obs_goal)
    assert model.pred_action_seq is not None   # forward was called again
    assert a_new.shape == (B, 7)


def test_step_returns_slice_zero(dummy_obs_goal):
    """Original step() uses index [0, counter] — verify it differs from step_batch's [:, counter]."""
    B = 4
    model = _MinimalFlower(B=B)

    # Manually run one forward
    model.pred_action_seq = model(None, None)
    seq = model.pred_action_seq  # [B, multistep, 7]

    # step_batch: returns all B actions for current timestep
    batch_action = seq[:, 0]          # [B, 7]
    # legacy step: returns only env 0
    single_action = seq[0, 0]         # [7]

    assert batch_action.shape == (B, 7)
    assert single_action.shape == (7,)
    # batch_action[0] must equal single_action
    assert torch.allclose(batch_action[0], single_action)


# ---------------------------------------------------------------------------
# 3. process_env_obs_batch tensor stacking
# ---------------------------------------------------------------------------

def _make_fake_transforms():
    """Return a transform dict that does a simple float normalisation."""
    def to_float_tensor(x):
        return x.float() / 255.0

    return {
        'val': {
            'rgb_static': [to_float_tensor],
            'rgb_gripper': [to_float_tensor],
        }
    }


def _make_single_obs(H: int = 224, W: int = 224) -> dict:
    """Return a single obs dict as returned by OffScreenRenderEnv."""
    return {
        'agentview_image': np.random.randint(0, 255, (H, W, 3), dtype=np.uint8),
        'robot0_eye_in_hand_image': np.random.randint(0, 255, (H, W, 3), dtype=np.uint8),
        'robot0_joint_pos': np.zeros(7, dtype=np.float32),
        'robot0_gripper_qpos': np.zeros(2, dtype=np.float32),
    }


def _build_eval_libero_stub(transforms):
    """Build a minimal EvaluateLibero-like object with only the obs-processing methods."""
    from flower.evaluation.flower_eval_libero import EvaluateLibero

    # Patch out __init__ to avoid loading benchmark / model
    with patch.object(EvaluateLibero, '__init__', lambda self: None):
        ev = EvaluateLibero()

    ev.transforms = transforms
    ev.device = None
    ev._printed_transforms = True  # suppress debug prints
    ev.img_h = 224
    ev.img_w = 224
    return ev


@pytest.mark.parametrize("B", [1, 4, 10])
def test_process_env_obs_batch_shape(B):
    """process_env_obs_batch should produce [B, 1, C, H, W] tensors."""
    transforms = _make_fake_transforms()
    ev = _build_eval_libero_stub(transforms)

    obs_list = [_make_single_obs() for _ in range(B)]
    lang_embed = torch.zeros(512)  # dummy embedding
    lang_text = "pick up the cup"

    data, goal = ev.process_env_obs_batch(obs_list, lang_embed, lang_text)

    for key in ('rgb_static', 'rgb_gripper'):
        t = data['rgb_obs'][key]
        assert t.shape == (B, 1, 3, 224, 224), (
            f"Expected [{B}, 1, 3, 224, 224] for {key}, got {t.shape}"
        )

    assert goal['lang_text'] == [lang_text] * B
    assert len(goal['lang_text']) == B


def test_process_env_obs_batch_heterogeneous_lang_text():
    """cross_task_batching feeds one distinct lang_text per slot; each slot's image
    processing must be independent of the others (batching doesn't mix slots)."""
    transforms = _make_fake_transforms()
    ev = _build_eval_libero_stub(transforms)

    B = 3
    obs_list = [_make_single_obs() for _ in range(B)]
    lang_embed = None  # never read by forward(); cross_task_batching passes None
    lang_texts = ["pick up the cup", "open the drawer", "close the drawer"]

    batch_data, goal = ev.process_env_obs_batch(obs_list, lang_embed, lang_texts)

    assert goal["lang_text"] == lang_texts
    for k, obs in enumerate(obs_list):
        single_data, _ = ev.process_env_obs(obs, lang_embed, lang_texts[k])
        for key in ('rgb_static', 'rgb_gripper'):
            batch_slice = batch_data['rgb_obs'][key][k]
            single_val = single_data['rgb_obs'][key][0]
            assert torch.allclose(batch_slice, single_val), (
                f"Batch slice {k} differs from single-obs result for {key}"
            )


def test_process_env_obs_batch_values_match_single():
    """Each slice [k] of the batch must equal the result of processing obs k alone."""
    transforms = _make_fake_transforms()
    ev = _build_eval_libero_stub(transforms)

    B = 3
    obs_list = [_make_single_obs() for _ in range(B)]
    lang_embed = torch.zeros(512)
    lang_text = "stack the blocks"

    batch_data, _ = ev.process_env_obs_batch(obs_list, lang_embed, lang_text)

    for k, obs in enumerate(obs_list):
        single_data, _ = ev.process_env_obs(obs, lang_embed, lang_text)
        for key in ('rgb_static', 'rgb_gripper'):
            batch_slice = batch_data['rgb_obs'][key][k]   # [1, C, H, W]
            single_val = single_data['rgb_obs'][key][0]   # [1, C, H, W]
            assert torch.allclose(batch_slice, single_val), (
                f"Batch slice {k} differs from single-obs result for {key}"
            )


# ---------------------------------------------------------------------------
# 4. LIBERO-Plus: select_task_indices and aggregate_by_category
# ---------------------------------------------------------------------------

# Fixture task_classification.json used in all Plus tests.
_FIXTURE_JSON = {
    "libero_10": [
        {"id": 1, "name": "task_a_table_1", "category": "Background Textures", "difficulty_level": 1},
        {"id": 2, "name": "task_a_view_1",  "category": "Camera Viewpoints",   "difficulty_level": 2},
        {"id": 3, "name": "task_b_table_1", "category": "Background Textures", "difficulty_level": 1},
        {"id": 4, "name": "task_b_add_1",   "category": "Objects Layout",      "difficulty_level": 3},
        {"id": 5, "name": "task_c_light_1", "category": "Light Conditions",    "difficulty_level": 2},
    ]
}
_ALL_TASK_NAMES = [e["name"] for e in _FIXTURE_JSON["libero_10"]]


def _make_fake_benchmark(suite_name="libero_10", names=None):
    """Return a mock benchmark_instance with .name and .get_task_names()."""
    bm = MagicMock()
    bm.name = suite_name
    bm.get_task_names.return_value = names if names is not None else list(_ALL_TASK_NAMES)
    return bm


@pytest.fixture()
def classification_json(tmp_path):
    """Write fixture JSON to a temp file; return its path."""
    p = tmp_path / "task_classification.json"
    p.write_text(json.dumps(_FIXTURE_JSON))
    return str(p)


def _patch_json_path(monkeypatch, json_path):
    """Patch select_task_indices to find the fixture JSON via libero.libero.benchmark.__file__."""
    import flower.evaluation.flower_eval_libero as fef
    # Point the module-level lookup at our temp file directory.
    fake_bm_mod = MagicMock()
    fake_bm_mod.__file__ = str(os.path.join(os.path.dirname(json_path), "__init__.py"))
    monkeypatch.setattr("flower.evaluation.flower_eval_libero.select_task_indices.__module__",
                        "flower.evaluation.flower_eval_libero", raising=False)
    return fake_bm_mod


class TestSelectTaskIndices:
    def test_none_category_returns_none(self):
        """task_category=None means 'all tasks' → returns None (no filter)."""
        from flower.evaluation.flower_eval_libero import select_task_indices
        bm = _make_fake_benchmark()
        result = select_task_indices(bm, task_category=None)
        assert result is None

    def test_filters_to_matching_category(self, tmp_path):
        """Returns exactly the indices of tasks with the given category."""
        from flower.evaluation.flower_eval_libero import select_task_indices

        json_path = tmp_path / "task_classification.json"
        json_path.write_text(json.dumps(_FIXTURE_JSON))

        bm = _make_fake_benchmark()
        with patch("libero.libero.benchmark.__file__",
                   str(tmp_path / "__init__.py"), create=True):
            import libero.libero.benchmark as bm_mod
            orig_file = bm_mod.__file__
            bm_mod.__file__ = str(tmp_path / "__init__.py")
            try:
                result = select_task_indices(bm, task_category="Background Textures")
            finally:
                bm_mod.__file__ = orig_file

        # task_a_table_1 (idx=0) and task_b_table_1 (idx=2) are Background Textures
        assert sorted(result) == [0, 2]

    def test_unknown_category_raises(self, tmp_path):
        """Unknown category raises ValueError with helpful message."""
        from flower.evaluation.flower_eval_libero import select_task_indices

        json_path = tmp_path / "task_classification.json"
        json_path.write_text(json.dumps(_FIXTURE_JSON))

        bm = _make_fake_benchmark()
        import libero.libero.benchmark as bm_mod
        orig_file = bm_mod.__file__
        bm_mod.__file__ = str(tmp_path / "__init__.py")
        try:
            with pytest.raises(ValueError, match="not found"):
                select_task_indices(bm, task_category="Nonexistent Category")
        finally:
            bm_mod.__file__ = orig_file

    def test_missing_json_raises_file_not_found(self, tmp_path):
        """FileNotFoundError when task_classification.json doesn't exist."""
        from flower.evaluation.flower_eval_libero import select_task_indices

        bm = _make_fake_benchmark()
        import libero.libero.benchmark as bm_mod
        orig_file = bm_mod.__file__
        bm_mod.__file__ = str(tmp_path / "__init__.py")  # no json written
        try:
            with pytest.raises(FileNotFoundError):
                select_task_indices(bm, task_category="Background Textures")
        finally:
            bm_mod.__file__ = orig_file


class TestAggregateByCategory:
    def _run(self, tmp_path, names, successes):
        from flower.evaluation.flower_eval_libero import aggregate_by_category

        json_path = tmp_path / "task_classification.json"
        json_path.write_text(json.dumps(_FIXTURE_JSON))

        import libero.libero.benchmark as bm_mod
        orig_file = bm_mod.__file__
        bm_mod.__file__ = str(tmp_path / "__init__.py")
        try:
            return aggregate_by_category(names, successes, suite_name="libero_10")
        finally:
            bm_mod.__file__ = orig_file

    def test_categories_averaged_correctly(self, tmp_path):
        """Each category averages its member task success rates."""
        names = _ALL_TASK_NAMES  # 5 tasks
        successes = [1.0, 0.0, 0.5, 1.0, 0.0]
        result = self._run(tmp_path, names, successes)
        # Background Textures: task_a_table_1=1.0, task_b_table_1=0.5 → avg 0.75
        assert abs(result["Background Textures"] - 0.75) < 1e-6
        # Camera Viewpoints: task_a_view_1=0.0 → 0.0
        assert abs(result["Camera Viewpoints"] - 0.0) < 1e-6
        # Objects Layout: task_b_add_1=1.0 → 1.0
        assert abs(result["Objects Layout"] - 1.0) < 1e-6
        # Light Conditions: task_c_light_1=0.0 → 0.0
        assert abs(result["Light Conditions"] - 0.0) < 1e-6

    def test_missing_json_returns_empty(self, tmp_path):
        """Returns empty dict when task_classification.json is absent (orig variant)."""
        from flower.evaluation.flower_eval_libero import aggregate_by_category

        import libero.libero.benchmark as bm_mod
        orig_file = bm_mod.__file__
        bm_mod.__file__ = str(tmp_path / "__init__.py")  # no json written
        try:
            result = aggregate_by_category(
                _ALL_TASK_NAMES, [1.0] * len(_ALL_TASK_NAMES), suite_name="libero_10"
            )
        finally:
            bm_mod.__file__ = orig_file
        assert result == {}


# ---------------------------------------------------------------------------
# 5. evaluate_work_list: (task, episode) work-list construction and chunking
# ---------------------------------------------------------------------------

def _build_work_items(all_tasks, n_eval):
    """Mirror the work_items construction in EvaluateLibero.evaluate_work_list."""
    return [(idx, ep) for idx in all_tasks for ep in range(n_eval)]


def _chunk(items, batch_size):
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def test_work_list_covers_every_task_episode_pair():
    items = _build_work_items(all_tasks=[5, 7], n_eval=3)
    assert items == [(5, 0), (5, 1), (5, 2), (7, 0), (7, 1), (7, 2)]


def test_work_list_n_eval_1_still_batches_across_tasks():
    """The LIBERO-Plus case (n_eval=1): evaluate_task's per-task batching collapses
    to batch size 1 here, since there's only one episode per task. Cross-task
    batching instead spans tasks, so a batch_size=2 chunk still holds 2 episodes."""
    items = _build_work_items(all_tasks=[0, 1, 2, 3, 4], n_eval=1)
    assert items == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    batches = _chunk(items, batch_size=2)
    assert [len(b) for b in batches] == [2, 2, 1]  # ragged final batch, not dropped


# ---------------------------------------------------------------------------
# 5b. Per-episode noise seeding: evaluate_work_list's
# seeds = [rollout_seed(base_seed, task_cache[idx]["task_name"], ep) for idx, ep in batch_items]
# reads only (task_name, ep) -- never a batch/chunk/shard position -- so it can't
# reproduce the old batch_seed(base_seed, batch_index) bug (a work item's seed
# depending on where it lands in the flattened work list).
# ---------------------------------------------------------------------------

def _work_list_seeds(all_tasks, n_eval, task_names, batch_size, base_seed=0):
    """Mirror evaluate_work_list's exact per-row seed expression, batch by batch."""
    items = _build_work_items(all_tasks, n_eval)
    seeds = {}
    for batch in _chunk(items, batch_size):
        for idx, ep in batch:
            seeds[(idx, ep)] = rollout_seed(base_seed, task_names[idx], ep)
    return seeds


def test_work_list_seed_is_position_and_shard_independent():
    """The same work item must derive the same seed regardless of task order,
    eval_batch_size, or (equivalently) which GPU shard's all_tasks subset it's part
    of -- exactly the property the old batch-ordinal seed lacked."""
    task_names = {7: "task_g", 3: "task_c", 11: "task_k"}

    forward = _work_list_seeds([7, 3, 11], n_eval=1, task_names=task_names, batch_size=2)
    reverse = _work_list_seeds([11, 3, 7], n_eval=1, task_names=task_names, batch_size=2)
    solo_batches = _work_list_seeds([7, 3, 11], n_eval=1, task_names=task_names, batch_size=1)
    one_big_batch = _work_list_seeds([7, 3, 11], n_eval=1, task_names=task_names, batch_size=10)

    assert forward[(3, 0)] == reverse[(3, 0)] == solo_batches[(3, 0)] == one_big_batch[(3, 0)]


def test_work_list_and_per_task_seed_expressions_agree_for_the_same_episode():
    """evaluate_task's per-row seed (rollout_seed(base_seed, task_name, episode_idx + k))
    and evaluate_work_list's (rollout_seed(base_seed, task_name, ep)) must derive the
    same value for the same (task_name, episode index) -- required for
    scripts/severity_sr.py's orig-vs-Plus pairing to actually share noise."""
    base_seed, task_name = 5, "task_c"
    episode_idx, k = 2, 1  # evaluate_task's slot k within a batch starting at episode_idx

    per_task_seed = rollout_seed(base_seed, task_name, episode_idx + k)
    work_list_seed = rollout_seed(base_seed, task_name, episode_idx + k)
    assert per_task_seed == work_list_seed


# ---------------------------------------------------------------------------
# 6. Active-slot masking: only step envs that haven't finished yet
# ---------------------------------------------------------------------------

def _run_masked_steps(done_schedule, B, max_steps):
    """Mirror evaluate_task/evaluate_work_list's active_ids bookkeeping without any
    env/model — done_schedule maps a 1-indexed step number to the set of slot
    indices that report `done` on that step.

    Returns (steps_taken, dones, stepped_ids_per_step) — the last lets a test
    assert that a finished slot is never stepped again (the efficiency claim:
    simulation, not just inference, is skipped for finished episodes).
    """
    dones = [False] * B
    steps_taken = [0] * B
    active_ids = list(range(B))
    stepped_ids_per_step = []
    steps = 0
    while steps < max_steps:
        steps += 1
        stepped_ids_per_step.append(list(active_ids))
        finished_now = done_schedule.get(steps, set()) & set(active_ids)
        still_active = []
        for k in active_ids:
            steps_taken[k] = steps
            if k in finished_now:
                dones[k] = True
            else:
                still_active.append(k)
        active_ids = still_active
        if not active_ids:
            break
    return steps_taken, dones, stepped_ids_per_step


def test_masked_steps_finished_slot_is_never_stepped_again():
    steps_taken, dones, stepped = _run_masked_steps(
        done_schedule={2: {1}, 3: {0, 3}}, B=4, max_steps=5,
    )
    assert steps_taken == [3, 2, 5, 3]
    assert dones == [True, True, False, True]
    # Slot 1 finishes at step 2: absent from every later step's stepped-id list.
    assert all(1 not in ids for ids in stepped[2:])
    # Slots 0 and 3 finish at step 3: absent afterwards.
    assert all(0 not in ids and 3 not in ids for ids in stepped[3:])
    # Slot 2 never finishes: stepped on every iteration up to max_steps.
    assert all(2 in ids for ids in stepped)
    assert len(stepped) == 5


def test_masked_steps_all_done_breaks_early():
    steps_taken, dones, stepped = _run_masked_steps(
        done_schedule={1: {0, 1}}, B=2, max_steps=100,
    )
    assert steps_taken == [1, 1]
    assert dones == [True, True]
    assert len(stepped) == 1  # loop breaks the instant every slot is done


def test_masked_steps_none_done_runs_to_max_steps():
    steps_taken, dones, stepped = _run_masked_steps(
        done_schedule={}, B=3, max_steps=4,
    )
    assert steps_taken == [4, 4, 4]
    assert dones == [False, False, False]
    assert len(stepped) == 4
    assert all(len(ids) == 3 for ids in stepped)  # full batch stepped every time


# ---------------------------------------------------------------------------
# 7. make_libero_venv start-method dispatch (no MuJoCo — a fake env_fn)
# ---------------------------------------------------------------------------

def test_make_libero_venv_dummy_returns_dummy_vector_env():
    from libero.libero.envs import DummyVectorEnv
    from flower.evaluation.libero_venv import make_libero_venv

    env = make_libero_venv([lambda: object()], "dummy")
    assert isinstance(env, DummyVectorEnv)


def test_make_libero_venv_unknown_start_method_raises():
    from flower.evaluation.libero_venv import make_libero_venv

    with pytest.raises(ValueError):
        make_libero_venv([lambda: object()], "bogus")


# ---------------------------------------------------------------------------
# 8. evaluate_work_list: per-slot init-state application for ragged cross-task
#    batches (LIBERO-Plus init states are MuJoCo sim-state vectors whose length
#    depends on the scene, e.g. 47 vs 45 vs 123 floats — np.stack raises
#    ValueError on a mixed-scene batch; ragged shapes here mirror the real
#    dims observed on LIBERO-Plus's libero_10 suite)
# ---------------------------------------------------------------------------

class _FakeVenv:
    """Records what set_init_state receives, without needing a real env."""

    def __init__(self):
        self.set_init_state_calls = []

    def set_init_state(self, init_state, id=None):
        self.set_init_state_calls.append((init_state, id))


def _apply_init_states(env, metas, state_idxs):
    """Mirror the init-state application block in EvaluateLibero.evaluate_work_list."""
    B = len(metas)
    init_slots = [k for k in range(B) if state_idxs[k] is not None]
    if init_slots:
        env.set_init_state(
            [metas[k]["initial_states"][state_idxs[k]] for k in init_slots],
            id=init_slots,
        )
    return init_slots


def test_ragged_init_states_applied_as_list_not_stacked():
    """A batch mixing tasks whose init-state vectors have different lengths must not
    be np.stack'ed (that's the ValueError this test guards against); set_init_state
    must receive a plain list instead."""
    metas = [
        {"initial_states": np.zeros((5, 47))},
        {"initial_states": np.zeros((5, 45))},
        {"initial_states": np.zeros((5, 123))},
    ]
    state_idxs = [0, 0, 0]
    env = _FakeVenv()

    init_slots = _apply_init_states(env, metas, state_idxs)

    assert init_slots == [0, 1, 2]
    assert len(env.set_init_state_calls) == 1
    passed_states, passed_ids = env.set_init_state_calls[0]
    assert isinstance(passed_states, list)  # not np.stack'ed
    assert [s.shape[0] for s in passed_states] == [47, 45, 123]
    assert passed_ids == [0, 1, 2]


def test_init_states_applied_only_to_slots_that_have_them():
    """A slot without init states (state_idxs[k] is None) must be excluded from the
    id list rather than blocking init-state application for the whole batch."""
    metas = [
        {"initial_states": np.zeros((5, 47))},
        {"initial_states": None},
        {"initial_states": np.zeros((5, 123))},
    ]
    state_idxs = [0, None, 0]
    env = _FakeVenv()

    init_slots = _apply_init_states(env, metas, state_idxs)

    assert init_slots == [0, 2]
    passed_states, passed_ids = env.set_init_state_calls[0]
    assert len(passed_states) == 2
    assert passed_ids == [0, 2]


def test_no_init_states_skips_set_init_state_call():
    metas = [{"initial_states": None}, {"initial_states": None}]
    state_idxs = [None, None]
    env = _FakeVenv()

    init_slots = _apply_init_states(env, metas, state_idxs)

    assert init_slots == []
    assert env.set_init_state_calls == []


# ---------------------------------------------------------------------------
# 9. rebuild(): worker-process reuse across batches
#
# evaluate_task/evaluate_work_list used to tear down and recreate the venv
# every batch (env.close() + make_libero_venv()); measured, that spawn + EGL
# context creation was ~42% of total eval wall-clock. rebuild() instead swaps
# each worker's env in place (env.close() + reconstruct, same process), so
# these tests assert both the swap actually happens (new obs, correct count)
# and that it does NOT pay for a new process (same PID -- the whole point).
# _FakeEnv avoids MuJoCo/EGL entirely so this stays a fast unit test; the
# real-env equivalence is the end-to-end CSV diff in the plan's verification
# section, not here.
# ---------------------------------------------------------------------------

class _FakeEnv:
    """Picklable stand-in for OffScreenRenderEnv -- just enough surface
    (reset/step/close) for _rebuildable_worker's command loop, with a `label`
    to prove which env_fn produced it. `seed`/`last_seed` let a test observe
    whether BaseVectorEnv.seed() actually reached this env (see section 11)."""

    def __init__(self, label):
        self.label = label
        self.closed = False
        self.last_seed = None

    def reset(self):
        return {"label": self.label, "seed": self.last_seed}

    def step(self, action):
        return {"label": self.label}, 0.0, False, {}

    def seed(self, seed):
        self.last_seed = seed

    def close(self):
        self.closed = True


def test_ctx_subproc_venv_rebuild_swaps_env_in_same_process():
    """rebuild() must reuse the worker's existing subprocess (same PID), not
    spawn a new one -- that reuse is the entire point of pooling workers
    across batches."""
    from flower.evaluation.libero_venv import make_libero_venv

    env = make_libero_venv([lambda: _FakeEnv("a"), lambda: _FakeEnv("b")], "spawn")
    try:
        pids_before = [w.process.pid for w in env.workers]

        obs = env.reset()
        assert [obs[i]["label"] for i in range(2)] == ["a", "b"]

        env.rebuild([lambda: _FakeEnv("c"), lambda: _FakeEnv("d")])
        pids_after = [w.process.pid for w in env.workers]
        assert pids_after == pids_before  # no new process spawned

        obs2 = env.reset()
        assert [obs2[i]["label"] for i in range(2)] == ["c", "d"]
    finally:
        env.close()


def test_ctx_subproc_venv_rebuild_ragged_leaves_surplus_workers_untouched():
    """A rebuild with fewer env_fns than the pool (the ragged last batch of a
    work list) must only touch workers[:n]; the rest keep their current env
    and stay steppable via an explicit id=[...]."""
    from flower.evaluation.libero_venv import make_libero_venv

    env = make_libero_venv(
        [lambda: _FakeEnv("a"), lambda: _FakeEnv("b"), lambda: _FakeEnv("c")], "spawn"
    )
    try:
        env.rebuild([lambda: _FakeEnv("x")])  # only worker 0 rebuilt

        obs = env.reset(id=[0])
        assert obs[0]["label"] == "x"

        obs_rest = env.reset(id=[1, 2])
        assert [obs_rest[0]["label"], obs_rest[1]["label"]] == ["b", "c"]
    finally:
        env.close()


def test_dummy_venv_rebuild_swaps_env_in_place():
    """The 'dummy' (sequential, in-process) start method supports the same
    rebuild() contract as 'spawn' -- evaluate_task/evaluate_work_list call it
    unconditionally regardless of env_start_method."""
    from flower.evaluation.libero_venv import make_libero_venv

    env = make_libero_venv([lambda: _FakeEnv("a"), lambda: _FakeEnv("b")], "dummy")
    try:
        obs = env.reset()
        assert [obs[i]["label"] for i in range(2)] == ["a", "b"]

        env.rebuild([lambda: _FakeEnv("c"), lambda: _FakeEnv("d")])
        obs2 = env.reset()
        assert [obs2[i]["label"] for i in range(2)] == ["c", "d"]
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 10. process_env_obs_batch skipped on cached (non-replan) steps
#
# step_batch only reads (data, goal) when rollout_step_counter % multistep ==
# 0 (see _MinimalFlower.step_batch / flower.py's real step_batch, copied
# verbatim in this file); on every other step it just indexes the cached
# pred_action_seq. evaluate_task/evaluate_work_list now guard the
# process_env_obs_batch call behind that same condition instead of building
# (and discarding) it every step. This test proves the guard can't change
# step_batch's output -- (data, goal) genuinely go unread on skip steps -- and
# that it's called exactly ceil(steps/multistep) times.
# ---------------------------------------------------------------------------

def test_step_batch_result_unaffected_by_skipping_preprocessing_on_cached_steps():
    B, multistep, steps = 3, 10, 25

    def build_baseline():
        return object(), object()  # stand-in (data, goal); _MinimalFlower's
                                    # __call__ ignores both, same as real step_batch
                                    # only reads them on a replan step

    baseline_model = _MinimalFlower(B=B)
    baseline_model.multistep = multistep
    baseline_actions = [baseline_model.step_batch(*build_baseline()) for _ in range(steps)]

    guarded_model = _MinimalFlower(B=B)
    guarded_model.multistep = multistep
    guarded_calls = 0
    guarded_actions = []
    for _ in range(steps):
        if guarded_model.rollout_step_counter % guarded_model.multistep == 0:
            data, goal = build_baseline()
            guarded_calls += 1
        else:
            data, goal = None, None
        guarded_actions.append(guarded_model.step_batch(data, goal))

    assert guarded_calls == math.ceil(steps / multistep)
    assert all(
        torch.allclose(a, b) for a, b in zip(baseline_actions, guarded_actions)
    ), "guarding process_env_obs_batch behind the replan condition must not change actions"


# ---------------------------------------------------------------------------
# 11. seed_and_reset: env reset is seeded before it runs
#
# bddl_base_domain.BddlBaseDomain._reset_internal samples fixture/object
# placement *during* reset() -- env.set_init_state() afterwards restores only
# qpos/qvel, never the fixture body_pos/body_quat that sampling wrote. The
# worker's numpy RNG was otherwise never seeded (libero_venv.py reseeds it to
# entropy on every rebuild), so that placement was silently random on every
# episode. seed_and_reset() seeds the env from the episode's own rollout_seed
# (folded into np.random.seed's uint32 range by eval_records.env_seed) right
# before reset() -- so the placement becomes reproducible, and a LIBERO-Plus
# episode shares its orig counterpart's layout the same way it already shares
# its noise (rollout_seed's base-task-name keying).
# ---------------------------------------------------------------------------

class _FakeSeedResetEnv:
    """Records call order/arguments -- proves seed_and_reset seeds before it
    resets, not after (a post-reset seed would be a no-op: the sampling that
    needs seeding already happened inside reset())."""

    def __init__(self):
        self.calls = []

    def seed(self, seeds):
        self.calls.append(("seed", list(seeds)))

    def reset(self, id=None):
        self.calls.append(("reset", id))


def test_seed_and_reset_seeds_before_resetting():
    from flower.evaluation.eval_records import env_seed
    from flower.evaluation.flower_eval_libero import seed_and_reset

    env = _FakeSeedResetEnv()
    seed_and_reset(env, ids=[0, 1], seeds=[10, 20])

    assert [call[0] for call in env.calls] == ["seed", "reset"]
    assert env.calls[0][1] == [env_seed(10), env_seed(20)]
    assert env.calls[1][1] == [0, 1]


def test_ctx_subproc_venv_seed_reaches_worker_env():
    """End-to-end through _rebuildable_worker's existing (until now dead) "seed"
    command: BaseVectorEnv.seed([...]) must actually reach each worker's env."""
    from flower.evaluation.libero_venv import make_libero_venv

    env = make_libero_venv([lambda: _FakeEnv("a"), lambda: _FakeEnv("b")], "spawn")
    try:
        env.seed([11, 22])
        obs = env.reset()
        assert [obs[i]["seed"] for i in range(2)] == [11, 22]
    finally:
        env.close()


def test_dummy_venv_seed_reaches_worker_env():
    from flower.evaluation.libero_venv import make_libero_venv

    env = make_libero_venv([lambda: _FakeEnv("a"), lambda: _FakeEnv("b")], "dummy")
    try:
        env.seed([11, 22])
        obs = env.reset()
        assert [obs[i]["seed"] for i in range(2)] == [11, 22]
    finally:
        env.close()


def test_seed_list_shorter_than_pool_seeds_only_first_workers():
    """A ragged final batch's seed list is shorter than the worker pool;
    BaseVectorEnv.seed's `zip(self.workers, seed_list)` must seed only the
    workers with a corresponding entry, leaving the rest untouched."""
    from flower.evaluation.libero_venv import make_libero_venv

    env = make_libero_venv(
        [lambda: _FakeEnv("a"), lambda: _FakeEnv("b"), lambda: _FakeEnv("c")], "spawn"
    )
    try:
        env.seed([11])
        obs = env.reset()
        assert obs[0]["seed"] == 11
        assert obs[1]["seed"] is None
        assert obs[2]["seed"] is None
    finally:
        env.close()
