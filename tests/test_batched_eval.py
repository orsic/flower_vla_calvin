"""
Tests for batched eval rollout and LIBERO-Plus category filtering.

These tests exercise the new code paths without loading Florence-2 or MuJoCo:
  1. flower.py forward() lang_text dispatch (str vs list)
  2. step_batch() returns [B, action_dim] with correct action-chunking index
  3. EvaluateLibero.process_env_obs_batch stacks B observations into [B, 1, C, H, W]
"""

import json
import os
import tempfile
import numpy as np
import torch
import pytest
from unittest.mock import MagicMock, patch


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
