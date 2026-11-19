"""VLAAdapterPolicy: VLA-Adapter behind the rollout interface the LIBERO/CALVIN harnesses call.

The model itself is stubbed (predict_chunk); these tests pin the parts that must match
VLA-Adapter's reference evaluators (experiments/robot/libero/run_libero_eval.py and
vla-scripts/{evaluate_calvin.py,vla_evaluation.py}): observation layout, replanning
cadence, CALVIN's temporal-ensembling schedule and the gripper conventions.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flower.evaluation.eval_records import apply_mean_proprio, blank_instruction, mean_pixel_views
from flower.evaluation.flower_evaluate import apply_eval_modalities
from flower.evaluation.vla_adapter_policy import (
    Modalities,
    VLAAdapterPolicy,
    layout_positions,
    mean_pixel,
    calvin_ensemble_action,
    calvin_gripper,
    calvin_observation,
    libero_gripper,
    libero_observation,
    select_unnorm_key,
)


def _libero_obs(seed=0, res=4):
    rng = np.random.default_rng(seed)
    return {
        "agentview_image": rng.integers(0, 255, (res, res, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": rng.integers(0, 255, (res, res, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, -0.2, 0.9]),
        # 90 degrees about z, (x, y, z, w)
        "robot0_eef_quat": np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]),
        "robot0_gripper_qpos": np.array([0.03, -0.03]),
    }


def _calvin_obs():
    return {
        "rgb_obs": {
            "rgb_static": np.zeros((200, 200, 3), dtype=np.uint8),
            "rgb_gripper": np.ones((84, 84, 3), dtype=np.uint8),
        },
        "robot_obs": np.arange(15, dtype=np.float64),
    }


class _Counter:
    """predict_chunk stub: chunk i is filled with i (gripper column too), records each call."""

    def __init__(self):
        self.calls = []

    def __call__(self, obs, instruction, mods):
        self.calls.append((obs, instruction, mods))
        return np.full((8, 7), float(len(self.calls) - 1))


def test_libero_observation_rotates_both_cameras_and_builds_the_8d_state():
    raw = _libero_obs()
    obs = libero_observation(raw)
    np.testing.assert_array_equal(obs["full_image"], raw["agentview_image"][::-1, ::-1])
    np.testing.assert_array_equal(obs["wrist_image"], raw["robot0_eye_in_hand_image"][::-1, ::-1])
    np.testing.assert_allclose(obs["state"], [0.1, -0.2, 0.9, 0.0, 0.0, np.pi / 2, 0.03, -0.03], atol=1e-7)
    assert set(obs) == {"full_image", "wrist_image", "state"}


def test_calvin_observation_keeps_raw_images_and_slices_tcp_pose_width_and_gripper_action():
    raw = _calvin_obs()
    obs = calvin_observation(raw)
    assert obs["full_image"] is raw["rgb_obs"]["rgb_static"]
    assert obs["wrist_image"] is raw["rgb_obs"]["rgb_gripper"]
    np.testing.assert_array_equal(obs["state"], [0, 1, 2, 3, 4, 5, 6, 14])


def test_libero_gripper_binarizes_then_inverts():
    # VLA-Adapter predicts 1 = open (RLDS convention); LIBERO expects -1 = open.
    chunk = np.zeros((3, 7))
    chunk[:, -1] = [0.9, 0.1, 0.5]
    out = libero_gripper(chunk)
    np.testing.assert_array_equal(out[:, -1], [-1.0, 1.0, -0.0])
    np.testing.assert_array_equal(out[:, :-1], chunk[:, :-1])


def test_calvin_gripper_matches_reference_sign_including_the_tie():
    # Input is the reference's already-flipped value (1 - g); the env gets -sign(2x - 1),
    # where a tie maps to +1 before the inversion.
    a = np.zeros((3, 7))
    a[:, -1] = [0.2, 0.8, 0.5]
    np.testing.assert_array_equal(calvin_gripper(a)[:, -1], [1.0, -1.0, -1.0])


def test_calvin_ensemble_reproduces_rollout_hi3_period():
    q = [np.full((8, 7), 10.0 * k) + np.arange(8)[:, None] for k in range(3)]
    expected = [
        q[0][0],
        (q[0][1] + q[1][0]) / 2,
        (q[0][2] + q[1][1] + q[2][0]) / 3,
        *[(q[0][t] + q[1][t - 1] + q[2][t - 2]) / 3 for t in range(2, 7)],
        (q[1][7] + q[2][6]) / 2,
        q[2][7],
    ]
    for p, want in enumerate(expected):
        np.testing.assert_allclose(calvin_ensemble_action(q, p), want)


def test_step_batch_replans_every_8_steps_per_sample_and_reset_restarts():
    stub = _Counter()
    policy = VLAAdapterPolicy(stub, "libero")
    data, goal = policy.process_raw_obs([_libero_obs(0), _libero_obs(1)], "open the drawer")
    assert goal == ["open the drawer", "open the drawer"]
    for step in range(9):
        if policy.rollout_step_counter % policy.multistep == 0:
            out = policy.step_batch(data, goal)
        else:
            out = policy.step_batch(None, None)
        assert isinstance(out, torch.Tensor) and out.shape == (2, 7)
    assert len(stub.calls) == 4  # two samples, replanned at steps 0 and 8
    assert out[0, 0].item() == 2.0 and out[1, 0].item() == 3.0
    policy.reset()
    assert policy.rollout_step_counter == 0


def test_step_batch_accepts_per_sample_instructions():
    stub = _Counter()
    policy = VLAAdapterPolicy(stub, "libero")
    data, goal = policy.process_raw_obs([_libero_obs(0), _libero_obs(1)], ["a", "b"])
    policy.step_batch(data, goal)
    assert [instr for _, instr, _ in stub.calls] == ["a", "b"]


def test_calvin_reference_schedule_queries_three_times_per_ten_steps():
    stub = _Counter()
    policy = VLAAdapterPolicy(stub, "calvin")
    actions = [policy.step(_calvin_obs(), {"lang_text": "lift the block"}) for _ in range(11)]
    assert len(stub.calls) == 4  # steps 0, 1, 2 and the next period's 0
    assert all(a.shape == (7,) for a in actions)
    assert actions[0][0] == 0.0 and actions[1][0] == 0.5 and actions[9][0] == 2.0 and actions[10][0] == 3.0
    assert stub.calls[0][1] == "lift the block"


def test_calvin_chunk_schedule_replans_every_8_steps():
    stub = _Counter()
    policy = VLAAdapterPolicy(stub, "calvin", calvin_schedule="chunk")
    for _ in range(9):
        policy.step(_calvin_obs(), {"lang_text": "x"})
    assert len(stub.calls) == 2


def test_calvin_actions_carry_a_binary_gripper():
    policy = VLAAdapterPolicy(lambda obs, instr, mods: np.full((8, 7), 0.9), "calvin")
    # Raw 0.9 -> flipped 0.1 -> -sign(2 * 0.1 - 1) = +1 (open in CALVIN)
    assert policy.step(_calvin_obs(), {"lang_text": "x"})[-1] == 1.0


FULL = {"rgb_static": True, "rgb_gripper": True, "language": True, "proprio": True}
CONDITIONS = {
    # removed: the modality's own flag off (scripts/eval_pipeline.py:modality_overrides)
    "no static": ({"rgb_static": False}, Modalities(static=False)),
    "no wrist": ({"rgb_gripper": False}, Modalities(wrist=False)),
    "no lang": ({"language": False}, Modalities(lang=False)),
    "no proprio": ({"proprio": False}, Modalities(proprio="off")),
    # masked: scripts/eval_pipeline.py:MASKED_VARIANTS
    "masked_static": ({"rgb_static_pixels": False}, Modalities(mean_static=True)),
    "masked_wrist": ({"rgb_gripper_pixels": False}, Modalities(mean_wrist=True)),
    "blank_instruction": ({"instruction": False}, Modalities(blank_instruction=True)),
    "masked_proprio": ({"proprio_state": False}, Modalities(proprio="mean")),
}


@pytest.mark.parametrize("name", list(CONDITIONS))
def test_harness_eval_modalities_reach_predict_chunk_as_the_matching_spec(name):
    """The CALVIN harness's apply_eval_modalities (the same attribute writes as
    EvaluateLibero.__init__) turns each single-input condition into the right spec."""
    overrides, expected = CONDITIONS[name]
    stub = _Counter()
    policy = VLAAdapterPolicy(stub, "calvin", proprio_mean=np.zeros(8))
    uses_proprio = apply_eval_modalities(policy, {**FULL, **overrides}, checkpoint="/ckpt/CALVIN-ABC-Pro")
    policy.step(_calvin_obs(), {"lang_text": "x"})
    assert stub.calls[0][2] == expected
    assert uses_proprio == (name != "no proprio")  # result.csv's use_proprio column


def test_all_inputs_is_the_default_spec():
    stub = _Counter()
    policy = VLAAdapterPolicy(stub, "libero")
    apply_eval_modalities(policy, FULL, checkpoint="/ckpt")
    data, goal = policy.process_raw_obs([_libero_obs()], "x")
    policy.step_batch(data, goal)
    assert stub.calls[0][2] == Modalities() and Modalities().complete


def test_masked_proprio_fill_is_the_policys_training_mean_not_a_proprio_mean_file():
    policy = VLAAdapterPolicy(_Counter(), "libero", proprio_mean=np.arange(8.0))
    apply_mean_proprio(policy, {**FULL, "proprio_state": False}, "/no/such/run/ckpt")
    np.testing.assert_array_equal(policy.eval_proprio_fill, np.arange(8.0))
    assert policy.modalities().proprio == "mean"


def test_layout_positions_keep_full_layout_positions_of_kept_tokens():
    n = 4  # patches per image
    full = layout_positions(3, True, True, True, n_patches=n)
    assert full == list(range(1 + 2 * n + 3 + 65))  # contiguous with everything present
    no_static = layout_positions(3, False, True, True, n_patches=n)
    assert no_static == [0] + list(range(1 + n, 1 + 2 * n + 3 + 65))
    no_wrist = layout_positions(3, True, False, True, n_patches=n)
    assert no_wrist == [0] + list(range(1, 1 + n)) + list(range(1 + 2 * n, 1 + 2 * n + 3 + 65))
    no_lang = layout_positions(3, True, True, False, n_patches=n)
    assert no_lang == list(range(1 + 2 * n)) + list(range(1 + 2 * n + 3, 1 + 2 * n + 3 + 65))


def test_mean_pixel_replaces_only_the_masked_images_channels_by_their_spatial_mean():
    pv = torch.arange(2 * 6 * 3 * 3, dtype=torch.float32).reshape(1, 12, 3, 3)
    out = mean_pixel(pv, [False, True])
    torch.testing.assert_close(out[:, :6], pv[:, :6])
    torch.testing.assert_close(out[:, 6:], pv[:, 6:].mean(dim=(-2, -1), keepdim=True).expand(1, 6, 3, 3))


def test_harness_attributes_match_what_the_evaluators_read_and_set():
    policy = VLAAdapterPolicy(_Counter(), "libero")
    assert policy.use_proprio is True
    assert policy.multistep == 8
    policy.set_eval_noise_seeds([1, 2])  # deterministic regression head: a no-op
    for attr in ("eval_modality_mask", "eval_proprio_fill"):
        assert getattr(policy, attr) is None


def test_select_unnorm_key():
    assert select_unnorm_key({"libero_10_no_noops": {}}, "libero") == "libero_10_no_noops"
    assert select_unnorm_key({"calvin_abc_rlds": {}, "x": {}}, "calvin") == "calvin_abc_rlds"
    with pytest.raises(ValueError):
        select_unnorm_key({"a": {}, "b": {}}, "libero")


def test_unknown_platform_rejected():
    with pytest.raises(ValueError):
        VLAAdapterPolicy(_Counter(), "aloha")
