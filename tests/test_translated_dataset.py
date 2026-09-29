"""Regression tests for TranslatedSequenceVLDataset after its move out of
libero_data_module.py into its own module with a parametrizable key_map.

These pin the exact sample shape/content LIBERO training has always produced, using a
fake sequence_dataset so no real hdf5/robomimic dependency is needed.
"""

import numpy as np
import pytest

from flower.datasets.translated_dataset import (
    DEFAULT_KEY_MAP,
    TranslatedSequenceVLDataset,
)


class _FakeSequenceDataset:
    """Minimal stand-in for robomimic's SequenceDataset, parametrized by obs key names."""

    def __init__(self, static_key, wrist_key, joints_key, gripper_key, seq_len=10):
        self.n_demos = 1
        self.total_num_sequences = 1
        self.goal_mode = None
        self._seq_len = seq_len
        self._keys = (static_key, wrist_key, joints_key, gripper_key)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        static_key, wrist_key, joints_key, gripper_key = self._keys
        return {
            "obs": {
                static_key: np.zeros((self._seq_len, 8, 8, 3), dtype=np.uint8),
                wrist_key: np.ones((self._seq_len, 8, 8, 3), dtype=np.uint8),
                joints_key: np.arange(self._seq_len * 7, dtype=np.float32).reshape(
                    self._seq_len, 7
                ),
                gripper_key: np.arange(self._seq_len * 2, dtype=np.float32).reshape(
                    self._seq_len, 2
                ),
            },
            "actions": np.zeros((self._seq_len, 7), dtype=np.float32),
        }


def _build(key_map, seq_dataset):
    return TranslatedSequenceVLDataset(
        seq_dataset,
        task_emb=None,
        task_description="do the thing",
        obs_seq_len=1,
        act_seq_len=10,
        transforms=None,
        key_map=key_map,
    )


def test_default_key_map_matches_libero_hdf5_keys():
    # Pins the exact keys LIBERO training has always used.
    assert DEFAULT_KEY_MAP == {
        "static": "agentview_rgb",
        "wrist": "eye_in_hand_rgb",
        "joints": "joint_states",
        "gripper": "gripper_states",
    }


def test_libero_default_keys_produce_expected_sample():
    seq_dataset = _FakeSequenceDataset(
        "agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states"
    )
    ds = _build(key_map=None, seq_dataset=seq_dataset)
    sample = ds[0]["lang"]

    assert sample["rgb_obs"]["rgb_static"].shape == (1, 8, 8, 3)
    assert sample["rgb_obs"]["rgb_gripper"].shape == (1, 8, 8, 3)
    assert sample["actions"].shape == (10, 7)
    assert sample["lang_text"] == "do the thing"
    # robot_obs = joint_states[:1] concatenated with gripper_states[0] -> [1, 9]
    assert sample["robot_obs"].shape == (1, 9)
    np.testing.assert_allclose(sample["robot_obs"][0, :7], np.arange(7, dtype=np.float32))
    np.testing.assert_allclose(sample["robot_obs"][0, 7:], np.array([0.0, 1.0], dtype=np.float32))


def test_custom_key_map_reads_mimicgen_style_keys():
    mimicgen_key_map = {
        "static": "agentview_image",
        "wrist": "robot0_eye_in_hand_image",
        "joints": "robot0_joint_pos",
        "gripper": "robot0_gripper_qpos",
    }
    seq_dataset = _FakeSequenceDataset(
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_joint_pos",
        "robot0_gripper_qpos",
    )
    ds = _build(key_map=mimicgen_key_map, seq_dataset=seq_dataset)
    sample = ds[0]["lang"]

    assert sample["rgb_obs"]["rgb_static"].shape == (1, 8, 8, 3)
    assert sample["robot_obs"].shape == (1, 9)


def test_wrong_key_map_raises_keyerror():
    seq_dataset = _FakeSequenceDataset(
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_joint_pos",
        "robot0_gripper_qpos",
    )
    # Default (LIBERO) key_map against MimicGen-style obs keys should fail loudly.
    ds = _build(key_map=None, seq_dataset=seq_dataset)
    with pytest.raises(KeyError):
        ds[0]
