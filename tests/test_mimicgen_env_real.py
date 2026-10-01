"""Integration test for flower_eval_mimicgen._create_mimicgen_env against a real
robomimic/MimicGen env (no mocks -- test_mimicgen_eval.py covers the loop with a fake).

Pins two contracts the mocked tests can't see:
- robomimic's obs-modality registry is initialized in the worker (get_observation
  raises TypeError on a None OBS_KEYS_TO_MODALITIES otherwise), and
- images come back as raw HWC uint8, the format the training data was rendered in
  (dataset_states_to_obs) and obs_translation.apply_transforms expects.

Skipped outside flower-vla-mimicgen:latest or without a GPU for EGL rendering.
"""

import json

import h5py
import numpy as np
import pytest

pytest.importorskip("mimicgen")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("offscreen rendering needs a GPU (EGL)", allow_module_level=True)

from robosuite import load_controller_config  # noqa: E402

from flower.evaluation import flower_eval_mimicgen as fem  # noqa: E402


@pytest.fixture
def square_dataset(tmp_path):
    env_meta = {
        "env_name": "Square_D0",
        "type": 1,  # robomimic EnvType.ROBOSUITE_TYPE
        "env_kwargs": {
            "robots": "Panda",
            "controller_configs": load_controller_config(default_controller="OSC_POSE"),
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": True,
            "use_object_obs": True,
            "use_camera_obs": True,
            "control_freq": 20,
            "reward_shaping": False,
        },
    }
    path = tmp_path / "square_d0.hdf5"
    with h5py.File(path, "w") as f:
        f.create_group("data").attrs["env_args"] = json.dumps(env_meta)
    return str(path)


def test_reset_returns_raw_hwc_uint8_images_and_proprio(square_dataset, monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "egl")
    env = fem._create_mimicgen_env(square_dataset, 64, 64)
    try:
        obs = env.reset()
        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            assert obs[key].dtype == np.uint8, key
            assert obs[key].shape == (64, 64, 3), key
        assert obs["robot0_joint_pos"].shape == (7,)
        assert obs["robot0_gripper_qpos"].shape == (2,)

        _, _, done, _ = env.step(np.zeros(7))
        assert done is False
    finally:
        env.close()  # the vector-env worker calls this on rebuild/shutdown
