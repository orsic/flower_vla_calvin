"""Integration test for flower_eval_mimicgen._create_mimicgen_env against a real
robomimic/MimicGen env (no mocks -- test_mimicgen_eval.py covers the loop with a fake).

Pins three contracts the mocked tests can't see:
- robomimic's obs-modality registry is initialized in the worker (get_observation
  raises TypeError on a None OBS_KEYS_TO_MODALITIES otherwise),
- images come back as raw HWC uint8, the format the training data was rendered in
  (dataset_states_to_obs) and obs_translation.apply_transforms expects, at the per-camera
  sizes recorded in the dataset's env_args, and
- on a real rendered dataset, the eval env reproduces the stored training frames
  pixel-for-pixel -- so training and rollout images go through identical rendering
  before the shared transform chain.

Skipped outside flower-vla-mimicgen:latest or without a GPU for EGL rendering.
"""

import json
import os

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
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "camera_heights": [64, 32],
            "camera_widths": [64, 32],
        },
    }
    path = tmp_path / "square_d0.hdf5"
    with h5py.File(path, "w") as f:
        f.create_group("data").attrs["env_args"] = json.dumps(env_meta)
    return str(path)


def test_reset_returns_raw_hwc_uint8_images_and_proprio(square_dataset, monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "egl")
    env = fem._create_mimicgen_env(square_dataset)
    try:
        obs = env.reset()
        for key, size in (("agentview_image", 64), ("robot0_eye_in_hand_image", 32)):
            assert obs[key].dtype == np.uint8, key
            assert obs[key].shape == (size, size, 3), key
        assert obs["robot0_joint_pos"].shape == (7,)
        assert obs["robot0_gripper_qpos"].shape == (2,)

        _, _, done, _ = env.step(np.zeros(7))
        assert done is False
    finally:
        env.close()  # the vector-env worker calls this on rebuild/shutdown


RENDERED_DATASET = "/mimicgen_hdf5/square_d0.hdf5"


@pytest.mark.skipif(not os.path.exists(RENDERED_DATASET), reason="needs a rendered square_d0")
def test_eval_env_reproduces_training_frames_pixel_exact(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "egl")
    with h5py.File(RENDERED_DATASET, "r") as f:
        demo = f["data/demo_0"]
        initial_state = {"states": demo["states"][0], "model": demo.attrs["model_file"]}
        stored = {k: demo["obs"][k][0] for k in ("agentview_image", "robot0_eye_in_hand_image")}

    env = fem._create_mimicgen_env(RENDERED_DATASET)
    try:
        env.reset()
        # Exactly how dataset_states_to_obs produced obs[0] of the training data.
        obs = env.reset_to(initial_state)
        for key, frame in stored.items():
            assert obs[key].shape == frame.shape, key
            assert np.array_equal(obs[key], frame), key
    finally:
        env.close()
