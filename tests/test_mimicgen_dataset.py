"""End-to-end test of MimicgenDataModule against synthetic MimicGen-shaped hdf5 files.

Builds a minimal robomimic-format hdf5 per dataset (the same layout
robomimic's dataset_states_to_obs.py produces) and asserts the datamodule emits samples
in the exact shape/content FLOWER's LIBERO path has always produced -- in particular
that `actions` pass through bit-identical to the source hdf5 (FLOWER applies zero action
normalization anywhere in its pipeline).
"""

import h5py
import numpy as np
import pytest

from flower.datasets import mimicgen_tasks
from flower.datasets.mimicgen_data_module import KEY_MAP, MimicgenDataModule

T = 20  # timesteps per synthetic demo


def _write_synthetic_dataset(path, seed):
    rng = np.random.RandomState(seed)
    with h5py.File(path, "w") as f:
        grp = f.create_group("data")
        demo = grp.create_group("demo_0")
        demo.attrs["num_samples"] = T
        demo.create_dataset(
            "actions", data=rng.uniform(-1, 1, size=(T, 7)).astype(np.float32)
        )
        obs = demo.create_group("obs")
        obs.create_dataset(
            KEY_MAP["static"], data=rng.randint(0, 255, size=(T, 8, 8, 3), dtype=np.uint8)
        )
        obs.create_dataset(
            KEY_MAP["wrist"], data=rng.randint(0, 255, size=(T, 8, 8, 3), dtype=np.uint8)
        )
        obs.create_dataset(
            KEY_MAP["joints"], data=rng.uniform(-1, 1, size=(T, 7)).astype(np.float32)
        )
        obs.create_dataset(
            KEY_MAP["gripper"], data=rng.uniform(-1, 1, size=(T, 2)).astype(np.float32)
        )


@pytest.fixture
def two_dataset_dir(tmp_path, monkeypatch):
    datasets = ("square_d0", "threading_d1")
    monkeypatch.setattr(mimicgen_tasks, "CORE_DATASETS", datasets)
    for i, name in enumerate(datasets):
        _write_synthetic_dataset(tmp_path / f"{name}.hdf5", seed=i)
    return tmp_path, datasets


def _datamodule(data_dir, **kwargs):
    from omegaconf import OmegaConf

    datasets_cfg = OmegaConf.create(
        {
            "lang_dataset": {
                "batch_size": 4,
                "action_seq_len": 10,
                "obs_seq_len": 1,
                "obs_space": {
                    "rgb_obs": [KEY_MAP["static"], KEY_MAP["wrist"]],
                    "state_obs": [KEY_MAP["joints"], KEY_MAP["gripper"]],
                },
            }
        }
    )
    observation_space = OmegaConf.create({})
    return MimicgenDataModule(
        datasets=datasets_cfg,
        observation_space=observation_space,
        num_workers=0,
        transforms=None,
        data_dir=str(data_dir),
        **kwargs,
    )


def test_datamodule_builds_one_dataset_per_registered_task(two_dataset_dir):
    data_dir, datasets = two_dataset_dir
    dm = _datamodule(data_dir)
    dm.setup()
    assert "lang" in dm.train_datasets
    assert len(dm.train_datasets["lang"]) == len(datasets) * T  # pad_seq_length: every timestep is a window start


def test_pad_seq_length_false_only_samples_full_windows(two_dataset_dir):
    """CALVIN's recipe (pad: false): an action window never runs past the episode end,
    so a T-step demo yields T - act_seq_len + 1 windows instead of T."""
    data_dir, datasets = two_dataset_dir
    dm = _datamodule(data_dir, pad_seq_length=False)
    dm.setup()
    assert len(dm.train_datasets["lang"]) == len(datasets) * (T - 10 + 1)


def test_sample_shape_matches_libero_convention(two_dataset_dir):
    data_dir, _ = two_dataset_dir
    dm = _datamodule(data_dir)
    dm.setup()
    sample = dm.train_datasets["lang"][0]["lang"]

    assert sample["rgb_obs"]["rgb_static"].shape == (1, 8, 8, 3)
    assert sample["rgb_obs"]["rgb_gripper"].shape == (1, 8, 8, 3)
    assert sample["actions"].shape == (10, 7)
    assert sample["robot_obs"].shape == (1, 9)  # 7 joints + 2 gripper
    assert isinstance(sample["lang_text"], str) and sample["lang_text"]


def test_actions_pass_through_unmodified(two_dataset_dir):
    """FLOWER applies no action normalization anywhere -- the sample's actions must be
    bit-identical to what's stored in the source hdf5."""
    data_dir, datasets = two_dataset_dir
    with h5py.File(data_dir / f"{datasets[0]}.hdf5", "r") as f:
        raw_actions = f["data/demo_0/actions"][:10]

    dm = _datamodule(data_dir)
    dm.setup()
    sample = dm.train_datasets["lang"][0]["lang"]
    np.testing.assert_array_equal(sample["actions"], raw_actions)


def test_language_is_per_family_instruction(two_dataset_dir):
    data_dir, datasets = two_dataset_dir
    dm = _datamodule(data_dir)
    dm.setup()
    # First T samples come from the first dataset (square_d0) since demos are concatenated in order.
    sample = dm.train_datasets["lang"][0]["lang"]
    assert sample["lang_text"] == mimicgen_tasks.language(datasets[0])
    last_sample = dm.train_datasets["lang"][len(dm.train_datasets["lang"]) - 1]["lang"]
    assert last_sample["lang_text"] == mimicgen_tasks.language(datasets[1])


def test_train_dataloader_is_dict_val_is_list(two_dataset_dir):
    data_dir, _ = two_dataset_dir
    dm = _datamodule(data_dir)
    dm.setup()
    train_loaders = dm.train_dataloader()
    val_loaders = dm.val_dataloader()
    assert isinstance(train_loaders, dict) and "lang" in train_loaders
    assert isinstance(val_loaders, list) and len(val_loaders) == 1


def test_dataset_names_override_restricts_to_subset(two_dataset_dir):
    """dataset_names lets a caller (e.g. a smoke test with only one dataset rendered)
    restrict the datamodule to a subset, without needing every entry in
    mimicgen_tasks.CORE_DATASETS to have a hdf5 on disk."""
    data_dir, datasets = two_dataset_dir
    from omegaconf import OmegaConf

    from flower.datasets.mimicgen_data_module import KEY_MAP, MimicgenDataModule

    datasets_cfg = OmegaConf.create(
        {
            "lang_dataset": {
                "batch_size": 4,
                "action_seq_len": 10,
                "obs_seq_len": 1,
                "obs_space": {
                    "rgb_obs": [KEY_MAP["static"], KEY_MAP["wrist"]],
                    "state_obs": [KEY_MAP["joints"], KEY_MAP["gripper"]],
                },
            }
        }
    )
    dm = MimicgenDataModule(
        datasets=datasets_cfg,
        observation_space=OmegaConf.create({}),
        num_workers=0,
        transforms=None,
        data_dir=str(data_dir),
        dataset_names=[datasets[0]],  # registry (mocked) has 2 datasets; restrict to 1
    )
    dm.setup()

    assert len(dm.train_datasets["lang"]) == T  # only the first dataset's demos
    sample = dm.train_datasets["lang"][0]["lang"]
    assert sample["lang_text"] == mimicgen_tasks.language(datasets[0])
