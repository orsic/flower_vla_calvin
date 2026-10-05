"""Hydra composition tests for conf/config_mimicgen.yaml.

Asserts the MimicGen training recipe follows CALVIN's (conf/config_calvin.yaml): optimizer
updates, resolution, preprocessing, trainer settings. CALVIN's global batch of 32 (8/GPU x
4 GPUs) is reached on the single GPU MimicGen is usually trained on as 16 x 2 gradient
accumulation, with limit_train_batches scaled so every accumulation setting gives the same
35k optimizer updates. Also checks the datamodule/model targets resolve to the MimicGen
classes.
"""

from hydra import compose, initialize
from omegaconf import OmegaConf

import flower.training_libero  # noqa: F401 -- registers the config's ${mul:...} resolver


def test_mimicgen_config_composes():
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config_mimicgen")
        assert cfg.datamodule._target_ == "flower.datasets.mimicgen_data_module.MimicgenDataModule"
        assert cfg.callbacks.checkpoint._target_ == "pytorch_lightning.callbacks.ModelCheckpoint"
        assert "rollout_lh" not in cfg.callbacks


def test_mimicgen_recipe_matches_calvin():
    with initialize(config_path="../conf"):
        calvin_cfg = compose(config_name="config_calvin")
        mimicgen_cfg = compose(config_name="config_mimicgen")

        for key in ("obs_seq_len", "act_seq_len", "multistep", "max_epochs"):
            assert mimicgen_cfg[key] == calvin_cfg[key], key
        for key in ("precision", "max_epochs", "sync_batchnorm", "limit_val_batches"):
            assert mimicgen_cfg.trainer[key] == calvin_cfg.trainer[key], key

        # CALVIN's global batch (8/GPU x 4 GPUs) on one GPU, as 16 x 2 accumulation, and
        # CALVIN's optimizer updates per epoch (limit_train_batches counts micro-batches).
        trainer = mimicgen_cfg.trainer
        assert (mimicgen_cfg.batch_size, trainer.accumulate_grad_batches) == (16, 2)
        assert mimicgen_cfg.batch_size * trainer.accumulate_grad_batches == calvin_cfg.batch_size * calvin_cfg.devices
        assert trainer.limit_train_batches // trainer.accumulate_grad_batches == calvin_cfg.trainer.limit_train_batches

        # Same image preprocessing, including the 224 model input resolution.
        assert mimicgen_cfg.datamodule.transforms == calvin_cfg.datamodule.transforms
        for split in ("train", "val"):
            for cam in ("rgb_static", "rgb_gripper"):
                assert mimicgen_cfg.datamodule.transforms[split][cam][0].size == 224

        # CALVIN only samples windows fully inside an episode (pad: false).
        assert calvin_cfg.datamodule.datasets.lang_dataset.pad is False
        assert mimicgen_cfg.datamodule.pad_seq_length is False

        assert mimicgen_cfg.proprio_dims == 9
        assert mimicgen_cfg.logger.project == calvin_cfg.logger.project
        assert mimicgen_cfg.logger.entity == calvin_cfg.logger.entity


def test_optimizer_updates_independent_of_grad_accumulation():
    """Overriding batch/accumulation (e.g. 32 x 1 on a GPU that fits it) keeps the same
    1000 optimizer updates per epoch -- and with them the LR schedule and EMA, which both
    count optimizer steps."""
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config_mimicgen", overrides=["batch_size=32", "accumulate_grad_batches=1"])
        assert cfg.trainer.accumulate_grad_batches == 1
        assert cfg.trainer.limit_train_batches == 1000

        cfg = compose(config_name="config_mimicgen", overrides=["batch_size=8", "accumulate_grad_batches=4"])
        assert cfg.trainer.limit_train_batches == 4000


def test_mimicgen_rollout_disabled_same_as_libero():
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config_mimicgen")
        assert cfg.rollout_lh_skip_epochs == cfg.max_epochs


def test_eval_mimicgen_composes_with_required_overrides():
    with initialize(config_path="../conf"):
        cfg = compose(
            config_name="eval_mimicgen",
            overrides=["train_folder=/fake/train", "checkpoint=/fake/last.ckpt"],
        )
        assert cfg.log_wandb is False
        assert cfg.datasets is None  # null -> all of mimicgen_tasks.CORE_DATASETS
        for key in ("rgb_static", "rgb_gripper", "language", "proprio"):
            assert cfg.eval_modalities[key] is True
        # get_default_mode_and_env reads this unconditionally even with
        # prep_dm_and_deps=False -- see the placeholder's comment in eval_mimicgen.yaml.
        assert cfg.eval_cfg_overwrite.datamodule.datasets.lang_dataset.lang_folder
