"""Hydra composition tests for conf/config_mimicgen.yaml.

Asserts the MimicGen training recipe (steps, batch size, precision, action/obs shapes)
is byte-identical to LIBERO's, per the plan's "same recipe as LIBERO" requirement, and
that the datamodule/model targets resolve to the new MimicGen classes.
"""

from hydra import compose, initialize


def test_mimicgen_config_composes():
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config_mimicgen")
        assert cfg.datamodule._target_ == "flower.datasets.mimicgen_data_module.MimicgenDataModule"
        assert cfg.callbacks.checkpoint._target_ == "pytorch_lightning.callbacks.ModelCheckpoint"
        assert "rollout_lh" not in cfg.callbacks


def test_mimicgen_recipe_matches_libero():
    with initialize(config_path="../conf"):
        libero_cfg = compose(config_name="config_libero")
        mimicgen_cfg = compose(config_name="config_mimicgen")

        for key in (
            "batch_size",
            "devices",
            "obs_seq_len",
            "act_seq_len",
            "max_epochs",
            "seed",
        ):
            assert mimicgen_cfg[key] == libero_cfg[key], key

        for key in (
            "precision",
            "max_epochs",
            "sync_batchnorm",
            "strategy",
            "limit_train_batches",
            "limit_val_batches",
        ):
            assert mimicgen_cfg.trainer[key] == libero_cfg.trainer[key], key

        assert mimicgen_cfg.proprio_dims == libero_cfg.proprio_dims == 9
        assert mimicgen_cfg.logger.project == libero_cfg.logger.project
        assert mimicgen_cfg.logger.entity == libero_cfg.logger.entity


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
