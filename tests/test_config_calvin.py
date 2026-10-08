"""Hydra composition tests for conf/config_calvin.yaml and conf/eval_calvin.yaml.

CALVIN trains with flower/training_libero.py (as MimicGen does) and evaluates after
training via ./run.sh pipeline-calvin, so the training config must: read the dataset from
the /calvin_data container mount, keep 1000 optimizer updates per epoch under any
gradient-accumulation setting (2 GPUs x 8 x 2 = CALVIN's global batch of 32), carry no
in-training rollout (calvin_env is only needed by the eval), and save last.ckpt as the
pipeline's input.
"""

from hydra import compose, initialize

import flower.training_libero  # noqa: F401 -- registers the config's ${mul:...}/${sub:...} resolvers


def test_calvin_reads_dataset_from_container_mount():
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config_calvin")
        assert cfg.root_data_dir == "/calvin_data/task_ABCD_D"
        assert cfg.datamodule.root_data_dir == "/calvin_data/task_ABCD_D"


def test_calvin_has_no_in_training_rollout_and_saves_last():
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config_calvin")
        assert "rollout_lh" not in cfg.callbacks
        assert cfg.callbacks.checkpoint.save_last is True
        assert cfg.callbacks.checkpoint.save_top_k == 0
        assert cfg.callbacks.checkpoint.get("monitor") is None


def test_calvin_optimizer_updates_independent_of_grad_accumulation():
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config_calvin")
        assert (cfg.batch_size, cfg.trainer.accumulate_grad_batches) == (8, 1)
        assert cfg.trainer.limit_train_batches == 1000

        cfg = compose(config_name="config_calvin", overrides=["accumulate_grad_batches=2"])
        assert cfg.trainer.accumulate_grad_batches == 2
        assert cfg.trainer.limit_train_batches == 2000


def test_eval_calvin_defaults():
    with initialize(config_path="../conf"):
        cfg = compose(
            config_name="eval_calvin",
            overrides=["train_folder=/fake/train", "checkpoint=/fake/last.ckpt"],
        )
        assert cfg.dataset_path == "/calvin_data/task_ABCD_D"
        assert cfg.log_dir == "/saves/eval_logs"
        assert cfg.num_sequences == 1000
        assert cfg.log_wandb is False
        assert cfg.reeval is False
        assert cfg.csv_dir is None
        for key in ("rgb_static", "rgb_gripper", "language", "proprio"):
            assert cfg.eval_modalities[key] is True
