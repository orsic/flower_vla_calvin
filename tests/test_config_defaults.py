"""
Unit tests for the config defaults that make rollout eval fire only at the final
training epoch (see flower/training_libero.py / flower/training_calvin.py, which
register the `sub` OmegaConf resolver used by rollout_lh_skip_epochs).
"""

import flower.training_libero  # noqa: F401 — registers the `sub` resolver
import flower.training_calvin  # noqa: F401 — same resolver, registering twice is a no-op
from hydra import compose, initialize


def test_libero_skip_epochs_is_final_epoch():
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config_libero")
        assert cfg.rollout_lh_skip_epochs == cfg.max_epochs - 1
        assert cfg.callbacks.rollout_lh.skip_epochs == cfg.max_epochs - 1


def test_libero_skip_epochs_tracks_max_epochs_override():
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config_libero", overrides=["max_epochs=20"])
        assert cfg.rollout_lh_skip_epochs == 19


def test_calvin_skip_epochs_is_final_epoch():
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config_calvin")
        assert cfg.rollout_lh_skip_epochs == cfg.max_epochs - 1
        assert cfg.callbacks.rollout_lh.skip_epochs == cfg.max_epochs - 1
