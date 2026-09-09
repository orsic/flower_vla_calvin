"""
Unit tests for the config defaults that gate rollout eval (see flower/training_libero.py /
flower/training_calvin.py, which register the `sub` OmegaConf resolver used by CALVIN's
rollout_lh_skip_epochs).

LIBERO rollout eval is disabled outright (skip_epochs == max_epochs, so
on_validation_epoch_end's skip branch always fires); evaluation runs after training via
./run.sh pipeline instead. CALVIN still evaluates at the final epoch.
"""

import flower.training_libero  # noqa: F401 — registers the `sub` resolver
import flower.training_calvin  # noqa: F401 — same resolver, registering twice is a no-op
from hydra import compose, initialize


def test_libero_skip_epochs_disables_rollout():
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config_libero")
        assert cfg.rollout_lh_skip_epochs == cfg.max_epochs
        assert cfg.callbacks.rollout_lh.skip_epochs == cfg.max_epochs


def test_libero_skip_epochs_tracks_max_epochs_override():
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config_libero", overrides=["max_epochs=20"])
        assert cfg.rollout_lh_skip_epochs == 20


def test_calvin_skip_epochs_is_final_epoch():
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config_calvin")
        assert cfg.rollout_lh_skip_epochs == cfg.max_epochs - 1
        assert cfg.callbacks.rollout_lh.skip_epochs == cfg.max_epochs - 1
