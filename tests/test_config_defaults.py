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


def test_shipped_config_still_means_12_pretrained_plus_6_random_appended_one_lr():
    """The shipped config's action-expert capacity knobs (see flower/models/utils.py's
    dit_layer_sources) must still mean today's behavior: 12 pretrained DiT blocks, 6
    randomly-initialized ones appended after them, and a single learning rate for the
    whole model -- except rope_theta, which changes from 32.0 to 1000.0 to match the
    persistent rope buffers the 12 pretrained blocks load (see conf/model/flower.yaml)."""
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config_libero")
        assert cfg.model.n_layers == 18
        assert cfg.model.pretrained_dit_layers == 12
        assert cfg.model.extra_layer_init == "random"
        assert cfg.model.extra_layer_placement == "append"
        assert cfg.model.lora_dim == 256
        assert cfg.model.mlp_hidden_dim is None
        assert cfg.model.extra_layer_lora_dim is None
        assert cfg.model.extra_layer_mlp_hidden_dim is None
        assert cfg.model.optimizer.dit_learning_rate is None
        assert cfg.model.optimizer.new_layer_learning_rate is None
        assert cfg.model.rope_theta == 1000.0
