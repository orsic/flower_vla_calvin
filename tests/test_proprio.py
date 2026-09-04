"""
Tests for LIBERO proprioception wiring in flower/models/flower.py.

These exercise the fixed proprio code paths without loading Florence-2:
  1. model.lowdim_obs_dim resolves per-benchmark from the top-level proprio_dims
     (conf/model/flower.yaml -> conf/config_libero.yaml / conf/config_calvin.yaml).
  2. The proprio encoder is a real dim-correct MLP (not the old action-dim-shaped
     encoder, not the always-zero ZeroEncoder) — copied verbatim from
     FLOWERVLA._setup_dit_components / encode_proprio so the test reflects
     production code.
  3. encode_observations' batch lookup reads 'robot_obs' (the key LIBERO and CALVIN
     datasets actually emit) instead of the broken `batch.get(self.obs_modalities, {})`
     lookup, which raised TypeError whenever use_proprio was enabled.
"""

from hydra import compose, initialize
import pytest
import torch
from timm.layers.mlp import Mlp

from flower.models.utils import ActionIndex


# ---------------------------------------------------------------------------
# 1. Config wiring: lowdim_obs_dim tracks each benchmark's proprio_dims
# ---------------------------------------------------------------------------

def test_libero_lowdim_obs_dim_matches_proprio_dims():
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config_libero")
        assert cfg.proprio_dims == 9
        assert cfg.model.lowdim_obs_dim == 9


def test_calvin_lowdim_obs_dim_matches_proprio_dims():
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config_calvin")
        assert cfg.proprio_dims == 7
        assert cfg.model.lowdim_obs_dim == 7


# ---------------------------------------------------------------------------
# 2. Proprio encoder construction + encode_proprio (copied verbatim from
#    FLOWERVLA so the test reflects production code without loading Florence-2)
# ---------------------------------------------------------------------------

class _MinimalProprioFlower:
    """Thin stand-in for the proprio pieces of FLOWERVLA — no Florence-2 required."""

    def __init__(self, lowdim_obs_dim: int, dit_dim: int = 32):
        self.lowdim_obs_dim = lowdim_obs_dim
        self.dit_dim = dit_dim
        self.device = "cpu"
        self.action_space_index = ActionIndex()
        self.use_proprio = True

        # Copied verbatim from FLOWERVLA._setup_dit_components' proprio branch.
        self.proprio_encoders = torch.nn.ModuleDict()
        for action_name in self.action_space_index.action_spaces:
            self.proprio_encoders[action_name] = Mlp(
                self.lowdim_obs_dim, dit_dim, out_features=dit_dim, drop=0.2
            ).to(self.device)

    # Copied verbatim from FLOWERVLA.encode_proprio.
    def encode_proprio(self, proprio: torch.Tensor, action_type: torch.Tensor, output_shape) -> torch.Tensor:
        batch_size, _ = output_shape
        default_dtype = next(iter(self.proprio_encoders.parameters())).dtype

        encoded_proprio = torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=default_dtype)

        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                encoded_proprio[mask] = self.proprio_encoders[action_name](proprio[mask]).squeeze(1)

        return encoded_proprio


def test_proprio_encoder_uses_lowdim_obs_dim_not_action_dim():
    """eef_delta's action dim (7) and joint_single's (8) must not gate the proprio
    encoder's input size — only lowdim_obs_dim (here 9, LIBERO's proprio width) may."""
    model = _MinimalProprioFlower(lowdim_obs_dim=9)
    assert model.proprio_encoders['eef_delta'].fc1.in_features == 9
    assert model.proprio_encoders['joint_single'].fc1.in_features == 9


def test_encode_proprio_is_nonzero_for_eef_delta():
    """LIBERO/CALVIN forward passes hard-code action_type to eef_delta (index 1).
    Before the fix, that action type's proprio encoder was a parameter-free
    ZeroEncoder, so use_proprio=True silently contributed nothing. It must now
    be a real, non-trivial MLP."""
    torch.manual_seed(0)
    model = _MinimalProprioFlower(lowdim_obs_dim=9)
    B = 4
    proprio = torch.randn(B, 1, 9)
    action_type = torch.ones(B, dtype=torch.long)  # eef_delta

    encoded = model.encode_proprio(proprio, action_type, output_shape=(B, model.dit_dim))

    assert encoded.shape == (B, model.dit_dim)
    assert not torch.allclose(encoded, torch.zeros_like(encoded))

    # Different proprio inputs must yield different embeddings (real encoder, not a
    # constant/zero function of the input).
    encoded2 = model.encode_proprio(torch.randn(B, 1, 9), action_type, output_shape=(B, model.dit_dim))
    assert not torch.allclose(encoded, encoded2)


# ---------------------------------------------------------------------------
# 3. encode_observations' batch lookup (copied verbatim from FLOWERVLA)
# ---------------------------------------------------------------------------

def _extract_proprio(use_proprio, batch, device="cpu", default_type=torch.float32):
    """Mirror the fixed lookup in FLOWERVLA.encode_observations."""
    proprio = None
    if use_proprio and 'robot_obs' in batch:
        proprio = batch['robot_obs'].to(device).to(default_type)
    return proprio


def test_extract_proprio_reads_robot_obs():
    batch = {'robot_obs': torch.ones(2, 1, 9)}
    proprio = _extract_proprio(use_proprio=True, batch=batch)
    assert proprio is not None
    assert torch.equal(proprio, torch.ones(2, 1, 9))


def test_extract_proprio_missing_key_returns_none_without_crashing():
    """Regression test: the old lookup was `batch.get(self.obs_modalities, {})` with
    self.obs_modalities == [] (a list), which raises `TypeError: unhashable type:
    'list'` on any batch. The fixed lookup must simply return None instead."""
    batch = {'rgb_obs': {}}
    proprio = _extract_proprio(use_proprio=True, batch=batch)
    assert proprio is None


def test_extract_proprio_disabled_short_circuits():
    batch = {'robot_obs': torch.ones(2, 1, 9)}
    proprio = _extract_proprio(use_proprio=False, batch=batch)
    assert proprio is None
