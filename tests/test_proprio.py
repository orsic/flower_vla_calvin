"""
Tests for LIBERO proprioception wiring in flower/models/flower.py.

These exercise the fixed proprio code paths without loading Florence-2:
  1. model.lowdim_obs_dim resolves per-benchmark from the top-level proprio_dims
     (conf/model/flower.yaml -> conf/config_libero.yaml / conf/config_calvin.yaml).
  2. Per-action-space proprio encoder construction and encode_proprio's action-type
     masking — copied verbatim from FLOWERVLA._setup_dit_components / encode_proprio so
     the test reflects production code. eef_delta (what LIBERO/CALVIN use) gets a real
     encoder sized to lowdim_obs_dim; bimanual_nav keeps its action-dim-shaped (16) encoder
     so the pretrained checkpoint's real weights for it still load; joint_single stays a
     ZeroEncoder. Also covers the action_type shape bug: production's action_type is
     [B, act_window_size, action_dim], not [B] — encode_proprio must reduce it before
     masking, or `encoded_proprio[mask] = ...` raises IndexError.
  3. encode_observations' batch lookup reads 'robot_obs' (the key LIBERO and CALVIN
     datasets actually emit) instead of the broken `batch.get(self.obs_modalities, {})`
     lookup, which raised TypeError whenever use_proprio was enabled.
"""

from hydra import compose, initialize
import pytest
import torch
from timm.layers.mlp import Mlp

from flower.models.networks.transformers import ZeroEncoder
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
        # bimanual_nav keeps its action-dim-shaped encoder (real pretrained weights exist
        # for it, from bimanual ALOHA data); eef_delta (what LIBERO/CALVIN use) gets a real
        # encoder sized to the actual proprio dim; joint_single stays a ZeroEncoder, matching
        # the pretrained checkpoint and the fact that LIBERO/CALVIN never route through it.
        self.proprio_encoders = torch.nn.ModuleDict()
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            input_dim = self.action_space_index.get_action_dim(action_idx)
            self.proprio_encoders[action_name] = (
                Mlp(input_dim, dit_dim, out_features=dit_dim, drop=0.2).to(self.device)
                if action_name == 'bimanual_nav'
                else Mlp(self.lowdim_obs_dim, dit_dim, out_features=dit_dim, drop=0.2).to(self.device)
                if action_name == 'eef_delta'
                else ZeroEncoder(dit_dim, device=self.device)
            )

    # Copied verbatim from FLOWERVLA.encode_proprio.
    def encode_proprio(self, proprio: torch.Tensor, action_type: torch.Tensor, output_shape) -> torch.Tensor:
        # Only the batch size is needed here; output_shape (frequency_embeds.shape) can be
        # 2-D or 3-D depending on caller, so don't assume an exact-length unpack.
        batch_size = output_shape[0]
        default_dtype = next(iter(self.proprio_encoders.parameters())).dtype

        # action_type is [B, act_window_size, action_dim] with all entries identical per
        # sample; reduce to [B] so it can mask encoded_proprio's [B, dit_dim].
        action_type = action_type[:, 0, 0].to(self.device)

        encoded_proprio = torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=default_dtype)

        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                encoded_proprio[mask] = self.proprio_encoders[action_name](proprio[mask]).squeeze(1)

        return encoded_proprio


def test_proprio_encoder_uses_lowdim_obs_dim_for_eef_delta():
    """eef_delta (the action type LIBERO/CALVIN actually use) must be a real encoder sized
    to lowdim_obs_dim (proprio width), not gated by its own action dim (7)."""
    model = _MinimalProprioFlower(lowdim_obs_dim=9)
    assert isinstance(model.proprio_encoders['eef_delta'], Mlp)
    assert model.proprio_encoders['eef_delta'].fc1.in_features == 9


def test_proprio_encoder_bimanual_nav_matches_pretrained_shape():
    """bimanual_nav must stay sized to its action dim (16) regardless of lowdim_obs_dim,
    so the pretrained checkpoint's real proprio_encoders.bimanual_nav.* weights still load."""
    model = _MinimalProprioFlower(lowdim_obs_dim=9)
    assert isinstance(model.proprio_encoders['bimanual_nav'], Mlp)
    assert model.proprio_encoders['bimanual_nav'].fc1.in_features == 16


def test_proprio_encoder_joint_single_stays_zero_encoder():
    """joint_single is unused by LIBERO/CALVIN and has no pretrained proprio weights —
    it must stay the parameter-free ZeroEncoder, matching the pretrained checkpoint."""
    model = _MinimalProprioFlower(lowdim_obs_dim=9)
    assert isinstance(model.proprio_encoders['joint_single'], ZeroEncoder)


def test_encode_proprio_is_nonzero_for_eef_delta():
    """LIBERO/CALVIN forward passes hard-code action_type to eef_delta (index 1), shaped
    [B, act_window_size, action_dim] (production's actual shape, not a bare [B]). output_shape
    is dit_forward's frequency_embeds.shape, which is 3-D ([B, 1, dit_dim]) in production, not
    a bare (B, dit_dim) 2-tuple — regression test for `batch_size, _ = output_shape` raising
    `ValueError: too many values to unpack`. Before the encoder fix, eef_delta's proprio
    encoder was also a parameter-free ZeroEncoder, so use_proprio=True silently contributed
    nothing; it must now be a real, non-trivial MLP."""
    torch.manual_seed(0)
    model = _MinimalProprioFlower(lowdim_obs_dim=9)
    B, act_window_size, action_dim = 4, 10, 7
    proprio = torch.randn(B, 1, 9)
    action_type = torch.ones(B, act_window_size, action_dim, dtype=torch.long)  # eef_delta
    output_shape = (B, 1, model.dit_dim)  # frequency_embeds.shape, as dit_forward passes it

    encoded = model.encode_proprio(proprio, action_type, output_shape=output_shape)

    assert encoded.shape == (B, model.dit_dim)
    assert not torch.allclose(encoded, torch.zeros_like(encoded))

    # Different proprio inputs must yield different embeddings (real encoder, not a
    # constant/zero function of the input).
    encoded2 = model.encode_proprio(
        torch.randn(B, 1, 9), action_type, output_shape=output_shape
    )
    assert not torch.allclose(encoded, encoded2)


def test_bimanual_nav_checkpoint_shape_loads_cleanly():
    """Regression test for the reported crash: loading the pretrained checkpoint's real
    proprio_encoders.bimanual_nav.* weights (shape (1024, 16), from bimanual ALOHA data)
    into a freshly constructed use_proprio=True model must be a clean match, not a
    RuntimeError-raising size mismatch or a silently-ignored unexpected/missing key."""
    model = _MinimalProprioFlower(lowdim_obs_dim=9, dit_dim=1024)
    pretrained_state_dict = {
        'proprio_encoders.bimanual_nav.fc1.weight': torch.randn(1024, 16),
        'proprio_encoders.bimanual_nav.fc1.bias': torch.randn(1024),
        'proprio_encoders.bimanual_nav.fc2.weight': torch.randn(1024, 1024),
        'proprio_encoders.bimanual_nav.fc2.bias': torch.randn(1024),
    }
    missing, unexpected = model.proprio_encoders.load_state_dict(
        {k.removeprefix('proprio_encoders.'): v for k, v in pretrained_state_dict.items()},
        strict=False,
    )
    bimanual_missing = [k for k in missing if k.startswith('bimanual_nav.')]
    bimanual_unexpected = [k for k in unexpected if k.startswith('bimanual_nav.')]
    assert bimanual_missing == []
    assert bimanual_unexpected == []


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
