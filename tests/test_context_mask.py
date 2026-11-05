"""The DiT's cross-attention ignores padded context tokens when given a context mask
(model.modality_dropout_strategy=subset pads each sample's kept tokens to the longest
row), and FLOWERVLA.dit_forward passes the encoder's context_mask to every block --
None (no mask, the previous behavior) for every other path."""

import types

import pytest
import torch

from flower.models.flower import FLOWERVLA
from flower.models.networks.transformers import FlowerCrossAttention


def test_masked_context_positions_do_not_change_the_output():
    torch.manual_seed(0)
    attn = FlowerCrossAttention(dim=16, n_heads=4, attn_pdrop=0.0, resid_pdrop=0.0).eval()
    x = torch.randn(2, 5, 16)
    context = torch.randn(2, 7, 16)
    mask = torch.tensor([[True] * 7, [True] * 3 + [False] * 4])
    other = context.clone()
    other[1, 3:] = torch.randn(4, 16) * 100  # change only row 1's padded positions

    out = attn(x, context, custom_attn_mask=mask)
    assert torch.allclose(out, attn(x, other, custom_attn_mask=mask), atol=1e-6)
    # Unmasked, the padded positions do leak into the output.
    assert not torch.allclose(attn(x, context), attn(x, other), atol=1e-3)


class _DitStub:
    dit_forward = FLOWERVLA.dit_forward

    def __init__(self, n_layers=3):
        self.device = torch.device("cpu")
        self._param = torch.zeros(1)
        self.use_proprio = False
        self.use_rope = True
        self.use_nope = False
        self.use_adaln_cond = False
        self.use_readout_token = False
        self.use_cross_attn = True
        self.action_type_adaln = False
        self.t_embedder = lambda t: torch.zeros(t.shape[0], 4)
        self.cond_norm = lambda c: c
        self.cond_linear = lambda c: c
        self.adaln = lambda c: None
        self.encode_actions = lambda z, a: (z, None)
        self.decode_actions = lambda cx, a, v: cx
        self.seen = []
        self.dit = [self._layer for _ in range(n_layers)]

    def parameters(self):
        yield self._param

    def _layer(self, cx, c, context=None, custom_cross_attn_mask=None, is_causal=False, global_adaln=None):
        self.seen.append(custom_cross_attn_mask)
        return cx


def _cond(B=2, S=6, mask=None):
    cond = {
        "features": torch.randn(B, S, 4),
        "frequency_embeds": torch.zeros(B, 1, 4),
        "action_type": torch.ones(B, 10, 7),
        "proprio": None,
    }
    if mask is not None:
        cond["context_mask"] = mask
    return cond


@pytest.mark.parametrize("with_mask", [True, False])
def test_dit_forward_passes_the_context_mask_to_every_block(with_mask):
    stub = _DitStub()
    mask = torch.tensor([[True] * 6, [True] * 2 + [False] * 4]) if with_mask else None
    stub.dit_forward(torch.zeros(2, 10, 4), torch.zeros(2), _cond(mask=mask))
    assert len(stub.seen) == 3
    for m in stub.seen:
        assert (m is mask) if with_mask else (m is None)
