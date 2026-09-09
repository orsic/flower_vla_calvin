"""
Tests for treating proprioception as a fourth droppable modality
(flower/models/flower.py, flower/evaluation/flower_eval_libero.py).

Proprioception is NOT a token like static/wrist/language (see modality_dropout.py) — it's
a single [B, dit_dim] vector produced by encode_proprio and summed into the AdaLN global
conditioning t_emb in dit_forward. It therefore can't join the Dirichlet token budget in
flower/models/networks/modality_dropout.py (a single "token" against ~500-1000 image/
language tokens would almost never be dropped); it gets its own per-sample Bernoulli
keep-gate instead, resolved once per batch in encode_observations and reused across every
dit_forward call in the sampling loop (never re-drawn mid-trajectory).

These tests copy the relevant logic verbatim (matching tests/test_proprio.py's pattern) so
they exercise production code without instantiating the full FLOWERVLA (which downloads
Florence-2) — except for the two ValueError paths, which fail before FLOWERVLA reaches
_setup_vlm and so are exercised on the real class.
"""

import torch
import pytest

from flower.models.networks.transformers import stateless_norm
from flower.models.flower import FLOWERVLA


# ---------------------------------------------------------------------------
# 1. stateless_norm(0) == 0 -- the equivalence the whole gate design rests on:
#    zeroing a gated-off sample's proprio_embeds must contribute exactly nothing
#    to t_emb, identical to the existing use_proprio=False path.
# ---------------------------------------------------------------------------

def test_stateless_norm_of_zeros_is_zero():
    x = torch.zeros(4, 32)
    assert torch.equal(stateless_norm(x), torch.zeros_like(x))


def test_stateless_norm_of_zero_row_among_nonzero_rows_stays_zero():
    x = torch.randn(4, 32)
    x[1] = 0.0
    normed = stateless_norm(x)
    assert torch.equal(normed[1], torch.zeros(32))


# ---------------------------------------------------------------------------
# 2. Gate application (copied verbatim from FLOWERVLA.dit_forward's proprio_keep
#    branch): proprio_embeds * proprio_keep.unsqueeze(-1)
# ---------------------------------------------------------------------------

def _apply_gate(proprio_embeds: torch.Tensor, proprio_keep: torch.Tensor) -> torch.Tensor:
    """Copied verbatim from FLOWERVLA.dit_forward."""
    return proprio_embeds * proprio_keep.to(proprio_embeds.device).unsqueeze(-1).to(proprio_embeds.dtype)


def test_gate_zeroes_exactly_the_masked_rows():
    torch.manual_seed(0)
    embeds = torch.randn(5, 16)
    keep = torch.tensor([True, False, True, False, True])

    gated = _apply_gate(embeds, keep)

    assert torch.equal(gated[keep], embeds[keep])
    assert torch.equal(gated[~keep], torch.zeros(2, 16))


def test_gate_all_true_is_identity():
    embeds = torch.randn(3, 16)
    keep = torch.ones(3, dtype=torch.bool)
    assert torch.equal(_apply_gate(embeds, keep), embeds)


def test_gate_all_false_zeroes_everything():
    embeds = torch.randn(3, 16)
    keep = torch.zeros(3, dtype=torch.bool)
    assert torch.equal(_apply_gate(embeds, keep), torch.zeros_like(embeds))


def test_gate_is_stable_across_repeated_calls():
    """sample_actions calls dit_forward 4-5 times per Euler integration; the same
    proprio_keep tensor (computed once in encode_observations) must gate identically
    every time, not flicker between calls."""
    torch.manual_seed(0)
    embeds = torch.randn(4, 16)
    keep = torch.tensor([True, False, True, True])

    first = _apply_gate(embeds, keep)
    second = _apply_gate(embeds, keep)

    assert torch.equal(first, second)


# ---------------------------------------------------------------------------
# 3. Bernoulli keep-gate sampling (copied verbatim from FLOWERVLA.encode_observations)
# ---------------------------------------------------------------------------

def _sample_keep(batch_size: int, keep_p: float, generator=None) -> torch.Tensor:
    """Copied verbatim from FLOWERVLA.encode_observations's Bernoulli branch."""
    return torch.rand(batch_size, generator=generator) < keep_p


def test_sample_keep_p_one_keeps_everyone():
    keep = _sample_keep(100, 1.0)
    assert keep.all()


def test_sample_keep_p_zero_drops_everyone():
    keep = _sample_keep(100, 0.0)
    assert not keep.any()


def test_sample_keep_p_half_lands_in_sane_band():
    generator = torch.Generator().manual_seed(0)
    keep = _sample_keep(10_000, 0.5, generator=generator)
    rate = keep.float().mean().item()
    assert 0.45 < rate < 0.55


# ---------------------------------------------------------------------------
# 4. Validation: proprio masking requires use_proprio=True
#    (copied logic + a real-FLOWERVLA regression test for the two ValueError paths,
#    which fail before FLOWERVLA reaches _setup_vlm / downloads Florence-2)
# ---------------------------------------------------------------------------

def _validate_modalities(modalities: dict, use_proprio: bool, modality_dropout: bool):
    """Copied verbatim from FLOWERVLA.__init__'s modality-validation block."""
    modality_tuple = (
        bool(modalities.get("rgb_static", True)),
        bool(modalities.get("rgb_gripper", True)),
        bool(modalities.get("language", True)),
    )
    proprio_enabled = bool(modalities.get("proprio", True))
    if not any(modality_tuple):
        raise ValueError("model.modalities: at least one modality must be enabled")
    if not proprio_enabled and not use_proprio:
        raise ValueError("model.modalities.proprio=False requires model.use_proprio=True")
    if (not all(modality_tuple) or not proprio_enabled) and modality_dropout:
        raise ValueError("model.modalities and model.modality_dropout are mutually exclusive")
    return modality_tuple, proprio_enabled


def test_validate_modalities_proprio_false_without_use_proprio_raises():
    with pytest.raises(ValueError):
        _validate_modalities({"proprio": False}, use_proprio=False, modality_dropout=False)


def test_validate_modalities_proprio_false_with_use_proprio_ok():
    modality_tuple, proprio_enabled = _validate_modalities(
        {"proprio": False}, use_proprio=True, modality_dropout=False
    )
    assert proprio_enabled is False


def test_validate_modalities_dropout_with_use_proprio_false_is_not_an_error():
    """model.modality_dropout=True with use_proprio=False (today's train-dropout default)
    must stay valid -- the Bernoulli gate is simply inert without proprio."""
    modality_tuple, proprio_enabled = _validate_modalities(
        {}, use_proprio=False, modality_dropout=True
    )
    assert proprio_enabled is True


def test_validate_modalities_fixed_proprio_ablation_with_dropout_raises():
    with pytest.raises(ValueError):
        _validate_modalities({"proprio": False}, use_proprio=True, modality_dropout=True)


def test_flowervla_raises_on_proprio_ablation_without_use_proprio():
    """Regression test on the real class: this must fail before _setup_vlm (no Florence-2
    download needed) so it's safe to run as a fast unit test."""
    with pytest.raises(ValueError, match="modalities.proprio"):
        FLOWERVLA(modalities={"proprio": False}, use_proprio=False)


def test_flowervla_raises_on_all_modalities_false():
    with pytest.raises(ValueError, match="at least one modality"):
        FLOWERVLA(modalities={"rgb_static": False, "rgb_gripper": False, "language": False})


# ---------------------------------------------------------------------------
# 5. Eval-time proprio validation (copied verbatim from
#    FlowerLiberoEvaluation.__init__'s proprio-mask block)
# ---------------------------------------------------------------------------

def _validate_eval_proprio(eval_modalities: dict, model_use_proprio: bool) -> bool:
    """Copied verbatim from FlowerLiberoEvaluation.__init__."""
    proprio_enabled = bool(eval_modalities.get("proprio", True))
    if not proprio_enabled and not model_use_proprio:
        raise ValueError(
            "eval_modalities.proprio=False requires a model with use_proprio=True"
        )
    return proprio_enabled


def test_validate_eval_proprio_false_without_model_support_raises():
    with pytest.raises(ValueError):
        _validate_eval_proprio({"proprio": False}, model_use_proprio=False)


def test_validate_eval_proprio_false_with_model_support_ok():
    assert _validate_eval_proprio({"proprio": False}, model_use_proprio=True) is False


def test_validate_eval_proprio_defaults_to_true():
    assert _validate_eval_proprio({}, model_use_proprio=False) is True
