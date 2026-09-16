"""
Tests for input-level modality skipping in FLOWERVLA.encode_observations
(flower/models/flower.py).

A modality withheld by a fixed combo (model.modalities / eval_modalities) is now
skipped at the very input -- its encoder (DaViT for an image view, the tokenizer +
embedding lookup for language) is never called -- instead of being run and then
physically removed just before Florence's fusion encoder. The Dirichlet
train-dropout path samples a different per-sample subset every step, so it still
needs every use_second_view-enabled modality materialized; it is unaffected.

These run the *real* FLOWERVLA.encode_observations unbound against a stub `self`
and fakes for `self.vlm`, following the pattern in tests/test_proprio.py (no
Florence-2 download). `self.vlm._encode_image` and `self.vlm.get_encoder()` are
faithfully faked (including Florence2LearnedPositionalEmbedding's arange-based
position add, so position_correct's cancellation is exercised for real);
`self.construct_prompts`/`self._get_text_embeddings` are stubbed as a single unit
(encode_observations never touches self.vlm for text) -- see _ObsStub for details.

Tests:
1. Compute saving  — a withheld view's DaViT call / withheld language's embedding
                     call are skipped entirely (the actual feature).
2. Cold-start probe — the one-time cost of sizing a never-encoded view's reserved
                     position span, and that it isn't paid again afterwards.
3. Equivalence      — retained tokens' fused features AND attention_mask exactly match
                     a full (nothing skipped) run's values at the same absolute
                     position. The default batch has ragged instruction lengths, so
                     this also covers a retained language pad landing at the right
                     compact position.
4. No NaN/Inf       — attention mask always has the always-kept prompt token, so no
                     row goes fully-masked (a real constraint given the ragged batch,
                     not a vacuous one); features stay finite; a skipped encoder is
                     never called with a zero-size batch; retained language pads are
                     masked, not silently omitted.
5. Fixed-mask parity — model.modalities (training-time) and the Dirichlet
                     train-dropout path get the same treatment the docstring above
                     promises.
"""

import types

import pytest
import torch

from flower.models.flower import FLOWERVLA
from flower.models.networks.modality_dropout import compact_layout


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeVLM:
    """Stand-in for Florence-2's `self.vlm` -- only the pieces encode_observations
    calls directly: `_encode_image` (DaViT) and `get_encoder()` (position table +
    fusion encoder). Deterministic given the same pixel input (a fixed random
    projection, lazily sized from the first call's C*H*W), mirroring a real vision
    tower's determinism in eval mode -- required so a "full" and a "skip" run of the
    same pixels can be compared token-for-token.
    """

    def __init__(self, tokens_per_image: int, embed_dim: int, pos_table_size: int = 128):
        self.tokens_per_image = tokens_per_image
        self.embed_dim = embed_dim
        self.encode_image_batch_sizes: list[int] = []
        self._proj = None
        self._token_offsets = None
        pos_weight = torch.randn(pos_table_size, embed_dim, generator=torch.Generator().manual_seed(1))
        self._encoder = _FakeEncoder(pos_weight, offset=2)
        # Deterministic (not drawn from the unseeded global RNG) and lives on the shared
        # vlm, not per-stub-instance -- a "full" and a "skip" run sharing one _FakeVLM
        # must see the identical <Flow> prompt token, exactly like two forward passes of
        # the same real, frozen model would.
        self.prompt_embeds = torch.randn(1, 1, embed_dim, generator=torch.Generator().manual_seed(2))

    def _encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        n = pixel_values.shape[0]
        self.encode_image_batch_sizes.append(n)
        if n == 0:
            # Mirrors the real crash: modeling_florence2.py:2607's
            # `x.view(batch_size * T, -1, x.shape[-1])` can't infer -1 for an empty batch.
            raise RuntimeError("would crash on a zero-size batch, like real Florence-2")
        if self._proj is None:
            gen = torch.Generator().manual_seed(0)
            flat_dim = pixel_values[0].numel()
            self._proj = torch.randn(flat_dim, self.embed_dim, generator=gen)
            self._token_offsets = torch.randn(self.tokens_per_image, self.embed_dim, generator=gen)
        base = pixel_values.reshape(n, -1) @ self._proj  # [n, D], deterministic in pixel_values
        return base.unsqueeze(1) + self._token_offsets.unsqueeze(0)  # [n, tokens_per_image, D]

    def get_encoder(self):
        return self._encoder


class _FakeEncoder:
    """Stand-in for Florence2Encoder: faithfully reproduces the one piece that matters
    here -- adding embed_positions(arange(seq_len) + offset) to inputs_embeds -- so
    that a real position_correct upstream nets to the correct absolute position."""

    def __init__(self, pos_weight: torch.Tensor, offset: int):
        self.embed_positions = types.SimpleNamespace(weight=pos_weight, offset=offset)

    def __call__(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        B, K, _ = inputs_embeds.shape
        compact_pos = torch.arange(K).unsqueeze(0).expand(B, -1)
        pos_add = self.embed_positions.weight[compact_pos + self.embed_positions.offset]
        return types.SimpleNamespace(last_hidden_state=inputs_embeds + pos_add)


class _ObsStub:
    """Thin stand-in for FLOWERVLA exposing exactly what encode_observations reads
    from self, so the real (unbound) FLOWERVLA.encode_observations runs against
    fakes instead of loading Florence-2."""

    encode_observations = FLOWERVLA.encode_observations

    def __init__(
        self,
        use_second_view: bool = True,
        modality_mask=None,
        eval_modality_mask=None,
        modality_dropout: bool = False,
        training: bool = False,
        vlm: _FakeVLM = None,
        tokens_per_image: int = 4,
        embed_dim: int = 8,
    ):
        self.device = torch.device("cpu")
        self._param = torch.zeros(1)
        self.act_window_size = 10
        self.use_second_view = use_second_view
        self.use_proprio = False
        self.modality_mask = modality_mask
        self.eval_modality_mask = eval_modality_mask
        self.proprio_mask = None
        self.eval_proprio_mask = None
        self.modality_dropout = modality_dropout
        self.modality_dropout_alphas = (1.0, 1.0, 1.0)
        self.modality_dropout_keep_fraction = 0.5
        self.training = training
        self._tokens_per_image = None

        self.vlm = vlm if vlm is not None else _FakeVLM(tokens_per_image, embed_dim)
        self.prompt_embeds = self.vlm.prompt_embeds
        self.vlm_token_dropout = lambda x: x
        self.frequency_embedder = lambda x: x

        self.text_embed_calls = 0

    def parameters(self):
        yield self._param

    def construct_prompts(self, batch):
        return batch["lang_text"]

    def _get_text_embeddings(self, texts, device):
        """Stands in for the tokenizer + get_input_embeddings lookup as a single unit
        (encode_observations never touches self.vlm for text) -- deterministic per
        instruction string, so a "full" and a "skip" run compare token-for-token."""
        self.text_embed_calls += 1
        lengths = [max(len(t.split()), 1) for t in texts]
        Lt = max(lengths)
        D = self.vlm.embed_dim
        embeds = torch.zeros(len(texts), Lt, D)
        mask = torch.zeros(len(texts), Lt, dtype=torch.long)
        for i, (t, n) in enumerate(zip(texts, lengths)):
            gen = torch.Generator().manual_seed(abs(hash(t)) % (2**31))
            embeds[i, :n] = torch.randn(n, D, generator=gen)
            mask[i, :n] = 1
        return embeds, mask


def _batch(B=2, T=1, C=3, H=6, W=6):
    # Ragged word counts (5, 3, 5, 3) so the default B=2 batch already has a pad slot,
    # exercising the fixed-mask path's full-padded-span language convention.
    instructions = [
        "pick up the red cube", "push the button",
        "open the drawer slowly please", "close the drawer",
    ][:B]
    while len(instructions) < B:
        instructions.append("push the button")
    return {
        "rgb_obs": {
            "rgb_static": torch.randn(B, T, C, H, W),
            "rgb_gripper": torch.randn(B, T, C, H, W),
        },
        "lang_text": instructions,
    }


# Every combo with at least one token modality on (the model.modalities/eval_modalities
# invariant enforced at FLOWERVLA construction time).
COMBOS = [
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, False, False),
    (False, True, False),
    (False, False, True),
]


# ---------------------------------------------------------------------------
# 1. Compute saving
# ---------------------------------------------------------------------------

class TestComputeSaving:
    @pytest.mark.parametrize("combo", COMBOS)
    def test_skipped_modality_encoder_is_never_called(self, combo):
        """Steady state (the common case -- an eval sweep against an already-trained
        checkpoint): a withheld view's DaViT call and withheld language's embedding
        call are both skipped entirely."""
        vlm = _FakeVLM(tokens_per_image=4, embed_dim=8)
        stub = _ObsStub(eval_modality_mask=combo, vlm=vlm)
        stub._tokens_per_image = vlm.tokens_per_image  # pre-warm, as after any prior forward

        stub.encode_observations(_batch())

        assert len(vlm.encode_image_batch_sizes) == int(combo[0]) + int(combo[1])
        assert stub.text_embed_calls == int(combo[2])

    def test_full_combo_still_calls_every_encoder_exactly_once(self):
        vlm = _FakeVLM(tokens_per_image=4, embed_dim=8)
        stub = _ObsStub(vlm=vlm)
        stub.encode_observations(_batch())
        assert len(vlm.encode_image_batch_sizes) == 2
        assert stub.text_embed_calls == 1


# ---------------------------------------------------------------------------
# 2. Cold-start probe
# ---------------------------------------------------------------------------

class TestColdStartProbe:
    def test_first_ever_call_with_both_views_off_does_one_small_probe(self):
        """The very first forward, before _tokens_per_image is known, needs exactly
        one tiny (batch-size-1, not B) probe to size the withheld views' reserved
        position span -- and only once, ever."""
        vlm = _FakeVLM(tokens_per_image=4, embed_dim=8)
        stub = _ObsStub(eval_modality_mask=(False, False, True), vlm=vlm)
        assert stub._tokens_per_image is None

        stub.encode_observations(_batch(B=5))

        assert vlm.encode_image_batch_sizes == [1]
        assert stub._tokens_per_image == vlm.tokens_per_image

        vlm.encode_image_batch_sizes.clear()
        stub.encode_observations(_batch(B=5))
        assert vlm.encode_image_batch_sizes == []  # not paid a second time


# ---------------------------------------------------------------------------
# 3. Equivalence to a full-encode-then-mask oracle
# ---------------------------------------------------------------------------

class TestEquivalence:
    @pytest.mark.parametrize("combo", COMBOS)
    def test_retained_tokens_match_a_full_run_at_the_same_absolute_position(self, combo):
        batch = _batch()
        vlm = _FakeVLM(tokens_per_image=4, embed_dim=8)  # shared -> identical raw embeddings

        full_out = _ObsStub(vlm=vlm).encode_observations(batch)
        skip_out = _ObsStub(eval_modality_mask=combo, vlm=vlm).encode_observations(batch)

        Ns = vlm.tokens_per_image
        Nw = vlm.tokens_per_image  # use_second_view=True by default
        Lt = full_out["features"].shape[1] - Ns - Nw - 1
        present = (combo[0], combo[1], combo[2])  # use_second_view=True here, so unaffected
        layout = compact_layout(Ns, Nw, Lt, present)

        assert skip_out["features"].shape[1] == layout.n_compact
        expected = full_out["features"][:, layout.abs_index, :]
        assert torch.allclose(skip_out["features"], expected, atol=1e-5)

        expected_mask = full_out["attention_mask"][:, layout.abs_index]
        assert torch.equal(skip_out["attention_mask"], expected_mask)


# ---------------------------------------------------------------------------
# 4. No NaN/Inf, no zero-size batch
# ---------------------------------------------------------------------------

class TestNoNaNAndNoEmptyBatch:
    @pytest.mark.parametrize("combo", COMBOS)
    def test_attention_mask_has_no_all_zero_row_and_features_are_finite(self, combo):
        stub = _ObsStub(eval_modality_mask=combo)
        out = stub.encode_observations(_batch())
        assert (out["attention_mask"].sum(dim=1) > 0).all()
        assert torch.isfinite(out["features"]).all()

    @pytest.mark.parametrize("combo", [c for c in COMBOS if c != (True, True, True)])
    def test_no_encoder_is_ever_called_with_a_zero_size_batch(self, combo):
        vlm = _FakeVLM(tokens_per_image=4, embed_dim=8)
        stub = _ObsStub(eval_modality_mask=combo, vlm=vlm)
        stub.encode_observations(_batch())
        assert all(n > 0 for n in vlm.encode_image_batch_sizes)

    def test_retained_language_pads_are_masked_not_omitted(self):
        """The fixed-mask path keeps the full padded language span (not the per-sample
        non-pad length) to stay rectangular, relying on attention_mask -- not omission --
        to exclude retained pad slots. A withheld view also confirms the pad slots land at
        the right *compact* positions, not absolute ones."""
        batch = _batch()
        lengths = [len(t.split()) for t in batch["lang_text"]]
        Lt = max(lengths)
        out = _ObsStub(eval_modality_mask=(True, False, True)).encode_observations(batch)
        lang_mask = out["attention_mask"][:, -Lt:]  # language is the trailing compact group
        for b, n in enumerate(lengths):
            assert lang_mask[b].tolist() == [1] * n + [0] * (Lt - n)


# ---------------------------------------------------------------------------
# 5. Fixed-mask parity (model.modalities) and Dirichlet train-dropout unaffected
# ---------------------------------------------------------------------------

class TestOtherFixedMaskPaths:
    def test_fixed_training_ablation_also_skips_at_input(self):
        """model.modalities (a training-time-and-always ablation, not just eval_modalities)
        shares the same `present` resolution -- just a different attribute supplying
        fixed_mask -- so it gets the same input-level skip."""
        vlm = _FakeVLM(tokens_per_image=4, embed_dim=8)
        stub = _ObsStub(modality_mask=(True, False, True), training=True, vlm=vlm)
        stub._tokens_per_image = vlm.tokens_per_image

        stub.encode_observations(_batch())

        assert len(vlm.encode_image_batch_sizes) == 1  # static only
        assert stub.text_embed_calls == 1  # language present

    def test_dirichlet_training_dropout_still_encodes_everything(self):
        """The Dirichlet train-dropout path samples a different per-sample subset every
        step -- it must still materialize every use_second_view-enabled modality; there
        is nothing to skip at input time for it."""
        vlm = _FakeVLM(tokens_per_image=4, embed_dim=8)
        stub = _ObsStub(modality_dropout=True, training=True, vlm=vlm)
        stub._tokens_per_image = vlm.tokens_per_image

        stub.encode_observations(_batch())

        assert len(vlm.encode_image_batch_sizes) == 2
        assert stub.text_embed_calls == 1
