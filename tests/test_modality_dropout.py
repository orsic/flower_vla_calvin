"""
Unit tests for flower/models/networks/modality_dropout.py

Tests:
1. Budget / rectangularity  — counts ≤ avail; total K identical per sample; prompt always present;
                              pads never selected.
2. Per-sample variation     — different samples receive different keep-sets.
3. Position preservation    — position_correct + encoder re-add nets to absolute positions.
4. Removal ≡ masking oracle — compacted+corrected path matches attention-mask path on Florence-2-base.
"""

import pytest
import torch

from flower.models.networks.modality_dropout import (
    sample_token_budget,
    build_keep_indices,
    position_correct,
    deterministic_keep_counts,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

B = 4
NS, NW, LT = 196, 196, 20   # static, wrist, language token counts (per-sample)
PROMPT_IDX = NS + NW         # index of <Flow> token in merged sequence
LANG_START = PROMPT_IDX + 1  # first language token absolute index

ALPHAS = torch.tensor([1.0, 1.0, 1.0])
KEEP_FRAC = 0.5


def make_avail(lang_valid: int = LT) -> torch.LongTensor:
    """avail tensor [B, 3] with fixed image groups and variable lang."""
    avail = torch.zeros(B, 3, dtype=torch.long)
    avail[:, 0] = NS
    avail[:, 1] = NW
    avail[:, 2] = lang_valid
    return avail


def make_group_spans() -> list[tuple[int, int]]:
    return [
        (0, NS),
        (NS, NS + NW),
        (LANG_START, LANG_START + LT),
    ]


def make_lang_valid_len(val: int = LT) -> torch.LongTensor:
    return torch.full((B,), val, dtype=torch.long)


# ---------------------------------------------------------------------------
# Test 1: Budget / rectangularity
# ---------------------------------------------------------------------------

class TestBudget:
    def test_counts_within_avail(self):
        avail = make_avail()
        counts = sample_token_budget(avail, KEEP_FRAC, ALPHAS)
        assert (counts <= avail).all(), "counts exceed available tokens in some group"

    def test_counts_nonnegative(self):
        avail = make_avail()
        counts = sample_token_budget(avail, KEEP_FRAC, ALPHAS)
        assert (counts >= 0).all()

    def test_total_count_constant_across_batch(self):
        avail = make_avail()
        counts = sample_token_budget(avail, KEEP_FRAC, ALPHAS)
        totals = counts.sum(dim=1)  # [B]
        assert totals.eq(totals[0]).all(), (
            f"total kept-count differs across batch: {totals}"
        )

    def test_prompt_always_present(self):
        avail = make_avail()
        counts = sample_token_budget(avail, KEEP_FRAC, ALPHAS)
        indices = build_keep_indices(
            make_group_spans(), counts, make_lang_valid_len(), PROMPT_IDX
        )
        for b in range(B):
            assert PROMPT_IDX in indices[b].tolist(), (
                f"sample {b}: prompt token missing from keep_indices"
            )

    def test_pad_tokens_never_selected(self):
        """Language group is 10 real tokens + 10 pads; no index ≥ LANG_START+10."""
        lang_valid = 10
        avail = make_avail(lang_valid=lang_valid)
        counts = sample_token_budget(avail, KEEP_FRAC, ALPHAS)
        # cap lang kept to what's available
        counts[:, 2] = counts[:, 2].clamp(max=lang_valid)
        lang_valid_len = torch.full((B,), lang_valid, dtype=torch.long)
        indices = build_keep_indices(
            make_group_spans(), counts, lang_valid_len, PROMPT_IDX
        )
        pad_start = LANG_START + lang_valid
        for b in range(B):
            lang_sel = [i for i in indices[b].tolist()
                        if LANG_START <= i < LANG_START + LT]
            assert all(i < pad_start for i in lang_sel), (
                f"sample {b}: pad token selected in language group: {lang_sel}"
            )

    def test_keep_indices_shape_rectangular(self):
        avail = make_avail()
        counts = sample_token_budget(avail, KEEP_FRAC, ALPHAS)
        indices = build_keep_indices(
            make_group_spans(), counts, make_lang_valid_len(), PROMPT_IDX
        )
        # All rows same length (K = total+1 for prompt).
        assert indices.shape[0] == B
        expected_K = int(counts[0].sum().item()) + 1
        assert indices.shape[1] == expected_K, (
            f"expected K={expected_K}, got {indices.shape[1]}"
        )


# ---------------------------------------------------------------------------
# Test 1b: deterministic_keep_counts (eval-time full-group in/out masking)
# ---------------------------------------------------------------------------

class TestDeterministicKeepCounts:
    def test_full_keep_returns_avail_unchanged(self):
        avail = make_avail()
        counts = deterministic_keep_counts(avail, [True, True, True])
        assert torch.equal(counts, avail)

    def test_full_drop_returns_zeros(self):
        avail = make_avail()
        counts = deterministic_keep_counts(avail, [False, False, False])
        assert (counts == 0).all()

    def test_mixed_mask_zeros_only_dropped_groups(self):
        avail = make_avail()
        counts = deterministic_keep_counts(avail, [True, False, True])
        assert torch.equal(counts[:, 0], avail[:, 0])
        assert (counts[:, 1] == 0).all()
        assert torch.equal(counts[:, 2], avail[:, 2])

    def test_output_shape(self):
        avail = make_avail()
        counts = deterministic_keep_counts(avail, [True, False, True])
        assert counts.shape == (B, 3)

    def test_usable_with_build_keep_indices(self):
        """Feeding deterministic counts through the same downstream path as training."""
        avail = make_avail()
        counts = deterministic_keep_counts(avail, [True, False, True])  # drop wrist
        indices = build_keep_indices(
            make_group_spans(), counts, make_lang_valid_len(), PROMPT_IDX
        )
        for b in range(B):
            sel = set(indices[b].tolist())
            assert any(i < NS for i in sel), "static tokens missing"
            assert not any(NS <= i < NS + NW for i in sel), "wrist tokens present despite drop"
            assert PROMPT_IDX in sel


# ---------------------------------------------------------------------------
# Test 2: Per-sample variation
# ---------------------------------------------------------------------------

class TestPerSampleVariation:
    def test_different_samples_different_sets(self):
        """With Dirichlet sampling, not all samples should select the same tokens."""
        avail = make_avail()
        counts = sample_token_budget(avail, KEEP_FRAC, ALPHAS)
        indices = build_keep_indices(
            make_group_spans(), counts, make_lang_valid_len(), PROMPT_IDX
        )
        row0 = set(indices[0].tolist())
        any_differ = any(set(indices[b].tolist()) != row0 for b in range(1, B))
        # With random selection over ~400 tokens keeping ~200, the probability all
        # rows are identical is astronomically small. Just assert it's not constant.
        # (If this fails stochastically, the test harness has a fixed seed issue.)
        assert any_differ, (
            "All batch samples selected identical token sets — per-sample sampling broken"
        )


# ---------------------------------------------------------------------------
# Test 3: Position preservation (analytic)
# ---------------------------------------------------------------------------

class TestPositionPreservation:
    def test_corrected_embeds_net_to_absolute_positions(self):
        """
        After position_correct, adding the encoder's internal compact positions
        should yield the same result as adding the absolute positions directly.

        i.e.: corrected + pos[arange(K)+off] == original + pos[keep_idx+off]
        """
        D = 64
        N_total = NS + NW + 1 + LT   # total merged sequence length
        offset = 2
        max_pos = N_total + offset + 4

        # Fake positional embedding table.
        pos_weight = torch.randn(max_pos, D)

        # Random original embeddings (as if vision-tower output, no positions added).
        kept_embeds = torch.randn(B, 50, D)

        # Random keep indices (sorted, no repeats, all < N_total).
        keep_indices = torch.stack([
            torch.sort(torch.randperm(N_total)[:50])[0]
            for _ in range(B)
        ])  # [B, 50]
        K = 50

        corrected = position_correct(kept_embeds, keep_indices, pos_weight, offset)

        # What the encoder will compute after our correction:
        compact_pos = torch.arange(K).unsqueeze(0).expand(B, -1)  # [B, K]
        net = corrected + pos_weight[compact_pos + offset]

        # What it should equal:
        target = kept_embeds + pos_weight[keep_indices + offset]

        assert torch.allclose(net, target, atol=1e-5), (
            "position_correct does not produce absolute-position equivalence"
        )


# ---------------------------------------------------------------------------
# Test 4: Removal ≡ masking oracle (integration, loads Florence-2-base)
# ---------------------------------------------------------------------------

class TestRemovalEquivalentToMask:
    """
    Verifies that the compacted+position-corrected token sequence produces
    identical encoder hidden states for retained tokens vs. passing the full
    sequence with an attention mask that zeros out the dropped tokens.

    This uses Florence-2-base (smaller, same positional logic as -large).
    Model is loaded once per session (cached by pytest).
    """

    @pytest.fixture(scope="class")
    def encoder_and_pos(self):
        """Load Florence-2-base encoder. Skipped if not reachable."""
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-base",
                trust_remote_code=True,
            )
        except Exception as e:
            pytest.skip(f"Florence-2-base not available: {e}")

        encoder = model.get_encoder().eval()
        pos_w = encoder.embed_positions.weight
        pos_off = encoder.embed_positions.offset
        # Delete decoder to save memory
        del model
        return encoder, pos_w, pos_off

    def test_removal_matches_mask_oracle(self, encoder_and_pos):
        encoder, pos_w, pos_off = encoder_and_pos
        D = encoder.embed_tokens.embedding_dim
        torch.manual_seed(0)

        # Build a tiny fake merged sequence [B, N, D] (no embed_tokens needed,
        # we pass raw embeddings directly as inputs_embeds).
        B_t = 2
        Ns_t, Nw_t, Lt_t = 16, 16, 8
        N = Ns_t + Nw_t + 1 + Lt_t   # static | wrist | prompt | text
        prompt_idx_t = Ns_t + Nw_t

        merged = torch.randn(B_t, N, D)

        # Choose which tokens to keep (same set for simplicity of the test).
        # Keep all static, none of wrist, all text, always prompt.
        keep_mask_1d = torch.zeros(N, dtype=torch.bool)
        keep_mask_1d[:Ns_t] = True          # static
        keep_mask_1d[prompt_idx_t] = True   # prompt
        keep_mask_1d[prompt_idx_t + 1:] = True  # text

        # Oracle: full sequence with attention_mask zeroing dropped positions.
        attn_mask_oracle = keep_mask_1d.unsqueeze(0).expand(B_t, -1).long()
        with torch.no_grad():
            oracle_out = encoder(
                inputs_embeds=merged,
                attention_mask=attn_mask_oracle,
            ).last_hidden_state  # [B, N, D]

        # Removal path: gather retained tokens, position-correct, run encoder.
        keep_indices_t = keep_mask_1d.nonzero(as_tuple=True)[0]  # [K]
        K_t = keep_indices_t.shape[0]
        keep_indices_bt = keep_indices_t.unsqueeze(0).expand(B_t, -1)  # [B, K]

        kept_embeds = merged[:, keep_indices_t, :]  # [B, K, D]
        corrected = position_correct(kept_embeds, keep_indices_bt, pos_w.float(), pos_off)
        attn_mask_removal = torch.ones(B_t, K_t, dtype=torch.long)
        with torch.no_grad():
            removal_out = encoder(
                inputs_embeds=corrected.float(),
                attention_mask=attn_mask_removal,
            ).last_hidden_state  # [B, K, D]

        # Match retained positions in oracle output to the removal output.
        oracle_retained = oracle_out[:, keep_indices_t, :]  # [B, K, D]

        # Use float32 for comparison (encoder may be fp32 by default).
        assert torch.allclose(
            removal_out.float(), oracle_retained.float(), atol=1e-4
        ), (
            f"Removal path does not match masking oracle.\n"
            f"Max abs diff: {(removal_out.float() - oracle_retained.float()).abs().max():.6f}"
        )
