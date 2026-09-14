"""
Unit tests for flower/models/networks/modality_dropout.py

Tests:
1. Budget / rectangularity  — counts ≤ avail; total K identical per sample; prompt always present;
                              pads never selected.
2. Per-sample variation     — different samples receive different keep-sets.
3. Position preservation    — position_correct + encoder re-add nets to absolute positions.
4. Removal ≡ masking oracle — compacted+corrected path matches attention-mask path on Florence-2-base.
5. Intra-batch variety      — regression guard for the vectorized build_keep_indices/
                              sample_token_budget: identical per-row counts must still
                              produce different per-row selections (a broadcast-noise bug
                              would pass every rectangularity/count check above while
                              masking the same positions in every batch element).
6. compact_layout            — absolute-position bookkeeping for input-level modality skipping.
7. Zero/one-token edge cases — a group with 0 or 1 available/kept tokens.
"""

import pytest
import torch

from flower.models.networks.modality_dropout import (
    sample_token_budget,
    build_keep_indices,
    position_correct,
    deterministic_keep_counts,
    gather_attention_mask,
    compact_layout,
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

    @pytest.mark.parametrize(
        "keep_mask",
        [
            (True, True, True),
            (True, True, False),
            (True, False, True),
            (False, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ],
    )
    def test_all_seven_combos_keep_only_selected_groups(self, keep_mask):
        avail = make_avail()
        counts = deterministic_keep_counts(avail, keep_mask)
        indices = build_keep_indices(
            make_group_spans(), counts, make_lang_valid_len(), PROMPT_IDX
        )
        for b in range(B):
            sel = set(indices[b].tolist())
            assert PROMPT_IDX in sel
            has_static = any(i < NS for i in sel)
            has_wrist = any(NS <= i < NS + NW for i in sel)
            has_lang = any(i >= LANG_START for i in sel)
            assert has_static == keep_mask[0]
            assert has_wrist == keep_mask[1]
            assert has_lang == keep_mask[2]


# ---------------------------------------------------------------------------
# Test 1c: fixed-modality-mask path with ragged (per-sample-varying) instruction lengths
#
# A training batch mixes tasks, so instructions (and hence non-pad language lengths) differ
# across the batch — unlike an eval batch, which is always one task's identical instruction.
# deterministic_keep_counts + build_keep_indices only stay rectangular if avail is constant
# across the batch for every kept group, so the fixed-mask path must pass the full padded
# language span (constant by construction), not the per-sample non-pad length.
# ---------------------------------------------------------------------------

class TestFixedMaskRaggedLanguage:
    def test_full_span_stays_rectangular_despite_ragged_instructions(self):
        """The fixed-mask path (flower.py) passes both avail[:, 2] and lang_valid_len as the
        full padded span LT — constant regardless of each sample's real instruction length —
        so build_keep_indices stays rectangular; raggedness is handled later by
        gather_attention_mask instead of by restricting the pool here."""
        avail = make_avail(lang_valid=LT)  # full padded span
        counts = deterministic_keep_counts(avail, [True, True, True])
        lang_valid_len = make_lang_valid_len(LT)  # constant, per the fixed-mask convention
        indices = build_keep_indices(
            make_group_spans(), counts, lang_valid_len, PROMPT_IDX
        )
        assert indices.shape == (B, NS + NW + LT + 1)

    def test_per_sample_nonpad_length_breaks_rectangularity(self):
        """The bug this design avoids: feeding the per-sample non-pad length as avail (as the
        eval-only path did) makes kept_counts[:, 2] vary across the batch when instructions
        have different lengths, so build_keep_indices cannot build a rectangular tensor."""
        avail = torch.zeros(B, 3, dtype=torch.long)
        avail[:, 0] = NS
        avail[:, 1] = NW
        avail[:, 2] = torch.tensor([20, 15, 20, 10], dtype=torch.long)  # ragged non-pad length
        counts = deterministic_keep_counts(avail, [True, True, True])
        lang_valid_len = avail[:, 2].clone()
        with pytest.raises(RuntimeError):
            build_keep_indices(make_group_spans(), counts, lang_valid_len, PROMPT_IDX)


# ---------------------------------------------------------------------------
# Test 1d: gather_attention_mask
# ---------------------------------------------------------------------------

class TestGatherAttentionMask:
    def test_masks_out_retained_language_pads(self):
        """Fixed-mask path: language avail is the full padded span, so retained language
        indices can land on pad positions — gather_attention_mask must zero those out."""
        lang_valid = 10  # 10 real tokens + 10 pads within LT=20
        avail = make_avail(lang_valid=LT)  # fixed-mask convention: full padded span
        counts = deterministic_keep_counts(avail, [True, True, True])
        lang_valid_len = torch.full((B,), LT, dtype=torch.long)
        indices = build_keep_indices(make_group_spans(), counts, lang_valid_len, PROMPT_IDX)

        text_mask = torch.zeros(B, LT, dtype=torch.long)
        text_mask[:, :lang_valid] = 1
        seq_len = NS + NW + 1 + LT
        attn = gather_attention_mask(text_mask, indices, seq_len, LANG_START)

        for b in range(B):
            for pos, idx in enumerate(indices[b].tolist()):
                if LANG_START <= idx < LANG_START + LT:
                    expected = 1 if (idx - LANG_START) < lang_valid else 0
                    assert attn[b, pos].item() == expected
                else:
                    assert attn[b, pos].item() == 1

    def test_all_ones_for_dropout_path(self):
        """The Dirichlet training-dropout path restricts language selection to non-pad
        tokens before building keep_indices, so this must reduce to all-ones there."""
        lang_valid = 10
        avail = make_avail(lang_valid=lang_valid)  # dropout convention: non-pad length
        counts = sample_token_budget(avail, KEEP_FRAC, ALPHAS)
        counts[:, 2] = counts[:, 2].clamp(max=lang_valid)
        lang_valid_len = torch.full((B,), lang_valid, dtype=torch.long)
        indices = build_keep_indices(make_group_spans(), counts, lang_valid_len, PROMPT_IDX)

        text_mask = torch.zeros(B, LT, dtype=torch.long)
        text_mask[:, :lang_valid] = 1
        seq_len = NS + NW + 1 + LT
        attn = gather_attention_mask(text_mask, indices, seq_len, LANG_START)
        assert (attn == 1).all()


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


# ---------------------------------------------------------------------------
# Test 5: Intra-batch variety (vectorization regression guard)
#
# The obvious way to get the vectorized build_keep_indices/sample_token_budget wrong is
# to draw one noise vector and broadcast it across the batch dimension: every count/
# shape/rectangularity assertion above would still pass, but every sample would mask the
# *same* positions. TestPerSampleVariation (above) doesn't catch this on its own because
# there the per-row counts also differ; here we pin identical per-row counts and still
# require different per-row selections, checked per group so a bug broadcasting only one
# group's noise doesn't slip through.
# ---------------------------------------------------------------------------

class TestIntraBatchVariety:
    def test_identical_counts_still_vary_per_group_across_batch(self):
        avail = make_avail()
        # Same counts for every row (bypasses sample_token_budget's per-row Dirichlet
        # variation) so the only remaining source of per-row difference is the noise
        # draw inside build_keep_indices itself.
        counts = deterministic_keep_counts(avail, [True, True, True])
        half = torch.tensor([NS // 2, NW // 2, LT // 2], dtype=torch.long)
        kept_counts = half.unsqueeze(0).expand(B, -1).clone()
        indices = build_keep_indices(
            make_group_spans(), kept_counts, make_lang_valid_len(), PROMPT_IDX
        )

        static_sets = [
            frozenset(i for i in indices[b].tolist() if i < NS) for b in range(B)
        ]
        wrist_sets = [
            frozenset(i for i in indices[b].tolist() if NS <= i < NS + NW) for b in range(B)
        ]
        lang_sets = [
            frozenset(i for i in indices[b].tolist() if i >= LANG_START) for b in range(B)
        ]
        for name, sets in [("static", static_sets), ("wrist", wrist_sets), ("lang", lang_sets)]:
            assert any(s != sets[0] for s in sets[1:]), (
                f"{name} group: every batch row selected the identical token set despite "
                f"identical counts — noise is likely broadcast across the batch instead of "
                f"drawn per-row: {sets}"
            )

    def test_same_generator_seed_reproduces_selection(self):
        """Complement of the above: variety must come from per-row sampling, not from an
        unseeded/non-reproducible source — same generator state -> same selection."""
        avail = make_avail()
        counts = deterministic_keep_counts(avail, [True, True, True])
        half = torch.tensor([NS // 2, NW // 2, LT // 2], dtype=torch.long)
        kept_counts = half.unsqueeze(0).expand(B, -1).clone()

        gen1 = torch.Generator().manual_seed(0)
        indices1 = build_keep_indices(
            make_group_spans(), kept_counts, make_lang_valid_len(), PROMPT_IDX, generator=gen1
        )
        gen2 = torch.Generator().manual_seed(0)
        indices2 = build_keep_indices(
            make_group_spans(), kept_counts, make_lang_valid_len(), PROMPT_IDX, generator=gen2
        )
        assert torch.equal(indices1, indices2)


# ---------------------------------------------------------------------------
# Test 6: compact_layout
# ---------------------------------------------------------------------------

class TestCompactLayout:
    def test_all_present_is_the_identity(self):
        layout = compact_layout(NS, NW, LT, (True, True, True))
        assert layout.n_compact == NS + NW + 1 + LT
        assert layout.prompt_idx == PROMPT_IDX
        assert layout.lang_start == LANG_START
        assert layout.group_spans == [(0, NS), (NS, NS + NW), (LANG_START, LANG_START + LT)]
        assert torch.equal(layout.abs_index, torch.arange(NS + NW + 1 + LT))

    @pytest.mark.parametrize(
        "present",
        [
            (True, True, True),
            (True, True, False),
            (True, False, True),
            (False, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ],
    )
    def test_abs_index_strictly_increasing_and_matches_present_groups(self, present):
        layout = compact_layout(NS, NW, LT, present)
        # abs_index is strictly increasing (a valid absolute-position gather source).
        if layout.n_compact > 1:
            assert bool((layout.abs_index[1:] > layout.abs_index[:-1]).all())
        # The prompt token's absolute index is always Ns+Nw, regardless of what's present.
        assert layout.abs_index[layout.prompt_idx].item() == PROMPT_IDX
        # Compact span widths match exactly what's present (0 for an absent group).
        static_w, wrist_w, lang_w = (e - s for s, e in layout.group_spans)
        assert static_w == (NS if present[0] else 0)
        assert wrist_w == (NW if present[1] else 0)
        assert lang_w == (LT if present[2] else 0)
        assert layout.n_compact == static_w + wrist_w + 1 + lang_w

    def test_absent_group_reserves_position_space_for_groups_after_it(self):
        """Dropping wrist shouldn't move the prompt's or language's *absolute* position —
        only static+wrist-together vs static-only would (wrist is a real modality span
        being skipped, not zero-width to begin with)."""
        layout = compact_layout(NS, NW, LT, (True, False, True))
        # Prompt sits right after static in the compact sequence...
        assert layout.prompt_idx == NS
        # ...but its absolute position still reserves the (skipped) wrist span.
        assert layout.abs_index[layout.prompt_idx].item() == NS + NW
        # Language's absolute positions likewise start after the reserved wrist span.
        lang_start_compact = layout.group_spans[2][0]
        assert layout.abs_index[lang_start_compact].item() == NS + NW + 1


# ---------------------------------------------------------------------------
# Test 7: Zero/one-token edge cases
# ---------------------------------------------------------------------------

class TestZeroAndOneTokenEdgeCases:
    def test_group_with_zero_available_contributes_nothing(self):
        avail = make_avail()
        avail[:, 1] = 0  # wrist unavailable (e.g. skipped at input)
        counts = deterministic_keep_counts(avail, [True, True, True])
        assert (counts[:, 1] == 0).all()
        indices = build_keep_indices(
            make_group_spans(), counts, make_lang_valid_len(), PROMPT_IDX
        )
        for b in range(B):
            sel = set(indices[b].tolist())
            assert not any(NS <= i < NS + NW for i in sel)
        assert indices.shape[1] == NS + 1 + LT  # no wrist contribution

    def test_group_with_single_available_token_selects_exactly_it(self):
        """A group of size 1 (kept_counts[:, g] == 1) must select rank-0 deterministically,
        not raise or silently drop it."""
        avail = torch.zeros(B, 3, dtype=torch.long)
        avail[:, 0] = 1  # single static token
        avail[:, 1] = 0
        avail[:, 2] = 0
        counts = deterministic_keep_counts(avail, [True, False, False])
        assert (counts[:, 0] == 1).all()
        group_spans = [(0, 1), (1, 1), (2, 2)]
        indices = build_keep_indices(group_spans, counts, torch.zeros(B, dtype=torch.long), prompt_idx=1)
        for b in range(B):
            assert set(indices[b].tolist()) == {0, 1}  # the single static token + prompt

    def test_all_groups_empty_keeps_only_the_prompt(self):
        avail = torch.zeros(B, 3, dtype=torch.long)
        counts = deterministic_keep_counts(avail, [False, False, False])
        indices = build_keep_indices(
            make_group_spans(), counts, make_lang_valid_len(0), PROMPT_IDX
        )
        assert indices.shape == (B, 1)
        assert (indices == PROMPT_IDX).all()

    def test_language_pool_of_size_one_respects_lang_valid_len(self):
        """pool_size == 1 for the (last, pad-aware) language group must not crash the
        padding-noise-masking branch, and must honor lang_valid_len == 0 vs 1."""
        group_spans = [(0, 0), (0, 0), (0, 1)]
        avail = torch.zeros(B, 3, dtype=torch.long)
        avail[:, 2] = 1
        counts = deterministic_keep_counts(avail, [False, False, True])
        # lang_valid_len alternates 0/1 across the batch.
        lang_valid_len = torch.tensor([1, 0, 1, 0], dtype=torch.long)
        indices = build_keep_indices(group_spans, counts, lang_valid_len, prompt_idx=1)
        for b in range(B):
            sel = set(indices[b].tolist())
            # The prompt (index 1) is always present; the single language slot (index 0)
            # is only actually a real (non-pad) token when lang_valid_len[b] == 1, but
            # deterministic_keep_counts keeps the full group regardless (pad exclusion is
            # gather_attention_mask's job, not build_keep_indices'), so it's selected either way.
            assert sel == {0, 1}
