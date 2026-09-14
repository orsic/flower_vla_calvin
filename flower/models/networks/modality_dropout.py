"""
Dirichlet modality-token dropout for FLOWER training.

Implements the 4M-style per-sample token-budget sampling (apple/ml-4m masking.py)
adapted for 3 vision/language modality groups: static view, wrist view, language.
Tokens are physically removed from the sequence, and the position-correction
function ensures retained tokens carry their original absolute positional embedding
after the encoder's internal arange re-add.

For the fixed-mask paths (`model.modalities`, `eval_modalities`), a withheld group's
encoder isn't even run — see `compact_layout`, which maps a compact (only-present-
modalities) sequence onto the same absolute-position layout a full sequence would
have, so `position_correct` still lands retained tokens at the position they'd carry
if nothing had been skipped.

`sample_token_budget` and `build_keep_indices` are fully vectorized (no per-sample
Python loop, no `.item()`/`.cpu().tolist()` GPU sync) — see each docstring for the
one documented behavioral difference from a naive per-sample loop (remainder
distribution in `sample_token_budget`).
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
from torch import LongTensor, Tensor
from torch.distributions import Dirichlet


def sample_token_budget(
    avail: LongTensor,
    keep_fraction: float,
    alphas: Tensor,
    generator: torch.Generator | None = None,
) -> LongTensor:
    """
    Sample per-sample, per-group kept-token counts following the 4M Dirichlet scheme.

    For each sample b in [B], draws proportions from Dirichlet(alphas), scales by a
    fixed total budget n_keep, floors and clamps to available counts, then distributes
    any remainder across groups in descending-proportion order (each group takes as
    much of the remainder as its remaining capacity allows before moving to the next).
    This differs from a strict one-token-at-a-time round robin only when the remainder
    exceeds a single group's capacity in one pass — in the common case (remainder small
    relative to capacity) the two agree. The returned total is exactly n_keep for every
    sample (guaranteed since n_keep <= min_avail <= avail[b].sum() for every b), making
    the keep-index tensor rectangular.

    Args:
        avail: [B, 3] available tokens per group (language may differ per sample due
               to padding; image groups are constant).
        keep_fraction: fraction of total available tokens to keep, in (0, 1].
        alphas: [3] Dirichlet concentration parameters, one per group.
        generator: unused (kept for interface symmetry with build_keep_indices) —
                   torch.distributions.Dirichlet has no generator parameter.

    Returns:
        counts: [B, 3] int64 tensor, counts[b, g] ∈ [0, avail[b, g]].
                counts[b].sum() == n_keep for every b.
    """
    B, G = avail.shape
    device = avail.device

    # Fixed budget: use minimum total capacity across batch so all samples can hit it.
    min_avail = int(avail.sum(dim=1).min().item())
    n_keep = max(1, round(keep_fraction * min_avail))

    dist = Dirichlet(alphas.float().to(device))
    props = dist.sample((B,))  # [B, G]

    counts = (props * n_keep).floor().long()
    counts = torch.minimum(counts, avail)

    # Distribute the remainder: process groups in each row's own descending-proportion
    # order, each taking as much of its remaining capacity as still needed. G is a
    # small fixed constant (3), so this is G vectorized passes, not a per-sample loop.
    order = torch.argsort(props, dim=1, descending=True)  # [B, G]
    rem = n_keep - counts.sum(dim=1)  # [B]
    for r in range(G):
        g = order[:, r : r + 1]  # [B, 1]
        cap = torch.gather(avail, 1, g) - torch.gather(counts, 1, g)  # [B, 1]
        take = torch.minimum(rem.clamp(min=0).unsqueeze(1), cap.clamp(min=0))  # [B, 1]
        counts.scatter_add_(1, g, take)
        rem = rem - take.squeeze(1)

    return counts


def deterministic_keep_counts(avail: LongTensor, keep_mask: Sequence[bool]) -> LongTensor:
    """
    Eval-time counterpart to `sample_token_budget`: keep a group entirely or drop it
    entirely, per a fixed 3-tuple, instead of drawing a Dirichlet budget.

    Rectangularity (constant total across the batch, required by `build_keep_indices`)
    is the caller's responsibility here — it holds automatically when `avail[:, g]` is
    itself constant across the batch for every kept group `g`. For the language group
    this means passing the full padded span (`Lt`), not the per-sample non-pad length:
    it's constant for LIBERO eval (one batch = one task's identical instruction) by
    coincidence, but not for a training batch that mixes instructions of different
    lengths — there, pass `Lt` and let the caller's attention mask exclude the pads
    (see `gather_attention_mask`).

    Args:
        avail: [B, 3] available tokens per group.
        keep_mask: length-3 sequence of bools, one per group (static, wrist, language).

    Returns:
        counts: [B, 3] int64 tensor — `avail` where kept, 0 where dropped.
    """
    mask = torch.tensor([1 if m else 0 for m in keep_mask], dtype=avail.dtype, device=avail.device)
    return avail * mask.unsqueeze(0)


class CompactLayout(NamedTuple):
    """
    Maps a compact sequence (only the modality groups that were actually encoded,
    concatenated in [static | wrist | <Flow> | text] order) onto the absolute
    positions that same layout would occupy if every group had been encoded.

    A withheld group contributes an empty (start, start) span in `group_spans` and
    is simply absent from `abs_index` — its span still reserves absolute position
    space for the groups after it, so a retained group's absolute positions (and
    hence `position_correct`'s output) are identical to what they'd be in the
    non-dropped sequence.

    Attributes:
        group_spans: [(start, end), (start, end), (start, end)] compact-index spans
                     for (static, wrist, language); (cursor, cursor) if absent.
        prompt_idx: compact index of the always-present <Flow> prompt token.
        lang_start: compact index of the first language-group slot (equals
                    group_spans[2][0]; provided for gather_attention_mask's signature).
        n_compact: total compact sequence length.
        abs_index: [n_compact] int64 — compact index -> absolute index in the virtual
                   full [static | wrist | <Flow> | text] layout of length ns+nw+1+lt.
    """

    group_spans: list[tuple[int, int]]
    prompt_idx: int
    lang_start: int
    n_compact: int
    abs_index: LongTensor


def compact_layout(
    ns: int,
    nw: int,
    lt: int,
    present: Sequence[bool],
    device: torch.device | None = None,
) -> CompactLayout:
    """
    Build a `CompactLayout` for the groups in `present` (static, wrist, language).

    `ns`, `nw`, `lt` are the absolute-layout group lengths regardless of `present` —
    an absent group still reserves its absolute-position span for the groups after it.
    The <Flow> prompt token is always present. When `present == (True, True, True)`
    this is the identity: `abs_index == arange(ns + nw + 1 + lt)`.
    """
    prompt_idx_abs = ns + nw
    lang_start_abs = prompt_idx_abs + 1
    abs_spans = [(0, ns), (ns, ns + nw), (lang_start_abs, lang_start_abs + lt)]

    pieces: list[Tensor] = []
    group_spans: list[tuple[int, int]] = []
    cursor = 0
    for is_present, (start, end) in zip(present[:2], abs_spans[:2]):
        if is_present and end > start:
            pieces.append(torch.arange(start, end, device=device))
            group_spans.append((cursor, cursor + (end - start)))
            cursor += end - start
        else:
            group_spans.append((cursor, cursor))

    prompt_idx = cursor
    pieces.append(torch.tensor([prompt_idx_abs], device=device))
    cursor += 1

    lang_start = cursor
    is_present, (start, end) = present[2], abs_spans[2]
    if is_present and end > start:
        pieces.append(torch.arange(start, end, device=device))
        group_spans.append((cursor, cursor + (end - start)))
        cursor += end - start
    else:
        group_spans.append((cursor, cursor))

    abs_index = torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.long, device=device)
    return CompactLayout(
        group_spans=group_spans,
        prompt_idx=prompt_idx,
        lang_start=lang_start,
        n_compact=cursor,
        abs_index=abs_index.long(),
    )


def build_keep_indices(
    group_spans: list[tuple[int, int]],
    kept_counts: LongTensor,
    lang_valid_len: LongTensor,
    prompt_idx: int,
    generator: torch.Generator | None = None,
) -> LongTensor:
    """
    Build keep-indices for each sample such that:
    - Within each group, kept_counts[b,g] tokens are drawn uniformly at random
      without replacement from the group's valid range (language excludes padding).
    - The <Flow> prompt token at prompt_idx is always included.
    - The returned tensor has shape [B, K] where K = kept_counts[0].sum() + 1,
      constant across b (rectangular), sorted ascending per row.

    Vectorized: for each group, per-row noise is argsorted to pick a random subset
    without a Python loop over the batch; a group's "pool" is padding-pushed-last for
    the language group (`lang_valid_len`) so the top-n picks by rank stay within the
    valid span whenever kept_counts[:, g] <= lang_valid_len (true along every call
    site's convention — see `deterministic_keep_counts` / `sample_token_budget`
    docstrings). The compacted boolean keep-mask is turned into sorted indices via a
    cumsum + scatter, avoiding `nonzero` (which syncs on non-rectangular results).

    Args:
        group_spans: list of (start, end) index ranges, one per group, in the same
                     coordinate space as prompt_idx (compact-layout indices when
                     called via compact_layout, or absolute indices for the
                     no-skip/oracle case). group 2 (language) should NOT include the
                     prompt token.
        kept_counts: [B, G] per-sample per-group kept counts.
        lang_valid_len: [B] number of non-padding language tokens per sample.
        prompt_idx: index of the <Flow> prompt token; always kept.
        generator: optional RNG.

    Returns:
        keep_indices: [B, K] sorted indices, in the same coordinate space as
                      group_spans/prompt_idx.
    """
    B, G = kept_counts.shape
    totals = kept_counts.sum(dim=1)
    if not bool((totals == totals[0]).all()):
        raise RuntimeError(
            "kept_counts total differs across the batch: build_keep_indices requires a "
            "rectangular result (see deterministic_keep_counts / sample_token_budget "
            "docstrings for how to keep the per-sample total constant)"
        )
    K = int(totals[0].item()) + 1  # +1 for the prompt
    device = kept_counts.device
    N = max(group_spans[-1][1], prompt_idx + 1)

    keep = torch.zeros(B, N, dtype=torch.bool, device=device)
    keep[:, prompt_idx] = True

    for g, (start, end) in enumerate(group_spans):
        pool_size = end - start
        if pool_size <= 0:
            continue

        noise = torch.rand(B, pool_size, generator=generator, device=device)
        if g == len(group_spans) - 1:
            # Language group: push padding positions to the back of the ranking so
            # they're never selected as long as kept_counts[:, g] <= lang_valid_len.
            arange_p = torch.arange(pool_size, device=device).unsqueeze(0)
            valid_len = lang_valid_len.clamp(max=pool_size).unsqueeze(1)
            noise = noise.masked_fill(arange_p >= valid_len, 2.0)  # > max possible rand()

        order = torch.argsort(noise, dim=1)  # [B, pool_size], ascending
        ranks = torch.arange(pool_size, device=device).unsqueeze(0).expand(B, -1)
        counts_g = kept_counts[:, g].clamp(max=pool_size).unsqueeze(1)
        sel = ranks < counts_g  # [B, pool_size], True for the lowest-noise `counts_g` ranks

        idx = start + order  # [B, pool_size]: absolute (group-space) position at each rank
        keep.scatter_(1, idx, sel)

    # Compact the boolean keep-mask into sorted indices without `nonzero`: each kept
    # position's destination rank is its running count of kept positions so far
    # (cumsum - 1); non-kept positions are routed to a scratch column (index K) that
    # gets discarded.
    dest = (keep.long().cumsum(dim=1) - 1).masked_fill(~keep, K)  # [B, N]
    buf = torch.zeros(B, K + 1, dtype=torch.long, device=device)
    src = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
    buf.scatter_(1, dest, src)
    return buf[:, :K]


def position_correct(
    kept_embeds: Tensor,
    abs_positions: LongTensor,
    pos_weight: Tensor,
    offset: int,
) -> Tensor:
    """
    Adjust kept token embeddings so that after the encoder re-adds its internal
    embed_positions(arange(K)) the net positional contribution equals the absolute
    position of each retained token.

    The correction is:
        kept_embeds + pos_weight[abs_positions + offset]
                    - pos_weight[arange(K)      + offset]

    where pos_weight is the nn.Embedding weight of Florence2LearnedPositionalEmbedding
    (the +offset accounts for the 2-slot offset in that class).

    Args:
        kept_embeds: [B, K, D] gathered token embeddings (without positions added).
        abs_positions: [B, K] absolute 0-based positions each retained token should
                       carry in the full (non-dropped) sequence — when tokens were
                       gathered from a compact (only-present-modalities) sequence,
                       this is `compact_layout(...).abs_index[keep_indices]`, not
                       `keep_indices` itself.
        pos_weight: [max_pos + offset, D] embedding weight table.
        offset: the .offset value of Florence2LearnedPositionalEmbedding (=2).

    Returns:
        corrected: [B, K, D] ready to pass to the encoder as inputs_embeds.
    """
    B, K, D = kept_embeds.shape
    device = kept_embeds.device

    # Absolute positions for retained tokens.
    abs_pos_emb = pos_weight[abs_positions + offset]          # [B, K, D]

    # Positions the encoder will add internally (arange(K) + offset).
    compact_pos = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)  # [B, K]
    compact_pos_emb = pos_weight[compact_pos + offset]       # [B, K, D]

    return kept_embeds + abs_pos_emb - compact_pos_emb


def gather_attention_mask(
    text_mask: LongTensor,
    keep_indices: LongTensor,
    seq_len: int,
    lang_start: int,
) -> LongTensor:
    """
    Attention mask for a compacted (gathered) sequence: 1 everywhere except retained
    language pad tokens.

    Needed when a group's kept-count was computed from a span that may include padding
    (e.g. the fixed-modality-mask path, which uses the full padded language span so that
    `build_keep_indices` stays rectangular across a batch of differing instruction
    lengths — see `deterministic_keep_counts`). For the Dirichlet training-dropout path,
    language is always restricted to non-pad tokens before selection, so this reduces to
    all-ones there.

    Args:
        text_mask: [B, Lt] tokenizer attention mask (1 = real token, 0 = pad).
        keep_indices: [B, K] index of each retained token, in the same coordinate
                      space as `lang_start`/`seq_len` (compact-layout indices when a
                      modality was skipped, absolute indices otherwise).
        seq_len: length of the sequence keep_indices was drawn from (before gather).
        lang_start: index of the first language token in that same coordinate space.

    Returns:
        attention_mask: [B, K] int64 tensor.
    """
    B, Lt = text_mask.shape
    device = text_mask.device

    full_mask = torch.ones(B, seq_len, dtype=text_mask.dtype, device=device)
    full_mask[:, lang_start:lang_start + Lt] = text_mask

    return torch.gather(full_mask, dim=1, index=keep_indices)
