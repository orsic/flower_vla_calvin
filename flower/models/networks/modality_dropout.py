"""
Dirichlet modality-token dropout for FLOWER training.

Implements the 4M-style per-sample token-budget sampling (apple/ml-4m masking.py)
adapted for 3 vision/language modality groups: static view, wrist view, language.
Tokens are physically removed from the sequence, and the position-correction
function ensures retained tokens carry their original absolute positional embedding
after the encoder's internal arange re-add.
"""

from __future__ import annotations

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
    any remainder by iterating over groups with remaining capacity. The returned total
    is exactly n_keep for every sample, making the keep-index tensor rectangular.

    Args:
        avail: [B, 3] available tokens per group (language may differ per sample due
               to padding; image groups are constant).
        keep_fraction: fraction of total available tokens to keep, in (0, 1].
        alphas: [3] Dirichlet concentration parameters, one per group.
        generator: optional RNG for reproducibility.

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

    # Sample Dirichlet proportions per sample [B, G].
    props = dist.sample((B,))  # [B, G]

    counts = torch.zeros(B, G, dtype=torch.long, device=device)

    for b in range(B):
        a = avail[b]  # [G]
        p = props[b]  # [G]

        # Base allocation: floor(prop * n_keep), clamped to available.
        c = (p * n_keep).floor().long()
        c = torch.minimum(c, a)

        # Distribute remainder one token at a time, cycling over groups by proportion.
        rem = n_keep - int(c.sum().item())
        if rem > 0:
            # Sort groups by descending original proportion for a greedy tie-break.
            order = torch.argsort(p, descending=True)
            added = 0
            while added < rem:
                progress = False
                for g in order.tolist():
                    if added >= rem:
                        break
                    if c[g] < a[g]:
                        c[g] += 1
                        added += 1
                        progress = True
                if not progress:
                    # All groups exhausted — n_keep ≤ min_avail so this can't happen
                    # in normal operation, but guard against float rounding.
                    break

        counts[b] = c

    return counts


def build_keep_indices(
    group_spans: list[tuple[int, int]],
    kept_counts: LongTensor,
    lang_valid_len: LongTensor,
    prompt_idx: int,
    generator: torch.Generator | None = None,
) -> LongTensor:
    """
    Build absolute keep-indices for each sample such that:
    - Within each group, kept_counts[b,g] tokens are drawn uniformly at random
      without replacement from the group's valid range (language excludes padding).
    - The <Flow> prompt token at prompt_idx is always included (appended last).
    - The returned tensor has shape [B, K] where K = kept_counts[0].sum() + 1,
      constant across b (rectangular).

    Args:
        group_spans: list of (start, end) absolute index ranges, one per group.
                     group 2 (language) should NOT include the prompt token.
        kept_counts: [B, G] per-sample per-group kept counts (from sample_token_budget).
        lang_valid_len: [B] number of non-padding language tokens per sample.
        prompt_idx: absolute index of the <Flow> prompt token; always kept.
        generator: optional RNG.

    Returns:
        keep_indices: [B, K] sorted absolute token indices.
    """
    B, G = kept_counts.shape
    K = int(kept_counts[0].sum().item()) + 1  # +1 for the prompt
    device = kept_counts.device

    keep_indices = torch.zeros(B, K, dtype=torch.long, device=device)

    for b in range(B):
        selected: list[int] = []
        for g, (start, end) in enumerate(group_spans):
            n = int(kept_counts[b, g].item())
            if n == 0:
                continue
            if g == len(group_spans) - 1:
                # Language group: restrict to valid (non-pad) tokens only.
                valid_end = start + int(lang_valid_len[b].item())
            else:
                valid_end = end
            pool_size = valid_end - start
            if pool_size <= 0 or n == 0:
                continue
            n = min(n, pool_size)
            # 4M-style: shuffle by random noise, take top-n.
            noise = torch.rand(pool_size, generator=generator, device=device)
            ids = torch.argsort(noise)[:n]
            selected.extend((start + ids).cpu().tolist())

        # Sort + append prompt.
        selected.sort()
        selected.append(prompt_idx)
        keep_indices[b] = torch.tensor(selected, dtype=torch.long, device=device)

    return keep_indices


def position_correct(
    kept_embeds: Tensor,
    keep_indices: LongTensor,
    pos_weight: Tensor,
    offset: int,
) -> Tensor:
    """
    Adjust kept token embeddings so that after the encoder re-adds its internal
    embed_positions(arange(K)) the net positional contribution equals the absolute
    position of each retained token.

    The correction is:
        kept_embeds + pos_weight[keep_indices + offset]
                    - pos_weight[arange(K)    + offset]

    where pos_weight is the nn.Embedding weight of Florence2LearnedPositionalEmbedding
    (the +offset accounts for the 2-slot offset in that class).

    Args:
        kept_embeds: [B, K, D] gathered token embeddings (without positions added).
        keep_indices: [B, K] absolute 0-based token indices in the original sequence.
        pos_weight: [max_pos + offset, D] embedding weight table.
        offset: the .offset value of Florence2LearnedPositionalEmbedding (=2).

    Returns:
        corrected: [B, K, D] ready to pass to the encoder as inputs_embeds.
    """
    B, K, D = kept_embeds.shape
    device = kept_embeds.device

    # Absolute positions for retained tokens.
    abs_pos_emb = pos_weight[keep_indices + offset]          # [B, K, D]

    # Positions the encoder will add internally (arange(K) + offset).
    compact_pos = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)  # [B, K]
    compact_pos_emb = pos_weight[compact_pos + offset]       # [B, K, D]

    return kept_embeds + abs_pos_emb - compact_pos_emb
