r"""Standalone simulation of FlowerVLA's training-time input sampling (modality dropout).

Reproduces, with numpy only, what flower/models/networks/modality_dropout.py and the proprio
gate in flower/models/flower.py (FLOWERVLA.encode_observations / dit_forward) do to every
training sample, with the token counts of our LIBERO-10, CALVIN (ABCD->D) and MimicGen runs.
It prints and plots, per modality: the distribution of kept tokens, and the share of samples
in which the modality is entirely dropped, partly kept or fully kept.

    python scripts/simulate_modality_sampling.py --dataset all --out plots/sampling

The algorithm, as implemented (one training micro-batch of B samples)
--------------------------------------------------------------------
Token groups g \in \{s, w, \ell\} (static camera, wrist camera, language). Sample b has
N_{b,s} = N_s, N_{b,w} = N_w image tokens and N_{b,\ell} = L_b non-padding prompt tokens
(template + instruction, incl. <s> and </s>). The <Flow> token is always kept and is not
part of the budget. Proprioception is no token: it is a vector added to the AdaLN
conditioning, kept (\pi_b = 1) or zeroed (\pi_b = 0).

Budget (Dirichlet and uniform; f = keep fraction):
    n = \max\left(1, \operatorname{round}\left(f \cdot \min_{b} \sum_g N_{b,g}\right)\right)

Dirichlet (strategy "dirichlet", \alpha = (1, 1, 1)):
    p_b \sim \operatorname{Dir}(\alpha), \qquad
    c^{0}_{b,g} = \min\left(\lfloor p_{b,g} \, n \rfloor, N_{b,g}\right)
  then the remainder r_b = n - \sum_g c^0_{b,g} is handed out in order of decreasing
  p_{b,g}: with \sigma_b that order and r^{(0)}_b = r_b, for k = 1, 2, 3, g = \sigma_b(k):
    t = \min\left(r^{(k-1)}_b, N_{b,g} - c^{0}_{b,g}\right), \quad c_{b,g} = c^0_{b,g} + t,
    \quad r^{(k)}_b = r^{(k-1)}_b - t
  so \sum_g c_{b,g} = n for every b. No modality is guaranteed to survive.

Uniform (strategy "uniform"): a uniformly random n-subset of all \sum_g N_{b,g} tokens,
  so (c_{b,s}, c_{b,w}, c_{b,\ell}) is multivariate hypergeometric.

Within a group (both strategies above), the c_{b,g} kept tokens are a uniformly random
subset of that group's tokens; kept tokens keep their original positions.

Proprio gate (Dirichlet / uniform): \pi_b \sim \operatorname{Bernoulli}(q), q = proprio keep p,
  independent of the token draw.

Subset (strategy "subset", whole inputs; M = \{s, w, \ell, \pi\}, or \{s, w, \ell\} without
proprio):
    P(S) = \frac{2^{|S|}}{\sum_{\emptyset \neq S' \subseteq M} 2^{|S'|}}
           = \frac{2^{|S|}}{3^{|M|} - 1}, \qquad \emptyset \neq S \subseteq M
    c_{b,g} = N_{b,g} \, [g \in S_b], \qquad \pi_b = [\pi \in S_b]
  (|M| = 4: full set 0.2, each 3-subset 0.1, each pair 0.05, each single input 0.025.)

No dropout: c_{b,g} = N_{b,g}, \pi_b = 1 (when the run uses proprio).

Token counts
------------
Florence-2's DaViT encoder gives 1 pooled + h x w spatial tokens per image, one frame
(obs_seq_len = 1): 224 px -> 7 x 7 + 1 = 50 (CALVIN, MimicGen), 112 px -> 4 x 4 + 1 = 17
(LIBERO). Language lengths are the Florence-2 tokenizer's counts of
generate_policy_prompt(instruction, prompt_style="minimal") (25 tokens of template + the
instruction), weighted by how many training samples carry each instruction: frames per
task (LIBERO-10, MimicGen core datasets), action windows per annotation (CALVIN
task_ABCD_D/training/lang_clip_resnet50/auto_lang_ann.npy). The micro-batch size matters
only through the batch minimum in n, and differs between runs (see --batch).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GROUPS = ("static", "wrist", "language", "proprio")


@dataclass(frozen=True)
class Dataset:
    name: str
    static_tokens: int
    wrist_tokens: int
    lang_hist: dict[int, int]  # prompt tokens -> number of training samples with that length
    batch: int  # per-GPU micro-batch (default config)


@dataclass(frozen=True)
class Experiment:
    name: str
    strategy: str  # none | dirichlet | uniform | subset
    keep_fraction: float = 1.0
    proprio: bool = False
    proprio_keep: float = 1.0


DATASETS = {
    "libero": Dataset(
        "LIBERO-10",
        static_tokens=17,
        wrist_tokens=17,
        lang_hist={33: 20794, 37: 56251, 38: 13476, 39: 12434, 40: 9470, 43: 12756, 44: 12909},
        batch=32,
    ),
    "calvin": Dataset(
        "CALVIN",
        static_tokens=50,
        wrist_tokens=50,
        lang_hist={
            28: 22385,
            29: 65294,
            30: 148198,
            31: 72282,
            32: 257961,
            33: 293755,
            34: 160889,
            35: 44940,
            36: 43664,
            37: 41538,
        },
        batch=8,
    ),
    "mimicgen": Dataset(
        "MimicGen",
        static_tokens=50,
        wrist_tokens=50,
        lang_hist={31: 256553, 32: 50936, 33: 45779, 34: 453111, 35: 35374},
        batch=16,
    ),
}

# The configurations we trained (docs/results.md, Setup and Reproduction).
EXPERIMENTS = {
    "libero": [
        Experiment("no dropout", "none"),
        Experiment("no dropout + proprio", "none", proprio=True),
        Experiment("Dirichlet 0.5", "dirichlet", 0.5),
        Experiment("Dirichlet 0.5, proprio keep 1.0", "dirichlet", 0.5, True, 1.0),
        Experiment("Dirichlet 0.5, proprio keep 0.75", "dirichlet", 0.5, True, 0.75),
        Experiment("Dirichlet 0.5, proprio keep 0.5", "dirichlet", 0.5, True, 0.5),
        Experiment("Dirichlet 0.5, proprio keep 0.25", "dirichlet", 0.5, True, 0.25),
        Experiment("subset", "subset", proprio=True),
    ],
    "calvin": [
        Experiment("no dropout", "none"),
        Experiment("no dropout + proprio", "none", proprio=True),
        Experiment("Dirichlet 0.75", "dirichlet", 0.75),
        Experiment("Dirichlet 0.5", "dirichlet", 0.5),
        Experiment("Dirichlet 0.25", "dirichlet", 0.25),
        Experiment("uniform 0.5", "uniform", 0.5),
        Experiment("Dirichlet 0.5, proprio keep 0.5", "dirichlet", 0.5, True, 0.5),
        Experiment("Dirichlet 0.5, proprio keep 1.0", "dirichlet", 0.5, True, 1.0),
        Experiment("subset", "subset", proprio=True),
    ],
    "mimicgen": [
        Experiment("no dropout", "none"),
        Experiment("no dropout + proprio", "none", proprio=True),
        Experiment("Dirichlet 0.75, proprio keep 0.5", "dirichlet", 0.75, True, 0.5),
        Experiment("Dirichlet 0.5, proprio keep 0.5", "dirichlet", 0.5, True, 0.5),
        Experiment("Dirichlet 0.5, proprio keep 1.0", "dirichlet", 0.5, True, 1.0),
        Experiment("uniform 0.5, proprio keep 0.5", "uniform", 0.5, True, 0.5),
        Experiment("subset", "subset", proprio=True),
    ],
}


def token_budget(avail: np.ndarray, keep_fraction: float) -> int:
    """n = max(1, round(f * min_b sum_g N_bg)): one total for the whole micro-batch."""
    return max(1, round(keep_fraction * int(avail.sum(axis=1).min())))


def dirichlet_counts(
    avail: np.ndarray,
    keep_fraction: float,
    rng: np.random.Generator,
    alphas: tuple[float, ...] = (1.0, 1.0, 1.0),
    props: np.ndarray | None = None,
) -> np.ndarray:
    """Kept tokens per group, [B, 3]; mirrors modality_dropout.sample_token_budget.
    `props` ([B, 3]) replaces the Dirichlet draw (for testing)."""
    B, G = avail.shape
    n_keep = token_budget(avail, keep_fraction)
    if props is None:
        props = rng.dirichlet(alphas, size=B)
    counts = np.minimum(np.floor(props * n_keep).astype(np.int64), avail)
    order = np.argsort(-props, axis=1, kind="stable")
    rem = n_keep - counts.sum(axis=1)
    rows = np.arange(B)
    for r in range(G):
        g = order[:, r]
        take = np.minimum(np.clip(rem, 0, None), np.clip(avail[rows, g] - counts[rows, g], 0, None))
        counts[rows, g] += take
        rem -= take
    return counts


def uniform_counts(avail: np.ndarray, keep_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Kept tokens per group, [B, 3], of a uniform n-subset of all tokens;
    mirrors modality_dropout.sample_uniform_counts."""
    B, G = avail.shape
    totals = avail.sum(axis=1)
    n_keep = token_budget(avail, keep_fraction)
    slots = np.arange(totals.max())[None, :]
    noise = np.where(slots < totals[:, None], rng.random((B, totals.max())), 2.0)
    sel = np.argsort(np.argsort(noise, axis=1), axis=1) < n_keep
    group = (slots[..., None] >= avail.cumsum(axis=1)[:, None, :]).sum(axis=-1).clip(max=G - 1)
    return np.stack([(sel & (group == g)).sum(axis=1) for g in range(G)], axis=1)


def subset_masks(B: int, n_modalities: int, rng: np.random.Generator) -> np.ndarray:
    """[B, n] bool, columns (static, wrist, language[, proprio]): a non-empty subset S per
    sample with P(S) proportional to 2^|S|; mirrors modality_dropout.sample_modality_subsets."""
    codes = np.arange(1, 2**n_modalities)
    bits = (codes[:, None] >> np.arange(n_modalities)) & 1
    weights = 2.0 ** bits.sum(axis=1)
    pick = rng.choice(len(codes), size=B, p=weights / weights.sum())
    return bits[pick].astype(bool)


def sample_batch(ds: Dataset, exp: Experiment, B: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One micro-batch: (available, kept), each [B, 4] over GROUPS. Proprio is 1 = present,
    0 = absent; without proprio in the run, both are 0."""
    lengths, weights = zip(*sorted(ds.lang_hist.items()))
    lang = rng.choice(lengths, size=B, p=np.array(weights) / sum(weights))
    avail = np.stack([np.full(B, ds.static_tokens), np.full(B, ds.wrist_tokens), lang], axis=1)
    proprio_avail = np.full(B, int(exp.proprio))
    if exp.strategy == "none":
        kept, proprio = avail.copy(), proprio_avail.copy()
    elif exp.strategy == "subset":
        mask = subset_masks(B, 4 if exp.proprio else 3, rng)
        kept = avail * mask[:, :3]
        proprio = mask[:, 3].astype(int) if exp.proprio else proprio_avail.copy()
    elif exp.strategy in ("dirichlet", "uniform"):
        if exp.strategy == "dirichlet":
            kept = dirichlet_counts(avail, exp.keep_fraction, rng)
        else:
            kept = uniform_counts(avail, exp.keep_fraction, rng)
        proprio = proprio_avail * (rng.random(B) < exp.proprio_keep)
    else:
        raise ValueError(f"unknown strategy {exp.strategy!r}")
    return np.column_stack([avail, proprio_avail]), np.column_stack([kept, proprio])


def simulate(
    ds: Dataset, exp: Experiment, n_samples: int, rng: np.random.Generator, batch: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(available, kept), each [n_samples, 4], from consecutive micro-batches."""
    B = batch or ds.batch
    out = [sample_batch(ds, exp, B, rng) for _ in range(-(-n_samples // B))]
    return np.concatenate([a for a, _ in out])[:n_samples], np.concatenate([k for _, k in out])[:n_samples]


def summarize(avail: np.ndarray, kept: np.ndarray) -> list[dict]:
    """Per modality: mean available / kept tokens, kept fraction, P(empty), P(partial), P(full).
    Proprio rows count presence (0/1); modalities a run doesn't use are skipped."""
    rows = []
    for g, name in enumerate(GROUPS):
        a, k = avail[:, g], kept[:, g]
        if not a.any():
            continue
        rows.append(
            {
                "modality": name,
                "available": a.mean(),
                "kept": k.mean(),
                "kept_fraction": k.sum() / a.sum(),
                "p_empty": (k == 0).mean(),
                "p_partial": ((k > 0) & (k < a)).mean(),
                "p_full": (k == a).mean(),
            }
        )
    return rows


def plot(dataset_key: str, results: list[tuple[Experiment, np.ndarray, np.ndarray]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = DATASETS[dataset_key]
    # (i) Kept tokens per token modality, one row per experiment.
    fig, axes = plt.subplots(len(results), 3, figsize=(10, 1.6 * len(results)), sharex="col", squeeze=False)
    for r, (exp, avail, kept) in enumerate(results):
        for g in range(3):
            ax = axes[r, g]
            hi = int(avail[:, g].max())
            ax.hist(kept[:, g], bins=np.arange(hi + 2) - 0.5, density=True, color=f"C{g}")
            ax.axvline(kept[:, g].mean(), color="k", lw=0.8, ls="--")
            ax.set_yticks([])
            if r == 0:
                ax.set_title(GROUPS[g])
            if g == 0:
                ax.set_ylabel(exp.name, rotation=0, ha="right", va="center", fontsize=8)
            if r == len(results) - 1:
                ax.set_xlabel("kept tokens")
    fig.suptitle(f"{ds.name}: kept tokens per training sample (dashed: mean)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{dataset_key}_kept_tokens.{ext}", dpi=150)
    plt.close(fig)

    # (ii) Share of samples where each modality is empty / partial / full.
    fig, axes = plt.subplots(1, 4, figsize=(14, 0.45 * len(results) + 1.5), sharey=True)
    names = [exp.name for exp, _, _ in results]
    y = np.arange(len(results))
    for g, ax in enumerate(axes):
        left = np.zeros(len(results))
        for part, color in (("p_empty", "#d62728"), ("p_partial", "#ff7f0e"), ("p_full", "#2ca02c")):
            vals = []
            for _, avail, kept in results:
                row = [s for s in summarize(avail, kept) if s["modality"] == GROUPS[g]]
                vals.append(row[0][part] if row else 0.0)
            ax.barh(y, vals, left=left, color=color, label=part[2:])
            left += np.array(vals)
        ax.set_title(GROUPS[g])
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of training samples")
    axes[0].set_yticks(y, names, fontsize=8)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(f"{ds.name}: input empty / partly kept / fully kept (blank: input not used)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{dataset_key}_presence.{ext}", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    p.add_argument("--experiment", help="only experiments whose name contains this")
    p.add_argument("--samples", type=int, default=100_000)
    p.add_argument("--batch", type=int, help="micro-batch size (default: the dataset's config)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, help="write summary.csv and the figures here")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    keys = list(DATASETS) if a.dataset == "all" else [a.dataset]
    csv_rows = []
    for key in keys:
        ds = DATASETS[key]
        results = []
        print(f"\n## {ds.name} (static {ds.static_tokens}, wrist {ds.wrist_tokens}, language "
              f"{min(ds.lang_hist)}-{max(ds.lang_hist)} tokens, micro-batch {a.batch or ds.batch})")
        print(f"{'experiment':34s} {'modality':9s} {'avail':>6s} {'kept':>6s} {'frac':>5s} "
              f"{'empty':>6s} {'part':>6s} {'full':>6s}")
        for exp in EXPERIMENTS[key]:
            if a.experiment and a.experiment not in exp.name:
                continue
            avail, kept = simulate(ds, exp, a.samples, rng, a.batch)
            results.append((exp, avail, kept))
            for s in summarize(avail, kept):
                print(f"{exp.name:34s} {s['modality']:9s} {s['available']:6.1f} {s['kept']:6.1f} "
                      f"{s['kept_fraction']:5.2f} {s['p_empty']:6.3f} {s['p_partial']:6.3f} {s['p_full']:6.3f}")
                csv_rows.append({"dataset": ds.name, "experiment": exp.name, **s})
        if a.out and results:
            a.out.mkdir(parents=True, exist_ok=True)
            plot(key, results, a.out)
    if a.out and csv_rows:
        with open(a.out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)


if __name__ == "__main__":
    main()
