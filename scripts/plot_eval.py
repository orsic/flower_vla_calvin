#!/usr/bin/env python3
"""Style + figures + CLI for scripts/plot_data.py's evaluation data, pulled from W&B.

Three subcommands, one per figure (`all` runs every one, sharing a single fetch):

  presence     Clean LIBERO-10: bars for all-4-modalities vs. each 1-modality-off,
               grouped by modality_dropout_proprio_keep_p.
  perturbation LIBERO-10-Plus: per-perturbation-category bars, one filled bar per
               keep_p, each paired with a hollow, 45-degree-hatched init-state-matched
               LIBERO original bar (full solid outline) in the same color.
  severity     All-modality vs. 1-left-out, each bar paired with its init-state-
               matched LIBERO original baseline the same way. Both physical severity
               and upstream difficulty_level get a per-category breakdown (faceted)
               and a pooled ("totals") view -- 4 PDFs.

Usage:
  python scripts/plot_eval.py presence --filters '{"config.modality_dropout": true}'
  python scripts/plot_eval.py all --filters '{"config.modality_dropout": true}'

Every subcommand shares one PoolConfig: --filters is the single mongo-style W&B
filter (same syntax as analyze_wandb.py's --filters) that defines the run pool for
every plot, so a --filters change moves consistently across all of them.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # PDF output only -- no display in this container
import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.patches import Patch

import analyze_wandb
import plot_data
import severity_sr
from plot_data import Bar

FONT_SCALE = 2.5

# Physical-Intelligence-paper-style categorical hues for the 5 inference-time
# modality configurations. Validated 2026-09-18 via the dataviz skill's
# validate_palette.js, adjacent-pair mode (the tool's own default for "stacks, bars,
# lines" -- these are always grouped/faceted bar charts, never scatter/map) against a
# white #fcfcfb surface: ALL CHECKS PASS for {no_wrist, no_static, no_proprio,
# no_lang} in both this order and the reference palette's dark steps.
#   node validate_palette.js "#2a78d6,#eb6834,#1baf7a,#4a3aa7" --mode light
MODALITY_COLORS = {
    # Neutral reference bar for "all modalities on" -- deliberately NOT one of the
    # validated hues above: it isn't competing for hue identity against the 4
    # withheld-modality bars, the same "gray the non-highlighted role" pattern as an
    # emphasis chart, just inverted (gray marks the baseline here, not the outlier).
    "all": "#4a4a4a",
    "no_wrist": "#2a78d6",
    "no_static": "#eb6834",
    "no_proprio": "#1baf7a",  # WARNed below 3:1 contrast on white by the validator --
    # always shown with a visible legend/tick label, per the relief rule, never
    # relying on this color alone.
    "no_lang": "#4a3aa7",
}
MODALITY_LABELS = {
    "all": "all modalities",
    "no_wrist": "no wrist cam",
    "no_static": "no static cam",
    "no_proprio": "no proprio",
    "no_lang": "no language",
}

# Sequential ramp for effective modality_dropout_proprio_keep_p -- an ordered
# magnitude, not an identity -- sampled from matplotlib's cividis (perceptually
# uniform, colorblind-safe by design) at t=[0.75, 0.5625, 0.375, 0.1875, 0], per
# explicit request instead of single-hue blue intensities, inverted so keep_p=1.0
# (least dropout) is the yellow end. keep_p=0.0 is a use_proprio=False run (never
# receives proprioception at all, regardless of its config's own
# modality_dropout_proprio_keep_p value -- see plot_data._effective_keep_p), not a
# literal keep_p=0.0 dropout setting. Validated with --ordinal against a white
# surface: lightness-monotone, adjacent-ΔL and light-end-contrast all PASS; "single
# hue" FAILs by design -- cividis deliberately spans two hues (navy -> olive/yellow),
# which is the accepted tradeoff of naming this specific colormap rather than a
# violation to fix.
#   node validate_palette.js "#bcae6c,#8c8878,#61656f,#32436d,#00224e" --ordinal --mode light
_KEEP_P_RAMP = {1.0: "#bcae6c", 0.75: "#8c8878", 0.5: "#61656f", 0.25: "#32436d", 0.0: "#00224e"}


def keep_p_color(keep_p: float) -> str:
    try:
        return _KEEP_P_RAMP[round(keep_p, 2)]
    except KeyError:
        raise ValueError(
            f"No color assigned for modality_dropout_proprio_keep_p={keep_p!r} -- "
            f"expected one of {sorted(_KEEP_P_RAMP)}"
        )


# Canonical facet order for the severity figure -- Language Instructions and
# Background Textures are absent by construction (see plot_data.prepare_severity's
# docstring: severity_sr.collect() never emits axis="severity" records for them).
SEVERITY_CATEGORY_ORDER = [
    "Camera Viewpoints",
    "Robot Initial States",
    "Sensor Noise",
    "Objects Layout",
    "Light Conditions",
]


def apply_style() -> None:
    """Physical-Intelligence-paper-style rcParams, font scale 2.5x default: white
    background, no top/right/left spines, light horizontal-only gridlines behind the
    bars, frameless legend, sans-serif."""
    plt.rcParams.update(
        {
            "font.size": 10 * FONT_SCALE,
            "font.family": "sans-serif",
            "axes.titlesize": 12 * FONT_SCALE,
            "axes.labelsize": 11 * FONT_SCALE,
            "xtick.labelsize": 9 * FONT_SCALE,
            "ytick.labelsize": 9 * FONT_SCALE,
            "legend.fontsize": 9 * FONT_SCALE,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 1.0,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
            # Default padding is sized for the default font; at 2.5x it's too tight
            # between a figure-level legend/title and the first subplot's own title.
            "figure.constrained_layout.h_pad": 0.04 * FONT_SCALE,
            "figure.constrained_layout.hspace": 0.02 * FONT_SCALE,
            "hatch.linewidth": 1.5,
        }
    )


def _bar_errors(bars: List[Optional[Bar]]) -> Tuple[List[int], List[float], List[float], List[float]]:
    """(present_indices, heights, lower_err, upper_err), all in percent, for a list
    of Bar|None -- present_indices skips a None/n==0 entry entirely (no run in the
    pool covers that combination) rather than drawing a fake zero-height bar, which
    would be indistinguishable from a genuine 0% success rate. lo/hi are clamped at 0
    -- ci_low/ci_high are mathematically within [rate's own bounds], but summing many
    runs' successes/n before computing the Wilson interval (pool_bars) can leave a
    sub-1e-9 float overshoot the same way severity_sr.wilson_interval's own docstring
    describes, which matplotlib's errorbar rejects outright as negative."""
    present, heights, lo, hi = [], [], [], []
    for i, bar in enumerate(bars):
        if bar is None or bar.n == 0:
            continue
        present.append(i)
        heights.append(bar.rate * 100)
        lo.append(max(0.0, (bar.rate - bar.ci_low) * 100))
        hi.append(max(0.0, (bar.ci_high - bar.rate) * 100))
    return present, heights, lo, hi


def _draw_grouped_bars(
    ax, group_labels: List[str], series: Dict[str, List[Optional[Bar]]], colors: Dict[str, str], labels: Dict[str, str]
) -> None:
    """One group of bars per group_labels entry, one bar per `series` key within
    each group -- used by plot_presence, the only figure with no paired orig-LIBERO
    baseline (its data source, libero_orig.csv's own 14-combo sweep, isn't itself paired
    against anything). A series with no data anywhere in this axes is never plotted, so
    it never appears in ax.legend()."""
    n_series = len(series)
    group_width = 0.8
    slot_width = group_width / n_series
    x = np.arange(len(group_labels))
    for i, (key, bars) in enumerate(series.items()):
        present, heights, lo, hi = _bar_errors(bars)
        if not present:
            continue
        offsets = x[present] - group_width / 2 + slot_width * (i + 0.5)
        ax.bar(
            offsets, heights, width=slot_width * 0.9, color=colors[key], label=labels[key],
            yerr=[lo, hi], capsize=3, error_kw={"elinewidth": 1.2, "alpha": 0.7, "ecolor": "#333333"},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success rate (%)")


def _draw_paired_grouped_bars(
    ax, group_labels: List[str], series: Dict[Any, List[Tuple[Optional[Bar], Optional[Bar]]]], colors: Dict[Any, str]
) -> None:
    """One group of bars per group_labels entry, one filled+hollow-hatched PAIR per
    `series` key within each group: the filled bar is the measurement (LIBERO-Plus, or
    clean-LIBERO under a keep_p), the hollow 45-degree-hatched bar in the same color is
    its paired init-state-matched LIBERO original baseline -- performance on the exact
    initial states the filled bar next to it entails (see plot_data.prepare_severity's
    docstring). Either half of a pair being None simply isn't drawn. Shared by every
    figure that pairs a series (keep_p, or a modality config) against that baseline:
    plot_perturbation, plot_severity_percategory/_pooled, plot_difficulty_percategory/
    _pooled. Legend handles are built by each caller (one shared "orig LIBERO" hatch
    entry, not one per series key), so no `label=` is set here."""
    n_series = len(series)
    group_width = 0.8
    slot_width = group_width / n_series
    bar_width = slot_width * 0.42
    gap = slot_width * 0.05
    x = np.arange(len(group_labels))
    for i, (key, pairs) in enumerate(series.items()):
        color = colors[key]
        slot_center = x - group_width / 2 + slot_width * (i + 0.5)
        plus_present, plus_heights, plus_lo, plus_hi = _bar_errors([p[0] for p in pairs])
        orig_present, orig_heights, orig_lo, orig_hi = _bar_errors([p[1] for p in pairs])
        if plus_present:
            ax.bar(
                slot_center[plus_present] - bar_width / 2 - gap, plus_heights, width=bar_width, color=color,
                yerr=[plus_lo, plus_hi], capsize=3, error_kw={"elinewidth": 1.2, "alpha": 0.7, "ecolor": "#333333"},
            )
        if orig_present:
            ax.bar(
                slot_center[orig_present] + bar_width / 2 + gap, orig_heights, width=bar_width, facecolor="none",
                edgecolor=color, linewidth=1.8, hatch="//",
                yerr=[orig_lo, orig_hi], capsize=3, error_kw={"elinewidth": 1.2, "alpha": 0.7, "ecolor": color},
            )
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success rate (%)")


def _orig_legend_handle() -> Patch:
    return Patch(facecolor="none", edgecolor="#333333", hatch="//", label="orig LIBERO (init-state matched)")


# ---------------------------------------------------------------------------
# Plot 1 -- clean LIBERO-10, all-4 vs. 1-modality-off, by keep_p
# ---------------------------------------------------------------------------


def plot_presence(pool_data: Dict[float, Dict[str, Bar]], outdir: Path) -> Path:
    apply_style()
    keep_ps = sorted(pool_data, reverse=True)
    group_labels = [f"{kp:g}" for kp in keep_ps]
    series = {cfg: [pool_data[kp].get(cfg) for kp in keep_ps] for cfg in plot_data.MODALITY_CONFIGS}

    fig, ax = plt.subplots(figsize=(6 * FONT_SCALE, 4.5 * FONT_SCALE))
    _draw_grouped_bars(ax, group_labels, series, MODALITY_COLORS, MODALITY_LABELS)
    ax.set_xlabel("modality_dropout_proprio_keep_p")
    # Title folded into the legend's own `title=` (not a separate ax.set_title) --
    # constrained_layout doesn't reliably space an outside legend and an axes title
    # apart when they're two independent top-margin claimants.
    fig.legend(
        loc="outside upper center", ncol=min(3, len(plot_data.MODALITY_CONFIGS)),
        title="Clean LIBERO-10", title_fontsize=12 * FONT_SCALE,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "presence_libero10.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 2 -- LIBERO-10-Plus per perturbation category, by keep_p, paired with the
# init-state-matched LIBERO original baseline
# ---------------------------------------------------------------------------


def plot_perturbation(pool_data: Dict[str, Dict[float, Tuple[Bar, Optional[Bar]]]], outdir: Path) -> Path:
    apply_style()
    categories = sorted(pool_data)
    keep_ps = sorted({kp for by_kp in pool_data.values() for kp in by_kp}, reverse=True)
    series = {kp: [pool_data[cat].get(kp, (None, None)) for cat in categories] for kp in keep_ps}
    colors = {kp: keep_p_color(kp) for kp in keep_ps}

    fig, ax = plt.subplots(figsize=(7 * FONT_SCALE, 5 * FONT_SCALE))
    _draw_paired_grouped_bars(ax, categories, series, colors)
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.annotate(
        "orig_n is deduplicated by (modality combo, base task) -- ≤10 for LIBERO-10, so the\n"
        "hatched bars carry wide CIs by construction, not by chance.",
        xy=(0.0, -0.55), xycoords="axes fraction", ha="left", va="top", fontsize=8 * FONT_SCALE * 0.55, color="#666666",
    )

    handles = [Patch(facecolor=keep_p_color(kp), label=f"keep_p={kp:g}") for kp in keep_ps]
    handles.append(_orig_legend_handle())
    fig.legend(
        handles=handles, loc="outside upper center", ncol=min(3, len(handles)),
        title="LIBERO-10-Plus vs. init-state-matched LIBERO original", title_fontsize=12 * FONT_SCALE,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "perturbation_libero10plus.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 3 -- all-modality vs. 1-left-out, each bar paired with its init-state-matched
# LIBERO original baseline. Both axes (severity, difficulty_level) get both a
# per-category breakdown and a pooled ("totals") view -- four PDFs.
# ---------------------------------------------------------------------------


def _present_configs(pairs_by_key: Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]]) -> List[str]:
    """Which of MODALITY_CONFIGS have a plus_bar somewhere in a pooled (non-faceted)
    figure's data -- a config with none is dropped from the legend, same rationale as
    _draw_grouped_bars/_draw_paired_grouped_bars silently skipping an empty series."""
    return [cfg for cfg in plot_data.MODALITY_CONFIGS if any(cfg in configs for configs in pairs_by_key.values())]


def _present_configs_faceted(by_category: Dict[str, Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]]]) -> List[str]:
    return [
        cfg for cfg in plot_data.MODALITY_CONFIGS
        if any(cfg in configs for bins in by_category.values() for configs in bins.values())
    ]


def _plot_category_facets(
    by_category: Dict[str, Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]]],
    outdir: Path,
    filename: str,
    title: str,
    category_order: Optional[List[str]] = None,
    robot_initstate_note: bool = False,
    thin_dense_ticks: bool = False,
) -> Path:
    """One stacked facet per category, x = that category's own bins, one filled+hollow-
    hatched pair per modality config -- shared by plot_severity_percategory (severity
    bins, curated 5-category order, the Sensor Noise tick-thinning and Robot Initial
    States caveat) and plot_difficulty_percategory (difficulty_level bins, every
    category, neither of the above applies)."""
    apply_style()
    categories = [c for c in category_order if c in by_category] if category_order else sorted(by_category)

    fig, axes = plt.subplots(
        len(categories), 1, figsize=(7 * FONT_SCALE, 3.5 * FONT_SCALE * len(categories)), squeeze=False
    )
    for ax, category in zip(axes[:, 0], categories):
        bins = list(by_category[category])
        series = {
            cfg: [by_category[category][bin_label].get(cfg, (None, None)) for bin_label in bins]
            for cfg in plot_data.MODALITY_CONFIGS
        }
        _draw_paired_grouped_bars(ax, bins, series, MODALITY_COLORS)
        ax.set_xlabel("")
        ax.set_title(category)
        ax.tick_params(axis="x", rotation=25)
        # Sensor Noise alone has ~50 bins (5 corruptions x 10 severities) -- every bar
        # is still drawn, but only every Nth tick label, so labels stay legible instead
        # of overlapping into an unreadable smear.
        if thin_dense_ticks and len(bins) > 20:
            step = max(1, len(bins) // 15)
            ax.set_xticklabels([b if i % step == 0 else "" for i, b in enumerate(bins)])
        if robot_initstate_note and category == "Robot Initial States":
            # Reserved headroom above the tallest possible bar (100%), not a corner of
            # the plot area -- with every bar now paired against its orig-LIBERO
            # baseline (consistently high across every bin), there is no bin left with
            # enough blank space near the bars to tuck this note into without risking
            # a collision. x is axes-fraction, y is data-space, via get_yaxis_transform,
            # so it's centered regardless of how many bins this facet has.
            ax.set_ylim(0, 118)
            ax.annotate(
                "known LIBERO-Plus limitation: this axis's perturbation is discarded before rollout\n"
                "(see severity_sr.ROBOT_INITSTATE_NOTE) -- not a real physical gradient",
                xy=(0.5, 103), xycoords=ax.get_yaxis_transform(), ha="center", va="bottom",
                fontsize=8 * FONT_SCALE * 0.55, color="#666666",
            )

    # A separate fig.suptitle() alongside this outside legend leaves constrained_layout
    # juggling two independent top-margin claimants, which it doesn't reliably space
    # apart -- folding the title into the legend's own `title=` keeps it one artist.
    handles = [Patch(facecolor=MODALITY_COLORS[cfg], label=MODALITY_LABELS[cfg]) for cfg in _present_configs_faceted(by_category)]
    handles.append(_orig_legend_handle())
    fig.legend(handles=handles, loc="outside upper center", ncol=min(3, len(handles)), title=title, title_fontsize=12 * FONT_SCALE)

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_severity_percategory(by_category: Dict[str, Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]]], outdir: Path) -> Path:
    return _plot_category_facets(
        by_category, outdir, "severity_modality_off_percategory.pdf",
        "LIBERO-10-Plus by physical perturbation severity",
        category_order=SEVERITY_CATEGORY_ORDER, robot_initstate_note=True, thin_dense_ticks=True,
    )


def plot_difficulty_percategory(by_category: Dict[str, Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]]], outdir: Path) -> Path:
    return _plot_category_facets(
        by_category, outdir, "difficulty_modality_off_percategory.pdf",
        "LIBERO-10-Plus by upstream difficulty_level, per category",
    )


def _plot_pooled_bars(
    pooled: Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]],
    outdir: Path,
    filename: str,
    title: str,
    xlabel: str,
    sort_key=None,
    rotate: bool = False,
) -> Path:
    """One axes, x = pooled's own keys (categories, or difficulty levels), one
    filled+hollow-hatched pair per modality config -- shared by plot_severity_pooled
    (pools each category's own severity bins) and plot_difficulty_pooled (pools across
    categories on the shared difficulty_level scale)."""
    apply_style()
    keys = sorted(pooled, key=sort_key) if sort_key else sorted(pooled)
    series = {cfg: [pooled[k].get(cfg, (None, None)) for k in keys] for cfg in plot_data.MODALITY_CONFIGS}

    fig, ax = plt.subplots(figsize=(7 * FONT_SCALE, 5 * FONT_SCALE))
    _draw_paired_grouped_bars(ax, keys, series, MODALITY_COLORS)
    ax.set_xlabel(xlabel)
    if rotate:
        ax.set_xticklabels(keys, rotation=25, ha="right")

    handles = [Patch(facecolor=MODALITY_COLORS[cfg], label=MODALITY_LABELS[cfg]) for cfg in _present_configs(pooled)]
    handles.append(_orig_legend_handle())
    fig.legend(handles=handles, loc="outside upper center", ncol=min(3, len(handles)), title=title, title_fontsize=12 * FONT_SCALE)

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_severity_pooled(pooled: Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]], outdir: Path) -> Path:
    return _plot_pooled_bars(
        pooled, outdir, "severity_modality_off_pooled.pdf",
        "LIBERO-10-Plus by perturbation category (pooled over each category's severity bins)",
        "perturbation category", rotate=True,
    )


def plot_difficulty_pooled(pooled: Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]], outdir: Path) -> Path:
    return _plot_pooled_bars(
        pooled, outdir, "difficulty_modality_off_pooled.pdf",
        "LIBERO-10-Plus by upstream difficulty_level (pooled across categories)",
        "difficulty_level (upstream annotation, pooled across categories)",
        sort_key=analyze_wandb._natural_key,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class PoolConfig:
    """Shared by every subcommand -- --filters is the one mongo-style W&B filter
    (same syntax as analyze_wandb.py's --filters) that defines the run pool for
    every plot."""

    filters: str = "{}"
    modalities: Optional[str] = None
    entity: Optional[str] = None
    project: Optional[str] = None
    cache_dir: Path = Path("/saves/plot_cache")
    outdir: Path = Path("/saves/plots")
    libero_plus_root: str = severity_sr.DEFAULT_LIBERO_PLUS_ROOT
    measure: str = "ccs"
    refresh: bool = False


def _fetch(cfg: PoolConfig):
    return plot_data.fetch_pool(
        cfg.filters, cfg.modalities, cfg.entity, cfg.project, cfg.cache_dir, cfg.measure, cfg.libero_plus_root,
        refresh=cfg.refresh,
    )


def presence(cfg: tyro.conf.OmitArgPrefixes[PoolConfig]) -> None:
    """Plot 1: clean LIBERO-10, all-4-modalities vs. 1-modality-off, by keep_p."""
    pool = _fetch(cfg)
    path = plot_presence(plot_data.prepare_presence(pool), cfg.outdir)
    print(f"wrote {path}")


def perturbation(cfg: tyro.conf.OmitArgPrefixes[PoolConfig]) -> None:
    """Plot 2: LIBERO-10-Plus per perturbation category, by keep_p, paired with the
    init-state-matched LIBERO original baseline."""
    pool = _fetch(cfg)
    path = plot_perturbation(plot_data.prepare_perturbation(pool), cfg.outdir)
    print(f"wrote {path}")


def _severity_plots(pool, outdir: Path) -> List[Path]:
    """The 4 PDFs shared by the `severity` subcommand and `all`: both axes (physical
    severity, upstream difficulty_level) get both a per-category breakdown and a
    pooled ("totals") view, each bar paired with its init-state-matched LIBERO
    original baseline."""
    severity_by_category = plot_data.prepare_severity(pool, axis="severity")
    difficulty_by_category = plot_data.prepare_severity(pool, axis="difficulty_level")
    return [
        plot_severity_percategory(severity_by_category, outdir),
        plot_severity_pooled(plot_data.prepare_severity_pooled(severity_by_category), outdir),
        plot_difficulty_percategory(difficulty_by_category, outdir),
        plot_difficulty_pooled(plot_data.prepare_difficulty_pooled(difficulty_by_category), outdir),
    ]


def severity(cfg: tyro.conf.OmitArgPrefixes[PoolConfig]) -> None:
    """Plot 3: all-modality vs. 1-left-out, each bar paired with its init-state-matched
    LIBERO original baseline. Both physical severity and upstream difficulty_level get
    a per-category breakdown and a pooled ("totals") view -- 4 PDFs."""
    pool = _fetch(cfg)
    for path in _severity_plots(pool, cfg.outdir):
        print(f"wrote {path}")


def all_plots(cfg: tyro.conf.OmitArgPrefixes[PoolConfig]) -> None:
    """Every plot, sharing a single fetch of the run pool."""
    pool = _fetch(cfg)
    print(f"wrote {plot_presence(plot_data.prepare_presence(pool), cfg.outdir)}")
    print(f"wrote {plot_perturbation(plot_data.prepare_perturbation(pool), cfg.outdir)}")
    for path in _severity_plots(pool, cfg.outdir):
        print(f"wrote {path}")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {"presence": presence, "perturbation": perturbation, "severity": severity, "all": all_plots}
    )
