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
import sys
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
# (least dropout) is the yellow end. keep_p=0.0 is a use_proprio=False dropout run
# (never receives proprioception at all, regardless of its config's own
# modality_dropout_proprio_keep_p value -- see plot_data._series_key), not a
# literal keep_p=0.0 dropout setting. Validated with --ordinal against a white
# surface: lightness-monotone, adjacent-ΔL and light-end-contrast all PASS; "single
# hue" FAILs by design -- cividis deliberately spans two hues (navy -> olive/yellow),
# which is the accepted tradeoff of naming this specific colormap rather than a
# violation to fix.
#   node validate_palette.js "#bcae6c,#8c8878,#61656f,#32436d,#00224e" --ordinal --mode light
_KEEP_P_RAMP = {1.0: "#bcae6c", 0.75: "#8c8878", 0.5: "#61656f", 0.25: "#32436d", 0.0: "#00224e"}

# Fixed identity colors for the two modality_dropout=False series (plot_data.
# BASELINE_PROPRIO/BASELINE_NO_PROPRIO) -- deliberately outside the keep_p ramp above,
# same "own hue, not a ramp step" rationale as MODALITY_COLORS. Validated
# 2026-09-23 via the dataviz skill's validate_palette.py (no node available in this
# environment; it's an exact Python port of validate_palette.js), categorical mode,
# against a white #fcfcfb surface AND against the ramp's own adjacent endpoint
# (keep_p=0.0, #00224e -- these two baselines sort to its right, see
# series_sort_key): CVD separation, normal-vision floor and contrast all PASS; only
# #00224e's own lightness-band/chroma-floor checks fail, which is the ramp's
# pre-existing, already-accepted ordinal-ramp tradeoff (see above), not something
# these two colors introduce.
#   python3 validate_palette.py "#00224e,#c1121f,#0e9594" --mode light
_BASELINE_COLORS = {
    plot_data.BASELINE_PROPRIO: "#c1121f",
    plot_data.BASELINE_NO_PROPRIO: "#0e9594",
}
_BASELINE_LABELS = {
    plot_data.BASELINE_PROPRIO: "no dropout (+proprio)",
    plot_data.BASELINE_NO_PROPRIO: "no dropout (no proprio)",
}
# Fixed legend/x-axis order for the two baseline series, placed after the (descending)
# keep_p ramp -- see series_sort_key.
_BASELINE_ORDER = [plot_data.BASELINE_PROPRIO, plot_data.BASELINE_NO_PROPRIO]


def series_color(key) -> str:
    """The color for one presence/perturbation series key -- a float
    modality_dropout_proprio_keep_p on the cividis ramp, or one of the two BASELINE_*
    strings on their own fixed hues (see _BASELINE_COLORS)."""
    if isinstance(key, str):
        try:
            return _BASELINE_COLORS[key]
        except KeyError:
            raise ValueError(f"No color assigned for series key {key!r} -- expected one of {sorted(_BASELINE_COLORS)}")
    try:
        return _KEEP_P_RAMP[round(key, 2)]
    except KeyError:
        raise ValueError(f"No color assigned for modality_dropout_proprio_keep_p={key!r} -- expected one of {sorted(_KEEP_P_RAMP)}")


def series_label(key) -> str:
    """Legend/tick label for one series key."""
    if isinstance(key, str):
        return _BASELINE_LABELS[key]
    return f"keep_p={key:g}"


def series_sort_key(key):
    """Sort order for a mix of float keep_p and BASELINE_* keys: keep_p descending
    (least dropout first, matching the pre-existing figures), then the two baselines
    in _BASELINE_ORDER."""
    if isinstance(key, str):
        return (1, _BASELINE_ORDER.index(key))
    return (0, -key)


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

# The one series the "clear" (presentation-grade, no-McNemar) severity/difficulty
# figures show: all-modality LIBERO-10-Plus vs. its all-modality init-state-matched
# LIBERO original -- every 1-modality-off comparison is dropped from those figures by
# design (see plot_severity_pooled_clear etc.).
_CLEAR_CONFIGS = ["all"]


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


_VALUE_LABEL_FONTSIZE = 6.5 * FONT_SCALE
# Small negative y-offset (points, not data units) from the bar's own top edge --
# pulls the label down just inside the bar instead of sitting exactly on the edge.
_VALUE_LABEL_PADDING = -4
_MCNEMAR_FONTSIZE = 8 * FONT_SCALE * 0.55  # same reduced scale as the file's other caption/caveat text
_MCNEMAR_COLOR = "#666666"
# Below-axis room for the rotated "p=... (n=...)" McNemar label every paired figure
# now draws under each Plus bar -- clears the category tick labels _plot_category_
# facets/_plot_pooled_bars rotate into that same space. A rotated ~15-character label
# ("p<0.001 (n=10)") at _MCNEMAR_FONTSIZE needs on the order of 100pt of vertical
# room; sized empirically against a rendered figure, not computed from font metrics.
_XTICK_PAD = 45 * FONT_SCALE
# Matplotlib's own default xtick pad, font-scaled -- used instead of _XTICK_PAD when a
# figure draws with mcnemar=False: there's no rotated p-label to clear underneath, so
# reserving _XTICK_PAD's room would just open a large empty gap under the axis.
_NO_MCNEMAR_XTICK_PAD = 3.5 * FONT_SCALE


def _annotate_bar_values(ax, xs, heights, color: str) -> None:
    """Print each bar's own value (already in percent, see _bar_errors) just inside
    its top edge, rotated 90 degrees -- a group can hold as many as 7 series (the
    perturbation figure's keep_p ramp plus its 2 baselines) x 2 bars, too narrow for a
    horizontal label. `color` is white for a filled Plus/presence bar, the bar's own
    series color for a hollow hatched orig-LIBERO bar (nothing to contrast against a
    fill that isn't there).

    Deliberately NOT ax.bar_label(): that method anchors an 'edge' label to the tip of
    the bar's own error bar when one is attached to its container (its `endpt`
    computation reads the errorbar's extent), not to the bar's own height -- every bar
    here has one, so the label would float above the error whisker instead of sitting
    on the bar. Anchoring directly at (x, height) and nudging inward with a small
    offset in points keeps the label attached to the value it's labeling."""
    for x, h in zip(xs, heights):
        ax.annotate(
            f"{h:.0f}", xy=(x, h), xytext=(0, _VALUE_LABEL_PADDING), textcoords="offset points",
            ha="center", va="top", rotation=90, color=color, fontsize=_VALUE_LABEL_FONTSIZE,
        )


def _annotate_mcnemar(ax, xs, bars: List[Bar]) -> None:
    """"p=<mcnemar_exact_p> (n=<mcnemar_n>)" beneath each drawn Plus bar whose paired
    orig baseline carries a task-level McNemar table (see plot_data.prepare_perturbation
    /prepare_severity, severity_sr.paired_baseline) -- skipped where mcnemar_n is None
    or 0 (no baseline, or every task-level pair in the group tied and was dropped). n
    is printed alongside p because LIBERO-10 has at most ~10 base tasks per category/
    bin, so the test's own power has to stay visible next to its result, not just the
    p-value on its own."""
    for x, bar in zip(xs, bars):
        if not bar.mcnemar_n:
            continue
        p = severity_sr.mcnemar_exact_p(bar.mcnemar_b, bar.mcnemar_c)
        label = f"p<0.001 (n={bar.mcnemar_n})" if p < 0.001 else f"p={p:.2f} (n={bar.mcnemar_n})"
        ax.annotate(
            label, xy=(x, 0), xycoords=ax.get_xaxis_transform(), xytext=(0, -4), textcoords="offset points",
            ha="center", va="top", rotation=90, fontsize=_MCNEMAR_FONTSIZE, color=_MCNEMAR_COLOR,
        )


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
        _annotate_bar_values(ax, offsets, heights, "white")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success rate (%)")


def _draw_paired_grouped_bars(
    ax, group_labels: List[str], series: Dict[Any, List[Tuple[Optional[Bar], Optional[Bar]]]], colors: Dict[Any, str],
    mcnemar: bool = True,
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
    entry, not one per series key), so no `label=` is set here.

    Every drawn bar also gets its own value label (_annotate_bar_values), and every
    drawn Plus bar with a paired McNemar table gets a "p=... (n=...)" label beneath the
    axis (_annotate_mcnemar) -- ax.tick_params' pad below reserves the room the rotated
    p-label and the group's own (often rotated) tick label both need so they don't
    collide. mcnemar=False (the presentation-grade "clear" figures) skips that label
    entirely and falls back to matplotlib's own default tick pad, since there's no
    p-label to clear room for."""
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
            plus_x = slot_center[plus_present] - bar_width / 2 - gap
            ax.bar(
                plus_x, plus_heights, width=bar_width, color=color,
                yerr=[plus_lo, plus_hi], capsize=3, error_kw={"elinewidth": 1.2, "alpha": 0.7, "ecolor": "#333333"},
            )
            _annotate_bar_values(ax, plus_x, plus_heights, "white")
            if mcnemar:
                _annotate_mcnemar(ax, plus_x, [pairs[j][0] for j in plus_present])
        if orig_present:
            orig_x = slot_center[orig_present] + bar_width / 2 + gap
            ax.bar(
                orig_x, orig_heights, width=bar_width, facecolor="none",
                edgecolor=color, linewidth=1.8, hatch="//",
                yerr=[orig_lo, orig_hi], capsize=3, error_kw={"elinewidth": 1.2, "alpha": 0.7, "ecolor": color},
            )
            _annotate_bar_values(ax, orig_x, orig_heights, color)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success rate (%)")
    ax.tick_params(axis="x", pad=_XTICK_PAD if mcnemar else _NO_MCNEMAR_XTICK_PAD)


def _orig_legend_handle() -> Patch:
    return Patch(facecolor="none", edgecolor="#333333", hatch="//", label="orig LIBERO (init-state matched)")


# ---------------------------------------------------------------------------
# Plot 1 -- clean LIBERO-10, all-4 vs. 1-modality-off, by keep_p
# ---------------------------------------------------------------------------


def plot_presence(pool_data: Dict[plot_data.SeriesKey, Dict[str, Bar]], outdir: Path) -> Path:
    apply_style()
    keys = sorted(pool_data, key=series_sort_key)
    group_labels = [series_label(k) for k in keys]
    series = {cfg: [pool_data[k].get(cfg) for k in keys] for cfg in plot_data.MODALITY_CONFIGS}

    fig, ax = plt.subplots(figsize=(6 * FONT_SCALE, 4.5 * FONT_SCALE))
    _draw_grouped_bars(ax, group_labels, series, MODALITY_COLORS, MODALITY_LABELS)
    # Rotated, same as every other figure's category/bin labels -- group_labels can now
    # include the BASELINE_* series' full "no dropout (...)" text, too wide to sit
    # horizontally at this figure's width without overlapping its neighbors.
    ax.set_xticklabels(group_labels, rotation=25, ha="right")
    ax.set_xlabel("training config")
    # Title folded into the legend's own `title=` (not a separate ax.set_title) --
    # constrained_layout doesn't reliably space an outside legend and an axes title
    # apart when they're two independent top-margin claimants.
    fig.legend(
        loc="outside upper center", ncol=min(3, len(plot_data.MODALITY_CONFIGS)),
        title="Clean LIBERO-10", title_fontsize=12 * FONT_SCALE,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "presence_libero10.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 2 -- LIBERO-10-Plus per perturbation category, by keep_p, paired with the
# init-state-matched LIBERO original baseline
# ---------------------------------------------------------------------------


# The only two training-config series the clear perturbation figure shows: dropout
# with proprioception always kept vs. no dropout at all (plot_data._series_key). Every
# other series key present in the pool (other keep_p values, the no-proprio baseline)
# is dropped from this figure by design -- it isolates the one dropout-vs-no-dropout
# comparison. Exempt from _single_series_key/_require_single_series_key below: this
# figure is deliberately built from 2 training configs, not 1.
_CLEAR_PERTURBATION_KEYS = [1.0, plot_data.BASELINE_PROPRIO]


def _single_series_key(pool) -> Optional[plot_data.SeriesKey]:
    """The one training-config series key the clear severity/difficulty figures can be
    drawn for, or None when the pool spans more than one (or is empty). Those figures
    come from plot_data.prepare_severity, which pools every run in `pool` into one bar
    regardless of its series key (see fetch_pool's (run_id, series_key, analysis)
    triples) -- so more than one config in the pool would have that bar silently
    average across trained models, exactly the ambiguity these figures exist to avoid."""
    keys = {series_key for _run_id, series_key, _analysis in pool}
    return next(iter(keys)) if len(keys) == 1 else None


def _require_single_series_key(pool) -> plot_data.SeriesKey:
    """_single_series_key, raising instead of returning None -- used by the explicit
    `clear` subcommand, where silently skipping the figures the user asked for would be
    worse than failing outright."""
    key = _single_series_key(pool)
    if key is not None:
        return key
    keys = sorted({series_key for _run_id, series_key, _analysis in pool}, key=series_sort_key)
    found = ", ".join(series_label(k) for k in keys) if keys else "(empty pool)"
    raise ValueError(
        "clear severity/difficulty figures need exactly one training config in the pool "
        f"(plot_data.prepare_severity pools every run into one bar) -- found {len(keys)}: {found}. "
        "Narrow --filters to one config.modality_dropout_proprio_keep_p value, or to "
        "config.modality_dropout=false plus config.use_proprio."
    )


def _clear_series_style(series_key: plot_data.SeriesKey) -> Tuple[Dict[str, str], Dict[str, str]]:
    """(colors, labels) for a clear severity/difficulty figure's single all-modality
    bar -- the training config's own fixed color/label (series_color/series_label), so
    the same config reads the same color across every figure in the clear set instead of
    the neutral MODALITY_COLORS["all"] gray."""
    return {"all": series_color(series_key)}, {"all": series_label(series_key)}


def _plot_perturbation_bars(
    pool_data: Dict[str, Dict[plot_data.SeriesKey, Tuple[Bar, Optional[Bar]]]],
    outdir: Path,
    filename: str,
    title: str,
    keys: Optional[List[plot_data.SeriesKey]] = None,
    mcnemar: bool = True,
    note: Optional[str] = None,
) -> Path:
    """LIBERO-10-Plus per perturbation category, one filled+hollow-hatched pair per
    training-config series key -- shared by plot_perturbation (every series key present
    in the pool, McNemar labels + full caveat) and plot_perturbation_clear (only
    _CLEAR_PERTURBATION_KEYS, no McNemar).

    `keys`, when given, is used verbatim (not intersected with what's actually present
    in `pool_data`) -- same "fixed slot list, some may end up empty" contract as
    `configs` on _plot_pooled_bars/_plot_category_facets, so a `--filters` pool that
    happens to carry neither of _CLEAR_PERTURBATION_KEYS still renders a (bar-less)
    figure with both slots reserved, rather than a wanted_keys intersection collapsing
    to empty and dividing by zero in _draw_paired_grouped_bars. The legend still drops
    any key with no bar anywhere (see _present_keys)."""
    apply_style()
    categories = sorted(pool_data)
    present_keys = {k for by_key in pool_data.values() for k in by_key}
    sorted_keys = sorted(present_keys, key=series_sort_key) if keys is None else sorted(keys, key=series_sort_key)
    series = {k: [pool_data[cat].get(k, (None, None)) for cat in categories] for k in sorted_keys}
    colors = {k: series_color(k) for k in sorted_keys}

    fig, ax = plt.subplots(figsize=(7 * FONT_SCALE, 5 * FONT_SCALE))
    _draw_paired_grouped_bars(ax, categories, series, colors, mcnemar=mcnemar)
    ax.set_xticklabels(categories, rotation=25, ha="right")
    if note:
        ax.annotate(
            note, xy=(0.0, -0.7 if mcnemar else -0.5), xycoords="axes fraction", ha="left", va="top",
            fontsize=8 * FONT_SCALE * 0.55, color="#666666",
        )

    handles = [Patch(facecolor=series_color(k), label=series_label(k)) for k in sorted_keys if k in present_keys]
    handles.append(_orig_legend_handle())
    fig.legend(
        handles=handles, loc="outside upper center", ncol=min(3, len(handles)),
        title=title, title_fontsize=12 * FONT_SCALE,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_perturbation(pool_data: Dict[str, Dict[plot_data.SeriesKey, Tuple[Bar, Optional[Bar]]]], outdir: Path) -> Path:
    return _plot_perturbation_bars(
        pool_data, outdir, "perturbation_libero10plus.pdf",
        "LIBERO-10-Plus vs. init-state-matched LIBERO original",
        note=(
            "orig_n is deduplicated by (modality combo, base task) -- ≤10 for LIBERO-10, so the\n"
            "hatched bars carry wide CIs by construction, not by chance. The p=.../n=... label\n"
            "below each bar is a task-level exact McNemar test against that same baseline (see\n"
            "severity_sr.paired_baseline) -- n is the number of base tasks it's built from, so\n"
            "the smallest reachable two-sided p is ~0.002 and most bars will show p near 1."
        ),
    )


def plot_perturbation_clear(pool_data: Dict[str, Dict[plot_data.SeriesKey, Tuple[Bar, Optional[Bar]]]], outdir: Path) -> Path:
    """The 'clear' (presentation-grade, no McNemar) perturbation figure: only
    keep_p=1.0 vs. the no-dropout(+proprio) baseline, across perturbation categories."""
    return _plot_perturbation_bars(
        pool_data, outdir, "perturbation_libero10plus_dropout_vs_nodropout.pdf",
        "LIBERO-10-Plus: dropout training (keep_p=1) vs. no dropout, vs. init-state-matched LIBERO original",
        keys=_CLEAR_PERTURBATION_KEYS, mcnemar=False,
        note="orig_n is deduplicated by (modality combo, base task) -- ≤10 for LIBERO-10, so the\nhatched bars carry wide CIs by construction, not by chance.",
    )


# ---------------------------------------------------------------------------
# Plot 3 -- all-modality vs. 1-left-out, each bar paired with its init-state-matched
# LIBERO original baseline. Both axes (severity, difficulty_level) get both a
# per-category breakdown and a pooled ("totals") view -- four PDFs.
# ---------------------------------------------------------------------------


def _present_configs(
    pairs_by_key: Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]], configs: Optional[List[str]] = None
) -> List[str]:
    """Which of `configs` (default plot_data.MODALITY_CONFIGS) have a plus_bar
    somewhere in a pooled (non-faceted) figure's data -- a config with none is dropped
    from the legend, same rationale as _draw_grouped_bars/_draw_paired_grouped_bars
    silently skipping an empty series."""
    configs = configs if configs is not None else plot_data.MODALITY_CONFIGS
    return [cfg for cfg in configs if any(cfg in present for present in pairs_by_key.values())]


def _present_configs_faceted(
    by_category: Dict[str, Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]]], configs: Optional[List[str]] = None
) -> List[str]:
    configs = configs if configs is not None else plot_data.MODALITY_CONFIGS
    return [
        cfg for cfg in configs
        if any(cfg in present for bins in by_category.values() for present in bins.values())
    ]


def _slug(text: str) -> str:
    """'Camera Viewpoints' -> 'camera_viewpoints', '(unclassified)' -> 'unclassified' --
    a filesystem-safe filename fragment for a per-category "clear" PDF."""
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def _draw_category_axes(
    ax, category: str, bins: List[str], by_category: Dict[str, Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]]],
    configs: List[str], mcnemar: bool, thin_dense_ticks: bool, robot_initstate_note: bool,
    colors: Optional[Dict[str, str]] = None,
) -> None:
    """One category's own bars, drawn onto `ax` -- the single-category body shared by
    _plot_category_facets (one subplot per category, stacked into one figure) and
    _plot_category_separate (one standalone figure per category). `colors` defaults to
    MODALITY_COLORS -- overridden by a clear figure's _clear_series_style, so its single
    "all" bar takes its training config's own color instead of the neutral gray."""
    colors = colors if colors is not None else MODALITY_COLORS
    series = {
        cfg: [by_category[category][bin_label].get(cfg, (None, None)) for bin_label in bins]
        for cfg in configs
    }
    _draw_paired_grouped_bars(ax, bins, series, colors, mcnemar=mcnemar)
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
        _draw_category_axes(
            ax, category, bins, by_category, plot_data.MODALITY_CONFIGS, True, thin_dense_ticks, robot_initstate_note,
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


def _plot_category_separate(
    by_category: Dict[str, Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]]],
    outdir: Path,
    filename_prefix: str,
    title: str,
    configs: Optional[List[str]] = None,
    mcnemar: bool = True,
    category_order: Optional[List[str]] = None,
    robot_initstate_note: bool = False,
    thin_dense_ticks: bool = False,
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
) -> List[Path]:
    """One standalone single-axes PDF per category -- the "clear" counterpart to
    _plot_category_facets's stacked-subplot figure, so each perturbation category can
    be read/embedded on its own. filename is
    f"{filename_prefix}_{_slug(category)}.pdf". `colors`/`labels` default to
    MODALITY_COLORS/MODALITY_LABELS -- overridden by a clear figure's
    _clear_series_style so its single "all" bar/legend entry names its training config."""
    configs = configs if configs is not None else plot_data.MODALITY_CONFIGS
    colors = colors if colors is not None else MODALITY_COLORS
    labels = labels if labels is not None else MODALITY_LABELS
    categories = [c for c in category_order if c in by_category] if category_order else sorted(by_category)

    paths = []
    outdir.mkdir(parents=True, exist_ok=True)
    for category in categories:
        apply_style()
        bins = list(by_category[category])
        fig, ax = plt.subplots(figsize=(7 * FONT_SCALE, 5 * FONT_SCALE))
        _draw_category_axes(ax, category, bins, by_category, configs, mcnemar, thin_dense_ticks, robot_initstate_note, colors=colors)
        ax.set_title("")

        handles = [
            Patch(facecolor=colors[cfg], label=labels[cfg])
            for cfg in configs if any(cfg in by_category[category][b] for b in bins)
        ]
        handles.append(_orig_legend_handle())
        fig.legend(
            handles=handles, loc="outside upper center", ncol=min(3, len(handles)),
            title=f"{title} — {category}", title_fontsize=12 * FONT_SCALE,
        )

        path = outdir / f"{filename_prefix}_{_slug(category)}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_severity_percategory(by_category: Dict[str, Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]]], outdir: Path) -> Path:
    return _plot_category_facets(
        by_category, outdir, "severity_modality_off_percategory.pdf",
        "LIBERO-10-Plus by physical perturbation severity",
        category_order=SEVERITY_CATEGORY_ORDER, robot_initstate_note=True, thin_dense_ticks=True,
    )


def plot_severity_percategory_clear(
    by_category: Dict[str, Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]]], outdir: Path,
    series_key: plot_data.SeriesKey,
) -> List[Path]:
    """The 'clear' (presentation-grade, no McNemar) counterpart to
    plot_severity_percategory: one standalone PDF per category, all-modality only,
    colored/labelled by `series_key` (the pool's one training config -- see
    _require_single_series_key/_single_series_key) instead of the neutral gray."""
    colors, labels = _clear_series_style(series_key)
    return _plot_category_separate(
        by_category, outdir, "severity_all_percategory",
        "LIBERO-10-Plus vs. LIBERO original (all modalities)",
        configs=_CLEAR_CONFIGS, mcnemar=False, colors=colors, labels=labels,
        category_order=SEVERITY_CATEGORY_ORDER, robot_initstate_note=True, thin_dense_ticks=True,
    )


def plot_difficulty_percategory(by_category: Dict[str, Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]]], outdir: Path) -> Path:
    return _plot_category_facets(
        by_category, outdir, "difficulty_modality_off_percategory.pdf",
        "LIBERO-10-Plus by upstream difficulty_level, per category",
    )


def plot_difficulty_percategory_clear(
    by_category: Dict[str, Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]]], outdir: Path,
    series_key: plot_data.SeriesKey,
) -> List[Path]:
    """The 'clear' (presentation-grade, no McNemar) counterpart to
    plot_difficulty_percategory: one standalone PDF per category, all-modality only,
    colored/labelled by `series_key` instead of the neutral gray."""
    colors, labels = _clear_series_style(series_key)
    return _plot_category_separate(
        by_category, outdir, "difficulty_all_percategory",
        "LIBERO-10-Plus vs. LIBERO original, by upstream difficulty_level (all modalities)",
        configs=_CLEAR_CONFIGS, mcnemar=False, colors=colors, labels=labels,
    )


def _plot_pooled_bars(
    pooled: Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]],
    outdir: Path,
    filename: str,
    title: str,
    xlabel: str,
    sort_key=None,
    rotate: bool = False,
    configs: Optional[List[str]] = None,
    mcnemar: bool = True,
    colors: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
) -> Path:
    """One axes, x = pooled's own keys (categories, or difficulty levels), one
    filled+hollow-hatched pair per modality config -- shared by plot_severity_pooled
    (pools each category's own severity bins) and plot_difficulty_pooled (pools across
    categories on the shared difficulty_level scale). `colors`/`labels` default to
    MODALITY_COLORS/MODALITY_LABELS -- overridden by a clear figure's
    _clear_series_style so its single "all" bar/legend entry names its training config."""
    apply_style()
    configs = configs if configs is not None else plot_data.MODALITY_CONFIGS
    colors = colors if colors is not None else MODALITY_COLORS
    labels = labels if labels is not None else MODALITY_LABELS
    keys = sorted(pooled, key=sort_key) if sort_key else sorted(pooled)
    series = {cfg: [pooled[k].get(cfg, (None, None)) for k in keys] for cfg in configs}

    fig, ax = plt.subplots(figsize=(7 * FONT_SCALE, 5 * FONT_SCALE))
    _draw_paired_grouped_bars(ax, keys, series, colors, mcnemar=mcnemar)
    ax.set_xlabel(xlabel)
    if rotate:
        ax.set_xticklabels(keys, rotation=25, ha="right")
    if mcnemar:
        ax.annotate(
            "p=.../n=... below a bar is McNemar's test vs. init-state-matched LIBERO original --\n"
            "pooled across bins/categories (plot_data.merge_bars), so it reuses the same base\n"
            "tasks' pairs several times over and is anti-conservative (n overstates independent\n"
            "evidence); see the un-pooled per-category/per-bin figure for the honest test.",
            xy=(0.0, -0.55), xycoords="axes fraction", ha="left", va="top", fontsize=8 * FONT_SCALE * 0.55, color="#666666",
        )

    handles = [Patch(facecolor=colors[cfg], label=labels[cfg]) for cfg in _present_configs(pooled, configs)]
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


def plot_severity_pooled_clear(
    pooled: Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]], outdir: Path, series_key: plot_data.SeriesKey,
) -> Path:
    """The 'clear' (presentation-grade, no McNemar) counterpart to plot_severity_pooled:
    only the all-modality series, across perturbation categories, colored/labelled by
    `series_key` instead of the neutral gray."""
    colors, labels = _clear_series_style(series_key)
    return _plot_pooled_bars(
        pooled, outdir, "severity_all_pooled.pdf",
        "LIBERO-10-Plus vs. LIBERO original, by perturbation category (all modalities)",
        "perturbation category", rotate=True, configs=_CLEAR_CONFIGS, mcnemar=False, colors=colors, labels=labels,
    )


def plot_difficulty_pooled(pooled: Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]], outdir: Path) -> Path:
    return _plot_pooled_bars(
        pooled, outdir, "difficulty_modality_off_pooled.pdf",
        "LIBERO-10-Plus by upstream difficulty_level (pooled across categories)",
        "difficulty_level (upstream annotation, pooled across categories)",
        sort_key=analyze_wandb._natural_key,
    )


def plot_difficulty_pooled_clear(
    pooled: Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]], outdir: Path, series_key: plot_data.SeriesKey,
) -> Path:
    """The 'clear' (presentation-grade, no McNemar) counterpart to plot_difficulty_pooled:
    only the all-modality series, across difficulty_level, colored/labelled by
    `series_key` instead of the neutral gray."""
    colors, labels = _clear_series_style(series_key)
    return _plot_pooled_bars(
        pooled, outdir, "difficulty_all_pooled.pdf",
        "LIBERO-10-Plus vs. LIBERO original, by upstream difficulty_level (all modalities)",
        "difficulty_level (upstream annotation, pooled across categories)",
        sort_key=analyze_wandb._natural_key, configs=_CLEAR_CONFIGS, mcnemar=False, colors=colors, labels=labels,
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


def _prepare_severity_axes(pool):
    """(severity_by_category, difficulty_by_category) -- prepared once and shared by
    every caller that needs both axes (_severity_plots, _clear_plots, all_plots), so
    plot_data.prepare_severity (and its warn_multi_run stderr lines) never runs twice
    for the same axis in one invocation."""
    return plot_data.prepare_severity(pool, axis="severity"), plot_data.prepare_severity(pool, axis="difficulty_level")


def _severity_plots(severity_by_category, difficulty_by_category, outdir: Path) -> List[Path]:
    """The 4 PDFs shared by the `severity` subcommand and `all`: both axes (physical
    severity, upstream difficulty_level) get both a per-category breakdown and a
    pooled ("totals") view, each bar paired with its init-state-matched LIBERO
    original baseline."""
    return [
        plot_severity_percategory(severity_by_category, outdir),
        plot_severity_pooled(plot_data.prepare_severity_pooled(severity_by_category), outdir),
        plot_difficulty_percategory(difficulty_by_category, outdir),
        plot_difficulty_pooled(plot_data.prepare_difficulty_pooled(difficulty_by_category), outdir),
    ]


def _clear_plots(
    perturbation_data, severity_by_category, difficulty_by_category, outdir: Path,
    series_key: Optional[plot_data.SeriesKey],
) -> List[Path]:
    """The presentation-grade, McNemar-free "clear" figures -- perturbation restricted
    to keep_p=1 vs. no-dropout(+proprio) (always drawn, exempt from the single-config
    requirement below), severity/difficulty restricted to the all-modality series,
    per-category figures split one-PDF-per-category. Written into outdir/clear so the
    audit-grade figures above are untouched.

    severity/difficulty come from plot_data.prepare_severity, which pools every run in
    the underlying pool into one bar regardless of its training config -- so they only
    mean something for a single-config pool. `series_key` is that config
    (_single_series_key/_require_single_series_key); when None (the pool spans more
    than one config), those figures are skipped with a stderr WARNING instead of
    silently averaging across trained models."""
    clear_outdir = outdir / "clear"
    paths = [plot_perturbation_clear(perturbation_data, clear_outdir)]
    if series_key is None:
        print(
            "WARNING: pool spans more than one training config -- skipping the clear "
            "severity/difficulty figures (they'd otherwise pool every config into one "
            "bar); narrow --filters to one config to get them.",
            file=sys.stderr,
        )
        return paths
    paths.append(plot_severity_pooled_clear(plot_data.prepare_severity_pooled(severity_by_category), clear_outdir, series_key))
    paths.append(plot_difficulty_pooled_clear(plot_data.prepare_difficulty_pooled(difficulty_by_category), clear_outdir, series_key))
    paths += plot_severity_percategory_clear(severity_by_category, clear_outdir, series_key)
    paths += plot_difficulty_percategory_clear(difficulty_by_category, clear_outdir, series_key)
    return paths


def severity(cfg: tyro.conf.OmitArgPrefixes[PoolConfig]) -> None:
    """Plot 3: all-modality vs. 1-left-out, each bar paired with its init-state-matched
    LIBERO original baseline. Both physical severity and upstream difficulty_level get
    a per-category breakdown and a pooled ("totals") view -- 4 PDFs."""
    pool = _fetch(cfg)
    severity_by_category, difficulty_by_category = _prepare_severity_axes(pool)
    for path in _severity_plots(severity_by_category, difficulty_by_category, cfg.outdir):
        print(f"wrote {path}")


def clear(cfg: tyro.conf.OmitArgPrefixes[PoolConfig]) -> None:
    """Presentation-grade figures with no McNemar labels/caveats: perturbation
    restricted to keep_p=1 vs. no dropout, severity/difficulty restricted to the
    all-modality series with one PDF per category -- written into outdir/clear.

    The severity/difficulty figures pool every run in --filters into one bar, so
    --filters must select exactly one training config (one
    config.modality_dropout_proprio_keep_p value, or config.modality_dropout=false plus
    config.use_proprio) -- raises otherwise, rather than silently averaging across
    trained models."""
    pool = _fetch(cfg)
    series_key = _require_single_series_key(pool)
    severity_by_category, difficulty_by_category = _prepare_severity_axes(pool)
    perturbation_data = plot_data.prepare_perturbation(pool)
    for path in _clear_plots(perturbation_data, severity_by_category, difficulty_by_category, cfg.outdir, series_key):
        print(f"wrote {path}")


def all_plots(cfg: tyro.conf.OmitArgPrefixes[PoolConfig]) -> None:
    """Every plot, sharing a single fetch of the run pool. Unlike `clear`, a --filters
    pool spanning more than one training config doesn't fail this subcommand -- it just
    skips the clear severity/difficulty figures (with a stderr WARNING) while every
    other figure, including the clear perturbation one, is still written."""
    pool = _fetch(cfg)
    print(f"wrote {plot_presence(plot_data.prepare_presence(pool), cfg.outdir)}")
    perturbation_data = plot_data.prepare_perturbation(pool)
    print(f"wrote {plot_perturbation(perturbation_data, cfg.outdir)}")
    severity_by_category, difficulty_by_category = _prepare_severity_axes(pool)
    for path in _severity_plots(severity_by_category, difficulty_by_category, cfg.outdir):
        print(f"wrote {path}")
    series_key = _single_series_key(pool)
    for path in _clear_plots(perturbation_data, severity_by_category, difficulty_by_category, cfg.outdir, series_key):
        print(f"wrote {path}")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {"presence": presence, "perturbation": perturbation, "severity": severity, "clear": clear, "all": all_plots}
    )
