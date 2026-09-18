#!/usr/bin/env python3
"""Gather / filter / prepare evaluation data for plot_eval.py, from W&B.

Pure data layer -- no matplotlib import, no plotting. Reuses analyze_wandb.py's
fetching and analysis rather than re-reading W&B or LIBERO-Plus CSVs: fetch_pool()
wraps wandb.Api() + analyze_wandb.analyze() into a persistent-cache version of the
same run-pool analyze_wandb.py's `filter` mode builds, keyed by
modality_dropout_proprio_keep_p instead of printed straight to stdout. The prepare_*()
functions turn that pool into exactly the (category/bin/modality-config) -> Bar (or
(plus_bar, orig_bar) pair) shapes plot_eval.py's figures need.

Every LIBERO-Plus bar (prepare_perturbation, prepare_severity, and the two pooled
variants) is paired with an orig_bar: the init-state-matched LIBERO original baseline
severity_sr.collect() already computes for that same group, matched on the row's OWN
modality combo -- so a withheld-modality bar's orig_bar reflects clean LIBERO
performance with that SAME modality ALSO withheld, not full-modality clean LIBERO.
orig_bar is None where no run's record carries one.

Every Bar's rate/CI is a 95% Wilson interval (severity_sr.wilson_interval) over
successes/n POOLED ACROSS every contributing run -- a macro average like
analyze_wandb.aggregate_scalars would hide how many episodes actually back a bar.
Pooling across more than one run is legitimate (more episodes, tighter CI) but also
means the bar no longer describes a single trained model; pool_bars() records which
runs contributed, and warn_multi_run() (called by every prepare_*() before returning)
prints one stderr line naming them whenever a bar pools more than one -- so that
pooling never happens silently.
"""
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import wandb

import analyze_wandb
import eval_pipeline
import perturbation_sr
import pid_modality
import severity_sr

# The five inference-time modality configurations plot 1 and plot 3 compare, in
# canonical (all-on, then wrist/static/proprio/lang -- eval_pipeline.MODALITY_OFF_VARIANTS'
# order) order.
MODALITY_CONFIGS = ["all", "no_wrist", "no_static", "no_proprio", "no_lang"]

# config name -> eval_pipeline.MODALITY_OFF_VARIANTS key, for the 4 withheld configs.
_CONFIG_TO_VARIANT = {
    variant.split("plus_", 1)[1]: variant for variant in eval_pipeline.MODALITY_OFF_VARIANTS
}


@dataclass
class Bar:
    """One bar's worth of pooled evaluation data: `runs` names every run.id that
    contributed at least one episode, in the order they were pooled -- `len(runs) > 1`
    is exactly the condition warn_multi_run() warns about."""

    label: str
    successes: int
    n: int
    rate: float
    ci_low: float
    ci_high: float
    runs: List[str] = field(default_factory=list)


def pool_bars(parts: List[Tuple[str, int, int]], label: str) -> Bar:
    """Bar from (run_id, successes, n) parts, pooling successes/n across parts before
    computing the Wilson interval (not averaging each part's own rate). Degenerates to
    exactly severity_sr's own numbers when one part contributes. `n == 0` (no part
    contributed, e.g. a config no run in the pool was evaluated under) yields a bar
    with rate/ci_low/ci_high == nan, same as severity_sr._record's empty-group case."""
    total_successes = sum(s for _rid, s, _n in parts)
    total_n = sum(n for _rid, _s, n in parts)
    rate = total_successes / total_n if total_n else float("nan")
    ci_low, ci_high = severity_sr.wilson_interval(total_successes, total_n)
    runs = [rid for rid, _s, n in parts if n > 0]
    return Bar(label=label, successes=total_successes, n=total_n, rate=rate, ci_low=ci_low, ci_high=ci_high, runs=runs)


def warn_multi_run(bars: Iterable[Bar]) -> None:
    """One stderr line per bar pooling more than one run -- naming them, so pooling
    across seeds/checkpoints never passes silently into a figure."""
    for bar in bars:
        if len(bar.runs) > 1:
            print(
                f"WARNING: bar {bar.label!r} pools {len(bar.runs)} runs ({', '.join(bar.runs)}) -- "
                "its Wilson CI reflects episodes pooled across runs, not one trained model.",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def _cached_or_download(run, cache_dir: Path, applicable: List[str], refresh: bool) -> Tuple[Optional[Path], List[str]]:
    """Skip analyze_wandb.download_run's network round trip when cache_dir/run.id
    already holds everything `applicable` (applicable_modality_off(run)) says this
    run's artifact should have -- an iterative plotting workflow would otherwise
    re-download every artifact on every invocation. --refresh forces the network path
    unconditionally."""
    run_dir = Path(cache_dir) / run.id
    if not refresh and run_dir.exists():
        missing = analyze_wandb.missing_members(run_dir, applicable)
        if not missing:
            return run_dir, missing
    return analyze_wandb.download_run(run, Path(cache_dir))


def _effective_keep_p(run) -> Optional[float]:
    """The modality_dropout_proprio_keep_p this run belongs on the presence/
    perturbation figures' x-axis under. use_proprio=False means the model never
    receives proprioception AT ALL -- there is nothing to keep or drop -- so its
    config's modality_dropout_proprio_keep_p (an inert FLOWERVLA.__init__ argument,
    typically just sitting at its 0.5 default) does not describe this run; the
    effective value is 0.0. A use_proprio=True run with no
    modality_dropout_proprio_keep_p in config at all returns None (can't be placed on
    the axis)."""
    if not bool(run.config.get("use_proprio", False)):
        return 0.0
    keep_p = run.config.get("modality_dropout_proprio_keep_p")
    return float(keep_p) if keep_p is not None else None


def fetch_pool(
    filters: str,
    modalities: Optional[str],
    entity: Optional[str],
    project: Optional[str],
    cache_dir: Path,
    measure: str,
    libero_plus_root: str,
    refresh: bool = False,
) -> List[Tuple[str, float, dict]]:
    """Every W&B run matching `filters` (a JSON mongo-style dict, same as
    analyze_wandb.py's --filters) and, if given, --modalities (same exact-combo
    semantics as analyze_wandb.parse_modality_spec), analyzed via analyze_wandb.analyze()
    and paired with its effective modality_dropout_proprio_keep_p (see
    _effective_keep_p: 0.0 for a use_proprio=False run, regardless of that config
    value) -- the single pool every plot in this module draws from. A use_proprio=True
    run whose config has no modality_dropout_proprio_keep_p is skipped (with a stderr
    warning): it can't be placed on any of this module's x-axes."""
    default_entity, default_project = analyze_wandb.default_entity_project()
    entity = entity or default_entity
    project = project or default_project

    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", filters=json.loads(filters)))

    if modalities is not None:
        wanted = analyze_wandb.parse_modality_spec(modalities)
        runs = [r for r in runs if analyze_wandb.run_modalities(r) == wanted]

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pool: List[Tuple[str, float, dict]] = []
    for run in runs:
        keep_p = _effective_keep_p(run)
        if keep_p is None:
            print(
                f"WARNING: {run.id} has no modality_dropout_proprio_keep_p in config -- skipped",
                file=sys.stderr,
            )
            continue

        applicable = analyze_wandb.applicable_modality_off(run)
        artifact_dir, missing = _cached_or_download(run, cache_dir, applicable, refresh)
        analysis = analyze_wandb.analyze(
            artifact_dir, missing, measure, libero_plus_root, modality_off_variants=applicable
        )
        pool.append((run.id, keep_p, analysis))

    return pool


# ---------------------------------------------------------------------------
# Prepare -- plot 1: clean LIBERO_10, all-4 vs. 1-modality-off, by keep_p
# ---------------------------------------------------------------------------


def _combo_for_config(config_name: str) -> Dict[str, bool]:
    if config_name == "all":
        return eval_pipeline.full_modality_combo()
    return eval_pipeline.modality_off_combo(_CONFIG_TO_VARIANT[config_name])


def _presence_key(combo: Dict[str, bool]) -> Tuple[int, int, int, int]:
    """combo (eval_pipeline's rgb_static/rgb_gripper/language/proprio keys) as the
    (static, wrist, lang, proprio) 0/1 tuple pid_modality.presence_success keys its
    result by (pid_modality.MODALITY_ORDER order)."""
    return (int(combo["rgb_static"]), int(combo["rgb_gripper"]), int(combo["language"]), int(combo["proprio"]))


def prepare_presence(pool: List[Tuple[str, float, dict]]) -> Dict[float, Dict[str, Bar]]:
    """keep_p -> modality_config -> Bar, from each run's libero_orig.csv (clean
    LIBERO_10). A dropout run's orig CSV already holds all 14 modality combos (7 token
    combos x proprio on/off, see eval_pipeline.py's module docstring), so every one of
    MODALITY_CONFIGS' 5 bars is read straight off it -- no separate clean-LIBERO
    modality-off eval is needed. Counts, not analyze_wandb.presence_values' bare rate,
    come from pid_modality.presence_success (successes = round(rate * n): rate is
    exactly successes/n, so this recovers the integer count without re-filtering rows)."""
    by_keep_p: Dict[float, List[Tuple[str, dict]]] = defaultdict(list)
    for run_id, keep_p, analysis in pool:
        by_keep_p[keep_p].append((run_id, analysis))

    result: Dict[float, Dict[str, Bar]] = {}
    for keep_p, entries in by_keep_p.items():
        bars: Dict[str, Bar] = {}
        for config_name in MODALITY_CONFIGS:
            key = _presence_key(_combo_for_config(config_name))
            parts = []
            for run_id, analysis in entries:
                if not analysis["orig_rows"]:
                    continue
                presence = pid_modality.presence_success(analysis["orig_rows"], pid_modality.MODALITY_ORDER)
                if key not in presence:
                    continue
                rate, n = presence[key]
                parts.append((run_id, round(rate * n), n))
            bars[config_name] = pool_bars(parts, config_name)
        result[keep_p] = bars

    warn_multi_run(bar for bars in result.values() for bar in bars.values())
    return result


# ---------------------------------------------------------------------------
# Prepare -- plot 2: LIBERO_10-Plus per perturbation category, by keep_p, paired
# with the init-state-matched LIBERO original baseline
# ---------------------------------------------------------------------------


def prepare_perturbation(pool: List[Tuple[str, float, dict]]) -> Dict[str, Dict[float, Tuple[Bar, Optional[Bar]]]]:
    """category -> keep_p -> (plus_bar, orig_bar). orig_bar is None where no run's
    severity breakdown carries an init-state-matched baseline for this category (the
    axis="total" record's orig_n == "", e.g. no libero_orig.csv was available at all).
    Reads the axis="total"/bin="ALL" record severity_sr.collect() emits once per
    category -- the category-level success rate, paired baseline included, without
    recomputing anything perturbation_sr/severity_sr don't already provide."""
    by_keep_p: Dict[float, List[Tuple[str, dict]]] = defaultdict(list)
    for run_id, keep_p, analysis in pool:
        by_keep_p[keep_p].append((run_id, analysis))

    categories = sorted(
        {
            r["category"]
            for _keep_p, entries in by_keep_p.items()
            for _run_id, analysis in entries
            if analysis["severity"]
            for r in analysis["severity"]
            if r["axis"] == "total"
        }
    )

    result: Dict[str, Dict[float, Tuple[Bar, Optional[Bar]]]] = defaultdict(dict)
    for category in categories:
        for keep_p, entries in by_keep_p.items():
            plus_parts, orig_parts = [], []
            for run_id, analysis in entries:
                if not analysis["severity"]:
                    continue
                rec = next(
                    (r for r in analysis["severity"] if r["axis"] == "total" and r["category"] == category), None
                )
                if rec is None:
                    continue
                plus_parts.append((run_id, rec["successes"], rec["n"]))
                if rec["orig_n"] != "":
                    orig_parts.append((run_id, rec["orig_successes"], rec["orig_n"]))
            plus_bar = pool_bars(plus_parts, category)
            orig_bar = pool_bars(orig_parts, f"{category} (orig)") if orig_parts else None
            result[category][keep_p] = (plus_bar, orig_bar)

    warn_multi_run(
        bar
        for keep_ps in result.values()
        for plus_bar, orig_bar in keep_ps.values()
        for bar in ([plus_bar] if orig_bar is None else [plus_bar, orig_bar])
    )
    return dict(result)


# ---------------------------------------------------------------------------
# Prepare -- plot 3: all-modality vs. 1-left-out, by severity / by difficulty_level
# ---------------------------------------------------------------------------


def _severity_records_for(analysis: dict, config_name: str) -> Optional[List[Dict]]:
    if config_name == "all":
        return analysis["severity"]
    variant = _CONFIG_TO_VARIANT[config_name]
    off = analysis.get("modality_off", {}).get(variant)
    return off["severity"] if off else None


def prepare_severity(
    pool: List[Tuple[str, float, dict]], axis: str
) -> Dict[str, Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]]]:
    """category -> bin -> modality_config -> (plus_bar, orig_bar), for `axis` in
    {"severity", "difficulty_level"}. Full-modality bars come from analysis["severity"];
    the 4 withheld-modality bars come from analysis["modality_off"][variant]["severity"].
    orig_bar is the init-state-matched LIBERO original baseline severity_sr.collect()
    computes for that SAME (category, axis, bin) group -- None where no run's record
    carries one (orig_n == "", e.g. no libero_orig.csv was available). Because
    severity_sr.paired_baseline matches on the row's OWN modality combo, a withheld-
    modality bar's orig_bar reflects clean LIBERO performance WITH THAT SAME MODALITY
    ALSO WITHHELD, not full-modality clean LIBERO -- the "orig bar reflects the initial
    states the bar next to it entails" pairing, same mechanism as
    prepare_perturbation's orig_bar. Bins are natural-sorted (analyze_wandb._natural_key)
    within each category, so e.g. fog_2 precedes fog_10.

    axis="severity" needs no category filtering here: severity_sr.collect() already
    only emits axis="severity" records for categories with a physical severity axis --
    Language Instructions, Background Textures and perturbation_sr.UNCLASSIFIED never
    appear (see severity_sr.py's UNORDERED_NOTE gate). axis="difficulty_level" keeps
    every category, including those three."""
    if axis not in ("severity", "difficulty_level"):
        raise ValueError(f"axis must be 'severity' or 'difficulty_level', got {axis!r}")

    keys = set()
    for _run_id, _keep_p, analysis in pool:
        for config_name in MODALITY_CONFIGS:
            records = _severity_records_for(analysis, config_name)
            if not records:
                continue
            for r in records:
                if r["axis"] == axis:
                    keys.add((r["category"], r["bin"]))

    by_category: Dict[str, List[str]] = defaultdict(list)
    for category, bin_label in keys:
        by_category[category].append(bin_label)
    for category in by_category:
        by_category[category].sort(key=analyze_wandb._natural_key)

    result: Dict[str, Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]]] = {
        category: {bin_label: {} for bin_label in bins} for category, bins in by_category.items()
    }
    for category, bins in by_category.items():
        for bin_label in bins:
            for config_name in MODALITY_CONFIGS:
                plus_parts, orig_parts = [], []
                for run_id, _keep_p, analysis in pool:
                    records = _severity_records_for(analysis, config_name)
                    if not records:
                        continue
                    rec = next(
                        (r for r in records if r["axis"] == axis and r["category"] == category and r["bin"] == bin_label),
                        None,
                    )
                    if rec is None:
                        continue
                    plus_parts.append((run_id, rec["successes"], rec["n"]))
                    if rec["orig_n"] != "":
                        orig_parts.append((run_id, rec["orig_successes"], rec["orig_n"]))
                if not plus_parts:
                    continue
                plus_bar = pool_bars(plus_parts, config_name)
                orig_bar = pool_bars(orig_parts, f"{config_name} (orig)") if orig_parts else None
                result[category][bin_label][config_name] = (plus_bar, orig_bar)

    warn_multi_run(
        bar
        for bins in result.values()
        for configs in bins.values()
        for pair in configs.values()
        for bar in pair
        if bar is not None
    )
    return result


def merge_bars(bars: List[Bar], label: str) -> Bar:
    """Combine several already-pooled Bars into one -- e.g. prepare_severity()'s
    per-category or per-bin bars, further pooled by prepare_severity_pooled()/
    prepare_difficulty_pooled() below. Sums each input Bar's own successes/n directly
    (not its rate) and unions their `runs` lists, so a run that already contributed to
    two categories is only listed once."""
    total_successes = sum(b.successes for b in bars)
    total_n = sum(b.n for b in bars)
    rate = total_successes / total_n if total_n else float("nan")
    ci_low, ci_high = severity_sr.wilson_interval(total_successes, total_n)
    runs: List[str] = []
    for b in bars:
        for run_id in b.runs:
            if run_id not in runs:
                runs.append(run_id)
    return Bar(label=label, successes=total_successes, n=total_n, rate=rate, ci_low=ci_low, ci_high=ci_high, runs=runs)


def _merge_pairs(pairs: List[Tuple[Bar, Optional[Bar]]], label: str) -> Tuple[Bar, Optional[Bar]]:
    """(plus_bar, orig_bar) merged across several (plus, orig) pairs -- shared by
    prepare_severity_pooled (collapses a category's own bins) and
    prepare_difficulty_pooled (collapses across categories)."""
    plus_bars = [p for p, _o in pairs]
    orig_bars = [o for _p, o in pairs if o is not None]
    plus_merged = merge_bars(plus_bars, label)
    orig_merged = merge_bars(orig_bars, f"{label} (orig)") if orig_bars else None
    return plus_merged, orig_merged


def prepare_severity_pooled(
    by_category: Dict[str, Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]]]
) -> Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]]:
    """category -> modality_config -> (plus_bar, orig_bar), pooling prepare_severity()'s
    per-bin pairs ACROSS A CATEGORY'S OWN BINS -- the "one bar (pair) per category"
    counterpart to prepare_severity's per-bin breakdown, analogous to
    prepare_perturbation's axis="total" bars but split by modality config instead of
    keep_p. See prepare_difficulty_pooled for the complementary collapse (across
    categories, keeping bins/levels apart)."""
    result: Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]] = {}
    for category, bins in by_category.items():
        by_config: Dict[str, List[Tuple[Bar, Optional[Bar]]]] = defaultdict(list)
        for configs in bins.values():
            for config_name, pair in configs.items():
                by_config[config_name].append(pair)
        result[category] = {config_name: _merge_pairs(pairs, config_name) for config_name, pairs in by_config.items()}

    warn_multi_run(bar for configs in result.values() for pair in configs.values() for bar in pair if bar is not None)
    return result


def prepare_difficulty_pooled(
    by_category: Dict[str, Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]]]
) -> Dict[str, Dict[str, Tuple[Bar, Optional[Bar]]]]:
    """difficulty_level -> modality_config -> (plus_bar, orig_bar), pooling
    prepare_severity(pool, axis="difficulty_level")'s per-category pairs ACROSS EVERY
    CATEGORY -- unlike the physical severity axis, difficulty_level is an upstream
    per-task annotation with the same 1-5 scale regardless of perturbation category, so
    this answers "how does success rate vary with difficulty, regardless of
    perturbation type" rather than the necessarily per-category severity breakdown."""
    by_bin: Dict[str, Dict[str, List[Tuple[Bar, Optional[Bar]]]]] = defaultdict(lambda: defaultdict(list))
    for bins in by_category.values():
        for bin_label, configs in bins.items():
            for config_name, pair in configs.items():
                by_bin[bin_label][config_name].append(pair)

    result = {
        bin_label: {config_name: _merge_pairs(pairs, config_name) for config_name, pairs in configs.items()}
        for bin_label, configs in sorted(by_bin.items(), key=lambda kv: analyze_wandb._natural_key(kv[0]))
    }
    warn_multi_run(bar for configs in result.values() for pair in configs.values() for bar in pair if bar is not None)
    return result
