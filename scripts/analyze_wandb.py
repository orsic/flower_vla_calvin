#!/usr/bin/env python3
"""Cross-run evaluation analysis pulled from W&B artifacts.

Reads the `evaluation`-type W&B artifact scripts/eval_pipeline.py's `upload` attaches
to a training run (libero_orig.csv, libero_plus.csv -- see that script's module
docstring), and reports:
  - the PID modality decomposition (scripts/pid_modality.py) over libero_orig.csv
  - the per-perturbation-category success rate (scripts/perturbation_sr.py) over
    libero_plus.csv
  - the success rate by physical perturbation severity and by upstream
    difficulty_level (scripts/severity_sr.py) over libero_plus.csv -- recomputed from
    libero_plus.csv rather than read from the artifact's own severity_sr.csv member
    (which may not exist for an artifact uploaded before this recomputation existed),
    same as the PID/perturbation reports above

Two modes:
  run     One or more W&B run IDs -- prints each run's two reports individually.
  filter  A W&B MongoDB-style filter over run config -- fetches every matching run
          and reports mean/min/max (+ how many runs contributed) per table cell.

W&B run config keys are FLAT (e.g. "modality_dropout"), not "model.modality_dropout" --
the only source of the training run's W&B config is FLOWERVLA.save_hyperparameters()
(flower/models/flower.py), which records its __init__ argument names as-is.

Usage:
  python scripts/analyze_wandb.py run <run_id> [<run_id>...] [--entity E] [--project P]
      [--measure ccs|broja|wb] [--config-keys k1,k2|all]

  python scripts/analyze_wandb.py filter [--filters '<mongo-json>'] [--modalities m1,m2,...]
      [--entity E] [--project P] [--measure ccs|broja|wb] [--config-keys k1,k2|all]

  Example:
    python scripts/analyze_wandb.py filter --filters \\
        '{"config.modality_dropout": true, "config.modality_dropout_proprio_keep_p": 0.5}'

    python scripts/analyze_wandb.py filter --modalities rgb_static,rgb_gripper,language,proprio

--modalities selects runs client-side by the *trained* modality set (every named modality
on, every other one off) -- W&B stores `modalities` as an unindexable repr string, so a
--filters entry can't express this (see run_modalities()'s docstring).

--entity/--project default to conf/config_libero.yaml's logger.entity/logger.project.
"""
import argparse
import ast
import itertools
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import wandb
from omegaconf import OmegaConf

import pid_modality
import perturbation_sr
import severity_sr
import eval_pipeline
from compare_eval_csvs import MODALITY_COLUMNS

REQUIRED_MEMBERS = ["libero_orig.csv", "libero_plus.csv"]
DEFAULT_CONFIG_KEYS = [
    "modality_dropout",
    "modality_dropout_keep_fraction",
    "modality_dropout_alphas",
    "modality_dropout_proprio_keep_p",
    "use_proprio",
    "modalities",
]


def default_entity_project() -> Tuple[str, str]:
    cfg = OmegaConf.load(Path(__file__).parents[1] / "conf" / "config_libero.yaml")
    return str(cfg.logger.entity), str(cfg.logger.project)


# ---------------------------------------------------------------------------
# Filtering by trained modality set
# ---------------------------------------------------------------------------


def run_modalities(run) -> Optional[Dict[str, bool]]:
    """The modality set this run's model was trained to observe, keyed like
    eval_pipeline.FLAG_COLUMNS (rgb_static, rgb_gripper, language, proprio), or None if
    the run's config can't be parsed.

    Mirrors FLOWERVLA.__init__ (flower/models/flower.py): a modality missing from
    `modalities` defaults to True, and proprio is additionally gated on use_proprio -- a
    model with use_proprio=False never receives proprio regardless of what `modalities`
    says. W&B stores `modalities` as a Python repr string (a DictConfig passed through
    str() on its way into the W&B config via FLOWERVLA.save_hyperparameters()), not a
    nested value, so this parses it rather than indexing into it -- the reason
    --filters 'config.modalities.rgb_static=...' can't work.
    """
    raw = run.config.get("modalities")
    if raw is None:
        raw = {}
    elif isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(raw, dict):
        return None
    modalities = {m: bool(raw.get(m, True)) for m in eval_pipeline.TOKEN_MODALITIES}
    modalities["proprio"] = bool(raw.get("proprio", True)) and bool(run.config.get("use_proprio", False))
    return modalities


def parse_modality_spec(spec: str) -> Dict[str, bool]:
    """Comma-separated modality names that must be ON; every other modality is required
    OFF -- "matches this exact combo" semantics, same as
    compare_eval_csvs.parse_modalities."""
    on = {m.strip() for m in spec.split(",") if m.strip()}
    unknown = on - set(eval_pipeline.FLAG_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown modalities {unknown}; choose from {sorted(eval_pipeline.FLAG_COLUMNS)}")
    return {m: (m in on) for m in eval_pipeline.FLAG_COLUMNS}


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def eval_artifact(run) -> Optional[Any]:
    """The most recently created `type=="evaluation"` artifact logged by `run`, or
    None if it has none (e.g. a run that was never through ./run.sh pipeline)."""
    candidates = [a for a in run.logged_artifacts() if a.type == "evaluation"]
    if not candidates:
        return None
    return max(candidates, key=lambda a: a.created_at)


def download_run(run, dest: Path) -> Tuple[Optional[Path], List[str]]:
    """Download run's evaluation artifact into dest/<run.id>/.

    Returns (artifact_dir, missing_members). artifact_dir is None if the run has no
    evaluation artifact at all, in which case missing_members == ["<no evaluation
    artifact>"]; otherwise missing_members lists which of REQUIRED_MEMBERS the
    artifact lacks (e.g. a dropout-only run's artifact may lack libero_plus.csv if
    eval-plus hadn't finished yet).
    """
    artifact = eval_artifact(run)
    if artifact is None:
        return None, ["<no evaluation artifact>"]
    artifact_dir = Path(artifact.download(root=str(dest / run.id)))
    missing = [m for m in REQUIRED_MEMBERS if not (artifact_dir / m).exists()]
    return artifact_dir, missing


def analyze(
    artifact_dir: Optional[Path],
    missing: List[str],
    measure: str,
    libero_plus_root: str = severity_sr.DEFAULT_LIBERO_PLUS_ROOT,
) -> dict:
    """{"orig_rows", "plus_rows", "pid", "perturbation", "severity", "missing"}.
    "pid"/"perturbation"/"severity" are None when the corresponding CSV wasn't in the
    artifact. severity_sr.collect() reads scene XMLs for Light Conditions
    (perturbation_severity.light_severity) -- if the LIBERO-Plus assets aren't checked
    out locally it raises, which is reported via `missing` rather than aborting the
    whole run's analysis."""
    orig_rows: List[Dict] = []
    plus_rows: List[Dict] = []
    pid_collected = None
    perturbation = None
    severity = None
    missing = list(missing)

    if artifact_dir is not None:
        orig_path = artifact_dir / "libero_orig.csv"
        plus_path = artifact_dir / "libero_plus.csv"
        if orig_path.exists():
            orig_rows = pid_modality.load_rows(str(orig_path))
            pid_collected = pid_modality.collect(orig_rows, measure=measure)
        if plus_path.exists():
            plus_rows = pid_modality.load_rows(str(plus_path))
            perturbation = perturbation_sr.category_sr(plus_rows)
            try:
                severity = severity_sr.collect(plus_rows, libero_plus_root)
            except Exception as exc:  # noqa: BLE001 -- see docstring
                missing.append(f"severity breakdown ({exc})")

    return {
        "orig_rows": orig_rows,
        "plus_rows": plus_rows,
        "pid": pid_collected,
        "perturbation": perturbation,
        "severity": severity,
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# Cross-run consistency -- catch runs that silently don't share the same
# evaluated coverage before averaging over them.
# ---------------------------------------------------------------------------


def combo_set(rows: List[Dict]) -> frozenset:
    return frozenset(perturbation_sr.modality_combos_present(rows)) if rows else frozenset()


def episode_key_set(rows: List[Dict]) -> frozenset:
    cols = [MODALITY_COLUMNS[m] for m in ("static", "wrist", "lang", "proprio")]
    return frozenset(
        (
            row.get("suite", ""),
            row.get("task_idx", ""),
            row.get("episode_idx", ""),
            tuple(int(row.get(col) or 0) for col in cols),
        )
        for row in rows
    )


def category_count_set(rows: List[Dict]) -> frozenset:
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.get("task_category") or perturbation_sr.UNCLASSIFIED] += 1
    return frozenset(counts.items())


def _check(per_run: Dict[str, dict], label: str, extract) -> List[str]:
    """Compare `extract(analysis)` across runs; runs whose value differs from the
    majority (the most common value) get one warning line each, with up to 5 example
    differences. Runs for which extract() is empty (no data on this axis, e.g. a run
    missing libero_plus.csv) are skipped -- that's already reported via "missing"."""
    values = {run_id: extract(analysis) for run_id, analysis in per_run.items()}
    values = {run_id: v for run_id, v in values.items() if v}
    if len(values) < 2:
        return []

    counts: Dict[frozenset, int] = defaultdict(int)
    for v in values.values():
        counts[v] += 1
    reference = max(counts, key=counts.get)

    lines = []
    for run_id, value in sorted(values.items()):
        if value == reference:
            continue
        extra = sorted(value - reference, key=str)[:5]
        missing = sorted(reference - value, key=str)[:5]
        detail = []
        if extra:
            detail.append(f"+{extra}")
        if missing:
            detail.append(f"-{missing}")
        lines.append(f"{run_id}: {label} mismatch vs. the majority ({', '.join(detail)})")
    return lines


def severity_bin_set(records: Optional[List[Dict[str, Any]]]) -> frozenset:
    return frozenset((r["category"], r["axis"], r["bin"]) for r in records) if records else frozenset()


def consistency_report(per_run: Dict[str, dict]) -> List[str]:
    lines = []
    lines += _check(per_run, "modality-combo coverage", lambda a: combo_set(a["orig_rows"]))
    lines += _check(per_run, "episode coverage", lambda a: episode_key_set(a["orig_rows"]))
    lines += _check(per_run, "perturbation-category coverage", lambda a: category_count_set(a["plus_rows"]))
    lines += _check(per_run, "severity-bin coverage", lambda a: severity_bin_set(a["severity"]))
    return lines


def print_problems(per_run: Dict[str, dict]) -> List[str]:
    lines = []
    for run_id, analysis in sorted(per_run.items()):
        for m in analysis["missing"]:
            lines.append(f"{run_id}: missing {m}")
    lines += consistency_report(per_run)
    if lines:
        print("WARNING:")
        for line in lines:
            print(f"  {line}")
        print()
    return lines


# ---------------------------------------------------------------------------
# Rendering -- single run
# ---------------------------------------------------------------------------


def _config_value(run, key: str):
    return run.config.get(key, "<absent>")


def print_run_header(run, config_keys: List[str]) -> None:
    print(f"run: {run.id}  (name={run.name})")
    if config_keys == ["all"]:
        print(f"  config: {json.dumps(dict(run.config), sort_keys=True, default=str)}")
    else:
        for key in config_keys:
            print(f"  {key} = {_config_value(run, key)}")


def print_single(run_id: str, analysis: dict, libero_plus_root: str) -> None:
    print(f"=== {run_id}: LIBERO (orig) ===")
    if analysis["orig_rows"]:
        pid_modality.report(analysis["pid"])
    else:
        print("  (no libero_orig.csv)")
    print()
    print(f"=== {run_id}: LIBERO-Plus ===")
    if analysis["plus_rows"]:
        perturbation_sr.report(analysis["plus_rows"])
        print()
        if analysis["severity"] is not None:
            severity_sr.report(analysis["plus_rows"], libero_plus_root)
        else:
            print("  (severity breakdown unavailable -- see WARNING above)")
    else:
        print("  (no libero_plus.csv)")


# ---------------------------------------------------------------------------
# Rendering -- filter (aggregate)
# ---------------------------------------------------------------------------


def print_run_table(runs, config_keys: List[str]) -> None:
    if config_keys == ["all"]:
        for run in runs:
            print(f"  {run.id}: {json.dumps(dict(run.config), sort_keys=True, default=str)}")
        return
    # "key=value, ..." per run rather than fixed-width columns -- config values (e.g.
    # modality_dropout_alphas' list, modalities' dict) vary wildly in string length,
    # so a fixed column width either truncates or runs headers/values together.
    for run in runs:
        cfg_str = ", ".join(f"{k}={_config_value(run, k)}" for k in config_keys)
        print(f"  {run.id}  ({cfg_str})")


def pid_values(rows: List[Dict], measure: str) -> Dict[Tuple[str, str], float]:
    """(pair_component) -> value for every MODALITY_ORDER pair whose 4 (x1,x2) cells
    are all present in `rows` -- e.g. a non-dropout run's single modality combo
    yields none. Held modalities are always the *other* two of pid_modality's fixed
    MODALITY_ORDER (not a per-run "active" set), so keys line up across runs."""
    values: Dict[Tuple[str, str], float] = {}
    for m1, m2 in itertools.combinations(pid_modality.MODALITY_ORDER, 2):
        held = pid_modality.held_modalities(m1, m2)
        try:
            dist = pid_modality.build_joint(rows, m1, m2, held)
        except pid_modality.IncompleteCombos:
            continue
        pid = pid_modality.decompose(dist, measure)
        pair = f"{m1}+{m2}"
        for component in ("I", "R", "U1", "U2", "S"):
            values[(pair, component)] = pid[component]
    return values


def presence_values(rows: List[Dict]) -> Dict[Tuple[int, int, int, int], float]:
    return {combo: sr for combo, (sr, _n) in pid_modality.presence_success(rows, pid_modality.MODALITY_ORDER).items()}


def perturbation_values(rows: List[Dict]) -> Dict[str, float]:
    """Per-category success rate plus an "OVERALL" entry -- the same total-successes /
    total-n figure perturbation_sr.report() prints as its OVERALL line."""
    sr = perturbation_sr.category_sr(rows)
    values = {cat: s for cat, (s, _n) in sr.items()}
    total_n = sum(n for _s, n in sr.values())
    if total_n:
        values["OVERALL"] = sum(s * n for s, n in sr.values()) / total_n
    return values


def severity_values(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], float]:
    """(category, axis, bin) -> success_rate for every severity_sr.collect() record."""
    return {(r["category"], r["axis"], r["bin"]): r["success_rate"] for r in records}


_NUM_RE = re.compile(r"(\d+)")


def _natural_key(s: str) -> tuple:
    """Order digit runs numerically (fog_1 < fog_2 < fog_10) instead of lexically
    (plain sorted() would put fog_10 between fog_1 and fog_2) -- ordering is the
    content of a severity table."""
    return tuple(int(tok) if tok.isdigit() else tok for tok in _NUM_RE.split(s))


def aggregate_scalars(per_run_values: Dict[str, Dict[Any, float]]) -> Dict[Any, dict]:
    """run_id -> {key: value} for every run -> key -> {mean, min, max, runs}. `runs`
    is how many of the input runs actually contributed a value for that key -- the
    signal that a cell is backed by fewer runs than the header claims."""
    combined: Dict[Any, List[float]] = defaultdict(list)
    for values in per_run_values.values():
        for key, value in values.items():
            combined[key].append(value)
    return {
        key: {"mean": mean(v), "min": min(v), "max": max(v), "runs": len(v)}
        for key, v in combined.items()
    }


def _print_aggregate_table(title: str, header_cols: List[str], rows: List[Tuple[str, dict]]) -> None:
    print(title)
    print(f"{'key':<32}" + "".join(f"{c:>10}" for c in header_cols) + f"{'runs':>8}")
    for key_label, agg in rows:
        print(
            f"{key_label:<32}"
            + f"{agg['mean']:>10.4f}{agg['min']:>10.4f}{agg['max']:>10.4f}"
            + f"{agg['runs']:>8}"
        )


def _print_presence_table(agg: Dict[Tuple[int, int, int, int], dict]) -> None:
    """Same per-modality columns as pid_modality.py's own presence table (rather than
    a bare (1, 1, 1, 1)-style tuple key, whose modality order isn't otherwise stated
    anywhere in this script's output)."""
    print("Modality-presence success rate (mean / min / max across runs):")
    print(
        " ".join(f"{m:>7}" for m in pid_modality.MODALITY_ORDER)
        + f"{'mean':>10}{'min':>10}{'max':>10}{'runs':>8}"
    )
    for combo in sorted(agg, reverse=True):  # binary-descending, all-ones first
        a = agg[combo]
        print(
            " ".join(f"{v:>7}" for v in combo)
            + f"{a['mean']:>10.4f}{a['min']:>10.4f}{a['max']:>10.4f}{a['runs']:>8}"
        )


def print_aggregate(per_run: Dict[str, dict], measure: str) -> None:
    pid_per_run = {rid: pid_values(a["orig_rows"], measure) for rid, a in per_run.items() if a["orig_rows"]}
    presence_per_run = {rid: presence_values(a["orig_rows"]) for rid, a in per_run.items() if a["orig_rows"]}
    perturbation_per_run = {rid: perturbation_values(a["plus_rows"]) for rid, a in per_run.items() if a["plus_rows"]}
    severity_per_run = {rid: severity_values(a["severity"]) for rid, a in per_run.items() if a["severity"]}

    pid_agg = aggregate_scalars(pid_per_run)
    presence_agg = aggregate_scalars(presence_per_run)
    perturbation_agg = aggregate_scalars(perturbation_per_run)
    severity_agg = aggregate_scalars(severity_per_run)

    if pid_agg:
        rows = sorted(pid_agg.items())
        _print_aggregate_table(
            "PID modality decomposition (mean / min / max across runs):",
            ["mean", "min", "max"],
            [(f"{pair} {component}", agg) for (pair, component), agg in rows],
        )
        print()

    if presence_agg:
        _print_presence_table(presence_agg)
        print()

    if perturbation_agg:
        # OVERALL last, like perturbation_sr.report()'s single-run table -- not
        # wherever it happens to sort alphabetically among the category names.
        rows = sorted((k, v) for k, v in perturbation_agg.items() if k != "OVERALL")
        if "OVERALL" in perturbation_agg:
            rows.append(("OVERALL", perturbation_agg["OVERALL"]))
        _print_aggregate_table(
            "Per-perturbation success rate (mean / min / max across runs):",
            ["mean", "min", "max"],
            rows,
        )
        print()

    if severity_agg:
        # axis in ("severity", "subtype") -- the measured breakdown -- vs.
        # "difficulty_level" -- the upstream annotation severity_sr disagrees with --
        # kept in separate tables, same as severity_sr.report()'s two titles per
        # category, rather than pooled into one and losing that distinction.
        physical = sorted(
            (k for k in severity_agg if k[1] in ("severity", "subtype")),
            key=lambda k: (k[0], _natural_key(k[2])),
        )
        _print_aggregate_table(
            "Success rate by physical perturbation severity (mean / min / max across runs):",
            ["mean", "min", "max"],
            [(f"{category} / {bin_label}", severity_agg[(category, axis, bin_label)]) for category, axis, bin_label in physical],
        )
        print()

        difficulty = sorted(
            (k for k in severity_agg if k[1] == "difficulty_level"),
            key=lambda k: (k[0], _natural_key(k[2])),
        )
        _print_aggregate_table(
            "Success rate by upstream difficulty_level (mean / min / max across runs):",
            ["mean", "min", "max"],
            [(f"{category} / {bin_label}", severity_agg[(category, axis, bin_label)]) for category, axis, bin_label in difficulty],
        )
        print()

    if not (pid_agg or presence_agg or perturbation_agg or severity_agg):
        print("No evaluation data available for any matched run.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_config_keys(spec: str, filters: Optional[dict]) -> List[str]:
    if spec.strip() == "all":
        return ["all"]
    keys = [k.strip() for k in spec.split(",") if k.strip()]
    if filters:
        for filter_key in filters:
            if filter_key.startswith("config."):
                key = filter_key.split(".", 1)[1]
                if key not in keys:
                    keys.append(key)
    return keys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    for name in ("run", "filter"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--entity", default=None)
        sub.add_argument("--project", default=None)
        sub.add_argument("--measure", default="ccs", choices=sorted(pid_modality.MEASURES))
        sub.add_argument("--config-keys", default=",".join(DEFAULT_CONFIG_KEYS))
        sub.add_argument(
            "--libero-plus-root", default=severity_sr.DEFAULT_LIBERO_PLUS_ROOT,
            help="Path to the LIBERO-Plus checkout (only needed to resolve Light Conditions scene XMLs).",
        )

    subparsers.choices["run"].add_argument("run_ids", nargs="+")
    subparsers.choices["filter"].add_argument("--filters", default="{}")
    subparsers.choices["filter"].add_argument(
        "--modalities",
        default=None,
        help="Comma-separated modality names (rgb_static,rgb_gripper,language,proprio) the "
        "matched runs must have been TRAINED with exactly -- every other modality must be "
        "off. Filters client-side on the effective set; see run_modalities()'s docstring "
        "for why --filters can't express this.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    default_entity, default_project = default_entity_project()
    entity = args.entity or default_entity
    project = args.project or default_project
    print(f"entity/project: {entity}/{project}")

    api = wandb.Api()

    if args.cmd == "run":
        print(f"run ids: {args.run_ids}")
        config_keys = parse_config_keys(args.config_keys, None)

        runs = []
        fetch_problems = []
        for run_id in args.run_ids:
            try:
                runs.append(api.run(f"{entity}/{project}/{run_id}"))
            except Exception as exc:  # noqa: BLE001 -- surfaced as a problem, not a crash
                fetch_problems.append(f"{run_id}: could not fetch run ({exc})")
        if not runs:
            print("WARNING:")
            for p in fetch_problems:
                print(f"  {p}")
            raise SystemExit(f"No runs could be fetched out of {args.run_ids}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            per_run = {}
            for run in runs:
                print_run_header(run, config_keys)
                artifact_dir, missing = download_run(run, tmp_path)
                per_run[run.id] = analyze(artifact_dir, missing, args.measure, args.libero_plus_root)
            print()

            all_problems = fetch_problems + print_problems(per_run)
            if fetch_problems:
                print("WARNING (unfetched runs):")
                for p in fetch_problems:
                    print(f"  {p}")
                print()

            for run in runs:
                print_single(run.id, per_run[run.id], args.libero_plus_root)
                print()

    else:  # filter
        filters = json.loads(args.filters)
        print(f"filters: {json.dumps(filters)}")
        config_keys = parse_config_keys(args.config_keys, filters)

        runs = list(api.runs(f"{entity}/{project}", filters=filters))

        modality_problems = []
        if args.modalities is not None:
            wanted = parse_modality_spec(args.modalities)
            print(f"modalities: {wanted}")
            kept = []
            for run in runs:
                modalities = run_modalities(run)
                if modalities is None:
                    modality_problems.append(f"{run.id}: could not parse trained modalities from config")
                elif modalities == wanted:
                    kept.append(run)
            print(f"  {len(runs) - len(kept)} of {len(runs)} run(s) dropped by --modalities")
            runs = kept

        if not runs:
            reason = f"filters={filters!r}"
            if args.modalities is not None:
                reason += f", --modalities={args.modalities!r}"
            raise SystemExit(f"No runs matched {reason} in {entity}/{project}")

        print(f"matched {len(runs)} run(s):")
        print_run_table(runs, config_keys)
        print()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            per_run = {}
            for run in runs:
                artifact_dir, missing = download_run(run, tmp_path)
                per_run[run.id] = analyze(artifact_dir, missing, args.measure, args.libero_plus_root)

            print_problems(per_run)
            if modality_problems:
                print("WARNING (unparseable modalities):")
                for p in modality_problems:
                    print(f"  {p}")
                print()
            print_aggregate(per_run, args.measure)


if __name__ == "__main__":
    main()
