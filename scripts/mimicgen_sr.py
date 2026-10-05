#!/usr/bin/env python3
"""MimicGen success rate per task family, and within a family per difficulty variant.

Reads the `mimicgen.csv` member of the `evaluation` W&B artifact scripts/mimicgen_pipeline.py's
`upload` attaches to a training run (one row per episode, eval_records.py's schema), and
prints success rate + 95% Wilson CI per (family, d0/d1/d2), a pooled per-family ALL row
(families with more than one variant), and an OVERALL row. A modality-dropout run's CSV
holds several modality combos (scripts/mimicgen_pipeline.py); each gets its own table,
all-modalities first.

Two modes, mirroring scripts/analyze_wandb.py:
  run     One or more W&B run IDs -- one table per run.
  filter  A W&B MongoDB-style filter over run config -- one table of mean/min/max of the
          per-run success rate per cell, plus how many runs contributed.

Usage:
  python scripts/mimicgen_sr.py run <run_id> [<run_id>...] [--entity E] [--project P]
  python scripts/mimicgen_sr.py filter --filters '<mongo-json>' [--entity E] [--project P]

--entity/--project default to conf/config_mimicgen.yaml's logger.entity/logger.project.
"""
import argparse
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import wandb
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from compare_eval_csvs import MODALITY_COLUMNS  # noqa: E402
from pid_modality import load_rows  # noqa: E402
from severity_sr import wilson_interval  # noqa: E402

from flower.datasets.mimicgen_tasks import family  # noqa: E402

MEMBER = "mimicgen.csv"
ALL = "ALL"
OVERALL = "OVERALL"

Key = Tuple[str, str]


def default_entity_project() -> Tuple[str, str]:
    cfg = OmegaConf.load(Path(__file__).parents[1] / "conf" / "config_mimicgen.yaml")
    return str(cfg.logger.entity), str(cfg.logger.project)


# ---------------------------------------------------------------------------
# Breakdown
# ---------------------------------------------------------------------------


def split_dataset(name: str) -> Key:
    """'three_piece_assembly_d2' -> ('three_piece_assembly', 'd2')."""
    return family(name), re.search(r"d\d+$", name).group(0)


def breakdown(rows: List[Dict]) -> Dict[Key, Tuple[int, int]]:
    """(family, variant) -> (successes, n), plus a pooled (family, ALL) per family and
    (OVERALL, ""). Every dataset gets the same n_eval, so pooling == mean over variants."""
    cells: Dict[Key, List[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        fam, variant = split_dataset(row["task_name"])
        success = int(row["success"])
        for key in ((fam, variant), (fam, ALL), (OVERALL, "")):
            cells[key][0] += success
            cells[key][1] += 1
    return {key: (s, n) for key, (s, n) in cells.items()}


def ordered_keys(cells: Dict[Key, Any]) -> List[Key]:
    """Print order: families alphabetically, d0 < d1 < d2 within each, then the family's
    ALL row if it has more than one variant; OVERALL last."""
    variants: Dict[str, List[str]] = defaultdict(list)
    for fam, variant in cells:
        if fam != OVERALL and variant != ALL:
            variants[fam].append(variant)
    keys: List[Key] = []
    for fam in sorted(variants):
        keys += [(fam, v) for v in sorted(variants[fam], key=lambda v: int(v[1:]))]
        if len(variants[fam]) > 1:
            keys.append((fam, ALL))
    if (OVERALL, "") in cells:
        keys.append((OVERALL, ""))
    return keys


def split_by_combo(rows: List[Dict]) -> Dict[str, List[Dict]]:
    """combo label (e.g. 'static+wrist+lang') -> its rows; all-modalities first, then by
    descending number of enabled modalities."""
    groups: Dict[Tuple[int, ...], List[Dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(int(row.get(col) or 0) for col in MODALITY_COLUMNS.values())].append(row)
    order = sorted(groups, key=lambda flags: (sum(flags), flags), reverse=True)
    return {"+".join(m for m, on in zip(MODALITY_COLUMNS, flags) if on): groups[flags] for flags in order}


def by_combo(sections: Dict[str, str]) -> str:
    """One labeled section per combo; a single combo prints its table alone, unlabeled."""
    if len(sections) == 1:
        return next(iter(sections.values()))
    return "\n\n".join(f"-- {label} --\n{text}" for label, text in sections.items())


def format_table(rows: List[Dict]) -> str:
    return by_combo({label: format_combo_table(group) for label, group in split_by_combo(rows).items()})


def format_combo_table(rows: List[Dict]) -> str:
    lines = [f"{'family':<22} {'variant':<7} {'success_rate':>12} {'ci_low':>7} {'ci_high':>7} {'n':>6}"]
    cells = breakdown(rows)
    for key in ordered_keys(cells):
        s, n = cells[key]
        lo, hi = wilson_interval(s, n)
        lines.append(f"{key[0]:<22} {key[1]:<7} {s / n:>12.3f} {lo:>7.3f} {hi:>7.3f} {n:>6}")
    return "\n".join(lines)


def aggregate_runs(per_run: Dict[str, List[Dict]]) -> Dict[Key, Tuple[float, float, float, int]]:
    """cell -> (mean, min, max, n_runs) of each run's own success rate in that cell."""
    rates: Dict[Key, List[float]] = defaultdict(list)
    for rows in per_run.values():
        for key, (s, n) in breakdown(rows).items():
            rates[key].append(s / n)
    return {key: (sum(v) / len(v), min(v), max(v), len(v)) for key, v in rates.items()}


def format_aggregate(agg: Dict[Key, Tuple[float, float, float, int]]) -> str:
    lines = [f"{'family':<22} {'variant':<7} {'mean':>7} {'min':>7} {'max':>7} {'n_runs':>6}"]
    for key in ordered_keys(agg):
        mean, lo, hi, n_runs = agg[key]
        lines.append(f"{key[0]:<22} {key[1]:<7} {mean:>7.3f} {lo:>7.3f} {hi:>7.3f} {n_runs:>6}")
    return "\n".join(lines)


def format_aggregate_by_combo(per_run: Dict[str, List[Dict]]) -> str:
    """format_aggregate per combo; a run contributes to every combo it was evaluated on."""
    per_combo: Dict[str, Dict[str, List[Dict]]] = defaultdict(dict)
    for run_id, rows in per_run.items():
        for label, group in split_by_combo(rows).items():
            per_combo[label][run_id] = group
    labels = list(split_by_combo([row for rows in per_run.values() for row in rows]))
    return by_combo({label: format_aggregate(aggregate_runs(per_combo[label])) for label in labels})


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def mimicgen_artifact(run) -> Optional[Any]:
    """Newest `evaluation` artifact of `run` that holds mimicgen.csv -- not just the newest
    evaluation artifact, which for a run also through ./run.sh pipeline may be LIBERO's."""
    candidates = [
        a for a in run.logged_artifacts() if a.type == "evaluation" and MEMBER in a.manifest.entries
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda a: a.created_at)


def fetch_rows(runs, dest: Path) -> Tuple[Dict[str, List[Dict]], List[str]]:
    """(run_id -> mimicgen.csv rows, problems) for every run that has the member."""
    per_run, problems = {}, []
    for run in runs:
        artifact = mimicgen_artifact(run)
        if artifact is None:
            problems.append(f"{run.id}: no evaluation artifact with {MEMBER}")
            continue
        artifact_dir = Path(artifact.download(root=str(dest / run.id)))
        per_run[run.id] = load_rows(str(artifact_dir / MEMBER))
    return per_run, problems


def print_problems(problems: List[str]) -> None:
    if problems:
        print("WARNING (skipped runs):")
        for p in problems:
            print(f"  {p}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    run_p = subparsers.add_parser("run")
    run_p.add_argument("run_ids", nargs="+")
    filter_p = subparsers.add_parser("filter")
    filter_p.add_argument("--filters", default="{}")
    for p in (run_p, filter_p):
        p.add_argument("--entity")
        p.add_argument("--project")
    args = parser.parse_args()

    default_entity, default_project = default_entity_project()
    entity = args.entity or default_entity
    project = args.project or default_project
    print(f"entity/project: {entity}/{project}")
    api = wandb.Api()

    problems = []
    if args.cmd == "run":
        runs = []
        for run_id in args.run_ids:
            try:
                runs.append(api.run(f"{entity}/{project}/{run_id}"))
            except Exception as exc:  # noqa: BLE001 -- surfaced as a problem, not a crash
                problems.append(f"{run_id}: could not fetch run ({exc})")
    else:
        filters = json.loads(args.filters)
        print(f"filters: {json.dumps(filters)}")
        runs = list(api.runs(f"{entity}/{project}", filters=filters))

    with tempfile.TemporaryDirectory() as tmp:
        per_run, fetch_problems = fetch_rows(runs, Path(tmp))
    print_problems(problems + fetch_problems)
    if not per_run:
        raise SystemExit("No run with a mimicgen.csv evaluation artifact.")

    names = {run.id: run.name for run in runs}
    if args.cmd == "run":
        for run_id, rows in per_run.items():
            print(f"== {run_id} ({names[run_id]}) ==")
            print(format_table(rows))
            print()
    else:
        print(f"{len(per_run)} run(s) with {MEMBER}:")
        for run_id in per_run:
            print(f"  {run_id}  {names[run_id]}")
        print()
        print(format_aggregate_by_combo(per_run))


if __name__ == "__main__":
    main()
