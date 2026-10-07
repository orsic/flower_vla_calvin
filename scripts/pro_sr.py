#!/usr/bin/env python3
"""Success rate per LIBERO-PRO perturbation, against an equated LIBERO original subset.

LIBERO-PRO perturbs what a task *is* -- which object, where it starts, how the
instruction is worded, what counts as success -- rather than how the scene is sensed
(LIBERO-Plus's axis). Each perturbation is its own 10-task suite whose task names, bddl
filenames and init filenames are the *original* libero_10 ones, so a PRO episode maps to
its orig baseline episode by identity: no prefix matching, no suffix grammar.

The measurement this script exists for is the comparison, not the raw PRO number. A PRO
suite's success rate on its own says little; what says something is how it moves relative
to the *same* episodes run unperturbed. So every record's baseline is not the whole
libero_orig.csv -- it is exactly the subset of orig rows whose (modality combo, task,
episode) keys are present on the PRO side. A PRO run covering 8 of 10 tasks is compared
against those 8 tasks' orig episodes and no others.

Two pairing strengths, chosen per record by which init-state arm produced the rows:

  episode  The pro_matched arm (libero_10_lan, libero_10_task) rolls out from the
           ORIGINAL suite's init states, so a PRO episode and its orig counterpart share
           a task, an init state and a rollout seed -- they differ only by the
           perturbation. One McNemar pair per (combo, task, episode): the strongest test
           available here, and the reason the matched arm is run at all.

  task     The native arm's init states were sampled independently of the originals'
           (and for libero_10_swap the init state IS the perturbation), so there is no
           episode-level counterpart. Pairs collapse to one per (combo, task) by strict
           majority vote on each side, ties dropped -- the same granularity
           severity_sr.paired_baseline uses, and for the same reason: one pair per
           episode against a reused orig outcome would inflate discordance by
           pseudo-replication rather than by evidence.

`perturbation_vector` records the suite's perturbation as the upstream 5-flag tuple
(swap, object, lan, task, env; see LIBERO-PRO's evaluation_config.yaml). Only one flag is
ever set for the published suites, but the column is a vector so a locally generated
multi-flag "_temp" suite would slot in without a schema change.

Usage:
    python scripts/pro_sr.py <pro result.csv> [more...] --orig-csv <libero_orig.csv>
    python scripts/pro_sr.py --out pro_sr.csv ...
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, Path(__file__).absolute().parent.as_posix())
from compare_eval_csvs import MODALITY_COLUMNS  # noqa: E402
from severity_sr import mcnemar_exact_p, wilson_interval  # noqa: E402

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
from flower.evaluation.libero_tasks import pro_suite_tag  # noqa: E402

# Upstream's flag order (LIBERO-PRO evaluation_config.yaml writes exactly this tuple into
# a generated suite's log.txt), so a perturbation_vector here reads the same way it does
# over there.
PERTURBATION_FLAGS = ("swap", "object", "lan", "task", "env")

# Which arm each record's rows came from -> how finely its pairs can be built.
EPISODE_PAIRING = "episode"
TASK_PAIRING = "task"

CSV_COLUMNS = [
    "libero_variant", "suite", "perturbation", "perturbation_vector",
    "successes", "n", "tasks", "success_rate", "ci_low", "ci_high",
    "paired_success_rate",
    "orig_successes", "orig_n", "orig_success_rate", "orig_ci_low", "orig_ci_high",
    "delta", "pairing", "mcnemar_b", "mcnemar_c", "mcnemar_n", "mcnemar_p",
    "modality_combo",
]


def load_rows(path: Any) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def perturbation_vector(suite: str) -> str:
    """"0,0,1,0,0" for libero_10_lan -- the (swap, object, lan, task, env) flag tuple.

    An unrecognised suite yields an all-zero vector rather than an error: the record
    still carries the suite name, and a zero vector is an honest "no perturbation flag
    identified" rather than a guess.
    """
    tag = pro_suite_tag(suite)
    return ",".join("1" if flag == tag else "0" for flag in PERTURBATION_FLAGS)


def combo(row: Dict[str, str]) -> Tuple[int, int, int, int]:
    """(static, wrist, lang, proprio) for one row -- the pairing key has to match on
    modality ablation too, since a dropout run's orig result.csv holds all 14 combos."""
    cols = [MODALITY_COLUMNS[m] for m in ("static", "wrist", "lang", "proprio")]
    return tuple(int(row.get(col) or 0) for col in cols)


def combo_label(modality_combo: Tuple[int, int, int, int]) -> str:
    """Compact mask for one (static, wrist, lang, proprio) combo: "SWLP" all on,
    "SW-P" with language withheld. A swept matched suite emits 14 records that differ
    only by this, so the report needs it to be readable at a glance; the raw tuple stays
    in the CSV's modality_combo column."""
    return "".join(
        letter if on else "-" for letter, on in zip("SWLP", modality_combo)
    )


def episode_key(row: Dict[str, str]) -> Tuple[Tuple[int, int, int, int], str, str]:
    return (combo(row), row["task_name"], str(row.get("episode_idx", "")))


def task_key(row: Dict[str, str]) -> Tuple[Tuple[int, int, int, int], str]:
    return (combo(row), row["task_name"])


def _majority(successes: Sequence[int]) -> Optional[int]:
    """1/0 by strict majority, None on an exact tie (dropped rather than broken in
    either direction)."""
    ones = sum(successes)
    zeros = len(successes) - ones
    if ones == zeros:
        return None
    return 1 if ones > zeros else 0


def equated_subset(
    pro_rows: List[Dict[str, str]],
    orig_rows: List[Dict[str, str]],
    pairing: str,
) -> List[Tuple[int, int]]:
    """The orig episodes equated to these PRO rows, as (pro_outcome, orig_outcome) pairs
    at the requested granularity.

    Orig rows outside the PRO side's key set never enter -- that restriction is the whole
    point of "equated": a PRO suite that covered 8 tasks must not be read against a
    baseline averaged over 10. Each side collapses to one outcome per key by strict
    majority, ties dropped; under episode pairing each key already holds a single row
    either side, so the vote is a no-op there and only catches a torn CSV's duplicates.
    """
    key = episode_key if pairing == EPISODE_PAIRING else task_key
    orig_by_key: Dict[Any, List[int]] = defaultdict(list)
    for row in orig_rows:
        orig_by_key[key(row)].append(int(row["success"]))

    pro_by_key: Dict[Any, List[int]] = defaultdict(list)
    for row in pro_rows:
        k = key(row)
        if k in orig_by_key:
            pro_by_key[k].append(int(row["success"]))

    pairs: List[Tuple[int, int]] = []
    for k, pro_successes in pro_by_key.items():
        pro_outcome = _majority(pro_successes)
        orig_outcome = _majority(orig_by_key[k])
        if pro_outcome is None or orig_outcome is None:
            continue
        pairs.append((pro_outcome, orig_outcome))
    return pairs


def record(
    libero_variant: str,
    suite: str,
    modality_combo: Tuple[int, int, int, int],
    pro_rows: List[Dict[str, str]],
    orig_rows: Optional[List[Dict[str, str]]],
) -> Dict[str, Any]:
    """One output row: the PRO group's own rate, its equated orig baseline, and the
    paired test between them. Baseline/test columns are empty strings when no
    libero_orig.csv was supplied -- the PRO rate alone is still worth printing."""
    successes = sum(int(r["success"]) for r in pro_rows)
    n = len(pro_rows)
    lo, hi = wilson_interval(successes, n)
    out: Dict[str, Any] = {
        "libero_variant": libero_variant,
        "suite": suite,
        "perturbation": pro_suite_tag(suite) or "",
        "perturbation_vector": perturbation_vector(suite),
        "successes": successes,
        "n": n,
        "tasks": len({r["task_name"] for r in pro_rows}),
        "success_rate": successes / n if n else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "modality_combo": ",".join(str(v) for v in modality_combo),
        "paired_success_rate": "",
        "orig_successes": "", "orig_n": "", "orig_success_rate": "",
        "orig_ci_low": "", "orig_ci_high": "", "delta": "",
        "pairing": "", "mcnemar_b": "", "mcnemar_c": "", "mcnemar_n": "", "mcnemar_p": "",
    }
    if orig_rows is None:
        return out

    pairing = EPISODE_PAIRING if libero_variant == "pro_matched" else TASK_PAIRING
    pairs = equated_subset(pro_rows, orig_rows, pairing)
    if not pairs:
        # No orig episode shares a key with this group -- e.g. a withheld-modality record
        # on a non-dropout run, whose orig_libero_10 only ever held full modality. That is
        # an absent baseline, not a zero one, so it reads the same as "no orig csv given"
        # rather than printing nan down the paired columns.
        return out
    orig_successes = sum(o for _p, o in pairs)
    pro_paired_successes = sum(p for p, _o in pairs)
    orig_n = len(pairs)
    orig_lo, orig_hi = wilson_interval(orig_successes, orig_n)
    b = sum(1 for p, o in pairs if p == 1 and o == 0)
    c = sum(1 for p, o in pairs if p == 0 and o == 1)
    # Both sides of `delta` are read off the same pairs, so it compares like with like.
    # The group's own `success_rate` above is the raw per-episode rate and is NOT the
    # right minuend here: under task pairing each side has already collapsed to one
    # majority outcome per task, over a task set that excludes dropped ties -- subtracting
    # a task-level rate from an episode-level one would be a different estimand in each
    # term. (delta == (b - c) / mcnemar_n, by construction.)
    out.update({
        "paired_success_rate": pro_paired_successes / orig_n if orig_n else float("nan"),
        "orig_successes": orig_successes,
        "orig_n": orig_n,
        "orig_success_rate": orig_successes / orig_n if orig_n else float("nan"),
        "orig_ci_low": orig_lo,
        "orig_ci_high": orig_hi,
        "delta": (pro_paired_successes - orig_successes) / orig_n if orig_n else float("nan"),
        "pairing": pairing,
        "mcnemar_b": b,
        "mcnemar_c": c,
        "mcnemar_n": orig_n,
        "mcnemar_p": mcnemar_exact_p(b, c),
    })
    return out


def collect(
    pro_rows: List[Dict[str, str]],
    orig_rows: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """One record per (libero_variant, suite, modality combo) present in pro_rows, in
    suite then arm then combo order."""
    groups: Dict[Tuple[str, str, Tuple[int, int, int, int]], List[Dict[str, str]]] = defaultdict(list)
    for row in pro_rows:
        groups[(row.get("libero_variant", ""), row.get("suite", ""), combo(row))].append(row)

    # Full modality first within each suite/arm (descending count of enabled modalities),
    # matching eval_pipeline.modality_combos' own all-on-first order -- a plain tuple sort
    # would lead with (0, 0, 1, 0).
    records = []
    for (variant, suite, modality_combo) in sorted(
        groups, key=lambda k: (k[1], k[0], -sum(k[2]), k[2])
    ):
        rows = groups[(variant, suite, modality_combo)]
        orig_subset = None
        if orig_rows is not None:
            orig_subset = [r for r in orig_rows if combo(r) == modality_combo]
        records.append(record(variant, suite, modality_combo, rows, orig_subset))
    return records


def write_csv(path: Any, records: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def report(records: List[Dict[str, Any]], full_modality_only: bool = False) -> None:
    """Print one line per record. full_modality_only keeps just the all-on combo, for
    callers (analyze_wandb's default view) that would otherwise gain 28 swept matched-arm
    lines per run."""
    if full_modality_only and records:
        # Model-relative: a use_proprio=False checkpoint records use_proprio=0 on every
        # row (EvaluateLibero.uses_proprio is what the model actually received), so its
        # full-modality combo is "1,1,1,0". Keying on a literal "1,1,1,1" would filter
        # such a run's report down to nothing.
        widest = max(sum(int(v) for v in r["modality_combo"].split(",")) for r in records)
        records = [
            r for r in records
            if sum(int(v) for v in r["modality_combo"].split(",")) == widest
        ]
    show_baseline = any(r["orig_n"] != "" for r in records)
    header = (
        f"{'suite':<20} {'arm':<12} {'mod':<5} {'success_rate':>12} {'95% CI':>16} "
        f"{'n':>5} {'tasks':>6}"
    )
    if show_baseline:
        header += (
            f"  |  {'paired_sr':>9} {'orig_sr':>8} {'delta':>7} {'pairing':>8} "
            f"{'b':>3} {'c':>3} {'n':>4} {'p':>7}"
        )
    print(header)
    for r in records:
        ci = f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
        mod = combo_label(tuple(int(v) for v in r["modality_combo"].split(",")))
        line = (
            f"{r['suite']:<20} {r['libero_variant']:<12} {mod:<5} "
            f"{r['success_rate']:>12.3f} {ci:>16} {r['n']:>5} {r['tasks']:>6}"
        )
        if show_baseline and r["orig_n"] != "":
            line += (
                f"  |  {r['paired_success_rate']:>9.3f} {r['orig_success_rate']:>8.3f} "
                f"{r['delta']:>+7.3f} {r['pairing']:>8} {r['mcnemar_b']:>3} "
                f"{r['mcnemar_c']:>3} {r['mcnemar_n']:>4} {r['mcnemar_p']:>7.3f}"
            )
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+", help="LIBERO-PRO result.csv files (one per suite/arm)")
    parser.add_argument("--orig-csv", default=None, help="libero_orig.csv for the equated baseline")
    parser.add_argument("--out", default=None, help="write the breakdown to this CSV")
    args = parser.parse_args()

    pro_rows: List[Dict[str, str]] = []
    for path in args.csv:
        pro_rows.extend(load_rows(path))
    orig_rows = load_rows(args.orig_csv) if args.orig_csv else None

    records = collect(pro_rows, orig_rows)
    report(records)
    if args.out:
        write_csv(args.out, records)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
