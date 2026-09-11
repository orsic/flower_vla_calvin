#!/usr/bin/env python3
"""Partial information decomposition of modality availability vs. episode success.

For each pair of modalities (static RGB, wrist RGB, language, proprioception), treats
availability of the two modalities as sources X1, X2 and episode success as target Y,
with the remaining modalities held ON so all four (x1, x2) conditions are present in
the data. Decomposes I(X1, X2 ; Y) into redundancy / unique-1 / unique-2 / synergy.

Uses dit's PID_CCS (Ince 2017, common change in surprisal) by default -- the same
measure as robince/partial-info-decomp's Iccs.m, dit being a pure-Python
reimplementation (that repo is MATLAB and depends on the Statistics Toolbox's
`changem`, which GNU Octave lacks).

Usage:
  python scripts/pid_modality.py <result.csv> [--measure ccs|broja|wb]
"""
import argparse
import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import networkx as _nx
import numpy as _np
import scipy.optimize as _sopt

# dit==1.2.3 (2019) is the last release supporting python 3.9 (see Containerfile),
# predating networkx 2.4's Graph.node -> Graph.nodes rename and a stricter x0-must-
# be-1-D check scipy's minimize() gained since. Both shims restore the old, more
# permissive behavior dit's code was written against -- no change in what's computed.
if not hasattr(_nx.DiGraph, "node"):
    _nx.DiGraph.node = property(lambda self: self.nodes)

import dit
import dit.algorithms.optimization as _dit_opt
from dit.pid import PID_BROJA, PID_CCS, PID_WB

_orig_minimize = _sopt.minimize


def _minimize_flatten_x0(fun, x0, *args, **kwargs):
    return _orig_minimize(fun, _np.asarray(x0).ravel(), *args, **kwargs)


_dit_opt.minimize = _minimize_flatten_x0

sys.path.insert(0, str(Path(__file__).parent))
from compare_eval_csvs import MODALITY_COLUMNS  # noqa: E402

# Fixed print order (also used to order held-modality tuples deterministically).
MODALITY_ORDER = ["static", "wrist", "lang", "proprio"]
MEASURES = {"ccs": PID_CCS, "broja": PID_BROJA, "wb": PID_WB}


def held_modalities(m1: str, m2: str, active=None) -> tuple:
    """The modalities other than the pair (m1, m2), held ON while (m1, m2) vary.
    `active` restricts the pool of "other" modalities (default: all of MODALITY_ORDER)
    -- with a 2-modality active set this returns an empty tuple, i.e. no holding."""
    order = MODALITY_ORDER if active is None else active
    held = set(order) - {m1, m2}
    return tuple(m for m in order if m in held)


class IncompleteCombos(SystemExit):
    """Raised by build_joint when the data doesn't cover all four (x1, x2) cells for a
    pair -- e.g. a no-proprio dropout sweep has no rows where proprio varies. Subclasses
    SystemExit so standalone build_joint() calls (and existing tests) still see a
    SystemExit; main() catches it specifically to skip just that pair."""


def load_rows(path: str) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def classify_modality(rows: list, modality: str) -> str:
    """Classify one modality column across the whole file: 'absent' (no row carries
    the column at all), 'constant-0' (present, never 1), 'constant-1' (present, never
    0), or 'varying'."""
    col = MODALITY_COLUMNS[modality]
    if not any(col in row for row in rows):
        return "absent"
    values = {int(row.get(col) or 0) for row in rows}
    if values == {0}:
        return "constant-0"
    if values == {1}:
        return "constant-1"
    return "varying"


def active_modalities(rows: list) -> tuple:
    """(modalities usable in this report, classification per modality). Drops
    'absent' and 'constant-0' modalities -- a modality that was never evaluated, or
    was evaluated but never turned on, cannot be a PID source or a meaningful held
    condition. Order follows MODALITY_ORDER."""
    status = {m: classify_modality(rows, m) for m in MODALITY_ORDER}
    active = [m for m in MODALITY_ORDER if status[m] not in ("absent", "constant-0")]
    return active, status


def held_rows(rows: list, held) -> list:
    """Rows with every modality in `held` on. `held` is a modality name or an
    iterable of them -- a bare string is treated as one modality name, not iterated
    character-by-character. A row missing a held column entirely (pre-use_proprio
    result.csv, see eval_records.py's module docstring) reads as that modality being
    off. An empty `held` returns every row unfiltered (the marginalized case)."""
    held = (held,) if isinstance(held, str) else tuple(held)
    held_cols = [MODALITY_COLUMNS[h] for h in held]
    return [row for row in rows if all(int(row.get(col) or 0) == 1 for col in held_cols)]


def build_joint(rows: list, m1: str, m2: str, held) -> "dit.Distribution":
    """Joint distribution of (avail_m1, avail_m2, success) over rows with every
    modality in `held` on -- an empty `held` pools every held configuration in the
    data (marginalizing the held modalities out). See held_rows() for the
    held-column handling."""
    held = (held,) if isinstance(held, str) else tuple(held)
    col1, col2 = MODALITY_COLUMNS[m1], MODALITY_COLUMNS[m2]
    counts = defaultdict(int)
    total = 0
    for row in held_rows(rows, held):
        x1, x2, y = int(row[col1]), int(row[col2]), int(row["success"])
        counts[(x1, x2, y)] += 1
        total += 1

    seen_cells = {key[:2] for key in counts}
    missing = [(x1, x2) for x1 in (0, 1) for x2 in (0, 1) if (x1, x2) not in seen_cells]
    if missing:
        held_label = "+".join(held) or "none"
        raise IncompleteCombos(
            f"build_joint({m1}, {m2} | {held_label}=on): missing (x1, x2) cell(s) {missing} "
            f"-- has this modality combo been evaluated?"
        )

    outcomes = [f"{x1}{x2}{y}" for x1, x2, y in counts]
    pmf = [count / total for count in counts.values()]
    return dit.Distribution(outcomes, pmf)


def decompose(dist: "dit.Distribution", measure: str = "ccs") -> dict:
    if measure not in MEASURES:
        raise SystemExit(f"Unknown measure {measure!r}; choose from {sorted(MEASURES)}")
    pid = MEASURES[measure](dist)
    return {
        "R": pid[((0,), (1,))],
        "U1": pid[((0,),)],
        "U2": pid[((1,),)],
        "S": pid[((0, 1),)],
        "I": dit.shannon.mutual_information(dist, [0, 1], [2]),
    }


def conditional_srs(rows: list, m1: str, m2: str, held) -> dict:
    """Same held-column handling as build_joint (see held_rows())."""
    held = (held,) if isinstance(held, str) else tuple(held)
    col1, col2 = MODALITY_COLUMNS[m1], MODALITY_COLUMNS[m2]
    outcomes = defaultdict(list)
    for row in held_rows(rows, held):
        outcomes[(int(row[col1]), int(row[col2]))].append(int(row["success"]))
    return {k: sum(v) / len(v) for k, v in outcomes.items()}


def presence_success(rows: list, modalities=None) -> dict:
    """Success rate per modality-presence combination actually present in the data,
    keyed by 0/1 tuples ordered per `modalities` (default MODALITY_ORDER). Same
    missing-column handling as build_joint/conditional_srs: absent columns read as
    that modality being off."""
    modalities = MODALITY_ORDER if modalities is None else modalities
    cols = [MODALITY_COLUMNS[m] for m in modalities]
    outcomes = defaultdict(list)
    for row in rows:
        key = tuple(int(row.get(col) or 0) for col in cols)
        outcomes[key].append(int(row["success"]))
    return {k: (sum(v) / len(v), len(v)) for k, v in outcomes.items()}


_DROP_REASONS = {"absent": "column absent", "constant-0": "present but never 1"}


def _print_pid_row(label: str, held_label: str, pid: dict, n: int) -> None:
    print(
        f"{label:<16} {held_label:<16} {pid['I']:>12.4f} {pid['R']:>8.4f} "
        f"{pid['U1']:>8.4f} {pid['U2']:>8.4f} {pid['S']:>8.4f} {n:>6}"
    )


def _print_srs_row(srs: dict) -> None:
    print(
        f"    SR: (0,0)={srs[(0, 0)]:.3f}  (1,0)={srs[(1, 0)]:.3f}  "
        f"(0,1)={srs[(0, 1)]:.3f}  (1,1)={srs[(1, 1)]:.3f}"
    )


def collect(rows: list, measure: str = "ccs", marginalize: bool = False) -> dict:
    """Compute everything main() prints, without printing it -- for reuse by callers
    that want the numbers (e.g. scripts/analyze_wandb.py) as well as by report().

    Returns {"active", "status", "dropped", "pairs", "presence"}. Each entry in
    "pairs" is {"m1", "m2", "held", ["error"] | ["pid", "srs", "n"], ["marginalized"]},
    "marginalized" (present only when marginalize=True and held is non-empty) is
    itself {"error"} | {"pid", "srs", "n"}. "error" carries the IncompleteCombos
    message for a pair/marginalization that wasn't evaluated, mirroring main()'s
    "not evaluated --" lines. "presence" is presence_success(rows, active).
    """
    active, status = active_modalities(rows)
    dropped = [m for m in MODALITY_ORDER if m not in active]

    pairs = []
    for m1, m2 in itertools.combinations(active, 2):
        held = held_modalities(m1, m2, active)
        entry = {"m1": m1, "m2": m2, "held": held}
        try:
            dist = build_joint(rows, m1, m2, held)
        except IncompleteCombos as exc:
            entry["error"] = str(exc)
            pairs.append(entry)
            continue
        entry["pid"] = decompose(dist, measure)
        entry["srs"] = conditional_srs(rows, m1, m2, held)
        entry["n"] = len(held_rows(rows, held))

        if marginalize and held:
            try:
                mdist = build_joint(rows, m1, m2, ())
            except IncompleteCombos as exc:
                entry["marginalized"] = {"error": str(exc)}
                pairs.append(entry)
                continue
            entry["marginalized"] = {
                "pid": decompose(mdist, measure),
                "srs": conditional_srs(rows, m1, m2, ()),
                "n": len(held_rows(rows, ())),
            }
        pairs.append(entry)

    return {
        "active": active,
        "status": status,
        "dropped": dropped,
        "pairs": pairs,
        "presence": presence_success(rows, active),
    }


def report(collected: dict) -> None:
    """Print collect()'s output in main()'s original format."""
    active, status, dropped = collected["active"], collected["status"], collected["dropped"]

    if all(status[m] == "varying" for m in active):
        active_label = ", ".join(active) + " (varying)"
    else:
        active_label = ", ".join(
            f"{m} ({status[m]})" if status[m] != "varying" else m for m in active
        )
    header = f"modalities: {active_label}"
    if dropped:
        header += " | dropped: " + ", ".join(f"{m} ({_DROP_REASONS[status[m]]})" for m in dropped)
    print(header)

    if len(active) < 2:
        # Not an error -- a non-dropout run legitimately evaluates one modality combo,
        # so no pair can vary. Print the explanation and fall through to the presence
        # table; exiting non-zero here would make upload()'s check=True subprocess call
        # abort before the result.csv artifacts get attached.
        print(f"no PID: needs >=2 modalities that vary across episodes; got {active}")
    else:
        print(f"{'pair':<16} {'held':<16} {'I(X1,X2;Y)':>12} {'R':>8} {'U1':>8} {'U2':>8} {'S':>8} {'n':>6}")
        for entry in collected["pairs"]:
            pair_label = entry["m1"] + "+" + entry["m2"]
            held = entry["held"]
            held_label = "+".join(held) if held else "none"
            if "error" in entry:
                print(f"{pair_label:<16} {held_label:<16} not evaluated -- {entry['error']}")
                continue
            _print_pid_row(pair_label, held_label, entry["pid"], entry["n"])
            _print_srs_row(entry["srs"])

            marg = entry.get("marginalized")
            if marg is not None:
                if "error" in marg:
                    print(f"{pair_label:<16} {'marginalized':<16} not evaluated -- {marg['error']}")
                else:
                    _print_pid_row(pair_label, "marginalized", marg["pid"], marg["n"])
                    _print_srs_row(marg["srs"])

    print()
    print(" ".join(f"{m:>7}" for m in active) + f" {'success_rate':>12} {'n':>6}")
    presence = collected["presence"]
    for combo in sorted(presence, reverse=True):  # binary-descending, all-ones first
        sr, n = presence[combo]
        print(" ".join(f"{v:>7}" for v in combo) + f" {sr:>12.3f} {n:>6}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_csv")
    parser.add_argument("--measure", default="ccs", choices=sorted(MEASURES))
    parser.add_argument(
        "--marginalize",
        action="store_true",
        help="Also report each pair's decomposition pooled across all held "
        "configurations present in the data (held modalities marginalized out), "
        "alongside the default held-all-on decomposition.",
    )
    args = parser.parse_args()

    rows = load_rows(args.result_csv)
    report(collect(rows, measure=args.measure, marginalize=args.marginalize))


if __name__ == "__main__":
    main()
