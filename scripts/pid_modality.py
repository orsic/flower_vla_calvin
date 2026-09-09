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
PAIRS = list(itertools.combinations(MODALITY_ORDER, 2))
MEASURES = {"ccs": PID_CCS, "broja": PID_BROJA, "wb": PID_WB}


def held_modalities(m1: str, m2: str) -> tuple:
    """The modalities other than the pair (m1, m2), held ON while (m1, m2) vary."""
    held = set(MODALITY_COLUMNS) - {m1, m2}
    return tuple(m for m in MODALITY_ORDER if m in held)


class IncompleteCombos(SystemExit):
    """Raised by build_joint when the data doesn't cover all four (x1, x2) cells for a
    pair -- e.g. a no-proprio dropout sweep has no rows where proprio varies. Subclasses
    SystemExit so standalone build_joint() calls (and existing tests) still see a
    SystemExit; main() catches it specifically to skip just that pair."""


def load_rows(path: str) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_joint(rows: list, m1: str, m2: str, held) -> "dit.Distribution":
    """Joint distribution of (avail_m1, avail_m2, success) over rows with every
    modality in `held` on. `held` is a modality name or an iterable of them --
    a bare string is treated as one modality name, not iterated character-by-
    character. A row missing a held column entirely (pre-use_proprio result.csv,
    see eval_records.py's module docstring) reads as that modality being off."""
    held = (held,) if isinstance(held, str) else tuple(held)
    col1, col2 = MODALITY_COLUMNS[m1], MODALITY_COLUMNS[m2]
    held_cols = [MODALITY_COLUMNS[h] for h in held]
    counts = defaultdict(int)
    total = 0
    for row in rows:
        if not all(int(row.get(col) or 0) == 1 for col in held_cols):
            continue
        x1, x2, y = int(row[col1]), int(row[col2]), int(row["success"])
        counts[(x1, x2, y)] += 1
        total += 1

    seen_cells = {key[:2] for key in counts}
    missing = [(x1, x2) for x1 in (0, 1) for x2 in (0, 1) if (x1, x2) not in seen_cells]
    if missing:
        held_label = "+".join(held)
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
    """Same held-column handling as build_joint (see its docstring)."""
    held = (held,) if isinstance(held, str) else tuple(held)
    col1, col2 = MODALITY_COLUMNS[m1], MODALITY_COLUMNS[m2]
    held_cols = [MODALITY_COLUMNS[h] for h in held]
    outcomes = defaultdict(list)
    for row in rows:
        if not all(int(row.get(col) or 0) == 1 for col in held_cols):
            continue
        outcomes[(int(row[col1]), int(row[col2]))].append(int(row["success"]))
    return {k: sum(v) / len(v) for k, v in outcomes.items()}


def presence_success(rows: list) -> dict:
    """Success rate per modality-presence combination actually present in the data,
    keyed by (static, wrist, lang, proprio) 0/1 tuples. Same missing-column handling
    as build_joint/conditional_srs: absent columns read as that modality being off."""
    cols = [MODALITY_COLUMNS[m] for m in MODALITY_ORDER]
    outcomes = defaultdict(list)
    for row in rows:
        key = tuple(int(row.get(col) or 0) for col in cols)
        outcomes[key].append(int(row["success"]))
    return {k: (sum(v) / len(v), len(v)) for k, v in outcomes.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_csv")
    parser.add_argument("--measure", default="ccs", choices=sorted(MEASURES))
    args = parser.parse_args()

    rows = load_rows(args.result_csv)

    print(f"{'pair':<16} {'held':<16} {'I(X1,X2;Y)':>12} {'R':>8} {'U1':>8} {'U2':>8} {'S':>8}")
    for m1, m2 in PAIRS:
        held = held_modalities(m1, m2)
        held_label = "+".join(held)
        try:
            dist = build_joint(rows, m1, m2, held)
        except IncompleteCombos as exc:
            print(f"{m1 + '+' + m2:<16} {held_label:<16} not evaluated -- {exc}")
            continue
        pid = decompose(dist, args.measure)
        srs = conditional_srs(rows, m1, m2, held)
        print(
            f"{m1 + '+' + m2:<16} {held_label:<16} {pid['I']:>12.4f} {pid['R']:>8.4f} "
            f"{pid['U1']:>8.4f} {pid['U2']:>8.4f} {pid['S']:>8.4f}"
        )
        print(
            f"    SR: (0,0)={srs[(0, 0)]:.3f}  (1,0)={srs[(1, 0)]:.3f}  "
            f"(0,1)={srs[(0, 1)]:.3f}  (1,1)={srs[(1, 1)]:.3f}"
        )

    print()
    print(f"{'static':>7} {'wrist':>7} {'lang':>7} {'proprio':>7} {'success_rate':>12} {'n':>6}")
    presence = presence_success(rows)
    for combo in sorted(presence, reverse=True):  # binary-descending, all-ones first
        sr, n = presence[combo]
        static, wrist, lang, proprio = combo
        print(f"{static:>7} {wrist:>7} {lang:>7} {proprio:>7} {sr:>12.3f} {n:>6}")


if __name__ == "__main__":
    main()
