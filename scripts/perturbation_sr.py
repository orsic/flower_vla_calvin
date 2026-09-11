#!/usr/bin/env python3
"""Per-perturbation-category success rate for a LIBERO-Plus result.csv.

Each row already carries the LIBERO-Plus perturbation category it belongs to (written
by flower_eval_libero.py from task_classification.json -- see eval_records.py's module
docstring for the CSV layout). This just groups by that column; no LIBERO import needed.

n_eval=1 for every LIBERO-Plus task (conf/eval_libero_plus.yaml), so a per-episode
(micro) average here equals flower_eval_libero.py's per-task (macro) aggregate_by_category.

Usage:
  python scripts/perturbation_sr.py <result.csv>
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from compare_eval_csvs import MODALITY_COLUMNS  # noqa: E402
from pid_modality import load_rows  # noqa: E402

UNCLASSIFIED = "(unclassified)"


def category_sr(rows: List[Dict]) -> Dict[str, Tuple[float, int]]:
    """Success rate + count per task_category. Rows with an empty task_category
    (e.g. a LIBERO orig CSV, which carries no perturbation categories) group under
    UNCLASSIFIED rather than being silently dropped."""
    by_cat = defaultdict(list)
    for row in rows:
        cat = row.get("task_category") or UNCLASSIFIED
        by_cat[cat].append(int(row["success"]))
    return {cat: (sum(v) / len(v), len(v)) for cat, v in by_cat.items()}


def modality_combos_present(rows: List[Dict]) -> set:
    """Distinct (use_rgb_static, use_rgb_gripper, use_language, use_proprio) combos in
    the data. A LIBERO-Plus result.csv is expected to hold exactly one (the full
    combo) -- more means per-category success rates below would silently pool
    different modality ablations together."""
    cols = [MODALITY_COLUMNS[m] for m in ("static", "wrist", "lang", "proprio")]
    return {tuple(int(row.get(col) or 0) for col in cols) for row in rows}


def report(rows: List[Dict]) -> None:
    combos = modality_combos_present(rows)
    if len(combos) > 1:
        print(
            f"WARNING: {len(combos)} distinct modality combos present in this file "
            f"({sorted(combos)}) -- success rates below pool them together."
        )

    print(f"{'category':<24} {'success_rate':>12} {'n':>6}")
    sr = category_sr(rows)
    total_successes = 0
    total_n = 0
    for cat in sorted(sr):
        s, n = sr[cat]
        print(f"{cat:<24} {s:>12.3f} {n:>6}")
        total_successes += s * n
        total_n += n
    overall = total_successes / total_n if total_n else float("nan")
    print(f"{'OVERALL':<24} {overall:>12.3f} {total_n:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_csv")
    args = parser.parse_args()

    report(load_rows(args.result_csv))


if __name__ == "__main__":
    main()
