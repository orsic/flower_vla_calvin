#!/usr/bin/env python3
"""Success rate by physical perturbation severity, for one LIBERO-Plus result.csv.

Groups episodes by perturbation category (already a result.csv column) and, within
each category, by the physical severity axis scripts/perturbation_severity.py recovers
from task_name/bddl_file -- not by task_classification.json's difficulty_level, which
is a coarse upstream annotation that disagrees with the true magnitude for Robot
Initial States roughly two-thirds of the time and is non-monotone for Objects Layout
(see perturbation_severity.py's module docstring). Both are printed side by side so you
can see where they diverge.

Language Instructions (rewrite index is an unordered LLM-rewrite id) and Background
Textures (T indexes a texture identity, not a magnitude) have no severity axis; they
get a categorical sub-type breakdown instead, with an explicit note.

n_eval=1 per LIBERO-Plus task (conf/eval_libero_plus.yaml), so most severity bins hold
only 1-2 episodes -- success rates below are reported as Wilson score intervals, not
point estimates to trust past the interval width.

Usage:
  python scripts/severity_sr.py <result.csv> [--libero-plus-root PATH]
"""
import argparse
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from perturbation_sr import UNCLASSIFIED, modality_combos_present  # noqa: E402
from pid_modality import load_rows  # noqa: E402
from perturbation_severity import (  # noqa: E402
    background_texture_subtype,
    camera_severity,
    light_severity,
    objects_layout_severity,
    robot_initial_state_severity,
    sensor_noise_severity,
)

DEFAULT_LIBERO_PLUS_ROOT = str(Path(__file__).parents[1] / "LIBERO-plus")

UNORDERED_NOTE = {
    "Language Instructions": "no severity axis -- rewrite index is an unordered LLM-rewrite id",
    "Background Textures": "no severity axis -- T indexes a texture identity, not a magnitude",
    UNCLASSIFIED: "no perturbation category on these rows (e.g. an orig LIBERO result.csv)",
}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion -- tighter and better-
    behaved than a normal approximation at the small n (often 1-2) each severity bin
    holds here."""
    if n == 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (center - margin) / denom, (center + margin) / denom


def _severity_bin(row: Dict[str, str], libero_plus_root: str) -> Optional[Tuple[str, tuple]]:
    """(bin_label, sort_key) for one row's physical severity, or None if this
    category/row has no severity axis (unordered category, or a row the parser
    couldn't match -- both are reported explicitly rather than silently dropped)."""
    cat = row.get("task_category") or UNCLASSIFIED
    name = row["task_name"]

    if cat == "Camera Viewpoints":
        result = camera_severity(name)
        if result is None:
            return None
        rotation_deg, scale = result
        if abs(scale - 1.0) > 1e-9:
            # scale is a near-continuous filename field (e.g. 1.19, 1.21, 1.22, ...)
            # -- round to the nearest 0.1x so bins hold more than one episode each,
            # same rationale as the 15-degree rotation buckets above.
            bucket = round(scale, 1)
            return f"scale~{bucket}x", (1, bucket)
        bucket = int(rotation_deg // 15) * 15
        return f"rot~{bucket}-{bucket + 15}deg", (0, bucket)

    if cat == "Robot Initial States":
        rad = robot_initial_state_severity(name)
        if rad is None:
            return None
        return f"{rad:.1f}rad", (rad,)

    if cat == "Sensor Noise":
        result = sensor_noise_severity(name)
        if result is None:
            return None
        corruption, sev = result
        return f"{corruption}_{sev}", (corruption, sev)

    if cat == "Objects Layout":
        result = objects_layout_severity(name)
        if result is None:
            return None
        subtype, level = result
        return f"{subtype}_{level}", (subtype, level)

    if cat == "Light Conditions":
        bddl_path = os.path.join(
            libero_plus_root, "libero", "libero", "bddl_files",
            row["problem_folder"], row["bddl_file"],
        )
        result = light_severity(bddl_path, libero_plus_root)
        if result is None:
            return None
        bucket = round(result["severity"], 1)
        return f"severity~{bucket}", (bucket,)

    return None  # Language Instructions, Background Textures, UNCLASSIFIED


def _print_sr_table(title: str, groups: Dict[Any, List[int]], order: List[Any]) -> None:
    print(f"  -- {title} --")
    print(f"    {'bin':<24} {'success_rate':>12} {'95% CI':>16} {'n':>6}")
    for key in order:
        successes = groups[key]
        n = len(successes)
        sr = sum(successes) / n if n else float("nan")
        lo, hi = wilson_interval(sum(successes), n)
        print(f"    {str(key):<24} {sr:>12.3f} {f'[{lo:.2f}, {hi:.2f}]':>16} {n:>6}")


def report_category(category: str, rows: List[Dict[str, str]], libero_plus_root: str) -> None:
    print(f"\n=== {category} (n={len(rows)}) ===")

    by_difficulty: Dict[str, List[int]] = defaultdict(list)
    for row in rows:
        level = row.get("difficulty_level") or UNCLASSIFIED
        by_difficulty[level].append(int(row["success"]))
    difficulty_order = sorted(
        by_difficulty, key=lambda k: (k == UNCLASSIFIED, k)
    )
    _print_sr_table("by difficulty_level (upstream annotation, see caveat below)", by_difficulty, difficulty_order)

    by_severity: Dict[str, List[int]] = defaultdict(list)
    sort_keys: Dict[str, tuple] = {}
    unparsed = 0
    for row in rows:
        result = _severity_bin(row, libero_plus_root)
        if result is None:
            unparsed += 1
            continue
        label, sort_key = result
        by_severity[label].append(int(row["success"]))
        sort_keys[label] = sort_key

    note = UNORDERED_NOTE.get(category)
    if note is not None:
        print(f"  -- by physical severity: {note} --")
        if category == "Background Textures":
            by_subtype: Dict[str, List[int]] = defaultdict(list)
            for row in rows:
                subtype = background_texture_subtype(row["task_name"]) or UNCLASSIFIED
                by_subtype[subtype].append(int(row["success"]))
            _print_sr_table("sub-type instead", by_subtype, sorted(by_subtype))
    elif by_severity:
        severity_order = sorted(by_severity, key=lambda k: sort_keys[k])
        _print_sr_table("by physical severity", by_severity, severity_order)
        if unparsed:
            print(f"    ({unparsed} row(s) in this category didn't match the expected name pattern -- excluded above)")
    else:
        print("  -- by physical severity: none of this category's rows matched the expected name pattern --")


def report(rows: List[Dict[str, str]], libero_plus_root: str) -> None:
    combos = modality_combos_present(rows)
    if len(combos) > 1:
        print(
            f"WARNING: {len(combos)} distinct modality combos present in this file "
            f"({sorted(combos)}) -- success rates below pool them together."
        )

    by_category: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_category[row.get("task_category") or UNCLASSIFIED].append(row)

    for category in sorted(by_category, key=lambda c: (c == UNCLASSIFIED, c)):
        report_category(category, by_category[category], libero_plus_root)

    print(
        "\nNote: difficulty_level is an upstream (LIBERO-Plus) per-task annotation of "
        "unstated provenance -- it disagrees with the physical magnitude computed above "
        "for several categories (most starkly Robot Initial States and Objects Layout), "
        "so treat it as informational, not as this model's ground-truth severity."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_csv")
    parser.add_argument(
        "--libero-plus-root", default=DEFAULT_LIBERO_PLUS_ROOT,
        help="Path to the LIBERO-Plus checkout (only needed to resolve Light Conditions scene XMLs).",
    )
    args = parser.parse_args()

    report(load_rows(args.result_csv), args.libero_plus_root)


if __name__ == "__main__":
    main()
