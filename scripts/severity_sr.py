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
  python scripts/severity_sr.py <result.csv> [--libero-plus-root PATH] [--csv OUT.csv]
"""
import argparse
import csv
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

# Severity-axis bin label for a row _severity_bin() couldn't parse -- reported as its
# own bin (successes/n included) rather than silently dropped, so a category's n
# reconciles across the difficulty_level and severity axes even when parsing misses.
UNPARSED = "<unparsed>"

CSV_COLUMNS = ["category", "axis", "bin", "successes", "n", "success_rate", "ci_low", "ci_high", "note"]


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
    lo, hi = (center - margin) / denom, (center + margin) / denom
    # Wilson bounds are mathematically in [0, 1]; clamp away the tiny negative/>1
    # floating-point overshoot (e.g. an all-failure n=10 bin prints -0.00 otherwise).
    return max(0.0, lo), min(1.0, hi)


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


def _record(category: str, axis: str, bin_label: Any, successes: List[int], note: str) -> Dict[str, Any]:
    n = len(successes)
    total = sum(successes)
    sr = total / n if n else float("nan")
    lo, hi = wilson_interval(total, n)
    return {
        "category": category,
        "axis": axis,
        "bin": str(bin_label),
        "successes": total,
        "n": n,
        "success_rate": sr,
        "ci_low": lo,
        "ci_high": hi,
        "note": note,
    }


def collect(rows: List[Dict[str, str]], libero_plus_root: str) -> List[Dict[str, Any]]:
    """One record per (category, axis, bin) -- the return-only counterpart of
    report()/report_category(), which render this same data. `note` carries
    UNORDERED_NOTE[category] on every record of a category with no severity axis
    (Language Instructions, Background Textures, UNCLASSIFIED), empty string
    otherwise. A category with a severity axis but rows _severity_bin() couldn't
    parse gets an explicit UNPARSED bin instead of silently dropping them."""
    records: List[Dict[str, Any]] = []

    by_category: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_category[row.get("task_category") or UNCLASSIFIED].append(row)

    for category in sorted(by_category, key=lambda c: (c == UNCLASSIFIED, c)):
        cat_rows = by_category[category]
        note = UNORDERED_NOTE.get(category, "")

        by_difficulty: Dict[str, List[int]] = defaultdict(list)
        for row in cat_rows:
            level = row.get("difficulty_level") or UNCLASSIFIED
            by_difficulty[level].append(int(row["success"]))
        for level in sorted(by_difficulty, key=lambda k: (k == UNCLASSIFIED, k)):
            records.append(_record(category, "difficulty_level", level, by_difficulty[level], note))

        if category == "Background Textures":
            by_subtype: Dict[str, List[int]] = defaultdict(list)
            for row in cat_rows:
                subtype = background_texture_subtype(row["task_name"]) or UNCLASSIFIED
                by_subtype[subtype].append(int(row["success"]))
            for subtype in sorted(by_subtype):
                records.append(_record(category, "subtype", subtype, by_subtype[subtype], note))
        elif category not in UNORDERED_NOTE:
            by_severity: Dict[str, List[int]] = defaultdict(list)
            sort_keys: Dict[str, tuple] = {}
            unparsed: List[int] = []
            for row in cat_rows:
                result = _severity_bin(row, libero_plus_root)
                if result is None:
                    unparsed.append(int(row["success"]))
                    continue
                label, sort_key = result
                by_severity[label].append(int(row["success"]))
                sort_keys[label] = sort_key
            for label in sorted(by_severity, key=lambda k: sort_keys[k]):
                records.append(_record(category, "severity", label, by_severity[label], note))
            if unparsed:
                records.append(_record(category, "severity", UNPARSED, unparsed, note))

    return records


def write_csv(path: Any, records: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _print_table(title: str, records: List[Dict[str, Any]]) -> None:
    print(f"  -- {title} --")
    print(f"    {'bin':<24} {'success_rate':>12} {'95% CI':>16} {'n':>6}")
    for r in records:
        ci = f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
        print(f"    {r['bin']:<24} {r['success_rate']:>12.3f} {ci:>16} {r['n']:>6}")


def report_category(category: str, cat_rows: List[Dict[str, str]], records: List[Dict[str, Any]]) -> None:
    print(f"\n=== {category} (n={len(cat_rows)}) ===")

    difficulty_records = [r for r in records if r["axis"] == "difficulty_level"]
    _print_table("by difficulty_level (upstream annotation, see caveat below)", difficulty_records)

    note = UNORDERED_NOTE.get(category)
    if note is not None:
        print(f"  -- by physical severity: {note} --")
        if category == "Background Textures":
            subtype_records = [r for r in records if r["axis"] == "subtype"]
            _print_table("sub-type instead", subtype_records)
        return

    severity_records = [r for r in records if r["axis"] == "severity"]
    normal = [r for r in severity_records if r["bin"] != UNPARSED]
    unparsed_rec = next((r for r in severity_records if r["bin"] == UNPARSED), None)
    if normal:
        _print_table("by physical severity", normal)
        if unparsed_rec:
            print(f"    ({unparsed_rec['n']} row(s) in this category didn't match the expected name pattern -- excluded above)")
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

    records_by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in collect(rows, libero_plus_root):
        records_by_category[record["category"]].append(record)

    for category in sorted(by_category, key=lambda c: (c == UNCLASSIFIED, c)):
        report_category(category, by_category[category], records_by_category[category])

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
    parser.add_argument(
        "--csv", default=None,
        help="Also write the full (category, axis, bin) breakdown to this path as CSV.",
    )
    args = parser.parse_args()

    rows = load_rows(args.result_csv)
    report(rows, args.libero_plus_root)
    if args.csv:
        write_csv(args.csv, collect(rows, args.libero_plus_root))


if __name__ == "__main__":
    main()
