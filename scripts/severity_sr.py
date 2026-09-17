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

Every group is also compared against an init-state-matched LIBERO original baseline,
when a `libero_orig.csv`/`--orig-csv` is supplied: LIBERO-Plus's own
get_task_init_states (LIBERO-plus/libero/libero/benchmark/__init__.py) strips the
perturbation suffix off a task_name and loads the *original* task's init file, so
(except Objects Layout, see NEWOBJ_INIT_NOTE below) every Plus episode runs original
init state 0 of its base task under the same modality combo. The baseline for a group
is that set of (combo, base_task) orig episodes, deduplicated by task -- its n is the
number of distinct base tasks the group covers, not the number of Plus rows in it, so
it is usually small and its Wilson CI correspondingly wide.

Pairing on init_state_idx==0 is correct, not just convenient, even though a Plus task's
own task_name/init_states_file carries its own trailing number (e.g. "_table_5",
"_initstate_50") -- that number is the perturbation's own parameter id (view sample,
robot qpos-offset sample, noise instance, texture/light id), applied by
LIBERO-plus/libero/libero/envs/env_wrapper.py at env-construction time (parsed straight
out of the bddl filename), not an index into the init-states array. n_eval=1 for every
Plus task means flower_eval_libero.py always loads array index 0 regardless, and for
every category but Objects Layout that array is the *unperturbed* original task's own
init file -- so index 0 there really is the same underlying layout as orig's
init_state_idx=0 episode, confirmed independently of just this convention.

Usage:
  python scripts/severity_sr.py <result.csv> [--libero-plus-root PATH] \
      [--orig-csv ORIG_RESULT.csv] [--csv OUT.csv]
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from compare_eval_csvs import MODALITY_COLUMNS  # noqa: E402
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

# Objects Layout's init states are generated fresh into libero_newobj/, not derived from
# the original task's own init file (unlike every other category) -- its paired baseline
# is therefore the task's *nominal* layout, not the same init state under a stripped
# suffix. Still the meaningful contrast for a layout perturbation, but flagged.
NEWOBJ_INIT_NOTE = (
    "paired orig episode is the task's nominal layout -- Objects Layout init states come "
    "from LIBERO-Plus's libero_newobj/, not the original task's init file"
)

# KNOWN LIBERO-PLUS UPSTREAM LIMITATION (verified against a real checkpoint, not just code
# reading -- see scripts/debug_robot_initstate_qpos*.py and
# scripts/debug_robot_initstate_severity_rollout.py): Robot Initial States' perturbation is
# a robot *class* substitution (Panda -> MountedPanda{N}/OnTheGroundPanda{N}, N in 1..500,
# see LIBERO-plus/libero/libero/envs/robots/new_init.py) whose only effect is a perturbed
# init_qpos, applied by robosuite's robot.reset(). flower_eval_libero.py then calls
# env.set_init_state() with the array get_task_init_states() returns for this category --
# the *unperturbed* base task's own recorded state (get_task_init_states() strips the
# "_view_..." suffix for this category) -- which overwrites that qpos via a full
# MuJoCo-state restore. Confirmed empirically: a fresh rollout of the same checkpoint on
# the same task gives a *different* per-severity-band success pattern than the original
# eval, with no reproducible trend -- consistent with every episode actually running from
# the same unperturbed state, not with a real physical effect. Every LIBERO-Plus reference
# script that calls set_init_state() this way (e.g.
# LIBERO-plus/benchmark_scripts/render_single_task.py) has the identical gap, so this is a
# LIBERO-Plus upstream issue, not something introduced by flower_eval_libero.py's usage of
# it -- and it is left unfixed here deliberately, to keep results comparable with other
# work evaluating on this benchmark as shipped.
ROBOT_INITSTATE_NOTE = (
    "known LIBERO-Plus upstream limitation -- this category's perturbed robot init_qpos is "
    "silently discarded by a later env.set_init_state() call (see scripts/severity_sr.py's "
    "ROBOT_INITSTATE_NOTE comment), so every episode currently runs from the unperturbed "
    "base task's own state; the severity bins below do not reflect a real physical "
    "perturbation. Left as-is to stay comparable with other work on this benchmark."
)

# Severity-axis bin label for a row _severity_bin() couldn't parse -- reported as its
# own bin (successes/n included) rather than silently dropped, so a category's n
# reconciles across the difficulty_level and severity axes even when parsing misses.
UNPARSED = "<unparsed>"

CSV_COLUMNS = [
    "category", "axis", "bin", "successes", "n", "success_rate", "ci_low", "ci_high",
    "orig_successes", "orig_n", "orig_success_rate", "orig_ci_low", "orig_ci_high",
    "note",
]


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


def _combo(row: Dict[str, str]) -> Tuple[int, int, int, int]:
    """Same (static, wrist, lang, proprio) tuple perturbation_sr.modality_combos_present
    builds, for one row -- the pairing key must match on modality ablation too, since an
    orig result.csv can hold all 14 combos."""
    cols = [MODALITY_COLUMNS[m] for m in ("static", "wrist", "lang", "proprio")]
    return tuple(int(row.get(col) or 0) for col in cols)


def base_task(task_name: str, orig_task_names: Sequence[str]) -> Optional[str]:
    """The original LIBERO task a LIBERO-Plus task_name was generated from, by longest
    prefix match (every Plus name is <orig task_name>_<perturbation suffix>, e.g.
    "..._table_1", "..._initstate_50", "..._rewrite_3"). None when no original task
    name is a prefix -- such a row is excluded from the paired baseline rather than
    guessed at. The "_" boundary keeps one task from matching as a spurious prefix of
    another task's name."""
    candidates = [t for t in orig_task_names if task_name == t or task_name.startswith(t + "_")]
    if not candidates:
        return None
    return max(candidates, key=len)


def orig_init0_index(orig_rows: List[Dict[str, str]]) -> Dict[Tuple[Tuple[int, int, int, int], str], int]:
    """(modality_combo, task_name) -> success, for orig rows at init_state_idx 0 only --
    the LIBERO original episode each LIBERO-Plus episode (Objects Layout excepted) was
    generated from."""
    index: Dict[Tuple[Tuple[int, int, int, int], str], int] = {}
    for row in orig_rows:
        try:
            if int(row.get("init_state_idx", -1)) != 0:
                continue
        except ValueError:
            continue
        index[(_combo(row), row["task_name"])] = int(row["success"])
    return index


def paired_baseline(
    plus_rows: List[Dict[str, str]],
    orig_index: Optional[Dict[Tuple[Tuple[int, int, int, int], str], int]],
    orig_task_names: Optional[Sequence[str]],
) -> Dict[str, Any]:
    """The five orig_* CSV_COLUMNS for one group of LIBERO-Plus rows: the distinct
    (modality_combo, base_task) orig episodes the group's rows were generated from,
    looked up in orig_index. Deduplicated by task, so orig_n is the number of distinct
    base tasks the group covers, not the number of plus_rows in it. Empty strings for
    all five when orig_index is None (no libero_orig.csv available)."""
    if orig_index is None:
        return {
            "orig_successes": "", "orig_n": "", "orig_success_rate": "",
            "orig_ci_low": "", "orig_ci_high": "",
        }

    pairs = set()
    for row in plus_rows:
        task = base_task(row["task_name"], orig_task_names)
        if task is None:
            continue
        key = (_combo(row), task)
        if key in orig_index:
            pairs.add(key)

    n = len(pairs)
    total = sum(orig_index[key] for key in pairs)
    sr = total / n if n else float("nan")
    lo, hi = wilson_interval(total, n)
    return {
        "orig_successes": total,
        "orig_n": n,
        "orig_success_rate": sr,
        "orig_ci_low": lo,
        "orig_ci_high": hi,
    }


def _record(
    category: str,
    axis: str,
    bin_label: Any,
    rows: List[Dict[str, str]],
    note: str,
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    successes = [int(r["success"]) for r in rows]
    n = len(successes)
    total = sum(successes)
    sr = total / n if n else float("nan")
    lo, hi = wilson_interval(total, n)
    record = {
        "category": category,
        "axis": axis,
        "bin": str(bin_label),
        "successes": total,
        "n": n,
        "success_rate": sr,
        "ci_low": lo,
        "ci_high": hi,
    }
    record.update(baseline)
    record["note"] = note
    return record


def collect(
    rows: List[Dict[str, str]],
    libero_plus_root: str,
    orig_rows: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """One record per (category, axis, bin) -- the return-only counterpart of
    report()/report_category(), which render this same data. `note` carries
    UNORDERED_NOTE[category] on every record of a category with no severity axis
    (Language Instructions, Background Textures, UNCLASSIFIED), empty string
    otherwise. A category with a severity axis but rows _severity_bin() couldn't
    parse gets an explicit UNPARSED bin instead of silently dropping them.

    Also emits one axis="total"/bin="ALL" record per category -- the category-level
    counterpart of the per-bin paired baseline described in this module's docstring.
    When `orig_rows` is given, every record additionally carries the init-state-matched
    LIBERO original baseline for its group (orig_* CSV_COLUMNS); UNCLASSIFIED never
    gets one (pairing it against itself would be meaningless)."""
    records: List[Dict[str, Any]] = []

    by_category: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_category[row.get("task_category") or UNCLASSIFIED].append(row)

    orig_index = orig_init0_index(orig_rows) if orig_rows is not None else None
    orig_task_names = sorted({r["task_name"] for r in orig_rows}) if orig_rows is not None else None

    for category in sorted(by_category, key=lambda c: (c == UNCLASSIFIED, c)):
        cat_rows = by_category[category]
        cat_index = orig_index if category != UNCLASSIFIED else None
        cat_task_names = orig_task_names if category != UNCLASSIFIED else None
        if category == "Objects Layout" and cat_index is not None:
            note = NEWOBJ_INIT_NOTE
        elif category == "Robot Initial States":
            note = ROBOT_INITSTATE_NOTE
        else:
            note = UNORDERED_NOTE.get(category, "")

        records.append(
            _record(category, "total", "ALL", cat_rows, note, paired_baseline(cat_rows, cat_index, cat_task_names))
        )

        by_difficulty: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in cat_rows:
            level = row.get("difficulty_level") or UNCLASSIFIED
            by_difficulty[level].append(row)
        for level in sorted(by_difficulty, key=lambda k: (k == UNCLASSIFIED, k)):
            group = by_difficulty[level]
            records.append(
                _record(category, "difficulty_level", level, group, note, paired_baseline(group, cat_index, cat_task_names))
            )

        if category == "Background Textures":
            by_subtype: Dict[str, List[Dict[str, str]]] = defaultdict(list)
            for row in cat_rows:
                subtype = background_texture_subtype(row["task_name"]) or UNCLASSIFIED
                by_subtype[subtype].append(row)
            for subtype in sorted(by_subtype):
                group = by_subtype[subtype]
                records.append(
                    _record(category, "subtype", subtype, group, note, paired_baseline(group, cat_index, cat_task_names))
                )
        elif category not in UNORDERED_NOTE:
            by_severity: Dict[str, List[Dict[str, str]]] = defaultdict(list)
            sort_keys: Dict[str, tuple] = {}
            unparsed: List[Dict[str, str]] = []
            for row in cat_rows:
                result = _severity_bin(row, libero_plus_root)
                if result is None:
                    unparsed.append(row)
                    continue
                label, sort_key = result
                by_severity[label].append(row)
                sort_keys[label] = sort_key
            for label in sorted(by_severity, key=lambda k: sort_keys[k]):
                group = by_severity[label]
                records.append(
                    _record(category, "severity", label, group, note, paired_baseline(group, cat_index, cat_task_names))
                )
            if unparsed:
                records.append(
                    _record(category, "severity", UNPARSED, unparsed, note, paired_baseline(unparsed, cat_index, cat_task_names))
                )

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
    show_baseline = bool(records) and records[0]["orig_n"] != ""
    print(f"  -- {title} --")
    header = f"    {'bin':<24} {'success_rate':>12} {'95% CI':>16} {'n':>6}"
    if show_baseline:
        header += f"  |  {'orig_sr':>12} {'orig 95% CI':>16} {'orig_n':>6}"
    print(header)
    for r in records:
        ci = f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
        line = f"    {r['bin']:<24} {r['success_rate']:>12.3f} {ci:>16} {r['n']:>6}"
        if show_baseline:
            orig_ci = f"[{r['orig_ci_low']:.2f}, {r['orig_ci_high']:.2f}]"
            line += f"  |  {r['orig_success_rate']:>12.3f} {orig_ci:>16} {r['orig_n']:>6}"
        print(line)


def report_category(category: str, cat_rows: List[Dict[str, str]], records: List[Dict[str, Any]]) -> None:
    print(f"\n=== {category} (n={len(cat_rows)}) ===")

    total_record = next((r for r in records if r["axis"] == "total"), None)
    if total_record is not None and total_record["orig_n"] != "":
        ci = f"[{total_record['ci_low']:.2f}, {total_record['ci_high']:.2f}]"
        orig_ci = f"[{total_record['orig_ci_low']:.2f}, {total_record['orig_ci_high']:.2f}]"
        print(
            f"  overall (paired vs. orig): plus {total_record['success_rate']:.3f} {ci} n={total_record['n']}"
            f"  |  orig {total_record['orig_success_rate']:.3f} {orig_ci} n={total_record['orig_n']}"
        )

    difficulty_records = [r for r in records if r["axis"] == "difficulty_level"]
    _print_table("by difficulty_level (upstream annotation, see caveat below)", difficulty_records)

    note = UNORDERED_NOTE.get(category)
    if note is not None:
        print(f"  -- by physical severity: {note} --")
        if category == "Background Textures":
            subtype_records = [r for r in records if r["axis"] == "subtype"]
            _print_table("sub-type instead", subtype_records)
        return

    if category == "Robot Initial States":
        print(f"  ** WARNING: {ROBOT_INITSTATE_NOTE} **")

    severity_records = [r for r in records if r["axis"] == "severity"]
    normal = [r for r in severity_records if r["bin"] != UNPARSED]
    unparsed_rec = next((r for r in severity_records if r["bin"] == UNPARSED), None)
    if normal:
        _print_table("by physical severity", normal)
        if unparsed_rec:
            print(f"    ({unparsed_rec['n']} row(s) in this category didn't match the expected name pattern -- excluded above)")
    else:
        print("  -- by physical severity: none of this category's rows matched the expected name pattern --")


def report(
    rows: List[Dict[str, str]],
    libero_plus_root: str,
    orig_rows: Optional[List[Dict[str, str]]] = None,
) -> None:
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
    for record in collect(rows, libero_plus_root, orig_rows=orig_rows):
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
        "--orig-csv", default=None,
        help="The matching LIBERO original result.csv; adds an init-state-matched baseline column to every row.",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Also write the full (category, axis, bin) breakdown to this path as CSV.",
    )
    args = parser.parse_args()

    rows = load_rows(args.result_csv)
    orig_rows = load_rows(args.orig_csv) if args.orig_csv else None
    report(rows, args.libero_plus_root, orig_rows=orig_rows)
    if args.csv:
        write_csv(args.csv, collect(rows, args.libero_plus_root, orig_rows=orig_rows))


if __name__ == "__main__":
    main()
