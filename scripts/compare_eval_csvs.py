#!/usr/bin/env python3
"""Compare two LIBERO eval result.csv files task-by-task and overall.

Each side is filtered to one modality combo (a result.csv may hold rows for several
combos — see eval_modalities in conf/eval_libero*.yaml). Comparing two combos within
the SAME file (e.g. full vs. static-only for one checkpoint) works the same as
comparing two different checkpoints' files.

Usage:
  python scripts/compare_eval_csvs.py <a.csv> <b.csv> \
      [--modalities-a static,wrist,lang] [--modalities-b static,wrist,lang]

Comparing a LIBERO-Plus file against an original-LIBERO file needs two more flags:
`--base-task` keys both sides on eval_records.base_task_name(task_name) instead of the
raw name, so a Plus file's per-variant task names (e.g. every "..._language_N" rewrite
of one base task) collapse onto the same key as that task's original-LIBERO row, and
`--init-state-idx 0` restricts the original-LIBERO side to init_state_idx==0, the only
row LIBERO-Plus's own get_task_init_states() ever draws from. With both flags, the
AVERAGE line becomes an unweighted mean over the shared base tasks on both sides —
i.e. task-weighted, not episode-weighted — which is what makes it comparable to a
LIBERO-Plus category's typically-unequal per-task episode counts (see the plan/README
this lands with):

  python scripts/compare_eval_csvs.py plus_libero_10_no_lang/result.csv orig_libero_10/result.csv \
      --base-task --init-state-idx 0 \
      --modalities-a static,wrist,proprio --modalities-b static,wrist,proprio

Restricting to one LIBERO-Plus perturbation category (e.g. to check a category that's
physically inert once its modality is withheld against the original-LIBERO baseline)
needs a per-side flag rather than a shared one: an original-LIBERO file's rows all carry
an empty task_category, so a single flag applied to both sides would filter that side to
nothing. `--task-category-a` (the LIBERO-Plus side, typically) leaves `--task-category-b`
unset:

  python scripts/compare_eval_csvs.py plus_libero_10_no_static/result.csv orig_libero_10/result.csv \
      --base-task --init-state-idx 0 --task-category-a "Camera Viewpoints" \
      --modalities-a wrist,lang,proprio --modalities-b wrist,lang,proprio
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
from flower.evaluation.eval_records import base_task_name  # noqa: E402

MODALITY_COLUMNS = {
    "static": "use_rgb_static",
    "wrist": "use_rgb_gripper",
    "lang": "use_language",
    "proprio": "use_proprio",
}
ALL_MODALITIES = set(MODALITY_COLUMNS)


def _flag(row: dict, col: str) -> bool:
    # use_proprio is missing entirely from result.csv files written before it existed
    # (see eval_records.py's module docstring) — read those rows as proprio-absent,
    # which is factually correct since every such eval ran with use_proprio=false.
    return bool(int(row.get(col) or 0))


def parse_modalities(spec: str) -> set:
    modalities = {m.strip() for m in spec.split(",") if m.strip()}
    unknown = modalities - ALL_MODALITIES
    if unknown:
        raise ValueError(f"Unknown modalities {unknown}; choose from {ALL_MODALITIES}")
    if not modalities:
        raise ValueError("At least one modality must be selected")
    return modalities


def per_task_successes(
    path: str,
    modalities: set,
    base_task: bool = False,
    init_state_idx: Optional[int] = None,
    task_category: Optional[str] = None,
) -> dict:
    """{task_key: [success, ...]} for rows matching `modalities` exactly.

    base_task=True keys on eval_records.base_task_name(task_name) instead of the raw
    name, so a LIBERO-Plus file's per-variant names (e.g. every "..._language_N"
    rewrite of one base task) collapse onto the same key as an original-LIBERO file's
    row for that task -- what makes the two files comparable at all.
    init_state_idx, given, restricts to rows with that exact init_state_idx (LIBERO-
    Plus rows are always 0 -- see README; this is how an original-LIBERO file, which
    has one row per init state, is narrowed to the one LIBERO-Plus itself draws from).
    task_category, given, restricts to rows with that exact task_category -- an
    original-LIBERO file's rows all have an empty task_category, so this is meant for
    the LIBERO-Plus side only (see module docstring).
    """
    by_task = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            matches = all(
                _flag(row, col) == (name in modalities)
                for name, col in MODALITY_COLUMNS.items()
            )
            if not matches:
                continue
            if init_state_idx is not None and int(row.get("init_state_idx", -1)) != init_state_idx:
                continue
            if task_category is not None and row.get("task_category") != task_category:
                continue
            key = base_task_name(row["task_name"]) if base_task else row["task_name"]
            by_task[key].append(int(row["success"]))
    return by_task


def per_task_sr(
    path: str,
    modalities: set,
    base_task: bool = False,
    init_state_idx: Optional[int] = None,
    task_category: Optional[str] = None,
) -> dict:
    """Average success rate per task, restricted to rows matching `modalities` exactly."""
    by_task = per_task_successes(path, modalities, base_task, init_state_idx, task_category)
    if not by_task:
        raise SystemExit(
            f"No rows in {path} match modalities={sorted(modalities)} "
            f"(has this combo been evaluated?)"
        )
    return {task: sum(v) / len(v) for task, v in by_task.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_a")
    parser.add_argument("csv_b")
    parser.add_argument("--modalities-a", default="static,wrist,lang")
    parser.add_argument("--modalities-b", default="static,wrist,lang")
    parser.add_argument(
        "--base-task", action="store_true",
        help="Key both sides on the base (perturbation-suffix-stripped) task name, so a "
             "LIBERO-Plus file's variants align with an original-LIBERO file's tasks.",
    )
    parser.add_argument(
        "--init-state-idx", type=int, default=None,
        help="Restrict both sides to this init_state_idx (LIBERO-Plus rows are always 0; "
             "use 0 to make an original-LIBERO file's per-init-state rows comparable).",
    )
    parser.add_argument(
        "--task-category-a", default=None,
        help="Restrict csv_a to this LIBERO-Plus task_category (e.g. 'Camera Viewpoints'). "
             "Per-side, not shared with --task-category-b, since an original-LIBERO file's "
             "rows all carry an empty task_category.",
    )
    parser.add_argument(
        "--task-category-b", default=None,
        help="Restrict csv_b to this LIBERO-Plus task_category. See --task-category-a.",
    )
    args = parser.parse_args()

    modalities_a = parse_modalities(args.modalities_a)
    modalities_b = parse_modalities(args.modalities_b)
    label_a = "+".join(sorted(modalities_a))
    label_b = "+".join(sorted(modalities_b))

    successes_a = per_task_successes(
        args.csv_a, modalities_a, args.base_task, args.init_state_idx, args.task_category_a
    )
    successes_b = per_task_successes(
        args.csv_b, modalities_b, args.base_task, args.init_state_idx, args.task_category_b
    )
    if not successes_a:
        raise SystemExit(f"No rows in {args.csv_a} match modalities={sorted(modalities_a)}")
    if not successes_b:
        raise SystemExit(f"No rows in {args.csv_b} match modalities={sorted(modalities_b)}")
    sr_a = {task: sum(v) / len(v) for task, v in successes_a.items()}
    sr_b = {task: sum(v) / len(v) for task, v in successes_b.items()}

    tasks = sorted(set(sr_a) | set(sr_b))
    print(f"{'task':<70} {label_a:>15} {'n_a':>5} {label_b:>15} {'n_b':>5} {'delta':>8}")
    for task in tasks:
        a, b = sr_a.get(task, float("nan")), sr_b.get(task, float("nan"))
        n_a, n_b = len(successes_a.get(task, [])), len(successes_b.get(task, []))
        print(f"{task:<70} {a:>15.2%} {n_a:>5} {b:>15.2%} {n_b:>5} {b - a:>+8.2%}")

    # Unweighted mean over task keys -- with --base-task this is task-weighted, not
    # episode-weighted, which is what makes it comparable to a LIBERO-Plus category's
    # typically-unequal per-task episode counts (module docstring).
    avg_a = sum(sr_a.values()) / len(sr_a)
    avg_b = sum(sr_b.values()) / len(sr_b)
    print(f"\n{'AVERAGE':<70} {avg_a:>15.2%} {'':>5} {avg_b:>15.2%} {'':>5} {avg_b - avg_a:>+8.2%}")


if __name__ == "__main__":
    main()
