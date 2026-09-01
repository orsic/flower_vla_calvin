#!/usr/bin/env python3
"""Compare two LIBERO eval result.csv files task-by-task and overall.

Each side is filtered to one modality combo (a result.csv may hold rows for several
combos — see eval_modalities in conf/eval_libero*.yaml). Comparing two combos within
the SAME file (e.g. full vs. static-only for one checkpoint) works the same as
comparing two different checkpoints' files.

Usage:
  python scripts/compare_eval_csvs.py <a.csv> <b.csv> \
      [--modalities-a static,wrist,lang] [--modalities-b static,wrist,lang]
"""
import argparse
import csv
from collections import defaultdict

MODALITY_COLUMNS = {
    "static": "use_rgb_static",
    "wrist": "use_rgb_gripper",
    "lang": "use_language",
}
ALL_MODALITIES = set(MODALITY_COLUMNS)


def parse_modalities(spec: str) -> set:
    modalities = {m.strip() for m in spec.split(",") if m.strip()}
    unknown = modalities - ALL_MODALITIES
    if unknown:
        raise ValueError(f"Unknown modalities {unknown}; choose from {ALL_MODALITIES}")
    if not modalities:
        raise ValueError("At least one modality must be selected")
    return modalities


def per_task_sr(path: str, modalities: set) -> dict:
    """Average success rate per task, restricted to rows matching `modalities` exactly."""
    by_task = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            matches = all(
                int(row[col]) == (name in modalities)
                for name, col in MODALITY_COLUMNS.items()
            )
            if matches:
                by_task[row["task_name"]].append(int(row["success"]))
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
    args = parser.parse_args()

    modalities_a = parse_modalities(args.modalities_a)
    modalities_b = parse_modalities(args.modalities_b)
    label_a = "+".join(sorted(modalities_a))
    label_b = "+".join(sorted(modalities_b))

    sr_a = per_task_sr(args.csv_a, modalities_a)
    sr_b = per_task_sr(args.csv_b, modalities_b)

    tasks = sorted(set(sr_a) | set(sr_b))
    print(f"{'task':<70} {label_a:>15} {label_b:>15} {'delta':>8}")
    for task in tasks:
        a, b = sr_a.get(task, float("nan")), sr_b.get(task, float("nan"))
        print(f"{task:<70} {a:>15.2%} {b:>15.2%} {b - a:>+8.2%}")

    avg_a = sum(sr_a.values()) / len(sr_a)
    avg_b = sum(sr_b.values()) / len(sr_b)
    print(f"\n{'AVERAGE':<70} {avg_a:>15.2%} {avg_b:>15.2%} {avg_b - avg_a:>+8.2%}")


if __name__ == "__main__":
    main()
