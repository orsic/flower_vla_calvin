#!/usr/bin/env python3
"""Compare two LIBERO eval result.csv files task-by-task and overall.

Usage: python scripts/compare_eval_csvs.py <baseline_result.csv> <dropout_result.csv>
"""
import csv
import sys
from collections import defaultdict


def per_task_sr(path):
    by_task = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["task_name"]].append(int(row["success"]))
    return {task: sum(v) / len(v) for task, v in by_task.items()}


def main():
    base_path, drop_path = sys.argv[1], sys.argv[2]
    base_sr = per_task_sr(base_path)
    drop_sr = per_task_sr(drop_path)

    tasks = sorted(set(base_sr) | set(drop_sr))
    print(f"{'task':<70} {'baseline':>10} {'dropout':>10} {'delta':>8}")
    for task in tasks:
        b, d = base_sr.get(task, float("nan")), drop_sr.get(task, float("nan"))
        print(f"{task:<70} {b:>10.2%} {d:>10.2%} {d - b:>+8.2%}")

    avg_b = sum(base_sr.values()) / len(base_sr)
    avg_d = sum(drop_sr.values()) / len(drop_sr)
    print(f"\n{'AVERAGE':<70} {avg_b:>10.2%} {avg_d:>10.2%} {avg_d - avg_b:>+8.2%}")


if __name__ == "__main__":
    main()
