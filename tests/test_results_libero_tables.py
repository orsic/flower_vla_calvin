"""Tests for scripts/results_libero_tables.py: severity-bin merging (the doubled-range
tables in docs/results.md's "LIBERO-Plus by perturbation size") and the masked-input
LIBERO-Plus breakdown (masked_breakdown, masked_severity)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import results_libero_tables  # noqa: E402
from results_libero_tables import (  # noqa: E402
    Artifacts,
    masked_breakdown,
    masked_severity,
    merge_bins,
    merged_counts,
    size_table,
)


@pytest.mark.parametrize(
    "bins, expected",
    [
        (["a"], [["a"]]),
        (["a", "b"], [["a", "b"]]),
        (["a", "b", "c"], [["a", "b", "c"]]),
        (list("123456"), [["1", "2"], ["3", "4"], ["5", "6"]]),
        (list("12345"), [["1", "2"], ["3", "4", "5"]]),
        (list("123456789"), [["1", "2"], ["3", "4"], ["5", "6"], ["7", "8", "9"]]),
    ],
)
def test_merge_bins_pairs_from_low_end_and_folds_odd_last_bin(bins, expected):
    assert merge_bins(bins) == expected


def test_merge_bins_covers_every_bin_once():
    bins = [f"b{i}" for i in range(10)]
    groups = merge_bins(bins)
    assert [b for g in groups for b in g] == bins
    assert len(groups) == 5


def test_merged_counts_sum_counts_then_divide():
    d = pd.DataFrame(
        {
            "successes": [9, 1, 3],
            "n": [10, 10, 4],
            "orig_successes": [5, 4, 2],
            "orig_n": [5, 5, 2],
        },
        index=["x", "y", "z"],
    )
    m = merged_counts(d, [["x", "y"], ["z"]], ["x–y", "z"])
    assert list(m.index) == ["x–y", "z"]
    assert m.loc["x–y", "n"] == 20
    assert m.loc["x–y", "successes"] / m.loc["x–y", "n"] == pytest.approx(0.5)  # not mean(0.9, 0.1)
    assert m.loc["z", "orig_successes"] == 2


def _sev(successes):
    """A severity_sr.csv slice: one category, two size bins, 10 tasks each."""
    return pd.DataFrame(
        {
            "category": ["Light Conditions"] * 2,
            "bin": ["severity~0.1", "severity~0.2"],
            "successes": successes,
            "n": [10, 10],
            "orig_successes": [9, 9],
            "orig_n": [10, 10],
        }
    )


def test_size_table_pairs_with_original_libero_by_default():
    sev = {"m": {42: _sev([5, 3]), 5: _sev([7, 1])}}
    tab = size_table(sev, "Light Conditions", lambda b: b, doubled=False)
    assert list(tab.columns) == ["tasks", "m", "orig, paired"]
    assert tab.loc["severity~0.1", "m"] == "60.0 ± 14.1"
    assert tab.loc["severity~0.1", "orig, paired"] == "90"


def test_size_table_without_pairing_omits_the_pooled_original_column():
    # models withheld on different inputs pair with different LIBERO-10 combinations,
    # so their original-LIBERO counts must not be pooled into one column
    sev = {"a": {42: _sev([5, 3])}, "b": {42: _sev([1, 0])}}
    tab = size_table(sev, "Light Conditions", lambda b: b, doubled=True, paired=False)
    assert list(tab.columns) == ["tasks", "a", "b"]
    assert tab.loc["severity~0.1-0.2", "tasks"] == 20
    assert tab.loc["severity~0.1-0.2", "a"] == "40.0"


def _plus_csv(path, successes, combo=(1, 1, 1, 1)):
    """A small LIBERO-Plus result.csv: one row per task, two categories, two levels."""
    rows = []
    for i, ok in enumerate(successes):
        rows.append(
            {
                "suite": "libero_10",
                "task_idx": i,
                "task_name": f"t{i}",
                "episode_idx": 0,
                "task_category": "Camera Viewpoints" if i < 2 else "Sensor Noise",
                "difficulty_level": 1 if i % 2 == 0 else 2,
                "use_rgb_static": combo[0],
                "use_rgb_gripper": combo[1],
                "use_language": combo[2],
                "use_proprio": combo[3],
                "eval_timestamp": "2026-11-01_00-00-00",
                "success": ok,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_masked_breakdown_removed_vs_masked_by_category_and_level(tmp_path):
    a = Artifacts(tmp_path)
    runs = {42: "r42", 5: "r5"}
    root = tmp_path / results_libero_tables.PROJECT
    # removed: static camera withheld; masked: static camera's frames replaced by their mean pixel
    _plus_csv(root / "r42" / "libero_plus_no_static.csv", [1, 0, 0, 0], combo=(0, 1, 1, 1))
    _plus_csv(root / "r5" / "libero_plus_no_static.csv", [0, 0, 1, 0], combo=(0, 1, 1, 1))
    _plus_csv(root / "r42" / "masked_static" / "libero_plus.csv", [1, 1, 0, 0])
    _plus_csv(root / "r5" / "masked_static" / "libero_plus.csv", [0, 1, 1, 1])
    b = masked_breakdown(a, runs, "static", n_tasks=4)
    assert b["removed"]["all"] == "25.0 ± 0.0"
    assert b["masked"]["all"] == "62.5 ± 17.7"
    assert b["masked"]["Camera Viewpoints"] == "75.0 ± 35.4"  # (100, 50)
    assert b["removed"]["level 2"] == "0.0 ± 0.0"
    # paired over both seeds: masked-only r42 t1, r5 t1, r5 t3; removed-only none
    assert b["only masked / only removed"] == "3 / 0"


def test_masked_severity_computes_once_then_reuses(tmp_path, monkeypatch):
    a = Artifacts(tmp_path)
    plus = tmp_path / results_libero_tables.PROJECT / "r42" / "masked_proprio" / "libero_plus.csv"
    _plus_csv(plus, [1, 0, 1, 0])
    calls = []

    def fake_collect(rows, root, orig_rows=None):
        calls.append(len(rows))
        return [
            {
                "category": "Sensor Noise",
                "axis": "severity",
                "bin": "level 1",
                "successes": 1,
                "n": 2,
                "orig_successes": 0,
                "orig_n": 0,
            }
        ]

    monkeypatch.setattr(results_libero_tables.severity_sr, "collect", fake_collect)
    first = masked_severity(a, "r42", "masked_proprio")
    second = masked_severity(a, "r42", "masked_proprio")
    assert calls == [4]  # computed once, from the masked LIBERO-Plus rows
    assert (plus.parent / "severity_sr.csv").exists()
    assert list(first.bin) == list(second.bin) == ["level 1"]
