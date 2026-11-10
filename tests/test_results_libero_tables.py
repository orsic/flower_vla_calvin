"""Tests for scripts/results_libero_tables.py's severity-bin merging (the doubled-range
tables in docs/results.md's "LIBERO-Plus by perturbation size")."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from results_libero_tables import merge_bins, merged_counts, size_table  # noqa: E402


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
