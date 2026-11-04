"""Tests for scripts/results_libero_tables.py's severity-bin merging (the doubled-range
tables in docs/results.md's "LIBERO-Plus by perturbation size")."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from results_libero_tables import merge_bins, merged_counts  # noqa: E402


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
