"""Tests for scripts/mimicgen_sr.py -- MimicGen success rate by task family x difficulty.

Pure logic only, plus mimicgen_artifact()'s selection over fake artifacts.
download/W&B fetching itself is a thin wrapper, same rationale as
tests/test_analyze_wandb.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import mimicgen_sr  # noqa: E402


def _row(task_name, success, static=1, wrist=1, lang=1, proprio=1):
    return {
        "task_name": task_name,
        "success": success,
        "use_rgb_static": static,
        "use_rgb_gripper": wrist,
        "use_language": lang,
        "use_proprio": proprio,
    }


def _rows(task_name, successes, failures):
    return [_row(task_name, 1)] * successes + [_row(task_name, 0)] * failures


# ---------------------------------------------------------------------------
# split_dataset / breakdown
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("square_d0", ("square", "d0")),
        ("three_piece_assembly_d2", ("three_piece_assembly", "d2")),
        ("coffee_preparation_d1", ("coffee_preparation", "d1")),
    ],
)
def test_split_dataset(name, expected):
    assert mimicgen_sr.split_dataset(name) == expected


def test_breakdown_counts_variants_family_and_overall():
    rows = _rows("square_d0", 3, 1) + _rows("square_d1", 1, 3) + _rows("stack_d0", 2, 0)
    cells = mimicgen_sr.breakdown(rows)
    assert cells[("square", "d0")] == (3, 4)
    assert cells[("square", "d1")] == (1, 4)
    assert cells[("square", "ALL")] == (4, 8)
    assert cells[("stack", "d0")] == (2, 2)
    assert cells[("stack", "ALL")] == (2, 2)
    assert cells[(mimicgen_sr.OVERALL, "")] == (6, 10)


def test_ordered_keys_sorts_families_and_variants_and_skips_single_variant_all():
    rows = (
        _rows("threading_d2", 1, 0)
        + _rows("pick_place_d0", 1, 0)
        + _rows("threading_d0", 1, 0)
        + _rows("coffee_d1", 1, 0)
        + _rows("coffee_d0", 1, 0)
    )
    keys = mimicgen_sr.ordered_keys(mimicgen_sr.breakdown(rows))
    assert keys == [
        ("coffee", "d0"),
        ("coffee", "d1"),
        ("coffee", "ALL"),
        ("pick_place", "d0"),
        ("threading", "d0"),
        ("threading", "d2"),
        ("threading", "ALL"),
        (mimicgen_sr.OVERALL, ""),
    ]


# ---------------------------------------------------------------------------
# format_table
# ---------------------------------------------------------------------------


def test_format_table_has_rate_ci_and_n_per_row():
    rows = _rows("square_d0", 3, 1) + _rows("square_d1", 0, 4)
    lines = mimicgen_sr.format_table(rows).splitlines()
    by_label = {tuple(line.split()[:2]): line.split() for line in lines[1:]}
    d0 = by_label[("square", "d0")]
    assert float(d0[2]) == pytest.approx(0.75)
    assert 0.0 <= float(d0[3]) <= 0.75 <= float(d0[4]) <= 1.0
    assert d0[5] == "4"
    d1 = by_label[("square", "d1")]
    assert float(d1[2]) == 0.0 and float(d1[3]) == 0.0
    overall = next(line.split() for line in lines if line.startswith(mimicgen_sr.OVERALL))
    assert float(overall[1]) == pytest.approx(0.375)
    assert overall[-1] == "8"


def test_format_table_warns_on_mixed_modality_combos():
    rows = _rows("square_d0", 1, 1) + [_row("square_d0", 1, lang=0)]
    assert "WARNING" in mimicgen_sr.format_table(rows)


def test_format_table_no_warning_for_single_combo():
    assert "WARNING" not in mimicgen_sr.format_table(_rows("square_d0", 1, 1))


# ---------------------------------------------------------------------------
# aggregate_runs / format_aggregate (filter mode)
# ---------------------------------------------------------------------------


def test_aggregate_runs_mean_min_max_and_n_runs():
    per_run = {
        "r1": _rows("square_d0", 1, 1) + _rows("stack_d0", 1, 0),
        "r2": _rows("square_d0", 0, 2),
    }
    agg = mimicgen_sr.aggregate_runs(per_run)
    mean, lo, hi, n_runs = agg[("square", "d0")]
    assert (mean, lo, hi, n_runs) == (pytest.approx(0.25), 0.0, 0.5, 2)
    assert agg[("stack", "d0")] == (1.0, 1.0, 1.0, 1)  # only present in r1
    mean, lo, hi, n_runs = agg[(mimicgen_sr.OVERALL, "")]
    assert (mean, lo, hi, n_runs) == (pytest.approx((2 / 3 + 0.0) / 2), 0.0, pytest.approx(2 / 3), 2)


def test_format_aggregate_lists_each_cell():
    per_run = {"r1": _rows("square_d0", 1, 1), "r2": _rows("square_d0", 0, 2)}
    text = mimicgen_sr.format_aggregate(mimicgen_sr.aggregate_runs(per_run))
    square = next(line.split() for line in text.splitlines() if line.startswith("square"))
    assert square[:2] == ["square", "d0"]
    assert [float(v) for v in square[2:5]] == [pytest.approx(0.25), 0.0, 0.5]
    assert square[5] == "2"


# ---------------------------------------------------------------------------
# mimicgen_artifact
# ---------------------------------------------------------------------------


class _FakeManifest:
    def __init__(self, members):
        self.entries = {m: object() for m in members}


class _FakeArtifact:
    def __init__(self, type_, created_at, members):
        self.type = type_
        self.created_at = created_at
        self.manifest = _FakeManifest(members)


class _FakeRun:
    def __init__(self, artifacts):
        self.id = "r1"
        self._artifacts = artifacts

    def logged_artifacts(self):
        return self._artifacts


def test_mimicgen_artifact_picks_newest_with_member_skipping_newer_libero():
    old = _FakeArtifact("evaluation", "2026-09-01", ["mimicgen.csv"])
    new = _FakeArtifact("evaluation", "2026-09-10", ["mimicgen.csv"])
    libero = _FakeArtifact("evaluation", "2026-09-20", ["libero_orig.csv", "libero_plus.csv"])
    model = _FakeArtifact("model", "2026-09-30", ["mimicgen.csv"])
    run = _FakeRun([old, libero, new, model])
    assert mimicgen_sr.mimicgen_artifact(run) is new


def test_mimicgen_artifact_none_when_no_member():
    run = _FakeRun([_FakeArtifact("evaluation", "2026-09-20", ["libero_orig.csv"])])
    assert mimicgen_sr.mimicgen_artifact(run) is None
