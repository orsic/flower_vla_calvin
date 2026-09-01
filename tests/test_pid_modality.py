"""Tests for scripts/pid_modality.py -- joint construction and PID decomposition."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from pid_modality import build_joint, decompose, held_modality  # noqa: E402

dit = pytest.importorskip("dit")


CSV_COLUMNS = ["use_rgb_static", "use_rgb_gripper", "use_language", "success"]


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _rows_from_cells(cells):
    """cells: dict[(x1, x2)] -> list of 0/1 successes, for a fixed held-on modality.

    Builds rows for pair (static, wrist) with lang held on.
    """
    rows = []
    for (x1, x2), successes in cells.items():
        for s in successes:
            rows.append(
                {
                    "use_rgb_static": x1,
                    "use_rgb_gripper": x2,
                    "use_language": 1,
                    "success": s,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# held_modality
# ---------------------------------------------------------------------------


def test_held_modality_picks_the_third():
    assert held_modality("static", "wrist") == "lang"
    assert held_modality("static", "lang") == "wrist"
    assert held_modality("wrist", "lang") == "static"


# ---------------------------------------------------------------------------
# build_joint
# ---------------------------------------------------------------------------


def test_build_joint_filters_to_held_on_rows(tmp_path):
    rows = _rows_from_cells({(0, 0): [0], (1, 0): [0], (0, 1): [0], (1, 1): [1]})
    # add a held-off row that must be excluded
    rows.append({"use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 0, "success": 1})
    dist = build_joint(rows, "static", "wrist", "lang")
    assert dist.outcome_length() == 3
    assert sum(dist.pmf) == pytest.approx(1.0)


def test_build_joint_uniform_marginal_on_balanced_input():
    rows = _rows_from_cells({(0, 0): [0], (1, 0): [0], (0, 1): [1], (1, 1): [1]})
    dist = build_joint(rows, "static", "wrist", "lang")
    marginal = dist.marginal([0, 1])
    for p in marginal.pmf:
        assert p == pytest.approx(0.25)


def test_build_joint_missing_cell_raises(tmp_path):
    rows = _rows_from_cells({(0, 0): [0], (1, 0): [0], (0, 1): [1]})  # (1,1) missing
    with pytest.raises(SystemExit):
        build_joint(rows, "static", "wrist", "lang")


# ---------------------------------------------------------------------------
# decompose -- canonical distributions with known PID
# ---------------------------------------------------------------------------


def test_decompose_xor_is_pure_synergy():
    dist = dit.Distribution(["000", "011", "101", "110"], [0.25] * 4)
    pid = decompose(dist, "ccs")
    assert pid["R"] == pytest.approx(0.0, abs=1e-9)
    assert pid["U1"] == pytest.approx(0.0, abs=1e-9)
    assert pid["U2"] == pytest.approx(0.0, abs=1e-9)
    assert pid["S"] == pytest.approx(1.0, abs=1e-9)


def test_decompose_copy_is_pure_redundancy():
    dist = dit.Distribution(["000", "111"], [0.5, 0.5])
    pid = decompose(dist, "ccs")
    assert pid["R"] == pytest.approx(1.0, abs=1e-9)
    assert pid["U1"] == pytest.approx(0.0, abs=1e-9)
    assert pid["U2"] == pytest.approx(0.0, abs=1e-9)
    assert pid["S"] == pytest.approx(0.0, abs=1e-9)


def test_decompose_unique_to_first_source():
    # Y = X1, X2 independent uniform noise -> all information is unique to X1.
    dist = dit.Distribution(["000", "010", "101", "111"], [0.25] * 4)
    pid = decompose(dist, "ccs")
    assert pid["U1"] == pytest.approx(1.0, abs=1e-9)
    assert pid["R"] == pytest.approx(0.0, abs=1e-9)
    assert pid["U2"] == pytest.approx(0.0, abs=1e-9)
    assert pid["S"] == pytest.approx(0.0, abs=1e-9)


def test_decompose_terms_sum_to_mutual_information_on_real_shaped_joint():
    rows = _rows_from_cells(
        {(0, 0): [0, 0, 0, 0], (1, 0): [0, 1, 1, 0], (0, 1): [1, 1, 0, 1], (1, 1): [1, 1, 1, 0]}
    )
    dist = build_joint(rows, "static", "wrist", "lang")
    pid = decompose(dist, "ccs")
    assert pid["R"] + pid["U1"] + pid["U2"] + pid["S"] == pytest.approx(pid["I"], abs=1e-6)
