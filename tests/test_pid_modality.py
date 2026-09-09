"""Tests for scripts/pid_modality.py -- joint construction and PID decomposition."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import itertools

import pid_modality
from pid_modality import build_joint, decompose, held_modalities, presence_success  # noqa: E402

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
# held_modalities
# ---------------------------------------------------------------------------


def test_held_modalities_picks_the_other_two():
    assert held_modalities("static", "wrist") == ("lang", "proprio")
    assert held_modalities("static", "lang") == ("wrist", "proprio")
    assert held_modalities("wrist", "lang") == ("static", "proprio")
    assert held_modalities("static", "proprio") == ("wrist", "lang")


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


def test_build_joint_accepts_multiple_held_modalities():
    """held as a tuple requires every held column to be 1 -- the shape held_modalities()
    now returns with a 4th modality (proprio) in play."""
    rows = [
        {"use_rgb_static": x1, "use_rgb_gripper": x2, "use_language": 1,
         "use_proprio": 1, "success": s}
        for (x1, x2), successes in
        {(0, 0): [0], (1, 0): [0], (0, 1): [0], (1, 1): [1]}.items()
        for s in successes
    ]
    # add a row with one held modality off -- must be excluded
    rows.append({"use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 1,
                 "use_proprio": 0, "success": 1})
    dist = build_joint(rows, "static", "wrist", ("lang", "proprio"))
    assert dist.outcome_length() == 3
    assert sum(dist.pmf) == pytest.approx(1.0)


def test_build_joint_missing_held_column_reads_as_off(tmp_path):
    """A result.csv predating use_proprio has no such column; holding proprio on must
    then exclude every row (missing == absent), not KeyError."""
    rows = _rows_from_cells({(0, 0): [0], (1, 0): [0], (0, 1): [1], (1, 1): [1]})
    with pytest.raises(SystemExit):
        build_joint(rows, "static", "wrist", ("lang", "proprio"))


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


# ---------------------------------------------------------------------------
# presence_success
# ---------------------------------------------------------------------------


def test_presence_success_groups_by_all_four_flags():
    rows = [
        {"use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 1, "use_proprio": 1, "success": 1},
        {"use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 1, "use_proprio": 1, "success": 0},
        {"use_rgb_static": 1, "use_rgb_gripper": 0, "use_language": 0, "use_proprio": 0, "success": 1},
    ]
    result = presence_success(rows)
    assert result[(1, 1, 1, 1)] == (0.5, 2)
    assert result[(1, 0, 0, 0)] == (1.0, 1)


def test_presence_success_missing_column_reads_as_off():
    """Rows from a result.csv predating use_proprio have no such key at all."""
    rows = [{"use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 1, "success": 1}]
    result = presence_success(rows)
    assert (1, 1, 1, 0) in result
    assert (1, 1, 1, 1) not in result


def test_presence_success_sorted_binary_descending_all_ones_first():
    rows = [
        {"use_rgb_static": s, "use_rgb_gripper": w, "use_language": l, "use_proprio": p, "success": 1}
        for s in (0, 1) for w in (0, 1) for l in (0, 1) for p in (0, 1)
    ]
    combos = sorted(presence_success(rows), reverse=True)
    assert combos[0] == (1, 1, 1, 1)
    assert combos[-1] == (0, 0, 0, 0)
    assert combos == sorted(combos, reverse=True)


# ---------------------------------------------------------------------------
# main() -- skips an incomplete pair instead of aborting the whole run (a no-proprio
# dropout sweep never varies proprio, so every proprio pair is incomplete by design)
# ---------------------------------------------------------------------------


def test_main_skips_incomplete_pairs_instead_of_crashing(tmp_path, monkeypatch, capsys):
    full_columns = ["use_rgb_static", "use_rgb_gripper", "use_language", "use_proprio", "success"]
    rows = [
        dict(zip(full_columns, (static, wrist, lang, 1, success)))  # proprio never varies
        for static, wrist, lang in itertools.product((0, 1), repeat=3)
        for success in (0, 1)
    ]
    csv_path = tmp_path / "result.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=full_columns)
        writer.writeheader()
        writer.writerows(rows)

    monkeypatch.setattr(sys, "argv", ["pid_modality.py", str(csv_path)])
    pid_modality.main()  # must not raise despite the proprio pairs being incomplete

    out = capsys.readouterr().out
    assert "static+proprio" in out and "not evaluated" in out
    assert "static+wrist" in out  # unaffected pair still gets a full PID line
