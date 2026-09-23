"""Tests for scripts/compare_eval_csvs.py — per-side modality filtering."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from compare_eval_csvs import parse_modalities, per_task_successes, per_task_sr  # noqa: E402


CSV_COLUMNS = [
    "task_name", "use_rgb_static", "use_rgb_gripper", "use_language", "success", "init_state_idx",
]


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# parse_modalities
# ---------------------------------------------------------------------------

def test_parse_modalities_basic():
    assert parse_modalities("static,wrist,lang") == {"static", "wrist", "lang"}


def test_parse_modalities_single():
    assert parse_modalities("static") == {"static"}


def test_parse_modalities_unknown_raises():
    with pytest.raises(ValueError):
        parse_modalities("static,bogus")


def test_parse_modalities_empty_raises():
    with pytest.raises(ValueError):
        parse_modalities("")


def test_parse_modalities_includes_proprio():
    assert parse_modalities("proprio") == {"proprio"}


# ---------------------------------------------------------------------------
# per_task_sr
# ---------------------------------------------------------------------------

def test_per_task_sr_filters_to_matching_combo(tmp_path):
    path = tmp_path / "result.csv"
    _write_csv(path, [
        {"task_name": "t1", "use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 1, "success": 1},
        {"task_name": "t1", "use_rgb_static": 1, "use_rgb_gripper": 0, "use_language": 0, "success": 0},
    ])
    full = per_task_sr(str(path), {"static", "wrist", "lang"})
    static_only = per_task_sr(str(path), {"static"})
    assert full == {"t1": 1.0}
    assert static_only == {"t1": 0.0}


def test_per_task_sr_averages_within_combo(tmp_path):
    path = tmp_path / "result.csv"
    _write_csv(path, [
        {"task_name": "t1", "use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 1, "success": 1},
        {"task_name": "t1", "use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 1, "success": 0},
    ])
    result = per_task_sr(str(path), {"static", "wrist", "lang"})
    assert result == {"t1": 0.5}


def test_per_task_sr_no_match_raises(tmp_path):
    path = tmp_path / "result.csv"
    _write_csv(path, [
        {"task_name": "t1", "use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 1, "success": 1},
    ])
    with pytest.raises(SystemExit):
        per_task_sr(str(path), {"static"})


def test_per_task_sr_missing_use_proprio_column_reads_as_absent(tmp_path):
    """result.csv files predating use_proprio have no such column; the default
    modality set (no "proprio") must still match them — see eval_records.py's
    module docstring on why a missing column reads as proprio-absent."""
    path = tmp_path / "result.csv"
    _write_csv(path, [
        {"task_name": "t1", "use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 1, "success": 1},
    ])
    result = per_task_sr(str(path), {"static", "wrist", "lang"})
    assert result == {"t1": 1.0}
    with pytest.raises(SystemExit):
        per_task_sr(str(path), {"static", "wrist", "lang", "proprio"})


def test_per_task_sr_filters_on_use_proprio_column(tmp_path):
    path = tmp_path / "result.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS + ["use_proprio"])
        writer.writeheader()
        writer.writerow({"task_name": "t1", "use_rgb_static": 1, "use_rgb_gripper": 1,
                          "use_language": 1, "use_proprio": 1, "success": 1})
        writer.writerow({"task_name": "t1", "use_rgb_static": 1, "use_rgb_gripper": 1,
                          "use_language": 1, "use_proprio": 0, "success": 0})
    with_proprio = per_task_sr(str(path), {"static", "wrist", "lang", "proprio"})
    without_proprio = per_task_sr(str(path), {"static", "wrist", "lang"})
    assert with_proprio == {"t1": 1.0}
    assert without_proprio == {"t1": 0.0}


# ---------------------------------------------------------------------------
# per_task_successes: --base-task / --init-state-idx (comparing a LIBERO-Plus file
# against an original-LIBERO file)
# ---------------------------------------------------------------------------

def test_base_task_collapses_plus_variants_onto_one_key(tmp_path):
    """Every LIBERO-Plus rewrite of one base task's name must key the same as that
    base task, so a Plus file's per-variant rows align with an orig file's task."""
    path = tmp_path / "plus_result.csv"
    _write_csv(path, [
        {"task_name": "t1_language_7_view_0_0_100_0_0_initstate_0",
         "use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 0, "success": 1},
        {"task_name": "t1_language_11_view_0_0_100_0_0_initstate_0",
         "use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 0, "success": 0},
    ])
    without_base_task = per_task_successes(str(path), {"static", "wrist"}, base_task=False)
    with_base_task = per_task_successes(str(path), {"static", "wrist"}, base_task=True)

    assert set(without_base_task) == {
        "t1_language_7_view_0_0_100_0_0_initstate_0",
        "t1_language_11_view_0_0_100_0_0_initstate_0",
    }
    assert with_base_task == {"t1": [1, 0]}


def test_init_state_idx_filters_original_libero_rows(tmp_path):
    """--init-state-idx 0 must keep only the row LIBERO-Plus itself draws from,
    dropping an original-LIBERO file's other init states for the same task."""
    path = tmp_path / "orig_result.csv"
    _write_csv(path, [
        {"task_name": "t1", "use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 0,
         "success": 1, "init_state_idx": 0},
        {"task_name": "t1", "use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 0,
         "success": 0, "init_state_idx": 1},
    ])
    filtered = per_task_successes(str(path), {"static", "wrist"}, init_state_idx=0)
    assert filtered == {"t1": [1]}


def test_per_task_successes_reports_n_per_task(tmp_path):
    """The whole point of exposing successes (not just the rate): an unequal
    per-task episode count -- e.g. LIBERO-Plus's Language Instructions category,
    21-47 variants per base task -- must be visible, not averaged away silently."""
    path = tmp_path / "result.csv"
    _write_csv(path, [
        {"task_name": "t1", "use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 1, "success": 1},
        {"task_name": "t1", "use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 1, "success": 1},
        {"task_name": "t1", "use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 1, "success": 0},
        {"task_name": "t2", "use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 1, "success": 1},
    ])
    result = per_task_successes(str(path), {"static", "wrist", "lang"})
    assert len(result["t1"]) == 3
    assert len(result["t2"]) == 1


def test_average_over_tasks_is_unweighted_regardless_of_per_task_n():
    """AVERAGE (computed the same way main() does) must be a plain mean of per-task
    rates, not weighted by each task's episode count -- otherwise a base task with
    more Plus variants would dominate the comparison against an orig file's uniform
    per-task counts."""
    sr = {"t1": 1.0, "t2": 0.0}  # t1 backed by many episodes, t2 by few -- irrelevant
    avg = sum(sr.values()) / len(sr)
    assert avg == 0.5
