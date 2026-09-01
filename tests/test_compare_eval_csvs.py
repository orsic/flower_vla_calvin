"""Tests for scripts/compare_eval_csvs.py — per-side modality filtering."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from compare_eval_csvs import parse_modalities, per_task_sr  # noqa: E402


CSV_COLUMNS = ["task_name", "use_rgb_static", "use_rgb_gripper", "use_language", "success"]


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
