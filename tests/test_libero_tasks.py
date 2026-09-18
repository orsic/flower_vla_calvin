"""Tests for flower.evaluation.libero_tasks.

base_task()'s own logic is exercised in tests/test_severity_sr.py (imported there via
severity_sr's re-export); this file covers original_task_names() and a direct import
of base_task from its new home.
"""
from flower.evaluation.libero_tasks import base_task, original_task_names


def test_base_task_longest_prefix_match():
    names = ["foo_bar", "foo"]
    assert base_task("foo_bar_1", names) == "foo_bar"
    assert base_task("foo_1", names) == "foo"
    assert base_task("unrelated_1", names) is None


def test_original_task_names_lists_only_pruned_init_stems(tmp_path):
    suite_dir = tmp_path / "libero_10"
    suite_dir.mkdir()
    (suite_dir / "task_a.pruned_init").touch()
    (suite_dir / "task_b.pruned_init").touch()
    (suite_dir / "task_a.init").touch()  # same stem, different extension -- not a match
    (suite_dir / "stray.bddl").touch()

    assert original_task_names(str(tmp_path), "libero_10") == ["task_a", "task_b"]


def test_original_task_names_missing_suite_dir_is_empty(tmp_path):
    assert original_task_names(str(tmp_path), "libero_10") == []
