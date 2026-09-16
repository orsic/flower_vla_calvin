"""Tests for flower.evaluation.flower_eval_libero.get_log_dir.

Concurrent ./run.sh pipeline runs (different seeds/train_folders) all pass the same
log_dir=/saves/eval_logs override, so two eval processes starting within the same wall-
clock second used to collide on os.makedirs(..., exist_ok=False) -- see the module
docstring at the call site for the reported FileExistsError.
"""
import flower.evaluation.flower_eval_libero as flower_eval_libero
from flower.evaluation.flower_eval_libero import get_log_dir


def test_get_log_dir_creates_a_logs_subdir(tmp_path):
    log_dir = get_log_dir(str(tmp_path))

    assert log_dir.exists()
    assert log_dir.parent.parent == tmp_path
    assert log_dir.parent.name == "logs"


def test_get_log_dir_two_calls_in_the_same_second_dont_collide(tmp_path, monkeypatch):
    monkeypatch.setattr(flower_eval_libero.time, "strftime", lambda *a, **k: "2026-01-01_00-00-00")

    first = get_log_dir(str(tmp_path))
    second = get_log_dir(str(tmp_path))

    assert first != second
    assert first.exists() and second.exists()
    assert first.name.startswith("2026-01-01_00-00-00_")
    assert second.name.startswith("2026-01-01_00-00-00_")
