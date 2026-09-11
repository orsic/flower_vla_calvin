"""Tests for scripts/perturbation_sr.py -- per-perturbation-category success rate."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from perturbation_sr import (  # noqa: E402
    UNCLASSIFIED,
    category_sr,
    modality_combos_present,
    report,
)

CSV_COLUMNS = [
    "use_rgb_static",
    "use_rgb_gripper",
    "use_language",
    "use_proprio",
    "task_category",
    "success",
]


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(category, success, static=1, wrist=1, lang=1, proprio=1):
    return {
        "use_rgb_static": static,
        "use_rgb_gripper": wrist,
        "use_language": lang,
        "use_proprio": proprio,
        "task_category": category,
        "success": success,
    }


def _read(path):
    import csv as _csv

    with open(path, newline="") as f:
        return list(_csv.DictReader(f))


# ---------------------------------------------------------------------------
# category_sr
# ---------------------------------------------------------------------------


def test_category_sr_groups_and_averages(tmp_path):
    path = tmp_path / "result.csv"
    _write_csv(
        path,
        [
            _row("Camera Viewpoints", 1),
            _row("Camera Viewpoints", 0),
            _row("Camera Viewpoints", 1),
            _row("Sensor Noise", 0),
            _row("Sensor Noise", 0),
        ],
    )
    sr = category_sr(_read(path))
    assert sr["Camera Viewpoints"] == (2 / 3, 3)
    assert sr["Sensor Noise"] == (0.0, 2)


def test_category_sr_empty_category_is_unclassified(tmp_path):
    path = tmp_path / "result.csv"
    _write_csv(path, [_row("", 1), _row("", 0)])
    sr = category_sr(_read(path))
    assert sr == {UNCLASSIFIED: (0.5, 2)}


# ---------------------------------------------------------------------------
# modality_combos_present
# ---------------------------------------------------------------------------


def test_modality_combos_present_single_combo(tmp_path):
    path = tmp_path / "result.csv"
    _write_csv(path, [_row("Sensor Noise", 1), _row("Sensor Noise", 0)])
    assert modality_combos_present(_read(path)) == {(1, 1, 1, 1)}


def test_modality_combos_present_detects_multiple_combos(tmp_path):
    path = tmp_path / "result.csv"
    _write_csv(
        path,
        [
            _row("Sensor Noise", 1, static=1, wrist=1, lang=1, proprio=1),
            _row("Sensor Noise", 0, static=1, wrist=0, lang=1, proprio=0),
        ],
    )
    combos = modality_combos_present(_read(path))
    assert combos == {(1, 1, 1, 1), (1, 0, 1, 0)}


# ---------------------------------------------------------------------------
# report -- overall line and warning
# ---------------------------------------------------------------------------


def test_report_overall_matches_total_success_rate(tmp_path, capsys):
    path = tmp_path / "result.csv"
    _write_csv(
        path,
        [
            _row("Camera Viewpoints", 1),
            _row("Camera Viewpoints", 0),
            _row("Camera Viewpoints", 1),
            _row("Sensor Noise", 0),
            _row("Sensor Noise", 0),
        ],
    )
    report(_read(path))
    out = capsys.readouterr().out
    assert "OVERALL" in out
    overall_line = [line for line in out.splitlines() if line.startswith("OVERALL")][0]
    assert f"{2 / 5:>12.3f}" in overall_line
    assert f"{5:>6}" in overall_line


def test_report_warns_on_multiple_modality_combos(tmp_path, capsys):
    path = tmp_path / "result.csv"
    _write_csv(
        path,
        [
            _row("Sensor Noise", 1, static=1, wrist=1, lang=1, proprio=1),
            _row("Sensor Noise", 0, static=1, wrist=0, lang=1, proprio=0),
        ],
    )
    report(_read(path))
    out = capsys.readouterr().out
    assert "WARNING" in out


def test_report_no_warning_for_single_combo(tmp_path, capsys):
    path = tmp_path / "result.csv"
    _write_csv(path, [_row("Sensor Noise", 1), _row("Sensor Noise", 0)])
    report(_read(path))
    out = capsys.readouterr().out
    assert "WARNING" not in out
