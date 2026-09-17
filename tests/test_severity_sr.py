"""Tests for scripts/severity_sr.py -- success rate by physical perturbation severity.

collect() is the return-only layer report()/report_category() render from (see that
module's docstring); these tests exercise collect()/write_csv() directly, plus a capsys
smoke test that the printed report still has the same shape after the refactor.
Light Conditions needs real BDDL/scene-XML fixtures (LIBERO-Plus's own assets), so it's
covered by tests/test_perturbation_severity.py instead -- these fixtures never use
"Light Conditions" as a category.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from severity_sr import (  # noqa: E402
    CSV_COLUMNS,
    NEWOBJ_INIT_NOTE,
    ROBOT_INITSTATE_NOTE,
    UNCLASSIFIED,
    UNORDERED_NOTE,
    UNPARSED,
    base_task,
    collect,
    report,
    wilson_interval,
    write_csv,
)

LIBERO_PLUS_ROOT = "/unused"  # only Light Conditions rows would ever read this path

CSV_ROW_COLUMNS = [
    "use_rgb_static",
    "use_rgb_gripper",
    "use_language",
    "use_proprio",
    "task_category",
    "task_name",
    "difficulty_level",
    "init_state_idx",
    "success",
]


def _write_rows_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_ROW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(category, task_name, difficulty_level, success, static=1, wrist=1, lang=1, proprio=1, init_state_idx=0):
    return {
        "use_rgb_static": static,
        "use_rgb_gripper": wrist,
        "use_language": lang,
        "use_proprio": proprio,
        "task_category": category,
        "task_name": task_name,
        "difficulty_level": difficulty_level,
        "init_state_idx": init_state_idx,
        "success": success,
    }


def _orig_row(task_name, success, static=1, wrist=1, lang=1, proprio=1, init_state_idx=0):
    """An orig LIBERO row: no task_category/difficulty_level, matching eval_records.py's
    ALL_COLUMNS (those columns are "" for the orig variant)."""
    return _row("", task_name, "", success, static=static, wrist=wrist, lang=lang, proprio=proprio, init_state_idx=init_state_idx)


def _read_rows(tmp_path, rows, name="result.csv"):
    csv_path = tmp_path / name
    _write_rows_csv(csv_path, rows)
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _records_by(records, axis=None, category=None):
    out = records
    if axis is not None:
        out = [r for r in out if r["axis"] == axis]
    if category is not None:
        out = [r for r in out if r["category"] == category]
    return out


# ---------------------------------------------------------------------------
# base_task / paired baseline -- init-state-matched LIBERO original comparison
# ---------------------------------------------------------------------------


def test_base_task_longest_prefix_match():
    names = ["foo", "foo_bar"]
    assert base_task("foo_bar_1", names) == "foo_bar"
    assert base_task("foo_1", names) == "foo"
    assert base_task("unrelated_1", names) is None


def test_paired_baseline_dedups_by_base_task(tmp_path):
    plus_rows = _read_rows(
        tmp_path,
        [
            _row("Sensor Noise", "foo_noise_1", "1", 1),
            _row("Sensor Noise", "foo_noise_2", "1", 0),
            _row("Sensor Noise", "bar_noise_1", "1", 1),
        ],
        name="plus.csv",
    )
    orig_rows = _read_rows(
        tmp_path,
        [_orig_row("foo", 1), _orig_row("bar", 0)],
        name="orig.csv",
    )
    total = _records_by(collect(plus_rows, LIBERO_PLUS_ROOT, orig_rows=orig_rows), axis="total", category="Sensor Noise")[0]
    assert total["n"] == 3
    assert total["orig_n"] == 2
    assert total["orig_successes"] == 1


def test_paired_baseline_excludes_orig_rows_at_other_init_states(tmp_path):
    plus_rows = _read_rows(tmp_path, [_row("Sensor Noise", "foo_noise_1", "1", 1)], name="plus.csv")
    orig_rows = _read_rows(tmp_path, [_orig_row("foo", 1, init_state_idx=1)], name="orig.csv")
    total = _records_by(collect(plus_rows, LIBERO_PLUS_ROOT, orig_rows=orig_rows), axis="total", category="Sensor Noise")[0]
    assert total["orig_n"] == 0


def test_paired_baseline_excludes_orig_rows_with_different_modality_combo(tmp_path):
    plus_rows = _read_rows(tmp_path, [_row("Sensor Noise", "foo_noise_1", "1", 1)], name="plus.csv")
    orig_rows = _read_rows(tmp_path, [_orig_row("foo", 1, proprio=0)], name="orig.csv")
    total = _records_by(collect(plus_rows, LIBERO_PLUS_ROOT, orig_rows=orig_rows), axis="total", category="Sensor Noise")[0]
    assert total["orig_n"] == 0


def test_paired_baseline_excludes_rows_with_no_matching_base_task(tmp_path):
    plus_rows = _read_rows(tmp_path, [_row("Sensor Noise", "unrelated_noise_1", "1", 1)], name="plus.csv")
    orig_rows = _read_rows(tmp_path, [_orig_row("foo", 1)], name="orig.csv")
    total = _records_by(collect(plus_rows, LIBERO_PLUS_ROOT, orig_rows=orig_rows), axis="total", category="Sensor Noise")[0]
    assert total["orig_n"] == 0


def test_collect_without_orig_rows_leaves_baseline_columns_empty(tmp_path):
    rows = _read_rows(tmp_path, [_row("Sensor Noise", "foo_noise_1", "1", 1)])
    total = _records_by(collect(rows, LIBERO_PLUS_ROOT), axis="total", category="Sensor Noise")[0]
    for key in ("orig_successes", "orig_n", "orig_success_rate", "orig_ci_low", "orig_ci_high"):
        assert total[key] == ""


def test_collect_emits_one_total_record_per_category(tmp_path):
    rows = _read_rows(
        tmp_path,
        [
            _row("Sensor Noise", "foo_noise_1", "1", 1),
            _row("Sensor Noise", "foo_noise_2", "1", 0),
            _row("Robot Initial States", "bar_initstate_50", "1", 1),
        ],
    )
    records = collect(rows, LIBERO_PLUS_ROOT)
    sensor_total = _records_by(records, axis="total", category="Sensor Noise")
    assert len(sensor_total) == 1 and sensor_total[0]["n"] == 2
    robot_total = _records_by(records, axis="total", category="Robot Initial States")
    assert len(robot_total) == 1 and robot_total[0]["n"] == 1


def test_objects_layout_records_carry_newobj_init_note_when_baseline_present(tmp_path):
    plus_rows = _read_rows(
        tmp_path,
        [
            _row("Objects Layout", "foo_add_1", "1", 1),
            _row("Sensor Noise", "foo_noise_1", "1", 1),
        ],
        name="plus.csv",
    )
    orig_rows = _read_rows(tmp_path, [_orig_row("foo", 1)], name="orig.csv")
    records = collect(plus_rows, LIBERO_PLUS_ROOT, orig_rows=orig_rows)
    objects_layout = _records_by(records, category="Objects Layout")
    assert objects_layout and all(r["note"] == NEWOBJ_INIT_NOTE for r in objects_layout)
    sensor_noise = _records_by(records, category="Sensor Noise")
    assert sensor_noise and all(r["note"] == "" for r in sensor_noise)


def test_objects_layout_note_empty_without_orig_rows(tmp_path):
    rows = _read_rows(tmp_path, [_row("Objects Layout", "foo_add_1", "1", 1)])
    records = collect(rows, LIBERO_PLUS_ROOT)
    assert all(r["note"] == "" for r in records)


def test_unclassified_category_gets_no_baseline(tmp_path):
    plus_rows = _read_rows(tmp_path, [_row("", "foo", "", 1)], name="plus.csv")
    orig_rows = _read_rows(tmp_path, [_orig_row("foo", 1)], name="orig.csv")
    records = collect(plus_rows, LIBERO_PLUS_ROOT, orig_rows=orig_rows)
    unclassified = _records_by(records, category=UNCLASSIFIED)
    assert unclassified and all(r["orig_n"] == "" for r in unclassified)


# ---------------------------------------------------------------------------
# difficulty_level axis -- every category, ordered or not
# ---------------------------------------------------------------------------


def test_collect_emits_one_difficulty_record_per_level(tmp_path):
    rows = _read_rows(
        tmp_path,
        [
            _row("Robot Initial States", "foo_initstate_50", "1", 1),
            _row("Robot Initial States", "foo_initstate_50", "1", 0),
            _row("Robot Initial States", "foo_initstate_150", "2", 1),
        ],
    )
    records = collect(rows, LIBERO_PLUS_ROOT)
    difficulty = _records_by(records, axis="difficulty_level", category="Robot Initial States")
    labels = [r["bin"] for r in difficulty]
    assert labels == ["1", "2"]
    level1 = difficulty[0]
    assert level1["successes"] == 1 and level1["n"] == 2
    # Robot Initial States carries the known-limitation caveat on every record (see
    # ROBOT_INITSTATE_NOTE); a genuinely uncaveated ordered category is covered below.
    assert level1["note"] == ROBOT_INITSTATE_NOTE


def test_collect_uncaveated_ordered_category_has_no_note(tmp_path):
    rows = _read_rows(
        tmp_path,
        [_row("Sensor Noise", "foo_noise_1", "1", 1)],
    )
    records = collect(rows, LIBERO_PLUS_ROOT)
    difficulty = _records_by(records, axis="difficulty_level", category="Sensor Noise")
    assert difficulty[0]["note"] == ""


def test_report_robot_initial_states_prints_known_limitation_warning(tmp_path, capsys):
    rows = _read_rows(
        tmp_path,
        [_row("Robot Initial States", "foo_initstate_50", "1", 1)],
    )
    report(rows, LIBERO_PLUS_ROOT)
    out = capsys.readouterr().out
    assert ROBOT_INITSTATE_NOTE in out


# ---------------------------------------------------------------------------
# severity axis, ordered category -- Robot Initial States (simplest parser)
# ---------------------------------------------------------------------------


def test_collect_severity_records_in_sort_key_order(tmp_path):
    rows = _read_rows(
        tmp_path,
        [
            _row("Robot Initial States", "foo_initstate_150", "3", 1),  # 0.2rad
            _row("Robot Initial States", "foo_initstate_50", "1", 1),  # 0.1rad
            _row("Robot Initial States", "foo_initstate_50", "1", 0),
        ],
    )
    severity = _records_by(collect(rows, LIBERO_PLUS_ROOT), axis="severity", category="Robot Initial States")
    assert [r["bin"] for r in severity] == ["0.1rad", "0.2rad"]
    assert severity[0]["successes"] == 1 and severity[0]["n"] == 2
    assert severity[1]["successes"] == 1 and severity[1]["n"] == 1


def test_collect_unparsed_row_gets_explicit_bin_and_reconciles_n(tmp_path):
    rows = _read_rows(
        tmp_path,
        [
            _row("Robot Initial States", "foo_initstate_50", "1", 1),
            _row("Robot Initial States", "no_match_here", "1", 0),  # doesn't parse
        ],
    )
    records = collect(rows, LIBERO_PLUS_ROOT)
    severity = _records_by(records, axis="severity", category="Robot Initial States")
    unparsed = [r for r in severity if r["bin"] == UNPARSED]
    assert len(unparsed) == 1
    assert unparsed[0]["successes"] == 0 and unparsed[0]["n"] == 1

    difficulty = _records_by(records, axis="difficulty_level", category="Robot Initial States")
    assert sum(r["n"] for r in difficulty) == sum(r["n"] for r in severity) == 2


# ---------------------------------------------------------------------------
# Background Textures -- subtype instead of severity
# ---------------------------------------------------------------------------


def test_collect_background_textures_gets_subtype_not_severity(tmp_path):
    rows = _read_rows(
        tmp_path,
        [
            _row("Background Textures", "foo_table_1", "2", 1),
            _row("Background Textures", "foo_tb_2", "2", 0),
        ],
    )
    records = collect(rows, LIBERO_PLUS_ROOT)
    assert _records_by(records, axis="severity", category="Background Textures") == []
    subtype = _records_by(records, axis="subtype", category="Background Textures")
    assert {r["bin"] for r in subtype} == {"table", "tb"}
    for r in subtype:
        assert r["note"] == UNORDERED_NOTE["Background Textures"]


# ---------------------------------------------------------------------------
# Language Instructions -- no severity axis, note on every record
# ---------------------------------------------------------------------------


def test_collect_language_instructions_has_no_severity_records(tmp_path):
    rows = _read_rows(
        tmp_path,
        [
            _row("Language Instructions", "foo_rewrite_3", "5", 1),
            _row("Language Instructions", "foo_rewrite_7", "5", 0),
        ],
    )
    records = collect(rows, LIBERO_PLUS_ROOT)
    assert _records_by(records, axis="severity", category="Language Instructions") == []
    difficulty = _records_by(records, axis="difficulty_level", category="Language Instructions")
    assert len(difficulty) == 1
    assert difficulty[0]["note"] == UNORDERED_NOTE["Language Instructions"]


# ---------------------------------------------------------------------------
# wilson_interval clamp
# ---------------------------------------------------------------------------


def test_wilson_interval_stays_within_unit_interval_at_extremes():
    lo, hi = wilson_interval(0, 10)
    assert lo >= 0.0 and hi <= 1.0
    lo, hi = wilson_interval(10, 10)
    assert lo >= 0.0 and hi <= 1.0


def test_collect_all_failure_bin_ci_within_unit_interval(tmp_path):
    rows = _read_rows(
        tmp_path,
        [_row("Robot Initial States", "foo_initstate_50", "1", 0) for _ in range(10)],
    )
    severity = _records_by(collect(rows, LIBERO_PLUS_ROOT), axis="severity", category="Robot Initial States")
    assert len(severity) == 1
    assert severity[0]["ci_low"] >= 0.0
    assert severity[0]["ci_high"] <= 1.0


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------


def test_write_csv_round_trips_with_exact_columns(tmp_path):
    rows = _read_rows(
        tmp_path,
        [
            _row("Robot Initial States", "foo_initstate_50", "1", 1),
            _row("Robot Initial States", "foo_initstate_150", "2", 0),
        ],
    )
    records = collect(rows, LIBERO_PLUS_ROOT)
    out_path = tmp_path / "severity_sr.csv"
    write_csv(out_path, records)

    with open(out_path, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_COLUMNS
        written = list(reader)
    assert len(written) == len(records)


# ---------------------------------------------------------------------------
# report() -- printed shape after the collect()-backed refactor
# ---------------------------------------------------------------------------


def test_report_prints_category_and_both_table_titles(tmp_path, capsys):
    rows = _read_rows(
        tmp_path,
        [
            _row("Robot Initial States", "foo_initstate_50", "1", 1),
            _row("Robot Initial States", "foo_initstate_150", "2", 0),
        ],
    )
    report(rows, LIBERO_PLUS_ROOT)
    out = capsys.readouterr().out
    assert "=== Robot Initial States (n=2) ===" in out
    assert "by difficulty_level (upstream annotation, see caveat below)" in out
    assert "by physical severity" in out
    assert "difficulty_level is an upstream" in out


def test_report_unordered_category_prints_note_and_no_severity_table(tmp_path, capsys):
    rows = _read_rows(
        tmp_path,
        [_row("Language Instructions", "foo_rewrite_3", "5", 1)],
    )
    report(rows, LIBERO_PLUS_ROOT)
    out = capsys.readouterr().out
    assert UNORDERED_NOTE["Language Instructions"] in out
    assert "sub-type instead" not in out


def test_report_all_unparsed_prints_none_matched_message(tmp_path, capsys):
    rows = _read_rows(
        tmp_path,
        [_row("Robot Initial States", "no_match_here", "1", 1)],
    )
    report(rows, LIBERO_PLUS_ROOT)
    out = capsys.readouterr().out
    assert "none of this category's rows matched the expected name pattern" in out


def test_report_with_orig_rows_prints_paired_columns(tmp_path, capsys):
    plus_rows = _read_rows(
        tmp_path,
        [
            _row("Robot Initial States", "foo_initstate_50", "1", 1),
            _row("Robot Initial States", "foo_initstate_150", "2", 0),
        ],
        name="plus.csv",
    )
    orig_rows = _read_rows(tmp_path, [_orig_row("foo", 1)], name="orig.csv")
    report(plus_rows, LIBERO_PLUS_ROOT, orig_rows=orig_rows)
    out = capsys.readouterr().out
    assert "overall (paired vs. orig)" in out
    assert "orig_sr" in out


def test_report_without_orig_rows_omits_paired_columns(tmp_path, capsys):
    rows = _read_rows(tmp_path, [_row("Robot Initial States", "foo_initstate_50", "1", 1)])
    report(rows, LIBERO_PLUS_ROOT)
    out = capsys.readouterr().out
    assert "overall (paired vs. orig)" not in out
    assert "orig_sr" not in out
