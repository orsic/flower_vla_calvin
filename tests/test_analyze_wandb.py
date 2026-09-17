"""Tests for scripts/analyze_wandb.py -- cross-run consistency checks and aggregation.

Pure logic / filesystem tests only. eval_artifact()/download_run() (the two functions
that actually touch the W&B network) are thin wrappers with no local logic to unit
test -- same rationale as scripts/eval_pipeline.py's upload() being untested.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import analyze_wandb  # noqa: E402
import pid_modality  # noqa: E402
import perturbation_sr  # noqa: E402

from flower.evaluation.eval_records import write_csv  # noqa: E402

dit = pytest.importorskip("dit")


def _combo_row(static, wrist, lang, proprio, success):
    return {
        "use_rgb_static": static,
        "use_rgb_gripper": wrist,
        "use_language": lang,
        "use_proprio": proprio,
        "success": success,
    }


class _FakeRun:
    def __init__(self, run_id, name, config):
        self.id = run_id
        self.name = name
        self.config = config


# ---------------------------------------------------------------------------
# combo_set / episode_key_set / category_count_set
# ---------------------------------------------------------------------------


def test_combo_set_basic():
    rows = [_combo_row(1, 1, 1, 1, 1)]
    assert analyze_wandb.combo_set(rows) == frozenset({(1, 1, 1, 1)})
    assert analyze_wandb.combo_set([]) == frozenset()


def test_episode_key_set_basic():
    rows = [{**_combo_row(1, 1, 1, 1, 1), "suite": "libero_10", "task_idx": "0", "episode_idx": "0"}]
    assert analyze_wandb.episode_key_set(rows) == frozenset({("libero_10", "0", "0", (1, 1, 1, 1))})


def test_category_count_set_groups_empty_as_unclassified():
    rows = [
        {"task_category": "A", "success": 1},
        {"task_category": "A", "success": 0},
        {"task_category": "", "success": 1},
    ]
    assert analyze_wandb.category_count_set(rows) == frozenset(
        {("A", 2), (perturbation_sr.UNCLASSIFIED, 1)}
    )


# ---------------------------------------------------------------------------
# _check / consistency_report
# ---------------------------------------------------------------------------


def test_check_flags_the_minority_run():
    per_run = {
        "r1": {"orig_rows": [_combo_row(1, 1, 1, 1, 1)]},
        "r2": {"orig_rows": [_combo_row(1, 1, 1, 1, 1)]},
        "r3": {"orig_rows": [_combo_row(1, 1, 1, 0, 1)]},
    }
    lines = analyze_wandb._check(
        per_run, "modality-combo coverage", lambda a: analyze_wandb.combo_set(a["orig_rows"])
    )
    assert len(lines) == 1
    assert "r3" in lines[0] and "modality-combo coverage" in lines[0]


def test_check_empty_when_uniform():
    per_run = {
        "r1": {"orig_rows": [_combo_row(1, 1, 1, 1, 1)]},
        "r2": {"orig_rows": [_combo_row(1, 1, 1, 1, 1)]},
    }
    assert analyze_wandb._check(per_run, "x", lambda a: analyze_wandb.combo_set(a["orig_rows"])) == []


def test_check_skips_when_fewer_than_two_runs_have_data():
    per_run = {"r1": {"orig_rows": [_combo_row(1, 1, 1, 1, 1)]}, "r2": {"orig_rows": []}}
    assert analyze_wandb._check(per_run, "x", lambda a: analyze_wandb.combo_set(a["orig_rows"])) == []


def test_consistency_report_flags_each_axis_independently():
    good_orig = [_combo_row(1, 1, 1, 1, 1), _combo_row(1, 1, 1, 1, 0)]
    bad_orig = [_combo_row(1, 1, 1, 0, 1)]
    good_plus = [{"task_category": "A", "success": 1}]
    bad_plus = [{"task_category": "B", "success": 1}]
    per_run = {
        "r1": {"orig_rows": good_orig, "plus_rows": good_plus, "severity": None},
        "r2": {"orig_rows": good_orig, "plus_rows": good_plus, "severity": None},
        "r3": {"orig_rows": bad_orig, "plus_rows": bad_plus, "severity": None},
    }
    lines = analyze_wandb.consistency_report(per_run)
    assert any("r3" in line and "modality-combo" in line for line in lines)
    assert any("r3" in line and "perturbation-category" in line for line in lines)


def test_consistency_report_flags_severity_bin_coverage():
    good_severity = [{"category": "Robot Initial States", "axis": "severity", "bin": "0.1rad"}]
    bad_severity = [{"category": "Robot Initial States", "axis": "severity", "bin": "0.2rad"}]
    per_run = {
        "r1": {"orig_rows": [], "plus_rows": [], "severity": good_severity},
        "r2": {"orig_rows": [], "plus_rows": [], "severity": good_severity},
        "r3": {"orig_rows": [], "plus_rows": [], "severity": bad_severity},
    }
    lines = analyze_wandb.consistency_report(per_run)
    assert any("r3" in line and "severity-bin coverage" in line for line in lines)


# ---------------------------------------------------------------------------
# analyze()
# ---------------------------------------------------------------------------


def test_analyze_both_csvs_present(tmp_path):
    artifact_dir = tmp_path / "artifact"
    orig_rows = [
        _combo_row(s, w, l, p, (s + w + l + p) % 2)
        for s in (0, 1)
        for w in (0, 1)
        for l in (0, 1)
        for p in (0, 1)
        if s or w or l  # flower_eval_libero.py never evaluates all-tokens-off
    ]
    write_csv(artifact_dir / "libero_orig.csv", orig_rows)
    write_csv(
        artifact_dir / "libero_plus.csv",
        [{**_combo_row(1, 1, 1, 1, 1), "task_category": "Camera Viewpoints"}],
    )

    result = analyze_wandb.analyze(artifact_dir, missing=[], measure="ccs")

    assert result["missing"] == []
    assert len(result["orig_rows"]) == len(orig_rows)
    assert result["pid"] is not None
    assert result["perturbation"] == {"Camera Viewpoints": (1.0, 1)}


def test_analyze_missing_plus_csv(tmp_path):
    artifact_dir = tmp_path / "artifact"
    write_csv(artifact_dir / "libero_orig.csv", [_combo_row(1, 1, 1, 1, 1)])

    result = analyze_wandb.analyze(artifact_dir, missing=["libero_plus.csv"], measure="ccs")

    assert result["missing"] == ["libero_plus.csv"]
    assert result["plus_rows"] == []
    assert result["perturbation"] is None
    assert result["orig_rows"]


def test_analyze_no_artifact_at_all():
    result = analyze_wandb.analyze(None, missing=["<no evaluation artifact>"], measure="ccs")
    assert result == {
        "orig_rows": [],
        "plus_rows": [],
        "pid": None,
        "perturbation": None,
        "severity": None,
        "missing": ["<no evaluation artifact>"],
    }


def test_analyze_passes_orig_rows_into_severity_baseline(tmp_path):
    """analyze() must thread its own orig_rows into severity_sr.collect() so the paired
    baseline (see severity_sr.py's module docstring) is populated, not left empty."""
    artifact_dir = tmp_path / "artifact"
    orig_rows = [
        _combo_row(s, w, l, p, (s + w + l + p) % 2)
        for s in (0, 1)
        for w in (0, 1)
        for l in (0, 1)
        for p in (0, 1)
        if s or w or l  # flower_eval_libero.py never evaluates all-tokens-off
    ]
    orig_rows.append({**_combo_row(1, 1, 1, 1, 1), "task_name": "foo", "init_state_idx": 0})
    write_csv(artifact_dir / "libero_orig.csv", orig_rows)
    write_csv(
        artifact_dir / "libero_plus.csv",
        [{
            **_combo_row(1, 1, 1, 1, 1),
            "task_category": "Robot Initial States",
            "task_name": "foo_initstate_50",
            "difficulty_level": "1",
        }],
    )

    result = analyze_wandb.analyze(artifact_dir, missing=[], measure="ccs")

    total = [r for r in result["severity"] if r["axis"] == "total" and r["category"] == "Robot Initial States"]
    assert total and total[0]["orig_n"] == 1


def test_analyze_severity_failure_degrades_instead_of_raising(tmp_path, monkeypatch):
    """A broken/missing LIBERO-Plus assets checkout must not abort the whole run's
    analysis -- severity_sr.collect()'s exception is caught, "severity" comes back
    None, and it's reported through the same "missing" list as an absent CSV member."""
    artifact_dir = tmp_path / "artifact"
    write_csv(
        artifact_dir / "libero_plus.csv",
        [{**_combo_row(1, 1, 1, 1, 1), "task_category": "Camera Viewpoints"}],
    )

    import severity_sr

    def _raise(*args, **kwargs):
        raise RuntimeError("LIBERO-Plus assets not downloaded")

    monkeypatch.setattr(severity_sr, "collect", _raise)

    result = analyze_wandb.analyze(artifact_dir, missing=[], measure="ccs")

    assert result["severity"] is None
    assert result["perturbation"] is not None  # unaffected by the severity failure
    assert any("severity" in m for m in result["missing"])


# ---------------------------------------------------------------------------
# pid_values / presence_values / perturbation_values -- wrap pid_modality /
# perturbation_sr exactly
# ---------------------------------------------------------------------------


def test_pid_values_matches_direct_decompose():
    cells = {(0, 0): [0, 0], (1, 0): [0, 1], (0, 1): [1, 0], (1, 1): [1, 1]}
    rows = [
        _combo_row(s, w, 1, 1, succ) for (s, w), successes in cells.items() for succ in successes
    ]

    values = analyze_wandb.pid_values(rows, "ccs")

    assert {pair for pair, _component in values} == {"static+wrist"}
    dist = pid_modality.build_joint(rows, "static", "wrist", ("lang", "proprio"))
    expected = pid_modality.decompose(dist, "ccs")
    for component in ("I", "R", "U1", "U2", "S"):
        assert values[("static+wrist", component)] == pytest.approx(expected[component])


def test_presence_values_matches_presence_success():
    rows = [_combo_row(1, 1, 1, 1, 1), _combo_row(0, 1, 1, 1, 0), _combo_row(1, 0, 1, 1, 1)]
    values = analyze_wandb.presence_values(rows)
    expected = pid_modality.presence_success(rows, pid_modality.MODALITY_ORDER)
    assert values == {k: sr for k, (sr, _n) in expected.items()}


def test_perturbation_values_matches_category_sr_plus_overall():
    rows = [
        {"task_category": "A", "success": 1},
        {"task_category": "A", "success": 0},
        {"task_category": "B", "success": 1},
    ]
    values = analyze_wandb.perturbation_values(rows)
    expected = perturbation_sr.category_sr(rows)
    assert {k: v for k, v in values.items() if k != "OVERALL"} == {
        k: sr for k, (sr, _n) in expected.items()
    }
    assert values["OVERALL"] == pytest.approx(2 / 3)  # 2 successes out of 3 rows total


# ---------------------------------------------------------------------------
# severity_values / _natural_key
# ---------------------------------------------------------------------------


def test_severity_values_keys_by_category_axis_bin():
    records = [
        {"category": "Sensor Noise", "axis": "severity", "bin": "fog_1", "success_rate": 0.8},
        {"category": "Sensor Noise", "axis": "difficulty_level", "bin": "1", "success_rate": 1.0},
    ]
    values = analyze_wandb.severity_values(records)
    assert values == {
        ("Sensor Noise", "severity", "fog_1"): 0.8,
        ("Sensor Noise", "difficulty_level", "1"): 1.0,
    }


def test_natural_key_orders_digit_runs_numerically():
    labels = ["fog_10", "fog_2", "fog_1"]
    assert sorted(labels, key=analyze_wandb._natural_key) == ["fog_1", "fog_2", "fog_10"]


def test_severity_orig_values_skips_records_without_a_baseline():
    records = [
        {
            "category": "Sensor Noise", "axis": "severity", "bin": "fog_1",
            "success_rate": 0.8, "n": 5, "orig_success_rate": 0.95, "orig_n": 10,
        },
        {
            "category": "Sensor Noise", "axis": "severity", "bin": "fog_2",
            "success_rate": 0.5, "n": 2, "orig_success_rate": "", "orig_n": "",
        },
    ]
    assert analyze_wandb.severity_orig_values(records) == {("Sensor Noise", "severity", "fog_1"): 0.95}
    assert analyze_wandb.severity_n_values(records) == {("Sensor Noise", "severity", "fog_1"): (5, 10)}


def test_severity_orig_values_tolerates_records_with_no_orig_keys_at_all():
    records = [{"category": "Sensor Noise", "axis": "severity", "bin": "fog_1", "success_rate": 0.8}]
    assert analyze_wandb.severity_orig_values(records) == {}
    assert analyze_wandb.severity_n_values(records) == {}


def test_aggregate_n_ranges_min_max_per_key():
    per_run = {"r1": {"a": (5, 10)}, "r2": {"a": (8, 10)}}
    assert analyze_wandb._aggregate_n_ranges(per_run) == {"a": ((5, 8), (10, 10))}


def test_n_label_collapses_equal_range():
    assert analyze_wandb._n_label((10, 10)) == "10"
    assert analyze_wandb._n_label((8, 10)) == "8-10"


def test_natural_key_orders_other_bin_vocabularies():
    assert sorted(["rot~15-30deg", "rot~0-15deg"], key=analyze_wandb._natural_key) == [
        "rot~0-15deg",
        "rot~15-30deg",
    ]
    assert sorted(["0.5rad", "0.1rad"], key=analyze_wandb._natural_key) == ["0.1rad", "0.5rad"]


# ---------------------------------------------------------------------------
# aggregate_scalars
# ---------------------------------------------------------------------------


def test_aggregate_scalars_mean_min_max_and_runs():
    per_run = {"r1": {"a": 1.0, "b": 2.0}, "r2": {"a": 3.0}, "r3": {"a": 5.0, "b": 4.0}}
    agg = analyze_wandb.aggregate_scalars(per_run)
    assert agg["a"] == {"mean": 3.0, "min": 1.0, "max": 5.0, "runs": 3}
    assert agg["b"] == {"mean": 3.0, "min": 2.0, "max": 4.0, "runs": 2}


# ---------------------------------------------------------------------------
# parse_config_keys
# ---------------------------------------------------------------------------


def test_parse_config_keys_merges_filter_keys_without_duplicating():
    keys = analyze_wandb.parse_config_keys(
        ",".join(analyze_wandb.DEFAULT_CONFIG_KEYS),
        {"config.modality_dropout": True, "config.foo": 1},
    )
    assert keys.count("modality_dropout") == 1
    assert "foo" in keys


def test_parse_config_keys_all_sentinel():
    assert analyze_wandb.parse_config_keys("all", {"config.modality_dropout": True}) == ["all"]


def test_parse_config_keys_custom_list_no_filters():
    assert analyze_wandb.parse_config_keys("a, b", None) == ["a", "b"]


# ---------------------------------------------------------------------------
# Filtering by trained modality set
# ---------------------------------------------------------------------------

ALL_ON = {"rgb_static": True, "rgb_gripper": True, "language": True, "proprio": True}


def test_run_modalities_four_key_string_all_on_but_use_proprio_false():
    # Config as W&B actually stores it: a Python repr string, not a nested dict. All four
    # keys True, but use_proprio=False means proprio is never actually received.
    run = _FakeRun(
        "r1", "n1",
        {"modalities": "{'rgb_static': True, 'rgb_gripper': True, 'language': True, 'proprio': True}",
         "use_proprio": False},
    )
    assert analyze_wandb.run_modalities(run) == {**ALL_ON, "proprio": False}


def test_run_modalities_three_key_string_with_use_proprio_true():
    # Pre-#042c754 config: no "proprio" key at all, but use_proprio=True -- the model
    # did receive proprio, so the effective set is all four on.
    run = _FakeRun(
        "r2", "n2",
        {"modalities": "{'rgb_static': True, 'rgb_gripper': True, 'language': True}",
         "use_proprio": True},
    )
    assert analyze_wandb.run_modalities(run) == ALL_ON


def test_run_modalities_absent_and_use_proprio_false():
    # No "modalities" key at all (older run) and proprio never enabled.
    run = _FakeRun("r3", "n3", {"use_proprio": False})
    assert analyze_wandb.run_modalities(run) == {**ALL_ON, "proprio": False}


def test_run_modalities_accepts_a_real_dict_value():
    run = _FakeRun(
        "r4", "n4",
        {"modalities": {"rgb_static": False, "rgb_gripper": True, "language": True}, "use_proprio": True},
    )
    assert analyze_wandb.run_modalities(run) == {**ALL_ON, "rgb_static": False}


def test_run_modalities_unparseable_string_returns_none():
    run = _FakeRun("r5", "n5", {"modalities": "not a dict", "use_proprio": True})
    assert analyze_wandb.run_modalities(run) is None


def test_run_modalities_non_dict_non_string_returns_none():
    run = _FakeRun("r6", "n6", {"modalities": 42, "use_proprio": True})
    assert analyze_wandb.run_modalities(run) is None


def test_parse_modality_spec_happy_path():
    assert analyze_wandb.parse_modality_spec("rgb_static, language") == {
        "rgb_static": True,
        "rgb_gripper": False,
        "language": True,
        "proprio": False,
    }


def test_parse_modality_spec_empty_means_all_off():
    assert analyze_wandb.parse_modality_spec("") == {
        "rgb_static": False,
        "rgb_gripper": False,
        "language": False,
        "proprio": False,
    }


def test_parse_modality_spec_unknown_name_raises():
    with pytest.raises(ValueError, match="rgb_statc"):
        analyze_wandb.parse_modality_spec("rgb_statc")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_print_run_header_prints_selected_keys(capsys):
    run = _FakeRun("run123", "libero_10_dropout/ts", {"modality_dropout": True, "use_proprio": False})
    analyze_wandb.print_run_header(run, ["modality_dropout", "use_proprio", "missing_key"])
    out = capsys.readouterr().out
    assert "run123" in out
    assert "modality_dropout = True" in out
    assert "missing_key = <absent>" in out


def test_print_run_header_all_sentinel_dumps_full_config(capsys):
    run = _FakeRun("run123", "name", {"a": 1, "b": 2})
    analyze_wandb.print_run_header(run, ["all"])
    out = capsys.readouterr().out
    assert '"a": 1' in out and '"b": 2' in out


def test_print_run_table_lists_every_run(capsys):
    runs = [_FakeRun("r1", "n1", {"modality_dropout": True}), _FakeRun("r2", "n2", {"modality_dropout": False})]
    analyze_wandb.print_run_table(runs, ["modality_dropout"])
    out = capsys.readouterr().out
    assert "r1" in out and "r2" in out


def test_print_problems_reports_missing_members(capsys):
    per_run = {
        "r1": {"missing": [], "orig_rows": [], "plus_rows": [], "severity": None},
        "r2": {"missing": ["libero_plus.csv"], "orig_rows": [], "plus_rows": [], "severity": None},
    }
    lines = analyze_wandb.print_problems(per_run)
    out = capsys.readouterr().out
    assert lines == ["r2: missing libero_plus.csv"]
    assert "r2: missing libero_plus.csv" in out


def test_print_problems_empty_when_nothing_wrong(capsys):
    per_run = {"r1": {"missing": [], "orig_rows": [], "plus_rows": [], "severity": None}}
    lines = analyze_wandb.print_problems(per_run)
    out = capsys.readouterr().out
    assert lines == []
    assert out == ""


def test_print_aggregate_reports_mean_min_max_and_runs(capsys):
    per_run = {
        "r1": {"orig_rows": [], "plus_rows": [{"task_category": "Camera Viewpoints", "success": 1}], "severity": None},
        "r2": {"orig_rows": [], "plus_rows": [{"task_category": "Camera Viewpoints", "success": 0}], "severity": None},
        "r3": {"orig_rows": [], "plus_rows": [{"task_category": "Sensor Noise", "success": 1}], "severity": None},
    }
    analyze_wandb.print_aggregate(per_run, "ccs")
    out = capsys.readouterr().out

    lines = [line for line in out.splitlines() if line.strip().startswith("Camera Viewpoints")]
    assert len(lines) == 1
    assert "0.5000" in lines[0]  # mean of 1.0 and 0.0
    assert lines[0].split()[-1] == "2"  # only 2 of 3 runs evaluated this category

    # OVERALL: each run's own total-successes/total-n (1.0, 0.0, 1.0) -> mean 2/3,
    # backed by all 3 runs, and printed last regardless of alphabetical sort order.
    perturbation_lines = [
        line
        for line in out.splitlines()
        if line.strip() and line.split()[0] in ("Camera", "Sensor", "OVERALL")
    ]
    assert perturbation_lines[-1].startswith("OVERALL")
    assert f"{2 / 3:.4f}" in perturbation_lines[-1]
    assert perturbation_lines[-1].split()[-1] == "3"


def test_print_aggregate_presence_table_shows_modality_order(capsys):
    per_run = {
        "r1": {"orig_rows": [_combo_row(1, 1, 1, 1, 1)], "plus_rows": [], "severity": None},
        "r2": {"orig_rows": [_combo_row(1, 1, 1, 1, 0)], "plus_rows": [], "severity": None},
    }
    analyze_wandb.print_aggregate(per_run, "ccs")
    out = capsys.readouterr().out

    header_line = next(line for line in out.splitlines() if "static" in line and "wrist" in line)
    assert header_line.split()[:4] == list(pid_modality.MODALITY_ORDER)


def test_print_aggregate_emits_both_severity_tables(capsys):
    per_run = {
        "r1": {
            "orig_rows": [],
            "plus_rows": [],
            "severity": [
                {"category": "Sensor Noise", "axis": "difficulty_level", "bin": "1", "success_rate": 1.0},
                {"category": "Sensor Noise", "axis": "severity", "bin": "fog_10", "success_rate": 0.5},
                {"category": "Sensor Noise", "axis": "severity", "bin": "fog_2", "success_rate": 0.9},
            ],
        },
    }
    analyze_wandb.print_aggregate(per_run, "ccs")
    out = capsys.readouterr().out

    assert "Success rate by physical perturbation severity (mean / min / max across runs):" in out
    assert "Success rate by upstream difficulty_level (mean / min / max across runs):" in out
    # fog_2 must print before fog_10 (natural-key order), not lexical order
    severity_lines = [line for line in out.splitlines() if "Sensor Noise / fog" in line]
    assert [line.split()[0] for line in severity_lines] == ["Sensor", "Sensor"]
    fog_2_idx = next(i for i, line in enumerate(severity_lines) if "fog_2" in line)
    fog_10_idx = next(i for i, line in enumerate(severity_lines) if "fog_10" in line)
    assert fog_2_idx < fog_10_idx


def test_print_aggregate_no_severity_tables_when_no_run_has_severity(capsys):
    per_run = {"r1": {"orig_rows": [], "plus_rows": [], "severity": None}}
    analyze_wandb.print_aggregate(per_run, "ccs")
    out = capsys.readouterr().out
    assert "physical perturbation severity" not in out
    assert "upstream difficulty_level" not in out


# ---------------------------------------------------------------------------
# print_aggregate -- paired (init-state-matched LIBERO original) tables
# ---------------------------------------------------------------------------


def _paired_severity_record(bin_label, axis, success_rate, n, orig_success_rate, orig_n):
    return {
        "category": "Sensor Noise", "axis": axis, "bin": bin_label,
        "success_rate": success_rate, "n": n,
        "orig_success_rate": orig_success_rate, "orig_n": orig_n,
    }


def test_print_aggregate_emits_paired_baseline_tables(capsys):
    per_run = {
        "r1": {
            "orig_rows": [], "plus_rows": [],
            "severity": [
                _paired_severity_record("ALL", "total", 0.8, 10, 0.95, 10),
                _paired_severity_record("fog_1", "severity", 0.8, 8, 0.95, 8),
                _paired_severity_record("1", "difficulty_level", 0.8, 10, 0.95, 10),
            ],
        },
        "r2": {
            "orig_rows": [], "plus_rows": [],
            "severity": [
                _paired_severity_record("ALL", "total", 0.6, 10, 0.90, 8),
                _paired_severity_record("fog_1", "severity", 0.6, 8, 0.90, 8),
                _paired_severity_record("1", "difficulty_level", 0.6, 10, 0.90, 8),
            ],
        },
    }
    analyze_wandb.print_aggregate(per_run, "ccs")
    out = capsys.readouterr().out
    lines = out.splitlines()

    total_title = "Per-perturbation success rate vs. init-state-matched LIBERO original (mean / min / max across runs):"
    severity_title = (
        "Success rate by physical perturbation severity vs. init-state-matched "
        "LIBERO original (mean / min / max across runs):"
    )
    difficulty_title = (
        "Success rate by upstream difficulty_level vs. init-state-matched LIBERO "
        "original (mean / min / max across runs):"
    )
    for title in (total_title, severity_title, difficulty_title):
        assert title in out

    total_idx = lines.index(total_title)
    total_line = lines[total_idx + 2]
    assert total_line.strip().startswith("Sensor Noise")
    assert "8-10" in total_line  # orig_n range across the two runs (10 vs. 8)
    assert "10" in total_line  # plus_n agrees across runs


def test_print_aggregate_no_paired_tables_when_no_run_has_baseline(capsys):
    per_run = {
        "r1": {
            "orig_rows": [], "plus_rows": [],
            "severity": [{"category": "Sensor Noise", "axis": "severity", "bin": "fog_1", "success_rate": 0.8, "n": 5}],
        },
    }
    analyze_wandb.print_aggregate(per_run, "ccs")
    out = capsys.readouterr().out
    assert "init-state-matched LIBERO original" not in out


def test_unpaired_severity_tables_exclude_total_axis_records(capsys):
    per_run = {
        "r1": {
            "orig_rows": [], "plus_rows": [],
            "severity": [
                _paired_severity_record("ALL", "total", 0.7, 20, 0.9, 10),
                {"category": "Sensor Noise", "axis": "severity", "bin": "fog_1", "success_rate": 0.8, "n": 5},
            ],
        },
    }
    analyze_wandb.print_aggregate(per_run, "ccs")
    out = capsys.readouterr().out
    lines = out.splitlines()

    title_idx = lines.index("Success rate by physical perturbation severity (mean / min / max across runs):")
    # only the fog_1 bin row follows (header, then data) -- no "ALL"/total row leaked in
    assert "ALL" not in lines[title_idx + 2]
    assert "fog_1" in lines[title_idx + 2]
