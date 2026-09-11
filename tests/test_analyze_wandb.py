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
        "r1": {"orig_rows": good_orig, "plus_rows": good_plus},
        "r2": {"orig_rows": good_orig, "plus_rows": good_plus},
        "r3": {"orig_rows": bad_orig, "plus_rows": bad_plus},
    }
    lines = analyze_wandb.consistency_report(per_run)
    assert any("r3" in line and "modality-combo" in line for line in lines)
    assert any("r3" in line and "perturbation-category" in line for line in lines)


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
        "missing": ["<no evaluation artifact>"],
    }


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
        "r1": {"missing": [], "orig_rows": [], "plus_rows": []},
        "r2": {"missing": ["libero_plus.csv"], "orig_rows": [], "plus_rows": []},
    }
    lines = analyze_wandb.print_problems(per_run)
    out = capsys.readouterr().out
    assert lines == ["r2: missing libero_plus.csv"]
    assert "r2: missing libero_plus.csv" in out


def test_print_problems_empty_when_nothing_wrong(capsys):
    per_run = {"r1": {"missing": [], "orig_rows": [], "plus_rows": []}}
    lines = analyze_wandb.print_problems(per_run)
    out = capsys.readouterr().out
    assert lines == []
    assert out == ""


def test_print_aggregate_reports_mean_min_max_and_runs(capsys):
    per_run = {
        "r1": {"orig_rows": [], "plus_rows": [{"task_category": "Camera Viewpoints", "success": 1}]},
        "r2": {"orig_rows": [], "plus_rows": [{"task_category": "Camera Viewpoints", "success": 0}]},
        "r3": {"orig_rows": [], "plus_rows": [{"task_category": "Sensor Noise", "success": 1}]},
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
        "r1": {"orig_rows": [_combo_row(1, 1, 1, 1, 1)], "plus_rows": []},
        "r2": {"orig_rows": [_combo_row(1, 1, 1, 1, 0)], "plus_rows": []},
    }
    analyze_wandb.print_aggregate(per_run, "ccs")
    out = capsys.readouterr().out

    header_line = next(line for line in out.splitlines() if "static" in line and "wrist" in line)
    assert header_line.split()[:4] == list(pid_modality.MODALITY_ORDER)
