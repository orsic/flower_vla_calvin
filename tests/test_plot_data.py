"""Tests for scripts/plot_data.py -- gather/filter/prepare logic, no W&B network calls.

Same rationale as tests/test_analyze_wandb.py: pure logic / filesystem tests only.
fetch_pool() itself (wandb.Api(), run.logged_artifacts(), artifact.download()) is a
thin wrapper with no local logic to unit test -- only its cache-skip decision
(_cached_or_download) is, via a duck-typed fake run and a monkeypatched
analyze_wandb.download_run so the network path is provably not taken.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import analyze_wandb  # noqa: E402
import eval_pipeline  # noqa: E402
import plot_data  # noqa: E402
import severity_sr  # noqa: E402

from flower.evaluation.eval_records import write_csv  # noqa: E402


class _FakeRun:
    def __init__(self, run_id, config):
        self.id = run_id
        self.config = config


def _combo_row(static, wrist, lang, proprio, success):
    return {
        "use_rgb_static": static,
        "use_rgb_gripper": wrist,
        "use_language": lang,
        "use_proprio": proprio,
        "success": success,
    }


def _severity_record(category, bin_label, successes, n, axis="severity", orig_successes="", orig_n=""):
    return {
        "category": category,
        "axis": axis,
        "bin": bin_label,
        "successes": successes,
        "n": n,
        "orig_successes": orig_successes,
        "orig_n": orig_n,
    }


# ---------------------------------------------------------------------------
# pool_bars / warn_multi_run
# ---------------------------------------------------------------------------


def test_pool_bars_sums_successes_and_n_before_the_wilson_interval():
    bar = plot_data.pool_bars([("r1", 3, 10), ("r2", 2, 5)], "label")
    assert bar.successes == 5
    assert bar.n == 15
    assert bar.rate == pytest.approx(5 / 15)
    assert (bar.ci_low, bar.ci_high) == severity_sr.wilson_interval(5, 15)
    assert bar.runs == ["r1", "r2"]


def test_pool_bars_single_part_matches_severity_sr_directly():
    bar = plot_data.pool_bars([("r1", 4, 10)], "label")
    assert (bar.ci_low, bar.ci_high) == severity_sr.wilson_interval(4, 10)
    assert bar.runs == ["r1"]


def test_pool_bars_no_parts_gives_nan_rate_and_no_runs():
    bar = plot_data.pool_bars([], "label")
    assert bar.n == 0
    assert math.isnan(bar.rate)
    assert bar.runs == []


def test_warn_multi_run_names_every_contributing_run(capsys):
    bars = [plot_data.Bar(label="x", successes=1, n=2, rate=0.5, ci_low=0.0, ci_high=1.0, runs=["r1", "r2"])]
    plot_data.warn_multi_run(bars)
    err = capsys.readouterr().err
    assert "r1" in err and "r2" in err and "x" in err


def test_warn_multi_run_silent_for_a_single_run_bar(capsys):
    bars = [plot_data.Bar(label="x", successes=1, n=2, rate=0.5, ci_low=0.0, ci_high=1.0, runs=["r1"])]
    plot_data.warn_multi_run(bars)
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# _cached_or_download
# ---------------------------------------------------------------------------


def test_cached_or_download_skips_the_network_when_cache_is_complete(tmp_path, monkeypatch):
    run = _FakeRun("r1", {"use_proprio": True})
    run_dir = tmp_path / "r1"
    write_csv(run_dir / "libero_orig.csv", [_combo_row(1, 1, 1, 1, 1)])
    write_csv(run_dir / "libero_plus.csv", [_combo_row(1, 1, 1, 1, 1)])
    for member in eval_pipeline.MODALITY_OFF_VARIANTS.values():
        write_csv(run_dir / f"libero_plus_{member[1]}.csv", [_combo_row(1, 1, 1, 1, 1)])

    def _fail(*args, **kwargs):
        raise AssertionError("download_run should not be called when the cache is complete")

    monkeypatch.setattr(analyze_wandb, "download_run", _fail)

    artifact_dir, missing = plot_data._cached_or_download(
        run, tmp_path, analyze_wandb.applicable_modality_off(run), refresh=False
    )
    assert artifact_dir == run_dir
    assert missing == []


def test_cached_or_download_redownloads_when_cache_is_incomplete(tmp_path, monkeypatch):
    run = _FakeRun("r1", {"use_proprio": True})
    (tmp_path / "r1").mkdir()  # present but empty -- missing every required member

    called = []
    monkeypatch.setattr(analyze_wandb, "download_run", lambda r, dest: called.append(r.id) or (tmp_path / "r1", []))

    plot_data._cached_or_download(run, tmp_path, analyze_wandb.applicable_modality_off(run), refresh=False)
    assert called == ["r1"]


def test_cached_or_download_refresh_forces_the_network_even_if_cache_is_complete(tmp_path, monkeypatch):
    run = _FakeRun("r1", {"use_proprio": True})
    run_dir = tmp_path / "r1"
    write_csv(run_dir / "libero_orig.csv", [_combo_row(1, 1, 1, 1, 1)])
    write_csv(run_dir / "libero_plus.csv", [_combo_row(1, 1, 1, 1, 1)])
    for member in eval_pipeline.MODALITY_OFF_VARIANTS.values():
        write_csv(run_dir / f"libero_plus_{member[1]}.csv", [_combo_row(1, 1, 1, 1, 1)])

    called = []
    monkeypatch.setattr(analyze_wandb, "download_run", lambda r, dest: called.append(r.id) or (run_dir, []))

    plot_data._cached_or_download(run, tmp_path, analyze_wandb.applicable_modality_off(run), refresh=True)
    assert called == ["r1"]


# ---------------------------------------------------------------------------
# _effective_keep_p
# ---------------------------------------------------------------------------


def test_effective_keep_p_is_zero_when_use_proprio_is_false():
    run = _FakeRun("r1", {"use_proprio": False, "modality_dropout_proprio_keep_p": 0.5})
    assert plot_data._effective_keep_p(run) == 0.0


def test_effective_keep_p_ignores_a_stray_config_value_when_use_proprio_is_false():
    # modality_dropout_proprio_keep_p is present but inert (there's no proprio to keep
    # or drop) -- the run is still 0.0, not that stray value.
    run = _FakeRun("r1", {"use_proprio": False, "modality_dropout_proprio_keep_p": 0.75})
    assert plot_data._effective_keep_p(run) == 0.0


def test_effective_keep_p_is_zero_when_use_proprio_is_false_and_config_key_is_absent():
    run = _FakeRun("r1", {"use_proprio": False})
    assert plot_data._effective_keep_p(run) == 0.0


def test_effective_keep_p_reads_config_when_use_proprio_is_true():
    run = _FakeRun("r1", {"use_proprio": True, "modality_dropout_proprio_keep_p": 0.75})
    assert plot_data._effective_keep_p(run) == 0.75


def test_effective_keep_p_is_none_when_use_proprio_true_and_config_missing():
    run = _FakeRun("r1", {"use_proprio": True})
    assert plot_data._effective_keep_p(run) is None


# ---------------------------------------------------------------------------
# prepare_presence
# ---------------------------------------------------------------------------


def test_prepare_presence_reads_exactly_the_five_intended_combos():
    rows = [_combo_row(int(c["rgb_static"]), int(c["rgb_gripper"]), int(c["language"]), int(c["proprio"]), 1)
            for c in eval_pipeline.modality_combos(use_proprio=True)]
    assert len(rows) == 14  # sanity: the full dropout sweep
    analysis = {"orig_rows": rows, "severity": None, "modality_off": {}}

    result = plot_data.prepare_presence([("r1", 0.5, analysis)])

    assert set(result[0.5]) == set(plot_data.MODALITY_CONFIGS)
    for config_name in plot_data.MODALITY_CONFIGS:
        bar = result[0.5][config_name]
        assert bar.n == 1  # each of the 5 intended combos matches exactly one of the 14 rows
        assert bar.rate == 1.0


def test_prepare_presence_pools_matching_keep_p_across_runs(capsys):
    all_on = eval_pipeline.full_modality_combo()
    row = _combo_row(int(all_on["rgb_static"]), int(all_on["rgb_gripper"]), int(all_on["language"]), int(all_on["proprio"]), 1)
    analysis1 = {"orig_rows": [row], "severity": None, "modality_off": {}}
    analysis2 = {"orig_rows": [row], "severity": None, "modality_off": {}}

    result = plot_data.prepare_presence([("r1", 0.5, analysis1), ("r2", 0.5, analysis2)])

    bar = result[0.5]["all"]
    assert bar.n == 2
    assert bar.runs == ["r1", "r2"]
    assert "r1" in capsys.readouterr().err  # warn_multi_run fired


def test_prepare_presence_keeps_different_keep_p_separate():
    all_on = eval_pipeline.full_modality_combo()
    row = _combo_row(int(all_on["rgb_static"]), int(all_on["rgb_gripper"]), int(all_on["language"]), int(all_on["proprio"]), 1)
    analysis = {"orig_rows": [row], "severity": None, "modality_off": {}}

    result = plot_data.prepare_presence([("r1", 1.0, analysis), ("r2", 0.5, analysis)])

    assert set(result) == {1.0, 0.5}
    assert result[1.0]["all"].runs == ["r1"]
    assert result[0.5]["all"].runs == ["r2"]


# ---------------------------------------------------------------------------
# prepare_perturbation
# ---------------------------------------------------------------------------


def test_prepare_perturbation_pairs_plus_with_orig_when_a_baseline_exists():
    records = [
        _severity_record("Camera Viewpoints", "ALL", 8, 10, axis="total", orig_successes=6, orig_n=8),
        _severity_record("Language Instructions", "ALL", 5, 10, axis="total"),  # no baseline: orig_n == ""
    ]
    analysis = {"severity": records}

    result = plot_data.prepare_perturbation([("r1", 0.5, analysis)])

    plus_bar, orig_bar = result["Camera Viewpoints"][0.5]
    assert (plus_bar.successes, plus_bar.n) == (8, 10)
    assert orig_bar is not None
    assert (orig_bar.successes, orig_bar.n) == (6, 8)

    plus_bar2, orig_bar2 = result["Language Instructions"][0.5]
    assert (plus_bar2.successes, plus_bar2.n) == (5, 10)
    assert orig_bar2 is None


def test_prepare_perturbation_skips_runs_with_no_severity_data():
    analysis = {"severity": None}
    result = plot_data.prepare_perturbation([("r1", 0.5, analysis)])
    assert result == {}


# ---------------------------------------------------------------------------
# prepare_severity
# ---------------------------------------------------------------------------


def test_prepare_severity_axis_severity_never_includes_severityless_categories():
    """severity_sr.collect() itself never emits axis="severity" for Language
    Instructions / Background Textures / UNCLASSIFIED (see severity_sr.py's
    UNORDERED_NOTE gate) -- this exercises that real invariant end to end, through
    prepare_severity, rather than re-asserting a manual filter that doesn't exist."""
    rows = [
        {
            "use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 1, "use_proprio": 1,
            "task_category": "Camera Viewpoints", "task_name": "task_a_view_0_0_100_2_6_initstate_0",
            "difficulty_level": "3", "success": 1,
        },
        {
            "use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 1, "use_proprio": 1,
            "task_category": "Language Instructions", "task_name": "task_a_rewrite_1",
            "difficulty_level": "3", "success": 1,
        },
    ]
    records = severity_sr.collect(rows, severity_sr.DEFAULT_LIBERO_PLUS_ROOT)
    analysis = {"severity": records, "modality_off": {}}

    severity_result = plot_data.prepare_severity([("r1", 0.5, analysis)], axis="severity")
    difficulty_result = plot_data.prepare_severity([("r1", 0.5, analysis)], axis="difficulty_level")

    assert "Language Instructions" not in severity_result
    assert "Camera Viewpoints" in severity_result
    assert "Language Instructions" in difficulty_result  # kept on the difficulty axis


def test_prepare_severity_rejects_an_unknown_axis():
    with pytest.raises(ValueError):
        plot_data.prepare_severity([], axis="bogus")


def test_prepare_severity_sorts_bins_naturally():
    records = [
        _severity_record("Sensor Noise", "fog_10", 1, 1),
        _severity_record("Sensor Noise", "fog_2", 1, 1),
        _severity_record("Sensor Noise", "fog_1", 1, 1),
    ]
    analysis = {"severity": records, "modality_off": {}}

    result = plot_data.prepare_severity([("r1", 0.5, analysis)], axis="severity")

    assert list(result["Sensor Noise"]) == ["fog_1", "fog_2", "fog_10"]


def test_prepare_severity_degrades_instead_of_raising_when_a_modality_off_variant_is_missing():
    """analysis["modality_off"]["plus_no_proprio"] is None when that variant is
    applicable but its CSV wasn't in the artifact (see analyze_wandb.analyze's
    docstring) -- prepare_severity must simply omit that bar, not raise."""
    records = [_severity_record("Sensor Noise", "fog_1", 1, 1)]
    analysis = {"severity": records, "modality_off": {"plus_no_proprio": None}}

    result = plot_data.prepare_severity([("r1", 0.5, analysis)], axis="severity")

    assert "all" in result["Sensor Noise"]["fog_1"]
    assert "no_proprio" not in result["Sensor Noise"]["fog_1"]


# ---------------------------------------------------------------------------
# merge_bars / prepare_difficulty_pooled
# ---------------------------------------------------------------------------


def test_merge_bars_sums_successes_and_unions_runs():
    bar_a = plot_data.Bar(label="a", successes=3, n=5, rate=0.6, ci_low=0.0, ci_high=1.0, runs=["r1"])
    bar_b = plot_data.Bar(label="b", successes=2, n=5, rate=0.4, ci_low=0.0, ci_high=1.0, runs=["r1", "r2"])

    merged = plot_data.merge_bars([bar_a, bar_b], "all")

    assert (merged.successes, merged.n) == (5, 10)
    assert (merged.ci_low, merged.ci_high) == severity_sr.wilson_interval(5, 10)
    assert merged.runs == ["r1", "r2"]  # r1 not duplicated


def test_prepare_difficulty_pooled_sums_across_categories():
    by_category = {
        "Camera Viewpoints": {"3": {"all": _pair(4, 5, orig_successes=3, orig_n=4)}},
        "Sensor Noise": {"3": {"all": _pair(1, 5, orig_successes=1, orig_n=4)}},
    }

    result = plot_data.prepare_difficulty_pooled(by_category)

    plus_bar, orig_bar = result["3"]["all"]
    assert (plus_bar.successes, plus_bar.n) == (5, 10)
    assert plus_bar.runs == ["r1"]  # same run in both categories -- not double-counted in `runs`
    assert (orig_bar.successes, orig_bar.n) == (4, 8)


def test_prepare_difficulty_pooled_sorts_bins_naturally():
    by_category = {
        "Camera Viewpoints": {
            "10": {"all": _pair(1, 1)},
            "2": {"all": _pair(1, 1)},
        },
    }
    result = plot_data.prepare_difficulty_pooled(by_category)
    assert list(result) == ["2", "10"]


def test_prepare_difficulty_pooled_orig_bar_is_none_when_no_bin_has_a_baseline():
    by_category = {"Camera Viewpoints": {"3": {"all": _pair(1, 1)}}}
    _plus_bar, orig_bar = plot_data.prepare_difficulty_pooled(by_category)["3"]["all"]
    assert orig_bar is None


def test_prepare_severity_reads_withheld_modality_bars():
    full_records = [_severity_record("Sensor Noise", "fog_1", 8, 10)]
    off_records = [_severity_record("Sensor Noise", "fog_1", 3, 10)]
    analysis = {
        "severity": full_records,
        "modality_off": {"plus_no_static": {"rows": [], "perturbation": {}, "severity": off_records}},
    }

    result = plot_data.prepare_severity([("r1", 0.5, analysis)], axis="severity")

    all_plus, all_orig = result["Sensor Noise"]["fog_1"]["all"]
    no_static_plus, no_static_orig = result["Sensor Noise"]["fog_1"]["no_static"]
    assert (all_plus.successes, all_plus.n) == (8, 10)
    assert (no_static_plus.successes, no_static_plus.n) == (3, 10)
    assert all_orig is None and no_static_orig is None  # no baseline in either fixture record


def test_prepare_severity_pairs_plus_with_orig_baseline():
    records = [_severity_record("Sensor Noise", "fog_1", 8, 10, orig_successes=6, orig_n=8)]
    analysis = {"severity": records, "modality_off": {}}

    result = plot_data.prepare_severity([("r1", 0.5, analysis)], axis="severity")

    plus_bar, orig_bar = result["Sensor Noise"]["fog_1"]["all"]
    assert (plus_bar.successes, plus_bar.n) == (8, 10)
    assert orig_bar is not None
    assert (orig_bar.successes, orig_bar.n) == (6, 8)


def test_prepare_severity_orig_bar_is_none_without_a_baseline():
    records = [_severity_record("Sensor Noise", "fog_1", 8, 10)]  # orig_n == "" (default)
    analysis = {"severity": records, "modality_off": {}}

    _plus_bar, orig_bar = plot_data.prepare_severity([("r1", 0.5, analysis)], axis="severity")["Sensor Noise"]["fog_1"]["all"]

    assert orig_bar is None


# ---------------------------------------------------------------------------
# prepare_severity_pooled
# ---------------------------------------------------------------------------


def _pair(successes, n, orig_successes=None, orig_n=None, runs=("r1",)):
    plus = plot_data.Bar(label="p", successes=successes, n=n, rate=successes / n, ci_low=0, ci_high=1, runs=list(runs))
    orig = None
    if orig_n is not None:
        orig = plot_data.Bar(
            label="o", successes=orig_successes, n=orig_n, rate=orig_successes / orig_n, ci_low=0, ci_high=1, runs=list(runs)
        )
    return plus, orig


def test_prepare_severity_pooled_sums_a_categorys_own_bins():
    by_category = {
        "Sensor Noise": {
            "fog_1": {"all": _pair(4, 5, orig_successes=3, orig_n=4)},
            "fog_2": {"all": _pair(1, 5, orig_successes=1, orig_n=4)},
        },
        "Camera Viewpoints": {
            "rot~0-15deg": {"all": _pair(9, 10)},
        },
    }

    result = plot_data.prepare_severity_pooled(by_category)

    plus_bar, orig_bar = result["Sensor Noise"]["all"]
    assert (plus_bar.successes, plus_bar.n) == (5, 10)
    assert (orig_bar.successes, orig_bar.n) == (4, 8)
    cv_plus, cv_orig = result["Camera Viewpoints"]["all"]
    assert (cv_plus.successes, cv_plus.n) == (9, 10)
    assert cv_orig is None  # neither bin had a baseline


def test_prepare_severity_pooled_keeps_categories_separate():
    by_category = {
        "Sensor Noise": {"fog_1": {"all": _pair(1, 1)}},
        "Camera Viewpoints": {"rot~0-15deg": {"all": _pair(1, 1)}},
    }
    result = plot_data.prepare_severity_pooled(by_category)
    assert set(result) == {"Sensor Noise", "Camera Viewpoints"}
