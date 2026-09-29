"""Smoke tests for scripts/plot_eval.py -- rendering, not the numbers (plot_data.py's
tests own those). Each test hand-builds the exact shape one prepare_*() function
would have returned and asserts a non-empty PDF comes out.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import plot_data  # noqa: E402
import plot_eval  # noqa: E402


def _bar(successes, n, runs=("r1",), mcnemar=None):
    rate = successes / n if n else float("nan")
    mcnemar_b, mcnemar_c, mcnemar_n = mcnemar if mcnemar else (None, None, None)
    return plot_data.Bar(
        label="x", successes=successes, n=n, rate=rate, ci_low=0.1, ci_high=0.9, runs=list(runs),
        mcnemar_b=mcnemar_b, mcnemar_c=mcnemar_c, mcnemar_n=mcnemar_n,
    )


def _pair(successes, n, orig=None):
    """(plus_bar, orig_bar_or_None) -- the shape every leaf in plot_severity_*/
    plot_difficulty_* data now takes."""
    return _bar(successes, n), (_bar(*orig) if orig else None)


def _assert_pdf(path: Path):
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.read_bytes()[:5] == b"%PDF-"


def test_plot_presence_writes_a_pdf(tmp_path):
    pool_data = {
        1.0: {cfg: _bar(8, 10) for cfg in plot_data.MODALITY_CONFIGS},
        0.5: {cfg: _bar(5, 10) for cfg in plot_data.MODALITY_CONFIGS},
    }
    path = plot_eval.plot_presence(pool_data, tmp_path)
    assert path.name == "presence_libero10.pdf"
    _assert_pdf(path)


def test_plot_perturbation_writes_a_pdf_with_and_without_orig_bar(tmp_path):
    pool_data = {
        "Camera Viewpoints": {1.0: (_bar(8, 10), _bar(6, 8)), 0.5: (_bar(5, 10), _bar(4, 8))},
        "Language Instructions": {1.0: (_bar(9, 10), None), 0.5: (_bar(7, 10), None)},
    }
    path = plot_eval.plot_perturbation(pool_data, tmp_path)
    assert path.name == "perturbation_libero10plus.pdf"
    _assert_pdf(path)


def test_plot_perturbation_with_baseline_series_and_mcnemar_writes_a_pdf(tmp_path):
    pool_data = {
        "Camera Viewpoints": {
            1.0: (_bar(8, 10, mcnemar=(2, 1, 8)), _bar(6, 8)),
            plot_data.BASELINE_PROPRIO: (_bar(7, 10), _bar(6, 8)),
            plot_data.BASELINE_NO_PROPRIO: (_bar(4, 10), None),
        },
    }
    path = plot_eval.plot_perturbation(pool_data, tmp_path)
    _assert_pdf(path)


def test_plot_severity_percategory_writes_a_pdf_with_a_facet_per_category(tmp_path):
    by_category = {
        "Camera Viewpoints": {
            "rot~0-15deg": {cfg: _pair(8, 10, orig=(6, 8)) for cfg in plot_data.MODALITY_CONFIGS},
            "rot~15-30deg": {cfg: _pair(5, 10) for cfg in plot_data.MODALITY_CONFIGS},
        },
        "Robot Initial States": {
            "0.1rad": {cfg: _pair(6, 10) for cfg in plot_data.MODALITY_CONFIGS},
        },
    }
    path = plot_eval.plot_severity_percategory(by_category, tmp_path)
    assert path.name == "severity_modality_off_percategory.pdf"
    _assert_pdf(path)


def test_plot_severity_percategory_handles_a_single_category(tmp_path):
    by_category = {
        "Sensor Noise": {"fog_1": {cfg: _pair(4, 10) for cfg in plot_data.MODALITY_CONFIGS}},
    }
    path = plot_eval.plot_severity_percategory(by_category, tmp_path)
    _assert_pdf(path)


def test_plot_severity_pooled_writes_a_pdf(tmp_path):
    pooled = {
        "Sensor Noise": {cfg: _pair(8, 10, orig=(6, 8)) for cfg in plot_data.MODALITY_CONFIGS},
        "Camera Viewpoints": {cfg: _pair(5, 10) for cfg in plot_data.MODALITY_CONFIGS},
    }
    path = plot_eval.plot_severity_pooled(pooled, tmp_path)
    assert path.name == "severity_modality_off_pooled.pdf"
    _assert_pdf(path)


def test_plot_difficulty_percategory_writes_a_pdf(tmp_path):
    by_category = {
        "Camera Viewpoints": {"1": {cfg: _pair(9, 10, orig=(8, 9)) for cfg in plot_data.MODALITY_CONFIGS}},
        "Sensor Noise": {"5": {cfg: _pair(2, 10) for cfg in plot_data.MODALITY_CONFIGS}},
    }
    path = plot_eval.plot_difficulty_percategory(by_category, tmp_path)
    assert path.name == "difficulty_modality_off_percategory.pdf"
    _assert_pdf(path)


def test_plot_difficulty_pooled_writes_a_pdf(tmp_path):
    pooled = {
        "1": {cfg: _pair(9, 10, orig=(8, 9)) for cfg in plot_data.MODALITY_CONFIGS},
        "5": {cfg: _pair(2, 10) for cfg in plot_data.MODALITY_CONFIGS},
    }
    path = plot_eval.plot_difficulty_pooled(pooled, tmp_path)
    assert path.name == "difficulty_modality_off_pooled.pdf"
    _assert_pdf(path)


# ---------------------------------------------------------------------------
# "clear" (presentation-grade, McNemar-free, all-modality) figures
# ---------------------------------------------------------------------------


def test_slug_lowercases_and_replaces_non_alnum():
    assert plot_eval._slug("Camera Viewpoints") == "camera_viewpoints"
    assert plot_eval._slug("(unclassified)") == "unclassified"


def test_clear_perturbation_keys_are_keep_p_1_and_baseline_proprio_only():
    assert plot_eval._CLEAR_PERTURBATION_KEYS == [1.0, plot_data.BASELINE_PROPRIO]


def test_single_series_key_returns_the_one_key_in_a_single_config_pool():
    pool = [("r1", 1.0, {}), ("r2", 1.0, {})]
    assert plot_eval._single_series_key(pool) == 1.0


def test_single_series_key_returns_none_for_a_multi_config_pool():
    pool = [("r1", 1.0, {}), ("r2", 0.5, {})]
    assert plot_eval._single_series_key(pool) is None


def test_single_series_key_returns_none_for_an_empty_pool():
    assert plot_eval._single_series_key([]) is None


def test_require_single_series_key_returns_the_key_when_unique():
    pool = [("r1", plot_data.BASELINE_PROPRIO, {})]
    assert plot_eval._require_single_series_key(pool) == plot_data.BASELINE_PROPRIO


def test_require_single_series_key_raises_naming_both_configs():
    pool = [("r1", 1.0, {}), ("r2", plot_data.BASELINE_PROPRIO, {})]
    with pytest.raises(ValueError) as exc_info:
        plot_eval._require_single_series_key(pool)
    message = str(exc_info.value)
    assert "keep_p=1" in message
    assert "proprio" in message
    assert "--filters" in message


def test_clear_series_style_colors_by_training_config_not_the_neutral_gray():
    colors, labels = plot_eval._clear_series_style(plot_data.BASELINE_PROPRIO)
    assert colors == {"all": plot_eval.series_color(plot_data.BASELINE_PROPRIO)}
    assert labels == {"all": plot_eval.series_label(plot_data.BASELINE_PROPRIO)}
    assert colors["all"] != plot_eval.MODALITY_COLORS["all"]


def test_clear_plots_skips_severity_difficulty_when_series_key_is_none(tmp_path):
    perturbation_data = {"Camera Viewpoints": {1.0: (_bar(8, 10), _bar(6, 8))}}
    severity_by_category = {"Sensor Noise": {"fog_1": {cfg: _pair(4, 10) for cfg in plot_data.MODALITY_CONFIGS}}}
    difficulty_by_category = {"Camera Viewpoints": {"1": {cfg: _pair(9, 10) for cfg in plot_data.MODALITY_CONFIGS}}}
    paths = plot_eval._clear_plots(perturbation_data, severity_by_category, difficulty_by_category, tmp_path, None)
    assert [p.name for p in paths] == ["perturbation_libero10plus_dropout_vs_nodropout.pdf"]


def test_clear_plots_writes_the_full_set_when_series_key_is_given(tmp_path):
    perturbation_data = {"Camera Viewpoints": {1.0: (_bar(8, 10), _bar(6, 8))}}
    severity_by_category = {"Sensor Noise": {"fog_1": {cfg: _pair(4, 10) for cfg in plot_data.MODALITY_CONFIGS}}}
    difficulty_by_category = {"Camera Viewpoints": {"1": {cfg: _pair(9, 10) for cfg in plot_data.MODALITY_CONFIGS}}}
    paths = plot_eval._clear_plots(perturbation_data, severity_by_category, difficulty_by_category, tmp_path, 1.0)
    names = {p.name for p in paths}
    assert "perturbation_libero10plus_dropout_vs_nodropout.pdf" in names
    assert "severity_all_pooled.pdf" in names
    assert "difficulty_all_pooled.pdf" in names
    assert "severity_all_percategory_sensor_noise.pdf" in names
    assert "difficulty_all_percategory_camera_viewpoints.pdf" in names
    for path in paths:
        _assert_pdf(path)


def test_plot_perturbation_clear_writes_a_pdf_dropping_non_clear_series(tmp_path):
    # Pool carries 4 training-config series; the clear figure must render fine while
    # silently restricting itself to just the 2 in _CLEAR_PERTURBATION_KEYS.
    pool_data = {
        "Camera Viewpoints": {
            1.0: (_bar(8, 10), _bar(6, 8)),
            0.5: (_bar(5, 10), _bar(4, 8)),
            plot_data.BASELINE_PROPRIO: (_bar(7, 10), _bar(6, 8)),
            plot_data.BASELINE_NO_PROPRIO: (_bar(4, 10), None),
        },
    }
    path = plot_eval.plot_perturbation_clear(pool_data, tmp_path)
    assert path.name == "perturbation_libero10plus_dropout_vs_nodropout.pdf"
    _assert_pdf(path)


def test_plot_perturbation_clear_survives_neither_clear_key_present(tmp_path):
    # A --filters pool that happens to carry neither keep_p=1.0 nor
    # BASELINE_PROPRIO (e.g. a modality_dropout-only filter with no keep_p=1.0 runs)
    # must not divide by zero in _draw_paired_grouped_bars -- regression test for that
    # crash.
    pool_data = {
        "Camera Viewpoints": {0.5: (_bar(5, 10), _bar(4, 8))},
        "Sensor Noise": {0.25: (_bar(3, 10), None)},
    }
    path = plot_eval.plot_perturbation_clear(pool_data, tmp_path)
    _assert_pdf(path)


def test_plot_severity_percategory_clear_writes_one_pdf_per_category(tmp_path):
    by_category = {
        "Camera Viewpoints": {
            "rot~0-15deg": {cfg: _pair(8, 10, orig=(6, 8)) for cfg in plot_data.MODALITY_CONFIGS},
        },
        "Sensor Noise": {
            "fog_1": {cfg: _pair(4, 10) for cfg in plot_data.MODALITY_CONFIGS},
        },
    }
    paths = plot_eval.plot_severity_percategory_clear(by_category, tmp_path, 1.0)
    names = {p.name for p in paths}
    assert names == {"severity_all_percategory_camera_viewpoints.pdf", "severity_all_percategory_sensor_noise.pdf"}
    for path in paths:
        _assert_pdf(path)


def test_plot_difficulty_percategory_clear_writes_one_pdf_per_category(tmp_path):
    by_category = {
        "Camera Viewpoints": {"1": {cfg: _pair(9, 10, orig=(8, 9)) for cfg in plot_data.MODALITY_CONFIGS}},
        "Sensor Noise": {"5": {cfg: _pair(2, 10) for cfg in plot_data.MODALITY_CONFIGS}},
    }
    paths = plot_eval.plot_difficulty_percategory_clear(by_category, tmp_path, plot_data.BASELINE_PROPRIO)
    names = {p.name for p in paths}
    assert names == {"difficulty_all_percategory_camera_viewpoints.pdf", "difficulty_all_percategory_sensor_noise.pdf"}
    for path in paths:
        _assert_pdf(path)


def test_plot_severity_pooled_clear_writes_a_pdf(tmp_path):
    pooled = {
        "Sensor Noise": {cfg: _pair(8, 10, orig=(6, 8)) for cfg in plot_data.MODALITY_CONFIGS},
        "Camera Viewpoints": {cfg: _pair(5, 10) for cfg in plot_data.MODALITY_CONFIGS},
    }
    path = plot_eval.plot_severity_pooled_clear(pooled, tmp_path, 1.0)
    assert path.name == "severity_all_pooled.pdf"
    _assert_pdf(path)


def test_plot_difficulty_pooled_clear_writes_a_pdf(tmp_path):
    pooled = {
        "1": {cfg: _pair(9, 10, orig=(8, 9)) for cfg in plot_data.MODALITY_CONFIGS},
        "5": {cfg: _pair(2, 10) for cfg in plot_data.MODALITY_CONFIGS},
    }
    path = plot_eval.plot_difficulty_pooled_clear(pooled, tmp_path, plot_data.BASELINE_PROPRIO)
    assert path.name == "difficulty_all_pooled.pdf"
    _assert_pdf(path)


def test_draw_paired_grouped_bars_mcnemar_false_adds_no_p_labels():
    fig, ax = plt.subplots()
    series = {"all": [(_bar(8, 10, mcnemar=(2, 1, 8)), _bar(6, 8))]}
    plot_eval._draw_paired_grouped_bars(ax, ["cat1"], series, {"all": "#000000"}, mcnemar=False)
    assert not any(t.get_text().startswith("p") for t in ax.texts)
    plt.close(fig)


def test_draw_paired_grouped_bars_mcnemar_true_adds_p_labels():
    fig, ax = plt.subplots()
    series = {"all": [(_bar(8, 10, mcnemar=(2, 1, 8)), _bar(6, 8))]}
    plot_eval._draw_paired_grouped_bars(ax, ["cat1"], series, {"all": "#000000"}, mcnemar=True)
    assert any(t.get_text().startswith("p") for t in ax.texts)
    plt.close(fig)


def test_series_color_known_values_are_distinct():
    # 0.0 is a use_proprio=False dropout run's series key (plot_data._series_key), not
    # a literal dropout setting -- it still needs its own color in this figure.
    colors = {
        plot_eval.series_color(kp)
        for kp in (1.0, 0.75, 0.5, 0.25, 0.0, plot_data.BASELINE_PROPRIO, plot_data.BASELINE_NO_PROPRIO)
    }
    assert len(colors) == 7


def test_series_color_raises_on_an_unexpected_value():
    with pytest.raises(ValueError):
        plot_eval.series_color(0.9)


def test_series_color_of_keep_p_1_is_the_yellow_end_of_cividis():
    # cividis(1.0) is yellow, cividis(0.0) is navy -- keep_p=1.0 must map to the
    # lighter/yellower end, keep_p=0.0 (use_proprio=False) to the darkest/navy end.
    import matplotlib.colors as mcolors

    lightest = mcolors.to_rgb(plot_eval.series_color(1.0))
    darkest = mcolors.to_rgb(plot_eval.series_color(0.0))
    assert sum(lightest) > sum(darkest)


def test_series_color_baselines_are_outside_the_keep_p_ramp():
    ramp_colors = {plot_eval.series_color(kp) for kp in (1.0, 0.75, 0.5, 0.25, 0.0)}
    assert plot_eval.series_color(plot_data.BASELINE_PROPRIO) not in ramp_colors
    assert plot_eval.series_color(plot_data.BASELINE_NO_PROPRIO) not in ramp_colors


def test_series_label_formats_keep_p_and_baselines_distinctly():
    assert plot_eval.series_label(0.5) == "keep_p=0.5"
    assert "proprio" in plot_eval.series_label(plot_data.BASELINE_PROPRIO)
    assert plot_eval.series_label(plot_data.BASELINE_PROPRIO) != plot_eval.series_label(plot_data.BASELINE_NO_PROPRIO)


def test_series_sort_key_orders_keep_p_descending_then_baselines():
    keys = [plot_data.BASELINE_NO_PROPRIO, 0.25, plot_data.BASELINE_PROPRIO, 1.0, 0.5]
    ordered = sorted(keys, key=plot_eval.series_sort_key)
    assert ordered == [1.0, 0.5, 0.25, plot_data.BASELINE_PROPRIO, plot_data.BASELINE_NO_PROPRIO]


def test_bar_errors_clamps_a_tiny_negative_float_overshoot():
    # pool_bars sums successes/n across runs before computing the Wilson interval;
    # a sub-epsilon float overshoot there must not reach matplotlib's errorbar, which
    # raises ValueError on any negative yerr.
    bar = plot_data.Bar(label="x", successes=1, n=1, rate=0.5, ci_low=0.5 + 1e-12, ci_high=0.5 - 1e-12, runs=["r1"])
    _present, _heights, lo, hi = plot_eval._bar_errors([bar])
    assert lo == [0.0]
    assert hi == [0.0]


def test_plot_perturbation_survives_a_tiny_negative_float_overshoot(tmp_path):
    pathological = plot_data.Bar(
        label="x", successes=1, n=1, rate=0.5, ci_low=0.5 + 1e-12, ci_high=0.5 - 1e-12, runs=["r1"]
    )
    pool_data = {"Camera Viewpoints": {1.0: (pathological, pathological)}}
    path = plot_eval.plot_perturbation(pool_data, tmp_path)
    _assert_pdf(path)


# ---------------------------------------------------------------------------
# bar value labels / McNemar annotations
# ---------------------------------------------------------------------------


def test_annotate_bar_values_adds_one_text_per_bar():
    fig, ax = plt.subplots()
    ax.bar([0, 1], [50.0, 80.0])
    plot_eval._annotate_bar_values(ax, [0, 1], [50.0, 80.0], "white")
    assert len(ax.texts) == 2
    plt.close(fig)


def test_annotate_mcnemar_skips_a_bar_with_no_mcnemar_n():
    fig, ax = plt.subplots()
    ax.bar([0], [50.0])
    plot_eval._annotate_mcnemar(ax, [0], [_bar(5, 10)])
    assert len(ax.texts) == 0
    plt.close(fig)


def test_annotate_mcnemar_adds_one_label_per_bar_carrying_a_table():
    fig, ax = plt.subplots()
    ax.bar([0], [50.0])
    plot_eval._annotate_mcnemar(ax, [0], [_bar(5, 10, mcnemar=(3, 1, 9))])
    assert len(ax.texts) == 1
    assert "n=9" in ax.texts[0].get_text()
    plt.close(fig)
