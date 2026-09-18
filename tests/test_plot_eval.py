"""Smoke tests for scripts/plot_eval.py -- rendering, not the numbers (plot_data.py's
tests own those). Each test hand-builds the exact shape one prepare_*() function
would have returned and asserts a non-empty PDF comes out.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import plot_data  # noqa: E402
import plot_eval  # noqa: E402


def _bar(successes, n, runs=("r1",)):
    rate = successes / n if n else float("nan")
    return plot_data.Bar(label="x", successes=successes, n=n, rate=rate, ci_low=0.1, ci_high=0.9, runs=list(runs))


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


def test_keep_p_color_known_values_are_distinct():
    # 0.0 is a use_proprio=False run's effective keep_p (plot_data._effective_keep_p),
    # not a literal dropout setting -- it still needs its own color in this figure.
    colors = {plot_eval.keep_p_color(kp) for kp in (1.0, 0.75, 0.5, 0.25, 0.0)}
    assert len(colors) == 5


def test_keep_p_color_raises_on_an_unexpected_value():
    with pytest.raises(ValueError):
        plot_eval.keep_p_color(0.9)


def test_keep_p_color_of_1_is_the_yellow_end_of_cividis():
    # cividis(1.0) is yellow, cividis(0.0) is navy -- keep_p=1.0 must map to the
    # lighter/yellower end, keep_p=0.0 (use_proprio=False) to the darkest/navy end.
    import matplotlib.colors as mcolors

    lightest = mcolors.to_rgb(plot_eval.keep_p_color(1.0))
    darkest = mcolors.to_rgb(plot_eval.keep_p_color(0.0))
    assert sum(lightest) > sum(darkest)


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
