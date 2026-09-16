"""Tests for scripts/perturbation_severity.py -- LIBERO-Plus perturbation magnitudes.

Table-driven against real task-name shapes from task_classification.json (see the
module docstring for how each was verified against the LIBERO-Plus source that applies
the perturbation). light_severity is exercised separately since it needs real BDDL/XML
fixtures (LIBERO-Plus's own scene assets), not just a task_name.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from perturbation_severity import (  # noqa: E402
    background_texture_subtype,
    camera_severity,
    light_severity,
    objects_layout_severity,
    robot_initial_state_severity,
    sensor_noise_severity,
)


# ---------------------------------------------------------------------------
# Camera Viewpoints
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "task_name, expected_rotation_deg, expected_scale",
    [
        # nominal in every field -> no perturbation at all
        ("X_view_0_0_100_0_0_initstate_0", 0.0, 1.0),
        # pure horizon (yaw) rotation equals the filename value directly
        ("X_view_11_0_100_0_0_initstate_0", 11.0, 1.0),
        # filename values wrap mod 360: 358 encodes -2 degrees, same magnitude as 2
        ("X_view_358_0_100_0_0_initstate_0", 2.0, 1.0),
        # scale is independent of rotation: 185 -> 1.85x, no rotation
        ("X_view_0_0_185_0_0_initstate_0", 0.0, 1.85),
        # a real Robot Initial States name carries this same nominal view prefix
        ("X_view_0_0_100_0_0_initstate_347", 0.0, 1.0),
    ],
)
def test_camera_severity(task_name, expected_rotation_deg, expected_scale):
    rotation_deg, scale = camera_severity(task_name)
    assert rotation_deg == pytest.approx(expected_rotation_deg, abs=1e-6)
    assert scale == pytest.approx(expected_scale, abs=1e-9)


def test_camera_severity_none_without_view_suffix():
    assert camera_severity("pick_up_the_thing_table_1") is None


# ---------------------------------------------------------------------------
# Robot Initial States
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "task_name, expected_rad",
    [
        ("X_view_0_0_100_0_0_initstate_0", 0.0),  # nominal robot, no swap
        ("X_view_0_0_100_0_0_initstate_1", 0.1),
        ("X_view_0_0_100_0_0_initstate_100", 0.1),
        ("X_view_0_0_100_0_0_initstate_101", 0.2),
        ("X_view_0_0_100_0_0_initstate_347", 0.4),
        ("X_view_0_0_100_0_0_initstate_500", 0.5),
    ],
)
def test_robot_initial_state_severity(task_name, expected_rad):
    assert robot_initial_state_severity(task_name) == pytest.approx(expected_rad, abs=1e-9)


def test_robot_initial_state_severity_none_without_initstate():
    assert robot_initial_state_severity("pick_up_the_thing_table_1") is None


# ---------------------------------------------------------------------------
# Sensor Noise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "task_name, expected",
    [
        ("X_noise_1", ("motion_blur", 1)),
        ("X_noise_10", ("motion_blur", 10)),
        ("X_noise_11", ("gaussian_blur", 1)),
        ("X_view_0_0_100_0_0_initstate_0_noise_37", ("fog", 7)),
        ("X_noise_41", ("glass_blur", 1)),
        ("X_noise_50", ("glass_blur", 10)),
    ],
)
def test_sensor_noise_severity(task_name, expected):
    assert sensor_noise_severity(task_name) == expected


def test_sensor_noise_severity_none_without_noise_suffix():
    assert sensor_noise_severity("pick_up_the_thing_table_1") is None


# ---------------------------------------------------------------------------
# Objects Layout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "task_name, expected",
    [
        ("X_add_1", ("distractors", 1)),
        ("X_add_6", ("distractors", 1)),
        ("X_add_7", ("distractors", 2)),
        ("X_add_23", ("distractors", 4)),
        ("X_add_25", ("distractors", 5)),
        ("X_add_30", ("distractors", 5)),
        ("X_level1_sample4", ("displacement", 1)),
        ("X_level4_sample2", ("displacement", 4)),
        ("X_level5_sample3", ("displacement", 5)),
    ],
)
def test_objects_layout_severity(task_name, expected):
    assert objects_layout_severity(task_name) == expected


def test_objects_layout_severity_none_without_add_or_level():
    assert objects_layout_severity("pick_up_the_thing_table_1") is None


@pytest.mark.parametrize("add_index", range(1, 31))
def test_objects_layout_distractor_count_matches_bddl_region_blocks(add_index):
    """Invariant behind n_distractors = ceil(A/6): each add_N BDDL region block adds
    exactly 2 <region> entries per distractor (verified by direct inspection of
    LIBERO-Plus's add_1..add_30 kitchen-scene BDDLs)."""
    libero_plus = pytest.importorskip("libero.libero", reason="needs the LIBERO-Plus submodule on sys.path")
    import os

    bddl_root = os.path.join(os.path.dirname(libero_plus.__file__), "bddl_files", "libero_10")
    bddl_path = os.path.join(
        bddl_root,
        f"KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_add_{add_index}.bddl",
    )
    if not os.path.exists(bddl_path):
        pytest.skip(f"fixture BDDL not present at {bddl_path} (LIBERO-Plus assets not checked out)")

    text = open(bddl_path).read()
    n_regions = text.count("add_object_region")
    _, expected_distractors = objects_layout_severity(f"X_add_{add_index}")
    assert n_regions == 2 * expected_distractors


# ---------------------------------------------------------------------------
# Background Textures (categorical, no magnitude)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "task_name, expected",
    [
        ("X_table_1", "table"),
        ("X_table_28", "table"),
        ("X_tb_11", "tb"),
        ("X_tb_23", "tb"),
    ],
)
def test_background_texture_subtype(task_name, expected):
    assert background_texture_subtype(task_name) == expected


def test_background_texture_subtype_none_for_other_categories():
    assert background_texture_subtype("X_language_1") is None
    assert background_texture_subtype("X_add_14") is None


# ---------------------------------------------------------------------------
# Light Conditions (needs real BDDL + scene XML fixtures)
# ---------------------------------------------------------------------------

def _libero_plus_root():
    libero_libero = pytest.importorskip("libero.libero", reason="needs the LIBERO-Plus submodule on sys.path")
    import os

    # libero.libero.__file__ is .../LIBERO-plus/libero/libero/__init__.py
    return os.path.dirname(os.path.dirname(os.path.dirname(libero_libero.__file__)))


def test_light_severity_none_for_non_light_task():
    root = _libero_plus_root()
    import os

    bddl_path = os.path.join(
        root,
        "libero/libero/bddl_files/libero_10",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_table_1.bddl",
    )
    if not os.path.exists(bddl_path):
        pytest.skip("LIBERO-Plus BDDL fixtures not present")
    assert light_severity(bddl_path, root) is None


@pytest.mark.parametrize(
    "bddl_name",
    [
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_light_1.bddl",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_light_10.bddl",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_light_10.bddl",
    ],
)
def test_light_severity_returns_composite_for_every_scene_prefix(bddl_name):
    """One fixture per scene prefix LIBERO-Plus's light generator produced for
    libero_10 (kitchen, liv, study) -- each must resolve to its own base-style XML and
    return a well-formed, non-degenerate composite."""
    root = _libero_plus_root()
    import os

    bddl_path = os.path.join(root, "libero/libero/bddl_files/libero_10", bddl_name)
    if not os.path.exists(bddl_path):
        pytest.skip("LIBERO-Plus BDDL fixtures not present")

    result = light_severity(bddl_path, root)
    assert result is not None
    assert 0.0 <= result["severity"] <= 1.0
    assert isinstance(result["shadow_flipped"], bool)
    assert result["dir_delta_deg"] >= 0.0
