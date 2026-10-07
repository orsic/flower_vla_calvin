"""Tests for flower_eval_libero's task-subset knobs (base_task / max_tasks).

These narrow the evaluation itself, unlike rerun.max_episodes which only caps recording.
Pure logic -- no MuJoCo, no model.
"""

import pytest

from flower.evaluation.flower_eval_libero import restrict_tasks, task_filter_label

# Two Camera-Viewpoint variants of one base task, one of another, plus an orig-style name.
TASK_NAMES = [
    "KITCHEN_SCENE3_turn_on_the_stove_view_0_0_100_2_6_initstate_0",
    "KITCHEN_SCENE3_turn_on_the_stove_view_1_15_100_0_0_initstate_0",
    "LIVING_ROOM_SCENE2_put_both_view_0_0_100_2_6_initstate_0",
    "KITCHEN_SCENE3_turn_on_the_stove",
]
ALL = list(range(len(TASK_NAMES)))


def test_no_filters_is_a_passthrough():
    assert restrict_tasks(ALL, TASK_NAMES) == ALL


def test_base_task_keeps_every_perturbation_variant():
    """The point of the knob: one base task's variants are the same episode under
    different perturbations, so this selects 'the same episode, many camera angles'."""
    got = restrict_tasks(ALL, TASK_NAMES, base_task="KITCHEN_SCENE3_turn_on_the_stove")
    assert got == [0, 1, 3]


def test_max_tasks_samples_evenly_rather_than_truncating():
    """A perturbation suite orders variants by parameter, so the first N would be a
    cluster of near-identical perturbations. Even spacing samples the actual range."""
    forty_eight = list(range(48))
    names = [f"T_view_0_0_{i}_0_0_initstate_0" for i in range(48)]
    got = restrict_tasks(forty_eight, names, max_tasks=5)
    assert got == [0, 12, 24, 35, 47]  # first and last always included


def test_max_tasks_is_deterministic_so_two_arms_pick_the_same_variants():
    names = [f"T_view_0_0_{i}_0_0_initstate_0" for i in range(48)]
    a = restrict_tasks(list(range(48)), names, max_tasks=10)
    b = restrict_tasks(list(range(48)), names, max_tasks=10)
    assert a == b and len(a) == 10


def test_max_tasks_of_one_takes_the_first():
    assert restrict_tasks(ALL, TASK_NAMES, max_tasks=1) == [0]


def test_max_tasks_at_or_above_the_count_keeps_everything():
    assert restrict_tasks(ALL, TASK_NAMES, max_tasks=len(ALL)) == ALL
    assert restrict_tasks(ALL, TASK_NAMES, max_tasks=99) == ALL


def test_base_task_and_max_tasks_compose():
    got = restrict_tasks(
        ALL, TASK_NAMES, base_task="KITCHEN_SCENE3_turn_on_the_stove", max_tasks=2
    )
    assert got == [0, 3]  # the base task's first and last variant


def test_restriction_applies_on_top_of_an_existing_category_subset():
    """select_task_indices runs first; restrict_tasks narrows what it returned."""
    assert restrict_tasks([1, 2], TASK_NAMES, base_task="KITCHEN_SCENE3_turn_on_the_stove") == [1]


def test_base_task_matching_nothing_raises():
    with pytest.raises(ValueError, match="matched no task"):
        restrict_tasks(ALL, TASK_NAMES, base_task="NO_SUCH_TASK")


def test_max_tasks_zero_or_none_is_no_cap():
    assert restrict_tasks(ALL, TASK_NAMES, max_tasks=0) == ALL
    assert restrict_tasks(ALL, TASK_NAMES, max_tasks=None) == ALL


# --- task_category_filter bookkeeping ----------------------------------------------


def test_label_is_empty_only_for_a_full_suite_run():
    """eval_pipeline.already_done() reads emptiness as 'suite complete', so every
    restriction must make this non-empty or --resume skips an unfinished suite."""
    assert task_filter_label(None) == ""
    assert task_filter_label(None, max_tasks=0) == ""
    assert task_filter_label("Camera Viewpoints") != ""
    assert task_filter_label(None, base_task="KITCHEN_SCENE3") != ""
    assert task_filter_label(None, max_tasks=5) != ""


def test_label_records_every_active_restriction():
    label = task_filter_label("Camera Viewpoints", "KITCHEN_SCENE3", 10)
    assert label == "Camera Viewpoints;base_task=KITCHEN_SCENE3;max_tasks=10"
