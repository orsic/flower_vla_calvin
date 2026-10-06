"""Tests for flower.evaluation.flower_eval_libero.task_language.

The model was fine-tuned on LIBERO's filename-derived instruction (training runs
LIBERO_VARIANT=orig, prompting with benchmark.get_task(i).language). LIBERO-Plus's
task.language is also filename-derived, but from the perturbation-suffixed filename, so
it leaks the perturbation into the prompt for every category except Language
Instructions. task_language() maps a Plus task back to the original LIBERO task it was
generated from and re-derives the instruction from that name instead -- see
flower_eval_libero.py's task_language docstring.
"""
from collections import namedtuple

import pytest

from flower.evaluation.flower_eval_libero import task_language

Task = namedtuple(
    "Task", ["name", "language", "problem", "problem_folder", "bddl_file", "init_states_file"]
)

ORIG_TASK_NAMES = [
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
]


def test_background_texture_variant_uses_base_task_instruction():
    task_i = Task(
        name="pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_table_1",
        language="pick up the black bowl between the plate and the ramekin and place it on the plate table 1",
        problem="Libero", problem_folder="libero_spatial",
        bddl_file="..._table_1.bddl", init_states_file="x",
    )
    assert task_language("plus", task_i, ORIG_TASK_NAMES) == (
        "pick up the black bowl between the plate and the ramekin and place it on the plate"
    )


def test_view_initstate_noise_variant_uses_base_task_instruction():
    task_i = Task(
        name="pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_view_0_0_100_2_6_initstate_0_noise_12",
        language="pick up the black bowl ... view 0 0 100 2 6 initstate 0 noise 12",
        problem="Libero", problem_folder="libero_spatial",
        bddl_file="..._view_0_0_100_2_6_initstate_0_noise_12.bddl", init_states_file="x",
    )
    assert task_language("plus", task_i, ORIG_TASK_NAMES) == (
        "pick up the black bowl between the plate and the ramekin and place it on the plate"
    )


@pytest.mark.parametrize(
    "suffix",
    ["_add_10", "_level1_sample3"],
)
def test_objects_layout_variant_uses_base_task_instruction(suffix):
    """The old BDDL-reading implementation handled neither Objects Layout suffix
    shape (its init states -- and hence its BDDL -- live under a different tree); the
    base-task mapping doesn't care about suffix shape at all."""
    task_i = Task(
        name="pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate" + suffix,
        language="pick up the black bowl ..." + suffix.replace("_", " "),
        problem="Libero", problem_folder="libero_spatial",
        bddl_file="..." + suffix + ".bddl", init_states_file="x",
    )
    assert task_language("plus", task_i, ORIG_TASK_NAMES) == (
        "pick up the black bowl between the plate and the ramekin and place it on the plate"
    )


def test_language_instructions_variant_keeps_its_own_rewrite():
    """Language Instructions tasks carry a compound "_language_K_view_..." suffix --
    the rewrite is the independent variable, so task_i.language (LIBERO-Plus's own
    BDDL-derived rewrite) is passed through untouched, not remapped to the base task."""
    task_i = Task(
        name="pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_language_2_view_0_0_100_0_0_initstate_0",
        language="grab the object between the plate and the ramekin",
        problem="Libero", problem_folder="libero_spatial",
        bddl_file="..._language_2_view_0_0_100_0_0_initstate_0.bddl", init_states_file="x",
    )
    result = task_language("plus", task_i, ORIG_TASK_NAMES)
    assert result == task_i.language
    assert result != "pick up the black bowl between the plate and the ramekin and place it on the plate"


def test_libero_10_scene_prefixed_task_derives_instruction():
    task_i = Task(
        name="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_table_1",
        language="turn on the stove and put the moka pot on it table 1",
        problem="Libero", problem_folder="libero_10",
        bddl_file="..._table_1.bddl", init_states_file="x",
    )
    assert task_language("plus", task_i, ORIG_TASK_NAMES) == "turn on the stove and put the moka pot on it"


def test_orig_variant_returns_task_i_language_unchanged():
    """LIBERO_VARIANT=orig must be a pure passthrough, even with no orig_task_names
    available (orig eval never consults the mapping)."""
    task_i = Task(
        name="x",
        language="pick up the black bowl and place it on the plate",
        problem="Libero", problem_folder="libero_10",
        bddl_file="does_not_exist.bddl", init_states_file="x",
    )
    assert task_language("orig", task_i, []) == task_i.language


def test_task_category_is_never_consulted():
    """Dispatch is purely on "_language_" in the task name -- no task_classification.json
    lookup, so a Plus task resolves correctly with no category information available at all."""
    task_i = Task(
        name="pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_table_1",
        language="irrelevant -- not read",
        problem="Libero", problem_folder="libero_spatial",
        bddl_file="..._table_1.bddl", init_states_file="x",
    )
    assert task_language("plus", task_i, ORIG_TASK_NAMES) == (
        "pick up the black bowl between the plate and the ramekin and place it on the plate"
    )


def test_unmatched_task_name_raises():
    task_i = Task(
        name="totally_unrelated_task_1",
        language="irrelevant",
        problem="Libero", problem_folder="libero_spatial",
        bddl_file="totally_unrelated_task_1.bddl", init_states_file="x",
    )
    with pytest.raises(ValueError, match="totally_unrelated_task_1"):
        task_language("plus", task_i, ORIG_TASK_NAMES)


# --- LIBERO-PRO -------------------------------------------------------------------
# PRO's task names are the *original* names, so task_i.language is always the
# training-matched instruction. That is the right control for the object and position
# suites (the perturbation is in the scene), but it would make the semantic and task
# suites inert -- their whole perturbation lives in the bddl's (:language ...) line.

_PRO_BDDL = """(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
  (:domain robosuite)
  (:language switch stove on and put moka pot on it)
  (:fixtures
    kitchen_table - kitchen_table
  )
  (:objects
    moka_pot_1 - moka_pot
  )
  (:obj_of_interest
    moka_pot_1
  )
  (:init
    (On moka_pot_1 kitchen_table_moka_pot_init_region)
  )
  (:goal
    (And (On moka_pot_1 kitchen_table))
  )
)
"""


@pytest.fixture
def pro_bddl(tmp_path):
    path = tmp_path / "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl"
    path.write_text(_PRO_BDDL)
    return str(path)


def _pro_task(suite):
    return Task(
        name="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        language="turn on the stove and put the moka pot on it",
        problem="Libero", problem_folder=suite,
        bddl_file="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl",
        init_states_file="x",
    )


@pytest.mark.parametrize("suite", ["libero_10_lan", "libero_10_task"])
def test_pro_perturbed_instruction_comes_from_the_bddl(suite, pro_bddl):
    """The semantic and task suites perturb (:language ...); prompting from the filename
    would hand the model the unperturbed instruction and make the suite a no-op."""
    assert task_language("pro", _pro_task(suite), [], pro_bddl) == (
        "switch stove on and put moka pot on it"
    )


@pytest.mark.parametrize("suite", ["libero_10_object", "libero_10_swap"])
def test_pro_scene_perturbations_keep_the_training_matched_instruction(suite, pro_bddl):
    """object/swap leave (:language ...) alone; task_i.language is the original, which is
    the correct control -- the bddl is not read at all."""
    task_i = _pro_task(suite)
    assert task_language("pro", task_i, [], pro_bddl) == task_i.language


def test_pro_matched_arm_dispatches_like_pro():
    """The matched arm folds its name into libero_variant ("pro_matched"); instruction
    dispatch must still see it as LIBERO-PRO."""
    task_i = _pro_task("libero_10_swap")
    assert task_language("pro_matched", task_i, [], None) == task_i.language


def test_pro_perturbed_suite_without_bddl_path_raises():
    with pytest.raises(ValueError, match="bddl_path"):
        task_language("pro", _pro_task("libero_10_lan"), [], None)
