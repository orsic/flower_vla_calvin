"""Tests for flower.evaluation.flower_eval_libero.task_language.

LIBERO-Plus derives the model's instruction from the BDDL *filename*
(libero.libero.benchmark.grab_language_from_filename), which leaks the perturbation id
into the prompt for every category except "Language Instructions" (e.g. "turn on the
stove ... table 1", "... view 0 0 100 2 6 initstate 0"). task_language() reads the
clean instruction from the BDDL's (:language ...) field instead, when
libero_variant == "plus" -- see flower_eval_libero.py's task_language docstring.

bddl.parsing (used by get_problem_info) lowercases every token, so real LIBERO-Plus
BDDLs already hold lowercase instructions; the fixtures below are written lowercase
to match.
"""
from collections import namedtuple

from flower.evaluation.flower_eval_libero import task_language

Task = namedtuple(
    "Task", ["name", "language", "problem", "problem_folder", "bddl_file", "init_states_file"]
)

_BDDL_TEMPLATE = """(define (problem {problem_name})
  (:domain robosuite)
  (:language {language})
    (:regions
    )
  (:fixtures
  )
  (:objects
  )
  (:obj_of_interest
  )
  (:init
  )
  (:goal
    (And)
  )
)
"""


def _write_bddl(suite_dir, filename, language, problem_name="test_problem"):
    (suite_dir / filename).write_text(
        _BDDL_TEMPLATE.format(problem_name=problem_name, language=language)
    )


def _suite_dir(tmp_path):
    d = tmp_path / "libero_10"
    d.mkdir()
    return d


def test_reads_clean_language_for_perturbation_suffixed_task(tmp_path):
    """A non-'_language_' variant (e.g. Background Textures' 'table_1') must read the
    clean instruction from the BDDL, not the perturbation-suffixed filename string
    LIBERO-Plus put in task_i.language."""
    suite_dir = _suite_dir(tmp_path)
    _write_bddl(suite_dir, "pick_up_the_thing_table_1.bddl", "pick up the thing")
    task_i = Task(
        name="pick_up_the_thing_table_1",
        language="pick up the thing table 1",  # what LIBERO-Plus currently emits
        problem="Libero",
        problem_folder="libero_10",
        bddl_file="pick_up_the_thing_table_1.bddl",
        init_states_file="x",
    )

    assert task_language("plus", str(tmp_path), task_i) == "pick up the thing"


def test_strips_view_initstate_suffix_before_lookup(tmp_path):
    """Camera/robot/noise variants carry '_view_..._initstate_N[_noise_M]' in the
    filename; the underlying BDDL is the un-suffixed base file (mirrors
    ControlEnv.__init__'s own filename split in LIBERO-Plus's env_wrapper.py)."""
    suite_dir = _suite_dir(tmp_path)
    _write_bddl(suite_dir, "pick_up_the_thing.bddl", "pick up the thing")
    task_i = Task(
        name="pick_up_the_thing_view_0_0_100_2_6_initstate_0",
        language="pick up the thing view 0 0 100 2 6 initstate 0",
        problem="Libero",
        problem_folder="libero_10",
        bddl_file="pick_up_the_thing_view_0_0_100_2_6_initstate_0.bddl",
        init_states_file="x",
    )

    assert task_language("plus", str(tmp_path), task_i) == "pick up the thing"


def test_language_instructions_variant_reads_its_own_bddl(tmp_path):
    """A '_language_K' rewrite's own BDDL already holds the rewritten instruction (no
    perturbation suffix to strip) -- task_language must resolve to that file as-is and
    return exactly what LIBERO-Plus already produces for this category."""
    suite_dir = _suite_dir(tmp_path)
    _write_bddl(suite_dir, "pick_up_the_thing_language_1.bddl", "grab the object")
    task_i = Task(
        name="pick_up_the_thing_language_1",
        language="grab the object",  # already clean today
        problem="Libero",
        problem_folder="libero_10",
        bddl_file="pick_up_the_thing_language_1.bddl",
        init_states_file="x",
    )

    assert task_language("plus", str(tmp_path), task_i) == "grab the object"


def test_orig_variant_returns_task_i_language_unchanged(tmp_path):
    """LIBERO_VARIANT=orig must be a pure passthrough -- no filesystem access, no
    behavior change to today's (already-correct) orig eval."""
    task_i = Task(
        name="x",
        language="pick up the black bowl and place it on the plate",
        problem="Libero",
        problem_folder="libero_10",
        bddl_file="does_not_exist.bddl",
        init_states_file="x",
    )

    assert task_language("orig", str(tmp_path), task_i) == task_i.language
