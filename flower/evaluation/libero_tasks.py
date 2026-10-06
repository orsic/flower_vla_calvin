"""Mapping a LIBERO-Plus task back to the original LIBERO task it was generated from.

Every LIBERO-Plus task name is "<original task name>_<perturbation suffix>"
("..._view_0_0_100_2_6_initstate_0", "..._table_1", "..._light_3", "..._add_10",
"..._level1_sample3", "..._language_2_view_..."), and the original names themselves are
never in the Plus task map. Longest-prefix matching against the original names recovers
the base task without enumerating the suffix shapes, so a Plus release that adds a new
perturbation family still maps correctly -- measured 0 unmatched out of
2402/2518/2591/2519 tasks in libero_spatial/object/goal/10.

Used by flower.evaluation.flower_eval_libero (to prompt with the base task's
training-matched instruction) and scripts/severity_sr.py (to pair a Plus episode with
its LIBERO original baseline episode).
"""
from pathlib import Path
from typing import List, Optional, Sequence

_INIT_SUFFIX = ".pruned_init"


def base_task(task_name: str, orig_task_names: Sequence[str]) -> Optional[str]:
    """The original LIBERO task a LIBERO-Plus task_name was generated from, by longest
    prefix match (every Plus name is <orig task_name>_<perturbation suffix>, e.g.
    "..._table_1", "..._initstate_50", "..._rewrite_3"). None when no original task
    name is a prefix -- such a row is excluded from the paired baseline rather than
    guessed at. The "_" boundary keeps one task from matching as a spurious prefix of
    another task's name."""
    candidates = [t for t in orig_task_names if task_name == t or task_name.startswith(t + "_")]
    if not candidates:
        return None
    return max(candidates, key=len)


def original_task_names(init_states_folder: str, problem_folder: str) -> List[str]:
    """The un-perturbed LIBERO task names of one suite, read off its init-state files.

    <init_states>/<suite>/ holds exactly one "<original task name>.pruned_init" per
    original task in both packages: LIBERO-Plus never writes a perturbed init file there
    (Benchmark.get_task_init_states strips the perturbation suffix, and Objects Layout's
    fresh init states live under libero_newobj/). Verified set-equal to upstream LIBERO's
    task list for libero_spatial/object/goal/10 -- which makes this the one source of the
    original task list that stays readable while LIBERO_VARIANT=plus has the upstream
    package off PYTHONPATH.
    """
    folder = Path(init_states_folder) / problem_folder
    if not folder.is_dir():
        return []
    return sorted(p.name[: -len(_INIT_SUFFIX)] for p in folder.glob("*" + _INIT_SUFFIX))


# LIBERO-PRO names each perturbed suite "<base suite>_<tag>" and keeps the base suite's
# task names, bddl filenames and init filenames verbatim, so a PRO task maps to its
# original by identity rather than by prefix matching.
#
# "scene identical" records whether the perturbed bddl compiles to the same MuJoCo model
# as the original -- i.e. whether the original init states can be loaded into it. Verified
# by diffing every published libero_10_* bddl against LIBERO's:
#   lan     (:language) only                                -> identical
#   task    (:language) (:goal) (:obj_of_interest)           -> identical
#   swap    (:init (On ...)) placements only                 -> identical, but the init
#           state IS the perturbation, so substituting the original's would undo it
#   object  object/fixture classes (moka_pot -> yellow_moka_pot) -> different meshes
#
# Confirmed in-simulator by scripts/debug_pro_scene_identity.py over all 10 libero_10
# tasks: lan/task/swap reproduce the original's state width *and* its joint and body
# name ordering exactly, while object keeps the width (its replacements are same-DOF
# bodies) but renames the joints and bodies. That last case is why matchability is a
# property of the tag and not something a shape check can decide -- an object-suite
# state would load without erroring and mean something else.
PRO_SUITE_TAGS = ("lan", "object", "swap", "task")
PRO_MATCHABLE_TAGS = ("lan", "task")


def pro_suite_tag(suite: str) -> Optional[str]:
    """The LIBERO-PRO perturbation tag of a suite name, or None for an un-perturbed suite.

    "libero_10_lan" -> "lan"; "libero_10" -> None.
    """
    for tag in PRO_SUITE_TAGS:
        if suite.endswith("_" + tag):
            return tag
    return None


def pro_base_suite(suite: str) -> str:
    """The original LIBERO suite a LIBERO-PRO suite was derived from.

    "libero_10_swap" -> "libero_10"; an un-perturbed suite name is returned unchanged.
    """
    tag = pro_suite_tag(suite)
    return suite[: -(len(tag) + 1)] if tag else suite


def matched_init_states(init_states_folder: str, suite: str, task_name: str):
    """The ORIGINAL suite's init states for a LIBERO-PRO task, or None if absent.

    Returns the array from <init_states>/<base suite>/<task_name>.pruned_init -- the same
    file the orig baseline rolls out from. Loading it into a PRO env whose scene is
    identical (see PRO_MATCHABLE_TAGS) makes every PRO episode an exact counterfactual of
    the corresponding libero_orig.csv episode: same task, same starting state, same
    rollout seed, differing only by the perturbation.
    """
    import torch

    path = Path(init_states_folder) / pro_base_suite(suite) / (task_name + _INIT_SUFFIX)
    if not path.is_file():
        return None
    return torch.load(path)
