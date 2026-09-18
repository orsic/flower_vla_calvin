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
