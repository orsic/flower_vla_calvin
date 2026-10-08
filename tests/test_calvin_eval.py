"""Unit tests for the pure helpers of flower/evaluation/flower_evaluate.py (CALVIN eval).

The long-horizon eval records one result.csv row per *attempted* subtask: a chain stops
at its first failure, so a sequence's chain length is the sum of its rows' `success`, and
avg_seq_len is that sum over all rows divided by the number of sequences. Sequences are
sharded round-robin across GPU workers.
"""

from types import SimpleNamespace

import pytest

from flower.evaluation.flower_evaluate import (
    apply_eval_modalities,
    chain_lengths,
    partition_sequences,
    subtask_rows,
)

CHAIN = ["open_drawer", "turn_on_lightbulb", "push_red_block_left", "rotate_blue_block_right", "stack_block"]
ANNOTATIONS = {task: [f"please {task}"] for task in CHAIN}


@pytest.mark.parametrize("n_sequences,n_gpus", [(1000, 2), (1000, 3), (8, 4), (3, 4), (5, 1)])
def test_partition_sequences_covers_every_index_once(n_sequences, n_gpus):
    splits = partition_sequences(n_sequences, n_gpus)
    assert len(splits) == n_gpus
    flat = sorted(i for split in splits for i in split)
    assert flat == list(range(n_sequences))
    assert max(map(len, splits)) - min(map(len, splits)) <= 1


def test_subtask_rows_only_for_attempted_subtasks():
    # Two successes then a failure: the chain stops, the last two subtasks get no rows.
    outcomes = [(True, 40), (True, 55), (False, 360)]
    rows = subtask_rows(7, CHAIN, outcomes, ANNOTATIONS, seeds=[11, 12, 13], common={"suite": "ABCD_D"})

    assert [r["episode_idx"] for r in rows] == [0, 1, 2]
    assert all(r["task_idx"] == 7 for r in rows)
    assert [r["task_name"] for r in rows] == CHAIN[:3]
    assert [r["language"] for r in rows] == [f"please {t}" for t in CHAIN[:3]]
    assert [r["success"] for r in rows] == [1, 1, 0]
    assert [r["steps_taken"] for r in rows] == [40, 55, 360]
    assert [r["rollout_seed"] for r in rows] == [11, 12, 13]
    assert all(r["suite"] == "ABCD_D" for r in rows)


def test_chain_lengths_and_avg_seq_len_from_rows():
    rows = (
        subtask_rows(0, CHAIN, [(True, 1)] * 5, ANNOTATIONS, seeds=[0] * 5, common={})
        + subtask_rows(1, CHAIN, [(False, 360)], ANNOTATIONS, seeds=[0], common={})
        + subtask_rows(2, CHAIN, [(True, 1), (False, 360)], ANNOTATIONS, seeds=[0, 0], common={})
    )
    # Rows read back from CSV hold strings -- must give the same result.
    as_strings = [{k: str(v) for k, v in row.items()} for row in rows]
    for r in (rows, as_strings):
        assert chain_lengths(r) == {0: 5, 1: 0, 2: 1}


class _FakeModel(SimpleNamespace):
    pass


def test_apply_eval_modalities_full_combo_sets_no_mask():
    model = _FakeModel(use_proprio=False, eval_modality_mask=None, eval_proprio_mask=None)
    uses_proprio = apply_eval_modalities(
        model, {"rgb_static": True, "rgb_gripper": True, "language": True, "proprio": True}
    )
    assert model.eval_modality_mask is None
    assert uses_proprio is False


def test_apply_eval_modalities_masks_withheld_modality():
    model = _FakeModel(use_proprio=False, eval_modality_mask=None, eval_proprio_mask=None)
    apply_eval_modalities(model, {"rgb_static": True, "rgb_gripper": False, "language": True, "proprio": True})
    assert model.eval_modality_mask == (True, False, True)


def test_apply_eval_modalities_rejects_all_off_and_proprio_off_without_proprio():
    model = _FakeModel(use_proprio=False, eval_modality_mask=None, eval_proprio_mask=None)
    with pytest.raises(ValueError):
        apply_eval_modalities(model, {"rgb_static": False, "rgb_gripper": False, "language": False})
    with pytest.raises(ValueError):
        apply_eval_modalities(model, {"rgb_static": True, "rgb_gripper": True, "language": True, "proprio": False})
