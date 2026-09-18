"""Tests for flower.evaluation.eval_records — per-episode CSV records.

Pure logic / filesystem tests, no MuJoCo or model dependencies.
"""

import os
import time

import pytest

from flower.evaluation.eval_records import (
    ALL_COLUMNS,
    base_task_name,
    checkpoint_name,
    merge_rank_csvs,
    merge_result_csv,
    merge_rows,
    read_csv,
    result_dir,
    rollout_seed,
    rotate_result_csv,
    write_csv,
)


def _row(
    task_idx, episode_idx, success, checkpoint_name_="last", variant="orig", suite="libero_10",
    use_rgb_static=1, use_rgb_gripper=1, use_language=1, use_proprio=1,
):
    row = {col: "" for col in ALL_COLUMNS}
    row.update(
        libero_variant=variant,
        suite=suite,
        task_idx=task_idx,
        episode_idx=episode_idx,
        checkpoint_name=checkpoint_name_,
        use_rgb_static=use_rgb_static,
        use_rgb_gripper=use_rgb_gripper,
        use_language=use_language,
        use_proprio=use_proprio,
        success=success,
    )
    return row


# ---------------------------------------------------------------------------
# checkpoint_name
# ---------------------------------------------------------------------------

def test_checkpoint_name_from_ckpt_file(tmp_path):
    f = tmp_path / "last.ckpt"
    f.write_text("x")
    assert checkpoint_name(f) == "last"


def test_checkpoint_name_from_named_epoch_file(tmp_path):
    f = tmp_path / "epoch=39_eval_lh.ckpt"
    f.write_text("x")
    assert checkpoint_name(f) == "epoch=39_eval_lh"


def test_checkpoint_name_from_directory(tmp_path):
    d = tmp_path / "libero_10"
    d.mkdir()
    assert checkpoint_name(d) == "libero_10"


# ---------------------------------------------------------------------------
# result_dir
# ---------------------------------------------------------------------------

def test_result_dir_orig_vs_plus_dont_collide(tmp_path):
    train_folder = tmp_path / "train_run"
    ckpt = train_folder / "seed_42" / "saved_models" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("x")

    orig_dir = result_dir(train_folder, ckpt, "orig", "libero_10")
    plus_dir = result_dir(train_folder, ckpt, "plus", "libero_10")

    assert orig_dir != plus_dir
    assert orig_dir.name == "orig_libero_10"
    assert plus_dir.name == "plus_libero_10"
    assert orig_dir.parent == plus_dir.parent == train_folder / "eval_logs" / "last"
    assert orig_dir.is_dir() and plus_dir.is_dir()


# ---------------------------------------------------------------------------
# base_task_name
# ---------------------------------------------------------------------------

_ORIG_BASE = "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"


@pytest.mark.parametrize(
    "plus_name",
    [
        f"{_ORIG_BASE}_view_0_0_100_2_6_initstate_0",  # Camera Viewpoints
        f"{_ORIG_BASE}_table_1",  # Background Textures
        f"{_ORIG_BASE}_tb_1",  # Background Textures (alt suffix)
        f"{_ORIG_BASE}_light_3",  # Light Conditions
        f"{_ORIG_BASE}_view_0_0_100_0_0_initstate_51",  # Robot Initial States
        f"{_ORIG_BASE}_view_0_0_100_0_0_initstate_0_noise_12",  # Sensor Noise
        f"{_ORIG_BASE}_add_10",  # Objects Layout
        f"{_ORIG_BASE}_level1_sample3",  # Objects Layout (alt suffix)
        f"{_ORIG_BASE}_language_2_view_0_0_100_0_0_initstate_0",  # Language Instructions
    ],
)
def test_base_task_name_strips_each_category_suffix(plus_name):
    assert base_task_name(plus_name) == _ORIG_BASE


def test_base_task_name_strips_moved_infix():
    """libero_goal's Objects Layout variants carry an extra "_moved" infix."""
    assert base_task_name("open_the_middle_drawer_of_the_cabinet_moved_level1_sample1") == (
        "open_the_middle_drawer_of_the_cabinet"
    )


def test_base_task_name_noop_on_original_names():
    assert base_task_name(_ORIG_BASE) == _ORIG_BASE


def test_base_task_name_does_not_strip_table_center():
    """Regression guard: a bare, non-digit-anchored "_table_" also appears inside an
    ORIGINAL libero_spatial task name -- must not be mistaken for the Background
    Textures perturbation marker (which is always digit-anchored, "_table_<N>")."""
    name = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
    assert base_task_name(name) == name


# ---------------------------------------------------------------------------
# rollout_seed
# ---------------------------------------------------------------------------

def test_rollout_seed_stable():
    assert rollout_seed(0, "task_a", 5) == rollout_seed(0, "task_a", 5)


def test_rollout_seed_distinct_across_task_episode_and_base_seed():
    seeds = {
        rollout_seed(0, "task_a", 0),
        rollout_seed(0, "task_b", 0),
        rollout_seed(0, "task_a", 1),
        rollout_seed(1, "task_a", 0),
    }
    assert len(seeds) == 4


def test_rollout_seed_shared_across_plus_variants_of_one_base_task():
    """The whole point: every LIBERO-Plus variant of a base task, across every
    perturbation category, must derive the SAME seed at the same episode index --
    common random numbers, so an orig-vs-Plus success delta is attributable to the
    perturbation rather than to an unrelated noise draw."""
    variants = [
        _ORIG_BASE,
        f"{_ORIG_BASE}_view_0_0_100_2_6_initstate_0",
        f"{_ORIG_BASE}_table_1",
        f"{_ORIG_BASE}_light_3",
        f"{_ORIG_BASE}_view_0_0_100_0_0_initstate_0_noise_12",
    ]
    seeds = {rollout_seed(0, name, 0) for name in variants}
    assert len(seeds) == 1


def test_rollout_seed_independent_of_pythonhashseed():
    """crc32, not hash(): str hashing is PYTHONHASHSEED-randomized, which would give a
    different seed on every process -- including between one run's multi-GPU eval
    workers."""
    import subprocess
    import sys

    code = (
        "from flower.evaluation.eval_records import rollout_seed; "
        "print(rollout_seed(0, 'some_task_name', 3))"
    )
    outs = []
    for hashseed in ("0", "1"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": hashseed},
            capture_output=True, text=True, check=True,
        )
        outs.append(result.stdout.strip())
    assert outs[0] == outs[1]


# ---------------------------------------------------------------------------
# merge_rows
# ---------------------------------------------------------------------------

def test_merge_rows_same_key_overwrites():
    existing = [_row(0, 0, success=0)]
    new = [_row(0, 0, success=1)]
    merged = merge_rows(existing, new)
    assert len(merged) == 1
    assert merged[0]["success"] == 1


def test_merge_rows_new_key_appends():
    existing = [_row(0, 0, success=1)]
    new = [_row(1, 0, success=0)]
    merged = merge_rows(existing, new)
    assert len(merged) == 2
    assert merged[0]["task_idx"] == 0
    assert merged[1]["task_idx"] == 1


def test_merge_rows_unrelated_prior_rows_survive():
    """Per-category accumulation: running category B must not drop category A's rows."""
    category_a = [_row(0, 0, success=1), _row(1, 0, success=0)]
    category_b = [_row(2, 0, success=1)]
    merged = merge_rows(category_a, category_b)
    assert len(merged) == 3
    assert {row["task_idx"] for row in merged} == {0, 1, 2}


def test_merge_rows_different_modality_combo_does_not_collide():
    """Same task/episode/checkpoint under a different modality combo is a new row."""
    full = [_row(0, 0, success=1, use_rgb_static=1, use_rgb_gripper=1, use_language=1)]
    static_only = [_row(0, 0, success=0, use_rgb_static=1, use_rgb_gripper=0, use_language=0)]
    merged = merge_rows(full, static_only)
    assert len(merged) == 2
    successes = {(r["use_rgb_gripper"], r["use_language"]): r["success"] for r in merged}
    assert successes[(1, 1)] == 1
    assert successes[(0, 0)] == 0


def test_merge_rows_same_modality_combo_still_overwrites():
    existing = [_row(0, 0, success=0, use_rgb_gripper=0, use_language=0)]
    rerun = [_row(0, 0, success=1, use_rgb_gripper=0, use_language=0)]
    merged = merge_rows(existing, rerun)
    assert len(merged) == 1
    assert merged[0]["success"] == 1


def test_merge_rows_different_proprio_combo_does_not_collide():
    """Same episode re-evaluated with proprio withheld is a new row, not an overwrite."""
    with_proprio = [_row(0, 0, success=1, use_proprio=1)]
    without_proprio = [_row(0, 0, success=0, use_proprio=0)]
    merged = merge_rows(with_proprio, without_proprio)
    assert len(merged) == 2
    successes = {r["use_proprio"]: r["success"] for r in merged}
    assert successes[1] == 1
    assert successes[0] == 0


def test_merge_rows_legacy_row_without_use_proprio_key_merges():
    """A row from a result.csv written before use_proprio existed has no such key at
    all (not even ""); _row_key must not KeyError on it."""
    legacy = _row(0, 0, success=1)
    del legacy["use_proprio"]
    new = [_row(1, 0, success=0)]
    merged = merge_rows([legacy], new)
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# merge_result_csv
# ---------------------------------------------------------------------------

def test_merge_result_csv_round_trip(tmp_path):
    path = tmp_path / "result.csv"
    merge_result_csv(path, [_row(0, 0, success=1)])
    merged = merge_result_csv(path, [_row(1, 0, success=0)])

    assert len(merged) == 2
    on_disk = read_csv(path)
    assert len(on_disk) == 2
    # CSV round-trips as strings; success values are still readable as ints.
    successes = {int(row["task_idx"]): int(row["success"]) for row in on_disk}
    assert successes == {0: 1, 1: 0}


def test_merge_result_csv_preserves_column_order(tmp_path):
    path = tmp_path / "result.csv"
    write_csv(path, [_row(0, 0, success=1)])
    with open(path) as f:
        header = f.readline().strip().split(",")
    assert header == ALL_COLUMNS


def test_merge_result_csv_rerun_same_key_updates_in_place(tmp_path):
    path = tmp_path / "result.csv"
    merge_result_csv(path, [_row(0, 0, success=0)])
    merge_result_csv(path, [_row(0, 0, success=1)])
    on_disk = read_csv(path)
    assert len(on_disk) == 1
    assert int(on_disk[0]["success"]) == 1


def test_batching_mode_round_trips(tmp_path):
    path = tmp_path / "result.csv"
    row = _row(0, 0, success=1)
    row["batching_mode"] = "cross_task"
    merge_result_csv(path, [row])
    on_disk = read_csv(path)
    assert on_disk[0]["batching_mode"] == "cross_task"


def test_old_row_without_batching_mode_still_merges(tmp_path):
    """A row dict missing the batching_mode key (old in-memory row) must not crash
    write_csv, and defaults to an empty value rather than erroring."""
    path = tmp_path / "result.csv"
    row = _row(0, 0, success=1)
    del row["batching_mode"]
    merge_result_csv(path, [row])
    on_disk = read_csv(path)
    assert len(on_disk) == 1
    assert on_disk[0]["batching_mode"] == ""


# ---------------------------------------------------------------------------
# merge_rank_csvs
# ---------------------------------------------------------------------------

def test_merge_rank_csvs_unions_shards_and_deletes_them(tmp_path):
    write_csv(tmp_path / "result_rank0.csv", [_row(0, 0, success=1)])
    write_csv(tmp_path / "result_rank1.csv", [_row(1, 0, success=0)])

    merged = merge_rank_csvs(tmp_path, world_size=2)

    assert len(merged) == 2
    assert not (tmp_path / "result_rank0.csv").exists()
    assert not (tmp_path / "result_rank1.csv").exists()
    assert (tmp_path / "result.csv").exists()


# ---------------------------------------------------------------------------
# rotate_result_csv
# ---------------------------------------------------------------------------

def test_rotate_result_csv_names_backup_from_file_mtime(tmp_path):
    path = tmp_path / "result.csv"
    write_csv(path, [_row(0, 0, success=1)])
    mtime = time.mktime((2026, 1, 2, 3, 4, 5, 0, 0, -1))
    os.utime(path, (mtime, mtime))

    backup = rotate_result_csv(path)

    assert backup == tmp_path / "results_2026-01-02_03-04-05.csv"
    assert backup.exists()
    assert not path.exists()
    assert read_csv(backup)[0]["task_idx"] == "0"


def test_rotate_result_csv_none_when_nothing_to_rotate(tmp_path):
    path = tmp_path / "result.csv"
    assert rotate_result_csv(path) is None
    assert not path.exists()


def test_rotate_result_csv_refuses_to_clobber_existing_backup(tmp_path):
    path = tmp_path / "result.csv"
    write_csv(path, [_row(0, 0, success=1)])
    mtime = time.mktime((2026, 1, 2, 3, 4, 5, 0, 0, -1))
    os.utime(path, (mtime, mtime))
    backup = tmp_path / "results_2026-01-02_03-04-05.csv"
    write_csv(backup, [_row(1, 0, success=0)])

    with pytest.raises(FileExistsError):
        rotate_result_csv(path)

    # the pre-existing backup and the not-yet-rotated result.csv both survive
    assert read_csv(backup)[0]["task_idx"] == "1"
    assert path.exists()


def test_rotate_then_merge_leaves_backup_intact(tmp_path):
    path = tmp_path / "result.csv"
    write_csv(path, [_row(0, 0, success=1)])
    mtime = time.mktime((2026, 1, 2, 3, 4, 5, 0, 0, -1))
    os.utime(path, (mtime, mtime))

    backup = rotate_result_csv(path)
    merge_result_csv(path, [_row(1, 0, success=0)])

    assert read_csv(backup)[0]["task_idx"] == "0"
    on_disk = read_csv(path)
    assert len(on_disk) == 1
    assert on_disk[0]["task_idx"] == "1"


def test_rotate_then_merge_rank_csvs_ignores_backup(tmp_path):
    path = tmp_path / "result.csv"
    write_csv(path, [_row(0, 0, success=1)])
    mtime = time.mktime((2026, 1, 2, 3, 4, 5, 0, 0, -1))
    os.utime(path, (mtime, mtime))
    backup = rotate_result_csv(path)

    write_csv(tmp_path / "result_rank0.csv", [_row(1, 0, success=1)])
    write_csv(tmp_path / "result_rank1.csv", [_row(2, 0, success=0)])
    merged = merge_rank_csvs(tmp_path, world_size=2)

    assert {int(row["task_idx"]) for row in merged} == {1, 2}
    assert backup.exists()
    assert read_csv(backup)[0]["task_idx"] == "0"
