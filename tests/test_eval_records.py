"""Tests for flower.evaluation.eval_records — per-episode CSV records.

Pure logic / filesystem tests, no MuJoCo or model dependencies.
"""

from flower.evaluation.eval_records import (
    ALL_COLUMNS,
    checkpoint_name,
    merge_rank_csvs,
    merge_result_csv,
    merge_rows,
    read_csv,
    result_dir,
    rollout_seed,
    write_csv,
)


def _row(
    task_idx, episode_idx, success, checkpoint_name_="last", variant="orig", suite="libero_10",
    use_rgb_static=1, use_rgb_gripper=1, use_language=1,
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
# rollout_seed
# ---------------------------------------------------------------------------

def test_rollout_seed_stable():
    assert rollout_seed(0, 3, 5) == rollout_seed(0, 3, 5)


def test_rollout_seed_distinct_across_task_and_episode():
    seeds = {
        rollout_seed(0, 0, 0),
        rollout_seed(0, 1, 0),
        rollout_seed(0, 0, 1),
        rollout_seed(1, 0, 0),
    }
    assert len(seeds) == 4


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
