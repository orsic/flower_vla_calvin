"""Per-episode CSV records for LIBERO evaluation.

One row = one episode, carrying every variable that deterministically defines it
(task, init state, seed, sampling config, which modalities were fed to the model).
Results are stored next to the checkpoint being evaluated:
    <train_folder>/eval_logs/<checkpoint_name>/<libero_variant>_<suite>/result.csv

Re-running (e.g. one LIBERO-Plus category at a time) merges into the existing file,
keyed on KEY_COLUMNS, with the newest run winning on a key collision.
"""
import csv
import time
from pathlib import Path
from typing import Dict, List, Union

# Columns that together identify one episode. Used as the merge/dedup key.
KEY_COLUMNS = [
    "libero_variant",
    "suite",
    "task_idx",
    "episode_idx",
    "checkpoint_name",
]

# Full column order written to CSV.
ALL_COLUMNS = KEY_COLUMNS + [
    "task_name",
    "problem_folder",
    "bddl_file",
    "init_states_file",
    "language",
    "task_category",
    "difficulty_level",
    "init_state_idx",
    "max_steps",
    "img_h",
    "img_w",
    "num_sampling_steps",
    "multistep",
    "eval_batch_size",
    "base_seed",
    "rollout_seed",
    "use_rgb_static",
    "use_rgb_gripper",
    "use_language",
    "checkpoint",
    "eval_timestamp",
    "steps_taken",
    "success",
]


def checkpoint_name(checkpoint: Union[str, Path]) -> str:
    """Derive a short, filesystem-friendly name for a checkpoint.

    A file (e.g. `.../last.ckpt`) -> its stem (`last`).
    A directory (HuggingFace layout) -> the directory name.
    """
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        return checkpoint.name
    return checkpoint.stem


def result_dir(
    train_folder: Union[str, Path],
    checkpoint: Union[str, Path],
    libero_variant: str,
    suite: str,
) -> Path:
    """<train_folder>/eval_logs/<checkpoint_name>/<variant>_<suite>/, created if missing."""
    out_dir = (
        Path(train_folder)
        / "eval_logs"
        / checkpoint_name(checkpoint)
        / f"{libero_variant}_{suite}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def rollout_seed(base_seed: int, task_idx: int, episode_idx: int) -> int:
    """Deterministic seed for the batch/episode starting at (task_idx, episode_idx)."""
    return base_seed * 1_000_003 + task_idx * 1_009 + episode_idx


def _row_key(row: Dict) -> tuple:
    # Stringify: rows freshly built in memory hold ints (e.g. task_idx=0) while rows
    # read back from CSV hold strings (task_idx="0") — without normalizing, the same
    # episode would get two different keys and merge_rows would append instead of
    # overwriting on a rerun.
    return tuple(str(row[c]) for c in KEY_COLUMNS)


def merge_rows(existing: List[Dict], new: List[Dict]) -> List[Dict]:
    """Merge new rows into existing, keyed on KEY_COLUMNS. New rows win on collision.

    Preserves the original relative order; new keys are appended at the end.
    """
    merged: Dict[tuple, Dict] = {_row_key(row): row for row in existing}
    order: List[tuple] = [_row_key(row) for row in existing]
    for row in new:
        key = _row_key(row)
        if key not in merged:
            order.append(key)
        merged[key] = row
    return [merged[key] for key in order]


def read_csv(path: Union[str, Path]) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Union[str, Path], rows: List[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in ALL_COLUMNS})


def merge_result_csv(path: Union[str, Path], rows: List[Dict]) -> List[Dict]:
    """Read the existing CSV (if any), merge `rows` in, write the result back."""
    for row in rows:
        row.setdefault("eval_timestamp", time.strftime("%Y-%m-%d_%H-%M-%S"))
    merged = merge_rows(read_csv(path), rows)
    write_csv(path, merged)
    return merged


def merge_rank_csvs(out_dir: Union[str, Path], world_size: int) -> List[Dict]:
    """Union result_rank<i>.csv shards (one per GPU worker) into result.csv, then delete them."""
    out_dir = Path(out_dir)
    rows: List[Dict] = []
    shard_paths = [out_dir / f"result_rank{rank}.csv" for rank in range(world_size)]
    for shard_path in shard_paths:
        rows.extend(read_csv(shard_path))
    merged = merge_result_csv(out_dir / "result.csv", rows)
    for shard_path in shard_paths:
        shard_path.unlink(missing_ok=True)
    return merged
