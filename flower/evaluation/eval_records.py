"""Per-episode CSV records for LIBERO evaluation.

One row = one episode, carrying every variable that deterministically defines it
(task, init state, seed, sampling config, which modalities were fed to the model).
Results are stored next to the checkpoint being evaluated:
    <train_folder>/eval_logs/<checkpoint_name>/<libero_variant>_<suite>/result.csv

Re-running (e.g. one LIBERO-Plus category at a time) merges into the existing file,
keyed on KEY_COLUMNS, with the newest run winning on a key collision.

use_proprio was added after use_rgb_static/use_rgb_gripper/use_language; rows in a
pre-existing result.csv have no such column, so _row_key reads it with a "" default
rather than indexing directly — those legacy rows read as proprio-absent, which is
factually correct (every eval predating this column ran with use_proprio=false).
"""
import csv
import re
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Union

# Columns that together identify one episode. Used as the merge/dedup key.
# The four use_* modality flags are part of the key so that evaluating the same
# task/episode/checkpoint under a different modality combo adds a new row instead of
# overwriting the previous combo's result.
KEY_COLUMNS = [
    "libero_variant",
    "suite",
    "task_idx",
    "episode_idx",
    "checkpoint_name",
    "use_rgb_static",
    "use_rgb_gripper",
    "use_language",
    "use_proprio",
]

# Full column order written to CSV.
ALL_COLUMNS = KEY_COLUMNS + [
    "batching_mode",
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


# Perturbation markers LIBERO-Plus appends to an original task name. Digit-anchored
# on purpose: a bare "_table_" also matches the *original* libero_spatial task
# "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate", which would
# then alias onto a different base task. LIBERO-Plus's own get_task_init_states uses
# re.sub(r'_table_\d+', ...) for the same reason.
_PERTURBATION_SUFFIX = re.compile(r"_(?:language|view|table|tb|light|add|noise|level)_?\d")


def base_task_name(task_name: str) -> str:
    """The original LIBERO task a LIBERO-Plus task_name was generated from.

    Every Plus name is <orig task name><perturbation suffix>; truncating at the first
    marker recovers the original. libero_goal's Objects Layout variants additionally
    carry a "_moved" infix ("..._cabinet_moved_level1_sample1"), stripped after.

    A no-op on an original LIBERO task name: no task name in any of the shipped suites
    contains a marker, which is what lets a LIBERO-Plus episode and its original-LIBERO
    counterpart (scripts/severity_sr.py's paired baseline) share this function's output
    as a noise-seeding key -- see rollout_seed below.
    """
    match = _PERTURBATION_SUFFIX.search(task_name)
    base = task_name[: match.start()] if match else task_name
    return base[: -len("_moved")] if base.endswith("_moved") else base


def rollout_seed(base_seed: int, task_name: str, episode_idx: int) -> int:
    """Deterministic seed for one episode's flow-matching noise stream.

    Keyed on the episode's *base* task name, not on task_idx: task_idx indexes the
    active benchmark instance (0-9 under LIBERO_VARIANT=orig, 0-2518 under plus,
    unrelated orderings), so it cannot relate a Plus episode to the original episode
    scripts/severity_sr.py's paired_baseline() compares it against, nor two Plus
    variants of one base task (e.g. two Camera-Viewpoint camera angles) to each other.
    Keying on the base name instead gives all of them one shared noise stream (common
    random numbers), which is what makes an orig-vs-Plus success delta attributable to
    the perturbation rather than to an unrelated noise draw -- see
    flower.models.flower.inference_noise, which consumes this per batch slot.

    crc32, not hash(): str hashing is PYTHONHASHSEED-randomized, so hash() would give a
    different seed on every process, including between one run's multi-GPU eval workers.
    """
    return base_seed * 1_000_003 + zlib.crc32(base_task_name(task_name).encode()) * 1_009 + episode_idx


def _row_key(row: Dict) -> tuple:
    # Stringify: rows freshly built in memory hold ints (e.g. task_idx=0) while rows
    # read back from CSV hold strings (task_idx="0") — without normalizing, the same
    # episode would get two different keys and merge_rows would append instead of
    # overwriting on a rerun.
    # .get(c, ""): use_proprio is missing from rows written before it existed (see
    # module docstring) — default rather than KeyError so old files still merge.
    return tuple(str(row.get(c, "")) for c in KEY_COLUMNS)


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


def rotate_result_csv(path: Union[str, Path]) -> Optional[Path]:
    """Move an existing result.csv aside so the next run starts from an empty file.

    Named for the backed-up file's own mtime, not now(): it records when those numbers
    were measured, which is what you need when comparing it to the run that replaced it.
    Returns the backup path, or None if there was nothing to rotate.
    """
    path = Path(path)
    if not path.exists():
        return None
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(path.stat().st_mtime))
    backup_path = path.parent / f"results_{stamp}.csv"
    if backup_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup {backup_path}")
    path.rename(backup_path)
    return backup_path


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
