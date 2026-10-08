from collections import Counter, defaultdict
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from omegaconf import OmegaConf

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
import hydra
import numpy as np
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
import torch.multiprocessing as tmp
from tqdm.auto import tqdm
import wandb
import torch.distributed as dist

from flower.evaluation.eval_records import (
    checkpoint_name,
    merge_rank_csvs,
    merge_result_csv,
    read_csv,
    result_dir,
    rollout_seed,
    rotate_result_csv,
    write_csv,
)
from flower.evaluation.multistep_sequences import get_sequences
from flower.evaluation.utils import get_default_mode_and_env, get_env_state_for_initial_condition, join_vis_lang
from flower.rollout.rollout_video import RolloutVideo

logger = logging.getLogger(__name__)

# result.csv's variant/suite columns: rows land in <train_folder>/eval_logs/<ckpt>/calvin_ABCD_D/.
LIBERO_VARIANT = "calvin"
SUITE = "ABCD_D"


def get_video_tag(i):
    if dist.is_available() and dist.is_initialized():
        i = i * dist.get_world_size() + dist.get_rank()
    return f"_long_horizon/sequence_{i}"


def get_log_dir(log_dir):
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")

    log_dir = log_dir / "logs" / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir, exist_ok=False)
    print(f"logging to {log_dir}")
    return log_dir


def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def partition_sequences(n_sequences: int, n_gpus: int) -> List[List[int]]:
    """Round-robin split of sequence indices across GPU workers."""
    return [list(range(rank, n_sequences, n_gpus)) for rank in range(n_gpus)]


def apply_eval_modalities(model, eval_modalities: Dict[str, bool]) -> bool:
    """Mask the modalities withheld from the model; returns whether it receives proprio.

    Same validation/masking as EvaluateMimicgen.__init__ (flower_eval_mimicgen.py).
    """
    modality_tuple = (
        bool(eval_modalities.get("rgb_static", True)),
        bool(eval_modalities.get("rgb_gripper", True)),
        bool(eval_modalities.get("language", True)),
    )
    if not any(modality_tuple):
        raise ValueError(f"eval_modalities: at least one modality must be enabled (got all-False: {eval_modalities})")
    if not all(modality_tuple):
        model.eval_modality_mask = modality_tuple

    proprio_enabled = bool(eval_modalities.get("proprio", True))
    if not proprio_enabled and not model.use_proprio:
        raise ValueError(
            "eval_modalities.proprio=False requires a model with use_proprio=True — this "
            "checkpoint never receives proprioception to withhold"
        )
    if not proprio_enabled:
        model.eval_proprio_mask = False
    return proprio_enabled and model.use_proprio


def subtask_rows(
    seq_idx: int,
    eval_sequence: Sequence[str],
    outcomes: List[Tuple[bool, int]],
    val_annotations,
    seeds: List[int],
    common: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """One result.csv row per attempted subtask of a chain (it stops at the first failure)."""
    return [
        {
            **common,
            "task_idx": seq_idx,
            "episode_idx": pos,
            "task_name": subtask,
            "language": val_annotations[subtask][0],
            "rollout_seed": seed,
            "steps_taken": steps,
            "success": int(success),
        }
        for pos, (subtask, (success, steps), seed) in enumerate(zip(eval_sequence, outcomes, seeds))
    ]


def chain_lengths(rows: List[Dict[str, Any]]) -> Dict[int, int]:
    """Successful subtasks in a row per sequence index (sum of its rows' success)."""
    lengths: Dict[int, int] = defaultdict(int)
    for row in rows:
        lengths[int(row["task_idx"])] += int(row["success"])
    return dict(lengths)


def print_and_save(rows, log_dir, log_wandb=False):
    """Print avg_seq_len, chain and per-task success rates; write them to log_dir/results.json."""
    lengths = list(chain_lengths(rows).values())
    avg_seq_len = np.mean(lengths)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(lengths))}
    print(f"Results over {len(lengths)} sequences:")
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")

    cnt_success = Counter()
    total = Counter()
    for row in rows:
        cnt_success[row["task_name"]] += int(row["success"])
        total[row["task_name"]] += 1
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}
    if log_wandb:
        wandb.log({"avrg_performance/avg_seq_len": avg_seq_len, "avrg_performance/chain_sr": chain_sr, "detailed_metrics/task_info": task_info})
    with open(log_dir / "results.json", "w") as file:
        json.dump(data, file, indent=2)
    return data


def evaluate_policy(model, env, lang_embeddings, cfg, sequence_indices, common, num_videos=0, save_dir=None):
    """Evaluate the chains at `sequence_indices` of get_sequences(cfg.num_sequences); returns result.csv rows."""
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations

    # video stuff
    if num_videos > 0:
        rollout_video = RolloutVideo(
            logger=logger,
            empty_cache=False,
            log_to_file=True,
            save_dir=save_dir,
            resolution_scale=1,
        )
    else:
        rollout_video = None

    all_sequences = get_sequences(cfg.num_sequences)

    rows = []
    results = []

    eval_sequences = sequence_indices
    if not cfg.debug:
        eval_sequences = tqdm(sequence_indices, position=0, leave=True)

    for n_done, i in enumerate(eval_sequences):
        initial_state, eval_sequence = all_sequences[i]
        record = n_done < num_videos
        # Keyed on subtask + sequence index only: the same chain draws the same noise under
        # every modality combo and every GPU sharding.
        seeds = [rollout_seed(cfg.base_seed, subtask, i) for subtask in eval_sequence]
        outcomes = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence, lang_embeddings, val_annotations, cfg, record, rollout_video, i, seeds
        )
        rows.extend(subtask_rows(i, eval_sequence, outcomes, val_annotations, seeds, common))
        results.append(sum(success for success, _ in outcomes))
        if record:
            rollout_video.write_to_tmp()
        if not cfg.debug:
            success_rates = count_success(results)
            average_rate = sum(success_rates) / len(success_rates) * 5
            description = " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_rates)])
            description += f" Average: {average_rate:.1f} |"
            eval_sequences.set_description(description)

    if num_videos > 0:
        # log rollout videos
        rollout_video._log_videos_to_file(0, save_as_video=False)
    return rows


def evaluate_sequence(
    env, model, task_checker, initial_state, eval_sequence, lang_embeddings, val_annotations, cfg, record, rollout_video, i, seeds
):
    """Run one chain; returns (success, steps_taken) per attempted subtask."""
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    if record:
        caption = " | ".join(eval_sequence)
        rollout_video.new_video(tag=get_video_tag(i), caption=caption)
    outcomes = []
    if cfg.debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask, seed in zip(eval_sequence, seeds):
        if record:
            rollout_video.new_subtask()
        model.set_eval_noise_seeds([seed])
        success, steps = rollout(env, model, task_checker, cfg, subtask, lang_embeddings, val_annotations, record, rollout_video)
        outcomes.append((success, steps))
        if record:
            rollout_video.draw_outcome(success)
        if not success:
            break
    return outcomes


def rollout(env, model, task_oracle, cfg, subtask, lang_embeddings, val_annotations, record=False, rollout_video=None):
    if cfg.debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    # get language goal embedding
    # goal = lang_embeddings.get_lang_goal(lang_annotation)
    goal = {}
    goal['lang_text'] = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    for step in range(cfg.ep_len):
        action = model.step(obs, goal)
        obs, _, _, current_info = env.step(action)
        if cfg.debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if record:
            # update video
            rollout_video.update(obs["rgb_obs"]["rgb_static"])
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if cfg.debug:
                print(colored("success", "green"), end=" ")
            if record:
                rollout_video.add_language_instruction(lang_annotation)
            return True, step + 1
    if cfg.debug:
        print(colored("fail", "red"), end=" ")
    if record:
        rollout_video.add_language_instruction(lang_annotation)
    return False, cfg.ep_len


def _evaluate(cfg, device_id, sequence_indices, log_dir, num_videos):
    """Load model + env on one GPU and evaluate `sequence_indices`; returns result.csv rows."""
    model, env, _, lang_embeddings = get_default_mode_and_env(
        cfg.train_folder,
        cfg.dataset_path,
        cfg.checkpoint,
        env=None,
        lang_embeddings=None,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite,
        device_id=device_id,
    )

    model = model.to(device_id)

    if cfg.num_sampling_steps is not None:
        model.num_sampling_steps = cfg.num_sampling_steps
    if cfg.multistep is not None:
        model.multistep = cfg.multistep
    print(model.num_sampling_steps, model.multistep)

    eval_modalities = OmegaConf.to_container(cfg.eval_modalities, resolve=True)
    uses_proprio = apply_eval_modalities(model, eval_modalities)
    model.eval()

    common = {
        "libero_variant": LIBERO_VARIANT,
        "suite": SUITE,
        "checkpoint_name": checkpoint_name(cfg.checkpoint),
        "batching_mode": "sequential",
        "max_steps": cfg.ep_len,
        "num_sampling_steps": model.num_sampling_steps,
        "multistep": model.multistep,
        "eval_batch_size": 1,
        "base_seed": cfg.base_seed,
        "use_rgb_static": int(eval_modalities.get("rgb_static", True)),
        "use_rgb_gripper": int(eval_modalities.get("rgb_gripper", True)),
        "use_language": int(eval_modalities.get("language", True)),
        "use_proprio": int(uses_proprio),
        "checkpoint": cfg.checkpoint,
    }
    return evaluate_policy(
        model, env, lang_embeddings, cfg, sequence_indices, common, num_videos=num_videos, save_dir=Path(log_dir)
    )


def _eval_worker(rank, cfg_dict, sequence_indices, log_dir, csv_dir):
    """Worker process: evaluate a shard of sequences on GPU `rank`, write a result CSV shard."""
    cfg = OmegaConf.create(cfg_dict)
    rows = _evaluate(cfg, rank, sequence_indices, log_dir, cfg.num_videos if rank == 0 else 0)
    write_csv(os.path.join(csv_dir, f"result_rank{rank}.csv"), rows)


@hydra.main(config_path="../../conf", config_name="eval_calvin")
def main(cfg):
    seed_everything(0, workers=True)
    log_dir = get_log_dir(cfg.log_dir)
    if cfg.log_wandb:
        run = wandb.init(project="mode_calvin_eval", entity=cfg.wandb_entity, config=OmegaConf.to_object(cfg))

    csv_dir = Path(cfg.csv_dir) if cfg.csv_dir else result_dir(cfg.train_folder, cfg.checkpoint, LIBERO_VARIANT, SUITE)
    csv_path = csv_dir / "result.csv"
    n_gpus = max(torch.cuda.device_count(), 1)
    splits = partition_sequences(cfg.num_sequences, n_gpus)

    if n_gpus == 1:
        rows = _evaluate(cfg, cfg.device, splits[0], log_dir, cfg.num_videos)
        if cfg.reeval:
            rotate_result_csv(csv_path)
        merge_result_csv(csv_path, rows)
    else:
        # Each worker loads its own model + CALVIN env on its GPU; the parent touches no GPU.
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        ctx = tmp.get_context("spawn")
        processes = []
        for rank in range(n_gpus):
            p = ctx.Process(target=_eval_worker, args=(rank, cfg_dict, splits[rank], str(log_dir), str(csv_dir)))
            p.start()
            processes.append(p)
        for rank, p in enumerate(processes):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Eval worker rank {rank} exited with code {p.exitcode}")
        if cfg.reeval:
            rotate_result_csv(csv_path)
        merge_rank_csvs(csv_dir, n_gpus)
        gc.collect()
    print(f"Wrote result rows to {csv_path}")

    # Summarize this eval's rows only: result.csv also holds the other modality combos.
    eval_modalities = OmegaConf.to_container(cfg.eval_modalities, resolve=True)
    combo = {
        "checkpoint_name": checkpoint_name(cfg.checkpoint),
        "use_rgb_static": str(int(eval_modalities.get("rgb_static", True))),
        "use_rgb_gripper": str(int(eval_modalities.get("rgb_gripper", True))),
        "use_language": str(int(eval_modalities.get("language", True))),
    }
    rows = [
        row for row in read_csv(csv_path)
        if all(row[k] == v for k, v in combo.items()) and int(row["task_idx"]) < cfg.num_sequences
    ]
    print_and_save(rows, log_dir, log_wandb=cfg.log_wandb)

    if cfg.log_wandb:
        run.finish()


if __name__ == "__main__":
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    # Set CUDA device IDs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    main()
