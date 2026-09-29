"""FLOWER evaluation on MimicGen `core` datasets.

Structurally a slimmer sibling of flower_eval_libero.py: same batched-rollout /
result-CSV / multi-GPU-worker shape, reusing flower.evaluation.eval_records (CSV
schema, seeding, merge/rotate), flower.evaluation.obs_translation (robosuite obs ->
model input) and flower.evaluation.libero_venv (spawn-context vector env) unchanged.

Deliberately does NOT import flower_eval_libero.py or anything under `libero.*`:
this module runs in the MimicGen container, whose robosuite/robomimic pins are newer
than LIBERO's (see scripts/podman/Containerfile.mimicgen) and may not be import-
compatible with libero.libero.envs. flower.evaluation.libero_venv already tolerates
this (file-path import fallback); this module simply never triggers the LIBERO-package
import path in the first place. robomimic/mimicgen themselves are imported lazily
(inside _create_mimicgen_env), not at module level, so the rest of this module --
row-building, CSV writing, dataset partitioning -- stays importable and unit-testable
without those packages installed.

MimicGen datasets use the same robosuite OSC_POSE delta-action convention LIBERO does
(see the plan this module lands with), so no action conversion happens anywhere on
this path either -- model output goes to env.step() exactly as in flower_eval_libero.py.
"""

import gc
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import hydra
import numpy as np
import torch
import torch.multiprocessing as tmp
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm

from flower.datasets import mimicgen_tasks
from flower.evaluation import obs_translation
from flower.evaluation.eval_records import (
    checkpoint_name,
    env_seed,
    merge_rank_csvs,
    merge_result_csv,
    result_dir,
    rollout_seed,
    rotate_result_csv,
    write_csv,
)
from flower.evaluation.libero_venv import make_libero_venv
from flower.evaluation.utils import get_default_mode_and_env, load_mode_from_safetensor

# result_dir()'s libero_variant/suite columns, reused as-is for a non-LIBERO benchmark
# (eval_records.py is benchmark-agnostic -- see the plan this module lands with).
LIBERO_VARIANT = "mimicgen"
SUITE = "core"


def get_log_dir(log_dir):
    """Duplicated from flower_eval_libero.py: that module imports `libero.*` at load
    time, which this module must not trigger (see module docstring)."""
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")

    stamp = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4().hex[:8]}"
    log_dir = log_dir / "logs" / stamp
    os.makedirs(log_dir, exist_ok=False)
    print(f"logging to {log_dir}")
    return log_dir


def seed_and_reset(env, ids: List[int], seeds: List[int]) -> None:
    """Duplicated from flower_eval_libero.py (see get_log_dir's docstring). Seed each
    slot's env, then reset -- MimicGen's random-reset path relies on this order to
    reproduce a given episode's initial state across runs, exactly like LIBERO's."""
    env.seed([env_seed(s) for s in seeds])
    env.reset(id=ids)


def _create_mimicgen_env(dataset_path: str, img_h: int, img_w: int):
    """Build one MimicGen/robosuite env from a rendered dataset's recorded env_args.

    Lazily imports mimicgen/robomimic (see module docstring): only reached inside a
    spawned worker subprocess, never at module import time.
    """
    import h5py
    import robomimic.envs.env_base as EB
    import robomimic.utils.env_utils as EnvUtils

    import mimicgen  # noqa: F401 -- registers MimicGen's robosuite env classes

    with h5py.File(dataset_path, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])

    env_type = EnvUtils.get_env_type(env_meta=env_meta)
    if env_type != EB.EnvType.ROBOSUITE_TYPE:
        raise ValueError(
            f"{dataset_path}: expected a robosuite-type MimicGen env, got env_type={env_type}"
        )

    env_kwargs = dict(env_meta["env_kwargs"])
    controller_configs = env_kwargs.get("controller_configs", {})
    if controller_configs.get("control_delta") is False:
        raise ValueError(
            f"{dataset_path}: env_args has control_delta=False (absolute-pose controller). "
            "FLOWER assumes MimicGen's default delta OSC_POSE convention, identical to "
            "LIBERO's -- see the plan this module lands with. Refusing to silently "
            "evaluate under a different action semantics."
        )
    # Render at eval resolution (matches flower_eval_libero.py's img_h=img_w=224,
    # independent of the 128x128 the training data was rendered at -- both get resized
    # to 112 by the same transform pipeline either way).
    env_kwargs["camera_names"] = ["agentview", "robot0_eye_in_hand"]
    env_kwargs["camera_heights"] = img_h
    env_kwargs["camera_widths"] = img_w

    class _MimicgenEnv(EnvUtils.get_env_class(env_type=env_type)):
        """Two behaviors the base robomimic env class doesn't provide, that the eval
        loop's BaseVectorEnv worker protocol (flower.evaluation.libero_venv) needs:

        - `seed()`: the base class has none, so a fresh vector-env worker's "seed"
          command would try `env.reset(seed=...)`, which the base reset() doesn't
          accept -- see LIBERO/libero/libero/envs/venv.py's _worker dispatch.
        - success-driven `done`: the base class's is_done() is hardcoded False
          ("robosuite envs always rollout to fixed horizon"), so the eval loop's
          early-exit-on-success and the result CSV's `success` column would never
          fire. Fold is_success()["task"] into `done` instead, matching how MimicGen's
          own rollout code (mimicgen/visuomotor-stack's extract_trajectory) checks it.
        """

        def seed(self, seed):
            np.random.seed(seed)

        def step(self, action):
            obs, reward, _done, info = super().step(action)
            return obs, reward, bool(self.is_success()["task"]), info

    return _MimicgenEnv(
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=True,
        use_image_obs=True,
        postprocess_visual_obs=True,
        **env_kwargs,
    )


class EvaluateMimicgen:
    """Batched rollout evaluator over flower.datasets.mimicgen_tasks.CORE_DATASETS.

    One robosuite env per dataset (not per task instance, unlike LIBERO's per-bddl
    envs); episodes reset randomly (MimicGen ships no per-episode init-state files the
    way LIBERO's benchmark package does), matching evaluate_task's own fallback path
    for when LIBERO initial states are unavailable.
    """

    def __init__(
        self,
        model,
        transforms,
        log_dir,
        data_dir: str,
        datasets: Optional[List[str]] = None,
        n_eval: int = 20,
        num_videos: int = 0,
        device=None,
        eval_batch_size: int = 10,
        checkpoint: str = "",
        base_seed: int = 0,
        eval_modalities: Optional[Dict[str, bool]] = None,
        env_start_method: str = "spawn",
        img_h: int = 224,
        img_w: int = 224,
    ):
        self.model = model
        self.transforms = transforms
        self.log_dir = log_dir
        self.data_dir = data_dir
        self.datasets = list(datasets) if datasets else list(mimicgen_tasks.CORE_DATASETS)
        self.n_eval = n_eval
        self.num_videos = num_videos
        self.device = device
        self.eval_batch_size = eval_batch_size
        self.checkpoint = str(checkpoint)
        self.base_seed = base_seed
        self.env_start_method = env_start_method
        self.img_h = img_h
        self.img_w = img_w

        self.eval_modalities = eval_modalities or {
            "rgb_static": True, "rgb_gripper": True, "language": True, "proprio": True
        }
        # Same validation/masking as EvaluateLibero.__init__ (flower_eval_libero.py) --
        # generic to FLOWERVLA, not LIBERO-specific.
        modality_tuple = (
            bool(self.eval_modalities.get("rgb_static", True)),
            bool(self.eval_modalities.get("rgb_gripper", True)),
            bool(self.eval_modalities.get("language", True)),
        )
        if not any(modality_tuple):
            raise ValueError(
                "eval_modalities: at least one modality must be enabled "
                f"(got all-False: {self.eval_modalities})"
            )
        if not all(modality_tuple):
            self.model.eval_modality_mask = modality_tuple

        proprio_enabled = bool(self.eval_modalities.get("proprio", True))
        if not proprio_enabled and not self.model.use_proprio:
            raise ValueError(
                "eval_modalities.proprio=False requires a model with use_proprio=True — this "
                "checkpoint never receives proprioception to withhold"
            )
        if not proprio_enabled:
            self.model.eval_proprio_mask = False
        self.uses_proprio = proprio_enabled and self.model.use_proprio

    def evaluate_policy(self, model, store_video=0) -> List[Dict[str, Any]]:
        all_rows: List[Dict[str, Any]] = []
        for idx, dataset_name in enumerate(self.datasets):
            print(f"starting to evaluate: {dataset_name}")
            rows = self.evaluate_dataset(model, dataset_name, idx, store_video=store_video)
            success_rate = sum(row["success"] for row in rows) / len(rows)
            print(f"Task {dataset_name} success rate: {success_rate:.4f}")
            all_rows.extend(rows)
        return all_rows

    def evaluate_dataset(self, model, dataset_name, idx, store_video=0) -> List[Dict[str, Any]]:
        """Evaluate one dataset, running eval_batch_size parallel episodes per batch.

        Mirrors flower_eval_libero.py's EvaluateLibero.evaluate_task, minus LIBERO's
        bddl/init-state machinery (MimicGen has neither): every episode is a random
        env.reset(), keyed by seed_and_reset just like LIBERO's own no-initial-states
        fallback path.
        """
        dataset_path = os.path.join(self.data_dir, f"{dataset_name}.hdf5")
        max_steps = mimicgen_tasks.max_steps(dataset_name)
        task_emb = None  # MimicGen's language comes from a fixed instruction string, no CLIP embedding
        language = mimicgen_tasks.language(dataset_name)

        rows: List[Dict[str, Any]] = []
        episode_idx = 0
        env = None

        with tqdm(total=self.n_eval, desc=f"Evaluating {dataset_name}") as pbar:
            while episode_idx < self.n_eval:
                current_batch_size = min(self.eval_batch_size, self.n_eval - episode_idx)
                ids = list(range(current_batch_size))

                env_fns = [
                    (lambda p=dataset_path: _create_mimicgen_env(p, self.img_h, self.img_w))
                    for _ in range(current_batch_size)
                ]
                if env is None:
                    env_creation = False
                    count = 0
                    while not env_creation and count < 5:
                        try:
                            env = make_libero_venv(env_fns, self.env_start_method)
                            env_creation = True
                        except Exception:
                            time.sleep(5)
                            count += 1
                    if not env_creation:
                        raise Exception("Failed to create environment")
                else:
                    env.rebuild(env_fns)

                seeds = [
                    rollout_seed(self.base_seed, dataset_name, episode_idx + k)
                    for k in range(current_batch_size)
                ]
                model.set_eval_noise_seeds(seeds)
                seed_and_reset(env, ids, seeds)
                torch.manual_seed(self.base_seed)
                np.random.seed(self.base_seed % (2**32))

                dummy = np.zeros((current_batch_size, 7))
                for _ in range(5):
                    obs, _, _, _ = env.step(dummy, id=ids)

                video_writers = {}
                video_frames = {}
                for k in range(current_batch_size):
                    global_ep = episode_idx + k
                    if global_ep < int(store_video):
                        video_filename = f"rollout_mimicgen_{dataset_name}_{global_ep}.mp4"
                        video_path = os.path.join(self.log_dir, video_filename)
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writers[k] = cv2.VideoWriter(
                            video_path, fourcc, 20.0, (self.img_w, self.img_h)
                        )
                        video_frames[k] = []

                dones = [False] * current_batch_size
                steps_taken = [0] * current_batch_size
                steps = 0
                model.reset()

                active_ids = list(range(current_batch_size))
                while steps < max_steps:
                    steps += 1
                    if model.rollout_step_counter % model.multistep == 0:
                        data, goal = self.process_env_obs_batch(obs, task_emb, language)
                    else:
                        data, goal = None, None
                    actions = model.step_batch(data, goal).cpu().numpy()  # [B, 7]
                    step_obs, _, step_done, _ = env.step(actions[active_ids], id=active_ids)

                    still_active = []
                    for pos, k in enumerate(active_ids):
                        obs[k] = step_obs[pos]
                        steps_taken[k] = steps
                        if bool(step_done[pos]):
                            dones[k] = True
                        else:
                            still_active.append(k)
                        if k in video_frames:
                            video_frames[k].append(obs[k]['agentview_image'])
                    active_ids = still_active

                    if not active_ids:
                        break

                for k, writer in video_writers.items():
                    for frame in video_frames[k]:
                        writer.write(frame)
                    writer.release()

                for k in range(current_batch_size):
                    rows.append({
                        "libero_variant": LIBERO_VARIANT,
                        "suite": SUITE,
                        "task_idx": idx,
                        "episode_idx": episode_idx + k,
                        "checkpoint_name": checkpoint_name(self.checkpoint),
                        "batching_mode": "per_task",
                        "task_name": dataset_name,
                        "language": language,
                        "task_category": mimicgen_tasks.family(dataset_name),
                        "init_state_idx": -1,
                        "max_steps": max_steps,
                        "img_h": self.img_h,
                        "img_w": self.img_w,
                        "num_sampling_steps": getattr(model, "num_sampling_steps", ""),
                        "multistep": getattr(model, "multistep", ""),
                        "eval_batch_size": current_batch_size,
                        "base_seed": self.base_seed,
                        "rollout_seed": seeds[k],
                        "use_rgb_static": int(self.eval_modalities.get("rgb_static", True)),
                        "use_rgb_gripper": int(self.eval_modalities.get("rgb_gripper", True)),
                        "use_language": int(self.eval_modalities.get("language", True)),
                        "use_proprio": int(self.uses_proprio),
                        "checkpoint": self.checkpoint,
                        "steps_taken": steps_taken[k],
                        "success": int(dones[k]),
                    })

                episode_idx += current_batch_size
                pbar.update(current_batch_size)

        if env is not None:
            env.close()
            gc.collect()

        return rows

    def process_env_obs_batch(self, obs_list, lang_embed, lang_text=None):
        return obs_translation.process_env_obs_batch(obs_list, lang_embed, lang_text, self.transforms, self.device)


def partition_datasets(datasets: List[str], n_gpus: int) -> List[List[str]]:
    """Round-robin partition, mirroring flower_eval_libero.py's task_splits."""
    splits: List[List[str]] = [[] for _ in range(n_gpus)]
    for i, name in enumerate(datasets):
        splits[i % n_gpus].append(name)
    return splits


def _eval_worker(
    rank: int,
    cfg_dict: dict,
    transforms_cfg,
    datasets: List[str],
    log_dir: str,
    csv_dir: str,
    n_gpus: int,
) -> None:
    """Worker process: evaluate a subset of datasets on one GPU, write a result CSV shard."""
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    phys_ids = [int(x.strip()) for x in cuda_vis.split(",") if x.strip()] if cuda_vis else list(range(n_gpus))
    phys_gpu = phys_ids[rank] if rank < len(phys_ids) else rank
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(phys_gpu)

    model_overwrite = cfg_dict.get("eval_cfg_overwrite", {}).get("model", {})
    model = load_mode_from_safetensor(Path(cfg_dict["checkpoint"]), overwrite_cfg=model_overwrite)
    model.freeze()
    model = model.cuda(rank)
    model.eval()

    transforms = hydra.utils.instantiate(transforms_cfg)

    evaluator = EvaluateMimicgen(
        model=model,
        transforms=transforms,
        log_dir=log_dir,
        data_dir=cfg_dict["data_dir"],
        datasets=datasets,
        n_eval=cfg_dict["n_eval"],
        num_videos=cfg_dict.get("num_videos", 0) if rank == 0 else 0,
        device=rank,
        eval_batch_size=cfg_dict.get("eval_batch_size", 10),
        checkpoint=cfg_dict["checkpoint"],
        base_seed=cfg_dict.get("seed", 0),
        eval_modalities=cfg_dict.get("eval_modalities"),
        env_start_method=cfg_dict.get("env_start_method", "spawn"),
    )
    rows = evaluator.evaluate_policy(
        model, store_video=cfg_dict.get("num_videos", 0) if rank == 0 else 0
    )
    write_csv(os.path.join(csv_dir, f"result_rank{rank}.csv"), rows)


@hydra.main(config_path="../../conf", config_name="eval_mimicgen")
def main(cfg):
    seed_everything(0, workers=True)
    model, _, dm, _ = get_default_mode_and_env(
        cfg.train_folder,
        cfg.data_dir,
        cfg.checkpoint,
        env=42,
        lang_embeddings=None,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite,
        device_id=cfg.device,
        prep_dm_and_deps=False,
    )
    model = model.to(cfg.device)
    model.eval()

    log_dir = get_log_dir(cfg.log_dir)
    transforms = hydra.utils.instantiate(dm.transforms)

    base_seed: int = OmegaConf.select(cfg, "seed", default=0)
    eval_modalities = OmegaConf.select(cfg, "eval_modalities", default=None)
    if eval_modalities is not None:
        eval_modalities = OmegaConf.to_container(eval_modalities, resolve=True)
    env_start_method: str = OmegaConf.select(cfg, "env_start_method", default="spawn")
    reeval: bool = OmegaConf.select(cfg, "reeval", default=False)

    datasets_cfg = OmegaConf.select(cfg, "datasets", default=None)
    datasets = list(datasets_cfg) if datasets_cfg is not None else list(mimicgen_tasks.CORE_DATASETS)

    csv_dir_override = OmegaConf.select(cfg, "csv_dir", default=None)
    csv_dir = Path(csv_dir_override) if csv_dir_override else result_dir(
        cfg.train_folder, cfg.checkpoint, LIBERO_VARIANT, SUITE
    )

    evaluator = EvaluateMimicgen(
        model=model,
        transforms=transforms,
        log_dir=log_dir,
        data_dir=cfg.data_dir,
        datasets=datasets,
        n_eval=cfg.n_eval,
        num_videos=cfg.get("num_videos", 0),
        device=cfg.device,
        eval_batch_size=cfg.get("eval_batch_size", 10),
        checkpoint=cfg.checkpoint,
        base_seed=base_seed,
        eval_modalities=eval_modalities,
        env_start_method=env_start_method,
    )

    n_gpus = torch.cuda.device_count()

    if n_gpus <= 1:
        rows = evaluator.evaluate_policy(model, store_video=cfg.get("num_videos", 0))
        csv_path = csv_dir / "result.csv"
        if reeval:
            rotate_result_csv(csv_path)
        merge_result_csv(csv_path, rows)
        print(f"Wrote {len(rows)} episode rows to {csv_path}")
    else:
        splits = partition_datasets(datasets, n_gpus)

        evaluator.model = None
        del model
        gc.collect()
        torch.cuda.empty_cache()

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        transforms_cfg = dm.transforms

        ctx = tmp.get_context("spawn")
        processes = []
        for rank in range(n_gpus):
            p = ctx.Process(
                target=_eval_worker,
                args=(rank, cfg_dict, transforms_cfg, splits[rank], str(log_dir), str(csv_dir), n_gpus),
            )
            p.start()
            processes.append(p)
        for rank, p in enumerate(processes):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Eval worker rank {rank} exited with code {p.exitcode}")

        if reeval:
            rotate_result_csv(csv_dir / "result.csv")
        merged_rows = merge_rank_csvs(csv_dir, n_gpus)
        print(f"Wrote merged rows to {csv_dir / 'result.csv'} ({len(merged_rows)} rows total)")

    print('done')


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    main()
