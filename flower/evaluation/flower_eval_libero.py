# Standard library imports
import gc
import json
import logging
import math
import multiprocessing
import os
import sys
import time
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import cv2
import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as tmp
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningModule, Trainer, seed_everything
from termcolor import colored
from tqdm import tqdm
from tqdm.auto import tqdm

# Add local repo to path when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

# LIBERO imports
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import DummyVectorEnv, OffScreenRenderEnv, SubprocVectorEnv
from libero.lifelong.metric import evaluate_multitask_training_success, raw_obs_to_tensor_obs
from libero.lifelong.utils import create_experiment_dir, get_task_embs, safe_device

# Local project imports
from flower.evaluation.eval_records import (
    checkpoint_name,
    merge_rank_csvs,
    merge_result_csv,
    read_csv,
    result_dir,
    rollout_seed,
    write_csv,
)
from flower.evaluation.multistep_sequences import get_sequences
from flower.evaluation.utils import (
    LangEmbeddings,
    get_default_mode_and_env,
    get_env_state_for_initial_condition,
    join_vis_lang,
    load_mode_from_safetensor,
)
from flower.rollout.rollout_video import RolloutVideo

logger = logging.getLogger(__name__)


def select_task_indices(
    benchmark_instance,
    task_category: Optional[str],
) -> Optional[List[int]]:
    """Return task indices matching a LIBERO-Plus perturbation category.

    Returns None when task_category is None (meaning: run all tasks).
    Matches by task *name* (robust to ordering differences).
    Requires LIBERO_VARIANT=plus and a valid task_classification.json in the
    active benchmark package; raises clearly if either precondition fails.
    """
    if task_category is None:
        return None

    import libero.libero.benchmark as _bm_mod
    json_path = os.path.join(os.path.dirname(_bm_mod.__file__), "task_classification.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"task_classification.json not found at {json_path}. "
            "Is LIBERO_VARIANT=plus and the LIBERO-Plus submodule on PYTHONPATH?"
        )

    with open(json_path) as f:
        data = json.load(f)

    suite_name = benchmark_instance.name
    if suite_name not in data:
        raise ValueError(
            f"Suite '{suite_name}' not in task_classification.json. "
            f"Available suites: {list(data.keys())}"
        )

    matching_names = {
        entry["name"]
        for entry in data[suite_name]
        if entry["category"] == task_category
    }
    if not matching_names:
        available = sorted({e["category"] for e in data[suite_name]})
        raise ValueError(
            f"Category '{task_category}' not found in suite '{suite_name}'. "
            f"Available categories: {available}"
        )

    task_names = benchmark_instance.get_task_names()
    indices = [i for i, name in enumerate(task_names) if name in matching_names]
    if not indices:
        raise ValueError(
            f"No benchmark tasks match category '{task_category}' in suite '{suite_name}'. "
            "Verify LIBERO_VARIANT=plus."
        )
    logger.info(
        f"Category filter '{task_category}': {len(indices)} tasks selected "
        f"out of {len(task_names)} in '{suite_name}'."
    )
    return indices


def aggregate_by_category(
    task_names: List[str],
    successes: List[float],
    suite_name: str,
) -> Dict[str, float]:
    """Average success rate per LIBERO-Plus perturbation category.

    Returns an empty dict when task_classification.json is absent (e.g. LIBERO_VARIANT=orig).
    """
    import libero.libero.benchmark as _bm_mod
    json_path = os.path.join(os.path.dirname(_bm_mod.__file__), "task_classification.json")
    if not os.path.exists(json_path):
        return {}

    with open(json_path) as f:
        data = json.load(f)

    if suite_name not in data:
        return {}

    name_to_cat = {entry["name"]: entry["category"] for entry in data[suite_name]}
    cat_srs: Dict[str, List[float]] = {}
    for name, sr in zip(task_names, successes):
        cat = name_to_cat.get(name)
        if cat:
            cat_srs.setdefault(cat, []).append(sr)

    return {cat: sum(srs) / len(srs) for cat, srs in cat_srs.items()}


def load_task_classification(suite_name: str) -> Dict[str, Dict[str, Any]]:
    """Map task name -> {"category": ..., "difficulty_level": ...} for one suite.

    Returns an empty dict when task_classification.json is absent (e.g. LIBERO_VARIANT=orig).
    """
    import libero.libero.benchmark as _bm_mod
    json_path = os.path.join(os.path.dirname(_bm_mod.__file__), "task_classification.json")
    if not os.path.exists(json_path):
        return {}

    with open(json_path) as f:
        data = json.load(f)

    return {
        entry["name"]: {"category": entry["category"], "difficulty_level": entry["difficulty_level"]}
        for entry in data.get(suite_name, [])
    }


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


class EvaluateLibero:
    def __init__(
        self,
        model,
        transforms,
        log_dir,
        benchmark_name,
        num_sequences,
        max_steps,
        num_videos,
        n_eval,
        task_embedding_format,
        device,
        eval_batch_size: int = 10,
        task_indices: Optional[List[int]] = None,
        checkpoint: str = "",
        base_seed: int = 0,
        eval_modalities: Optional[Dict[str, bool]] = None,
    ):
        self.model = model
        self.transforms = transforms
        self.log_dir = log_dir
        self.checkpoint = str(checkpoint)
        self.base_seed = base_seed
        self.eval_modalities = eval_modalities or {"rgb_static": True, "rgb_gripper": True, "language": True}
        self.libero_variant = os.environ.get("LIBERO_VARIANT", "orig")

        self.device = device
        self.task_order = 0
        self.bddl_folder = get_libero_path("bddl_files")
        self.init_states_folder = get_libero_path("init_states")
        self.task_embedding_format =task_embedding_format
        self.benchmark_name = benchmark_name
        self.benchmark_dict = benchmark.get_benchmark_dict()
        self.benchmark_instance = self.benchmark_dict[self.benchmark_name]()
        self.num_tasks = self.benchmark_instance.get_num_tasks()
        self.num_videos = num_videos
        self.task_names = self.benchmark_instance.get_task_names()
        self.task_classification = load_task_classification(self.benchmark_name)
        self.benchmark = get_benchmark(self.benchmark_name)(self.task_order)
        self.n_eval = n_eval
        self.eval_batch_size = eval_batch_size
        self.img_h = 224
        self.img_w = 224
        self.rank = None
        self.world_size = None
        self.num_sequences = num_sequences
        self.max_steps = max_steps
        # self.save_dir = save_dir
        self.device = None
        self.eval_sequences = None
        self.init_states_paths = []
        self.cfg = {}
        self.descriptions = []
        self.create_cfg_for_libero(self.task_embedding_format)
        for i in range(self.num_tasks):
            self.descriptions.append(self.benchmark_instance.get_task(i).language)
        with torch.no_grad():
            task_embs = get_task_embs(self.cfg, self.descriptions)
        self.benchmark_instance.set_task_embs(task_embs)

        # task_indices restricts which tasks this instance evaluates (used for multi-GPU).
        if task_indices is not None:
            self.all_tasks = task_indices
        else:
            self.all_tasks = list(range(self.benchmark_instance.n_tasks))

    def setup(self) -> None:
        if self.benchmark is None:
            self.eval_sequences = get_sequences(self.num_sequences)
            self.benchmark = get_benchmark(self.benchmark_name)(self.eval_sequences)

    def start(self) -> List[float]:
        rows = self.evaluate_policy(self.model, store_video=self.num_videos)
        self.last_rows = rows

        per_task_success: Dict[int, List[int]] = defaultdict(list)
        for row in rows:
            per_task_success[row["task_idx"]].append(row["success"])
        successes = [
            sum(per_task_success[idx]) / len(per_task_success[idx]) for idx in self.all_tasks
        ]

        result_array = sum(successes) / len(successes)
        evaluated_names = [self.task_names[idx] for idx in self.all_tasks]

        if wandb.run is not None:
            wandb.log({"eval_lh/avg_seq_len": torch.tensor(result_array)})
            for success, task_name in zip(successes, evaluated_names):
                wandb.log({f"eval_lh/sr_{task_name}": success})

        logger.info(f"eval_lh/avg_seq_len success rate {torch.tensor(result_array)}")
        for success, task_name in zip(successes, evaluated_names):
            logger.info(f"eval_lh/sr_{task_name} with success {success}")

        print('done')
        print()
        return successes

    def evaluate_policy(self, model, store_video=False) -> List[Dict[str, Any]]:
        """Run every task in self.all_tasks; return one row per episode across all tasks."""
        all_rows: List[Dict[str, Any]] = []

        for idx in self.all_tasks:  # Distribute tasks across GPUs
            task_name = self.task_names[idx]
            task_i = self.benchmark_instance.get_task(idx)
            task_emb = self.benchmark_instance.task_embs[idx]
            task_str = f"k{self.all_tasks[-1]}_p{idx}"
            logger.info(f"starting to evaluate: {task_name}")
            rows = self.evaluate_task(model, task_i, task_emb, task_str, idx, store_video=store_video)
            success_rate = sum(row["success"] for row in rows) / len(rows)
            print(f"Task {task_name} success rate: {success_rate:.4f}")
            logger.info(f"Task {task_name} success rate: {success_rate:.4f}")
            all_rows.extend(rows)

        return all_rows

    def evaluate_task(self, model, task_i, task_emb, task_str, idx, sim_states=None, store_video=0):
        """Evaluate a task, running eval_batch_size parallel episodes per batch.

        Each batch uses a SubprocVectorEnv (one MuJoCo subprocess per episode) so
        rendering is isolated per process, and model inference is batched across all
        episodes in the batch.
        """
        env_args = {
            "bddl_file_name": os.path.join(
                self.bddl_folder, task_i.problem_folder, task_i.bddl_file
            ),
            "camera_heights": self.img_h,
            "camera_widths": self.img_w,
        }

        try:
            initial_states = self.benchmark_instance.get_task_init_states(idx)
            n_states = len(initial_states)
            print(f"Using LIBERO native initial states, count: {n_states}")
        except Exception as e:
            print(f"Could not get LIBERO initial states: {e}, using random resets")
            initial_states = None
            n_states = 0

        task_name = self.task_names[idx]
        task_meta = self.task_classification.get(task_name, {})

        rows: List[Dict[str, Any]] = []
        episode_idx = 0

        with tqdm(total=self.n_eval, desc="Evaluating") as pbar:
            while episode_idx < self.n_eval:
                current_batch_size = min(self.eval_batch_size, self.n_eval - episode_idx)

                # DummyVectorEnv runs all envs sequentially in the parent process.
                # This avoids the fork-after-CUDA hazard: SubprocVectorEnv forks the parent
                # (which has CUDA initialized), and forked children fail with EGL_BAD_ALLOC
                # when trying to create EGL rendering contexts.  DummyVectorEnv keeps all
                # OffScreenRenderEnv instances in the same process where EGL + CUDA coexist
                # fine (same as the original sequential code).  Batched model inference is
                # still possible since all B observations are stacked into a single forward.
                env_creation = False
                count = 0
                while not env_creation and count < 5:
                    try:
                        env = DummyVectorEnv(
                            [lambda: OffScreenRenderEnv(**env_args)
                             for _ in range(current_batch_size)]
                        )
                        env_creation = True
                    except Exception:
                        time.sleep(5)
                        count += 1
                if not env_creation:
                    raise Exception("Failed to create environment")

                # Reset and set initial states for this batch
                env.reset()
                state_idxs = None
                if initial_states is not None:
                    state_idxs = np.array(
                        [(episode_idx + k) % n_states for k in range(current_batch_size)]
                    )
                    env.set_init_state(initial_states[state_idxs])

                # Seed once per batch: the model draws one shared noise tensor for the
                # whole batch on each replan (flower.py forward()), so a seed narrower
                # than "one batch" wouldn't change what gets sampled. Recorded as
                # rollout_seed below so the batch is exactly reproducible.
                seed = rollout_seed(self.base_seed, idx, episode_idx)
                torch.manual_seed(seed)
                np.random.seed(seed % (2**32))

                # Dummy warmup steps — obs from last warmup step is the starting obs
                dummy = np.zeros((current_batch_size, 7))
                for _ in range(5):
                    obs, _, _, _ = env.step(dummy)

                # Open video writers for episodes that should be recorded
                video_writers: Dict[int, cv2.VideoWriter] = {}
                video_frames: Dict[int, list] = {}
                for k in range(current_batch_size):
                    global_ep = episode_idx + k
                    if global_ep < int(store_video):
                        video_filename = f"rollout_{task_str}_nmp_{global_ep}.mp4"
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

                while steps < self.max_steps:
                    steps += 1
                    data, goal = self.process_env_obs_batch(obs, task_emb, task_i.language)
                    actions = model.step_batch(data, goal).cpu().numpy()  # [B, 7]
                    obs, _, done, _ = env.step(actions)

                    for k in range(current_batch_size):
                        if not dones[k]:
                            steps_taken[k] = steps
                        dones[k] = dones[k] or bool(done[k])
                        if k in video_frames:
                            video_frames[k].append(obs[k]['agentview_image'])

                    if all(dones):
                        break

                # Write videos
                for k, writer in video_writers.items():
                    for frame in video_frames[k]:
                        writer.write(frame)
                    writer.release()

                for k in range(current_batch_size):
                    rows.append({
                        "libero_variant": self.libero_variant,
                        "suite": self.benchmark_name,
                        "task_idx": idx,
                        "episode_idx": episode_idx + k,
                        "checkpoint_name": checkpoint_name(self.checkpoint),
                        "task_name": task_name,
                        "problem_folder": task_i.problem_folder,
                        "bddl_file": task_i.bddl_file,
                        "init_states_file": task_i.init_states_file,
                        "language": task_i.language,
                        "task_category": task_meta.get("category", ""),
                        "difficulty_level": task_meta.get("difficulty_level", ""),
                        "init_state_idx": int(state_idxs[k]) if state_idxs is not None else -1,
                        "max_steps": self.max_steps,
                        "img_h": self.img_h,
                        "img_w": self.img_w,
                        "num_sampling_steps": getattr(model, "num_sampling_steps", ""),
                        "multistep": getattr(model, "multistep", ""),
                        "eval_batch_size": current_batch_size,
                        "base_seed": self.base_seed,
                        "rollout_seed": seed,
                        "use_rgb_static": int(self.eval_modalities.get("rgb_static", True)),
                        "use_rgb_gripper": int(self.eval_modalities.get("rgb_gripper", True)),
                        "use_language": int(self.eval_modalities.get("language", True)),
                        "checkpoint": self.checkpoint,
                        "steps_taken": steps_taken[k],
                        "success": int(dones[k]),
                    })

                env.close()
                gc.collect()
                episode_idx += current_batch_size
                pbar.update(current_batch_size)

        return rows

    def create_cfg_for_libero(self, task_embedding_format):
        self.cfg = DictConfig({'task_embedding_format': task_embedding_format,
                               'data': {'max_word_len': 77}})

        self.cfg.policy = OmegaConf.create()
        self.cfg.policy.language_encoder = OmegaConf.create()
        self.cfg.policy.language_encoder.network_kwargs = OmegaConf.create()


    def translate_obs_space(self, obs_space):

        translated_dict = {}
        translated_dict['rgb_obs'] = {}
        translated_dict['rgb_obs']['rgb_static'] = obs_space['agentview_image']
        translated_dict["rgb_obs"]['rgb_gripper'] = obs_space['robot0_eye_in_hand_image']
        translated_dict['robot_obs'] = obs_space['robot0_joint_pos']
        translated_dict['gripper_states'] = obs_space['robot0_gripper_qpos']
        translated_dict['depth_obs'] = {}

        return translated_dict

    def translate_obs_space(self, obs_space):
        """Convert LIBERO environment observations to the format expected by the model"""
        translated_dict = {}
        translated_dict['rgb_obs'] = {}
        
        # Map environment camera observations to expected keys
        # The environment uses 'agentview_image' but model expects 'rgb_static'
        if 'agentview_image' in obs_space:
            translated_dict['rgb_obs']['rgb_static'] = obs_space['agentview_image']
        # The environment uses 'robot0_eye_in_hand_image' but model expects 'rgb_gripper'
        if 'robot0_eye_in_hand_image' in obs_space:
            translated_dict['rgb_obs']['rgb_gripper'] = obs_space['robot0_eye_in_hand_image']
        
        # Map robot state observations
        if 'robot0_joint_pos' in obs_space:
            translated_dict['robot_obs'] = obs_space['robot0_joint_pos']
        if 'robot0_gripper_qpos' in obs_space:
            translated_dict['gripper_states'] = obs_space['robot0_gripper_qpos']
        
        # Empty dict for depth since not used
        translated_dict['depth_obs'] = {}
        
        return translated_dict

    def apply_transforms(self, data, train=False):
        """Apply validation transforms to the observations"""
        # Determine which transform set to use (use 'val' for evaluation)
        transform_set = 'train' if train else 'val'
        
        # Print available transform keys for debugging
        if not hasattr(self, '_printed_transforms'):
            print(f"Transform structure: {type(self.transforms)}")
            if hasattr(self.transforms, 'keys'):
                print(f"Top-level transform keys: {list(self.transforms.keys())}")
                if transform_set in self.transforms:
                    print(f"{transform_set} transform keys: {list(self.transforms[transform_set].keys())}")
            self._printed_transforms = True
        
        # Ensure we're accessing the right transform subset
        if transform_set in self.transforms:
            transforms_to_use = self.transforms[transform_set]
        else:
            print(f"Warning: '{transform_set}' not found in transforms. Available keys: {list(self.transforms.keys())}")
            transforms_to_use = self.transforms  # Fall back to top level
        
        # Process each observation
        for key in data['rgb_obs']:
            x = data['rgb_obs'][key]
            if len(x.shape) == 3:
                x = np.expand_dims(x, axis=0)
            x = torch.from_numpy(x).byte().permute(0, 3, 1, 2)
            
            # Try to find the right transform key
            transform_found = False
            
            # Check direct key match
            if key in transforms_to_use:
                for transform in transforms_to_use[key]:
                    x = transform(x)
                transform_found = True
            else:
                # Try common alternative keys
                alternative_keys = {
                    'rgb_static': ['rgb', 'agentview', 'static', 'agentview_rgb'],
                    'rgb_gripper': ['gripper', 'eye_in_hand', 'hand', 'eye_in_hand_rgb']
                }
                
                if key in alternative_keys:
                    for alt_key in alternative_keys[key]:
                        if alt_key in transforms_to_use:
                            for transform in transforms_to_use[alt_key]:
                                x = transform(x)
                            transform_found = True
                            break
            
            if not transform_found:
                print(f"Warning: No transform found for {key}. Using default normalization.")
                x = x.float() / 255.0  # Default normalization
            
            data['rgb_obs'][key] = x.unsqueeze(0).to(self.device)
        
        # Ensure robot_obs and gripper_states are properly formatted tensors
        if 'robot_obs' in data and not isinstance(data['robot_obs'], torch.Tensor):
            data['robot_obs'] = torch.tensor(data['robot_obs'], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        if 'gripper_states' in data and not isinstance(data['gripper_states'], torch.Tensor):
            data['gripper_states'] = torch.tensor(data['gripper_states'], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        return data

    def process_env_obs(self, env_obs, lang_embed, lang_text=None):
        return_obs = self.translate_obs_space(env_obs)
        return_obs = self.apply_transforms(return_obs)

        goal = {}
        goal['lang_text'] = lang_text
        goal['lang'] = lang_embed

        return return_obs, goal

    def process_env_obs_batch(self, obs_list, lang_embed, lang_text=None):
        """Process a list of B obs dicts (from SubprocVectorEnv) into a batched model input.

        Returns data with rgb_obs tensors of shape [B, 1, C, H, W] and goal with
        lang_text as a list of B identical strings (handled by model.forward).
        """
        data_list = [self.process_env_obs(obs, lang_embed, lang_text)[0] for obs in obs_list]
        batch_data = {
            'rgb_obs': {
                key: torch.cat([d['rgb_obs'][key] for d in data_list], dim=0)
                for key in data_list[0]['rgb_obs']
            }
        }
        goal = {
            'lang_text': [lang_text] * len(obs_list),
            'lang': lang_embed,
        }
        return batch_data, goal

def _eval_worker(
    rank: int,
    cfg_dict: dict,
    transforms_cfg,
    task_indices: List[int],
    log_dir: str,
    csv_dir: str,
    n_gpus: int,
) -> None:
    """Worker process: evaluate a subset of tasks on one GPU and write a result CSV shard.

    Called via torch.multiprocessing.spawn or Process for multi-GPU eval.
    Each worker points MUJOCO_EGL_DEVICE_ID at the physical GPU that matches its
    CUDA rank so rendering and inference share the same GPU.
    """
    # Map PyTorch rank (index into CUDA_VISIBLE_DEVICES) to physical GPU id for EGL.
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    phys_ids = [int(x.strip()) for x in cuda_vis.split(",") if x.strip()] if cuda_vis else list(range(n_gpus))
    phys_gpu = phys_ids[rank] if rank < len(phys_ids) else rank
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(phys_gpu)

    # Load model on this worker's GPU
    model_overwrite = cfg_dict.get("eval_cfg_overwrite", {}).get("model", {})
    model = load_mode_from_safetensor(Path(cfg_dict["checkpoint"]), overwrite_cfg=model_overwrite)
    model.freeze()
    model = model.cuda(rank)
    model.eval()

    # Instantiate transforms (hydra.utils.instantiate does not need GlobalHydra context)
    transforms = hydra.utils.instantiate(transforms_cfg)

    eval_libero = EvaluateLibero(
        model=model,
        transforms=transforms,
        log_dir=log_dir,
        benchmark_name=cfg_dict["benchmark_name"],
        num_sequences=cfg_dict["num_sequences"],
        num_videos=cfg_dict.get("num_videos", 0) if rank == 0 else 0,
        max_steps=cfg_dict["max_steps"],
        n_eval=cfg_dict["n_eval"],
        task_embedding_format=cfg_dict["task_embedding_format"],
        device=rank,
        eval_batch_size=cfg_dict.get("eval_batch_size", 10),
        task_indices=task_indices,
        checkpoint=cfg_dict["checkpoint"],
        base_seed=cfg_dict.get("seed", 0),
        eval_modalities=cfg_dict.get("eval_modalities"),
    )
    eval_libero.setup()

    # evaluate_policy already iterates self.all_tasks, which the constructor set to
    # task_indices, prints a per-task success rate, and returns one row per episode —
    # reusing the exact same rollout/row-building logic as the single-GPU path.
    rows = eval_libero.evaluate_policy(
        model, store_video=cfg_dict.get("num_videos", 0) if rank == 0 else 0
    )
    write_csv(os.path.join(csv_dir, f"result_rank{rank}.csv"), rows)


@hydra.main(config_path="../../conf", config_name="eval_libero")
def main(cfg):
    seed_everything(0, workers=True)
    model, _, dm, _ = get_default_mode_and_env(
        cfg.train_folder,
        cfg.dataset_path,
        cfg.checkpoint,
        env=42,
        lang_embeddings=None,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite,
        device_id=cfg.device,
        prep_dm_and_deps=False
    )

    model = model.to(cfg.device)
    model.eval()

    log_dir = get_log_dir(cfg.log_dir)
    transforms = hydra.utils.instantiate(dm.transforms)

    task_category: Optional[str] = OmegaConf.select(cfg, "task_category", default=None)
    base_seed: int = OmegaConf.select(cfg, "seed", default=0)
    eval_modalities = OmegaConf.select(cfg, "eval_modalities", default=None)
    if eval_modalities is not None:
        eval_modalities = OmegaConf.to_container(eval_modalities, resolve=True)

    eval_libero = EvaluateLibero(
        model=model,
        transforms=transforms,
        log_dir=log_dir,
        benchmark_name=cfg.benchmark_name,
        num_sequences=cfg.num_sequences,
        num_videos=cfg.num_videos,
        max_steps=cfg.max_steps,
        n_eval=cfg.n_eval,
        task_embedding_format=cfg.task_embedding_format,
        device=cfg.device,
        eval_batch_size=cfg.eval_batch_size,
        checkpoint=cfg.checkpoint,
        base_seed=base_seed,
        eval_modalities=eval_modalities,
    )
    csv_dir_override = OmegaConf.select(cfg, "csv_dir", default=None)
    if csv_dir_override:
        csv_dir = Path(csv_dir_override)
        csv_dir.mkdir(parents=True, exist_ok=True)
    else:
        csv_dir = result_dir(cfg.train_folder, cfg.checkpoint, eval_libero.libero_variant, cfg.benchmark_name)

    # Apply per-category filter (no-op when task_category is None or LIBERO_VARIANT=orig).
    task_subset = select_task_indices(eval_libero.benchmark_instance, task_category)
    if task_subset is not None:
        eval_libero.all_tasks = task_subset

    if cfg.log_wandb:
        os.makedirs(log_dir / "wandb", exist_ok=False)
        run = wandb.init(
            project='mode_libero_eval',
            entity=cfg.wandb_entity,
            config=OmegaConf.to_object(cfg),
        )

    n_gpus = torch.cuda.device_count()

    if n_gpus <= 1:
        # Single-GPU path: all tasks on one GPU
        eval_libero.setup()
        successes = eval_libero.start()

        csv_path = csv_dir / "result.csv"
        merge_result_csv(csv_path, eval_libero.last_rows)
        print(f"Wrote {len(eval_libero.last_rows)} episode rows to {csv_path}")

        # Per-category breakdown (LIBERO-Plus only; no-op for orig).
        evaluated_names = [eval_libero.task_names[idx] for idx in eval_libero.all_tasks]
        cat_results = aggregate_by_category(
            evaluated_names, successes, eval_libero.benchmark_instance.name
        )
        if cat_results:
            print("\nPer-category success rates:")
            for cat, sr in sorted(cat_results.items()):
                print(f"  {cat}: {sr:.4f}")
                if cfg.log_wandb:
                    wandb.log({f"eval_lh/cat_{cat.replace(' ', '_').lower()}": sr})
    else:
        # Multi-GPU path: partition tasks across workers, one process per GPU.
        all_tasks = eval_libero.all_tasks  # already filtered by task_subset above
        task_splits: List[List[int]] = [[] for _ in range(n_gpus)]
        for i, idx in enumerate(all_tasks):
            task_splits[i % n_gpus].append(idx)

        # Free the parent's GPU0 model — each worker loads its own copy.
        eval_libero.model = None
        del model
        gc.collect()
        torch.cuda.empty_cache()

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        transforms_cfg = dm.transforms  # raw OmegaConf DictConfig, picklable

        ctx = tmp.get_context("spawn")
        processes = []
        for rank in range(n_gpus):
            p = ctx.Process(
                target=_eval_worker,
                args=(rank, cfg_dict, transforms_cfg, task_splits[rank], str(log_dir), str(csv_dir), n_gpus),
            )
            p.start()
            processes.append(p)

        for rank, p in enumerate(processes):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Eval worker rank {rank} exited with code {p.exitcode}")

        # Aggregate per-episode rows from all workers' shards. Read directly (not via
        # merge_rank_csvs yet) so this run's summary only reflects this run's tasks,
        # not older rows already accumulated in result.csv from a prior category run.
        current_rows: List[Dict[str, Any]] = []
        for rank in range(n_gpus):
            current_rows.extend(read_csv(csv_dir / f"result_rank{rank}.csv"))

        task_names = eval_libero.task_names
        evaluated_names = [task_names[idx] for idx in all_tasks]
        per_task_success: Dict[int, List[int]] = defaultdict(list)
        for row in current_rows:
            per_task_success[int(row["task_idx"])].append(int(row["success"]))
        successes = [
            sum(per_task_success[idx]) / len(per_task_success[idx]) for idx in all_tasks
        ]
        result_array = sum(successes) / len(successes)

        if wandb.run is not None:
            wandb.log({"eval_lh/avg_seq_len": torch.tensor(result_array)})
            for success, task_name in zip(successes, evaluated_names):
                wandb.log({f"eval_lh/sr_{task_name}": success})
        logger.info(f"eval_lh/avg_seq_len success rate {torch.tensor(result_array)}")
        for success, task_name in zip(successes, evaluated_names):
            print(f"Task {task_name} success rate: {success:.4f}")

        # Per-category breakdown (LIBERO-Plus only; no-op for orig).
        cat_results = aggregate_by_category(
            evaluated_names, successes, eval_libero.benchmark_instance.name
        )
        if cat_results:
            print("\nPer-category success rates:")
            for cat, sr in sorted(cat_results.items()):
                print(f"  {cat}: {sr:.4f}")
                if wandb.run is not None:
                    wandb.log({f"eval_lh/cat_{cat.replace(' ', '_').lower()}": sr})

        # Merge this run's shards into the accumulated result.csv, then delete them.
        merged_rows = merge_rank_csvs(csv_dir, n_gpus)
        print(f"Wrote {len(current_rows)} episode rows to {csv_dir / 'result.csv'} "
              f"({len(merged_rows)} rows total)")

        print('done')

    if cfg.log_wandb:
        run.finish()

if __name__ == "__main__":
    # Set CUDA device IDs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    main()