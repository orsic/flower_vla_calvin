"""VLA-Adapter (github.com/OpenHelix-Team/VLA-Adapter) behind the rollout interface the
LIBERO (flower_eval_libero.py) and CALVIN (flower_evaluate.py) harnesses call on FLOWERVLA,
so its evals share this repo's seeds, init states, result.csv records and multi-GPU split.

Selected with the eval config key `policy=vla_adapter`; `checkpoint` is a downloaded
VLA-Adapter checkpoint directory (./run.sh download-vla-adapter). Inference goes through
VLA-Adapter's own loaders and get_action (the VLA-Adapter/ submodule), so only the
evaluator-side glue lives here, mirroring its reference evaluators:

- LIBERO (experiments/robot/libero/run_libero_eval.py): both cameras rotated 180 degrees,
  state = eef pos + axis-angle + gripper qpos, an 8-step chunk executed open-loop, the
  gripper binarized and inverted.
- CALVIN (vla-scripts/evaluate_calvin.py:rollout_hi3, vla_evaluation.py): raw static/wrist
  images, state = robot_obs[:7] + robot_obs[-1:], gripper flipped to 1 - g, and a 10-step
  period that queries the model at its first three steps and averages the overlapping
  chunks (calvin_ensemble_action). calvin_schedule="chunk" runs plain 8-step open-loop.

Both platforms resize to 224 with VLA-Adapter's JPEG + lanczos resize and center-crop
(inside get_action). Modality masking is not wired yet: set masking attributes raise.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

CHUNK = 8
PLATFORMS = ("libero", "calvin")
CALVIN_SCHEDULES = ("reference", "chunk")
CALVIN_PERIOD = 10

PredictChunk = Callable[[Dict[str, np.ndarray], str], np.ndarray]


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quaternion to axis-angle; VLA-Adapter's libero_utils copy of robosuite's."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def libero_observation(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """VLA-Adapter's LIBERO input from a robosuite obs dict (run_libero_eval.prepare_observation)."""
    return {
        "full_image": raw["agentview_image"][::-1, ::-1],
        "wrist_image": raw["robot0_eye_in_hand_image"][::-1, ::-1],
        "state": np.concatenate(
            (raw["robot0_eef_pos"], quat2axisangle(raw["robot0_eef_quat"]), raw["robot0_gripper_qpos"])
        ),
    }


def calvin_observation(raw: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """VLA-Adapter's CALVIN input from a raw calvin_env obs (DualSystemCalvinEvaluation.step):
    tcp pos (3), euler (3), gripper width (1) and the gripper action (last entry)."""
    return {
        "full_image": raw["rgb_obs"]["rgb_static"],
        "wrist_image": raw["rgb_obs"]["rgb_gripper"],
        "state": np.concatenate([raw["robot_obs"][:7], raw["robot_obs"][-1:]]),
    }


def libero_gripper(actions: np.ndarray) -> np.ndarray:
    """[0, 1] (1 = open) to LIBERO's -1 = open: robot_utils.normalize_gripper_action(binarize)
    then invert_gripper_action."""
    out = np.array(actions, dtype=np.float64)
    out[..., -1] = -np.sign(2.0 * out[..., -1] - 1.0)
    return out


def calvin_gripper(actions: np.ndarray) -> np.ndarray:
    """The reference's process_action on an already-flipped (1 - g) gripper value:
    normalize to [-1, 1], binarize with a tie going to +1, then invert."""
    out = np.array(actions, dtype=np.float64)
    sign = np.sign(2.0 * out[..., -1] - 1.0)
    sign[sign == 0.0] = 1.0
    out[..., -1] = -sign
    return out


def calvin_ensemble_action(chunks: Sequence[np.ndarray], p: int) -> np.ndarray:
    """Action at step p of rollout_hi3's 10-step period, from the chunks queried at its
    steps 0, 1 and 2 (`chunks[k]` is used only once p >= k). Reproduces the reference
    exactly, including that steps 2 and 3 both execute the same three-way average."""
    q = chunks
    if p == 0:
        return q[0][0]
    if p == 1:
        return (q[0][1] + q[1][0]) / 2
    if p <= 7:
        t = max(p - 1, 2)
        return (q[0][t] + q[1][t - 1] + q[2][t - 2]) / 3
    if p == 8:
        return (q[1][7] + q[2][6]) / 2
    if p == 9:
        return q[2][7]
    raise ValueError(f"period step must be in [0, {CALVIN_PERIOD}), got {p}")


def select_unnorm_key(norm_stats: Dict[str, Any], platform: str) -> str:
    """The checkpoint's action/proprio statistics key: its only key, else the platform's."""
    if len(norm_stats) == 1:
        return next(iter(norm_stats))
    preferred = {"libero": "libero_10_no_noops", "calvin": "calvin_abc_rlds"}[platform]
    if preferred in norm_stats:
        return preferred
    raise ValueError(f"Cannot pick an unnorm key for {platform} from {sorted(norm_stats)}")


class VLAAdapterPolicy:
    """Rollout interface of FLOWERVLA (step / step_batch / reset / rollout_step_counter /
    multistep / eval_* attributes) around a `predict_chunk(obs, instruction) -> (8, 7)`
    callable returning VLA-Adapter's un-normalized chunk (gripper in [0, 1], 1 = open)."""

    def __init__(self, predict_chunk: PredictChunk, platform: str, calvin_schedule: str = "reference"):
        if platform not in PLATFORMS:
            raise ValueError(f"platform must be one of {PLATFORMS}, got {platform!r}")
        if calvin_schedule not in CALVIN_SCHEDULES:
            raise ValueError(f"calvin_schedule must be one of {CALVIN_SCHEDULES}, got {calvin_schedule!r}")
        self.predict_chunk = predict_chunk
        self.platform = platform
        self.calvin_schedule = calvin_schedule
        self.multistep = CHUNK
        self.num_sampling_steps = 1  # one deterministic forward pass per chunk
        self.use_proprio = True
        # Set by the harnesses (FLOWERVLA's masking hooks); only the all-inputs defaults
        # are supported until masking is implemented for VLA-Adapter.
        self.eval_modality_mask: Optional[tuple] = None
        self.eval_blank_instruction = False
        self.eval_mean_pixel: tuple = ()
        self.eval_proprio_mask = True
        self.eval_proprio_fill: Optional[torch.Tensor] = None
        self.reset()

    def reset(self) -> None:
        self.rollout_step_counter = 0
        self._chunks: List[np.ndarray] = []

    def set_eval_noise_seeds(self, seeds: Sequence[int]) -> None:
        """No noise to seed: the L1-regression head is deterministic."""

    def _check_unmasked(self) -> None:
        if (
            self.eval_modality_mask is not None
            or self.eval_blank_instruction
            or any(self.eval_mean_pixel)
            or not self.eval_proprio_mask
            or self.eval_proprio_fill is not None
        ):
            raise NotImplementedError("Modality masking is not implemented for VLA-Adapter yet")

    # --- LIBERO (batched harness) ---
    def process_raw_obs(self, obs_list: Sequence[Dict[str, np.ndarray]], lang_text: Union[str, Sequence[str]]):
        """Used by EvaluateLibero.process_env_obs_batch in place of FlowerVLA's transforms."""
        texts = [lang_text] * len(obs_list) if isinstance(lang_text, str) else list(lang_text)
        return [libero_observation(o) for o in obs_list], texts

    def step_batch(self, data, goal) -> torch.Tensor:
        self._check_unmasked()
        t = self.rollout_step_counter % self.multistep
        if t == 0:
            self._chunks = [libero_gripper(self.predict_chunk(obs, text)) for obs, text in zip(data, goal)]
        self.rollout_step_counter += 1
        return torch.from_numpy(np.stack([chunk[t] for chunk in self._chunks]))

    # --- CALVIN (sequential harness, raw calvin_env observations) ---
    def _query_calvin(self, obs: Dict[str, Any], instruction: str) -> np.ndarray:
        chunk = np.array(self.predict_chunk(calvin_observation(obs), instruction), dtype=np.float64)
        chunk[:, -1] = 1.0 - chunk[:, -1]
        return chunk

    def step(self, obs: Dict[str, Any], goal: Dict[str, str]) -> np.ndarray:
        self._check_unmasked()
        instruction = goal["lang_text"]
        if self.calvin_schedule == "chunk":
            t = self.rollout_step_counter % CHUNK
            if t == 0:
                self._chunks = [self._query_calvin(obs, instruction)]
            action = self._chunks[0][t]
        else:
            p = self.rollout_step_counter % CALVIN_PERIOD
            if p == 0:
                self._chunks = []
            if p < 3:
                self._chunks.append(self._query_calvin(obs, instruction))
            action = calvin_ensemble_action(self._chunks, p)
        self.rollout_step_counter += 1
        return calvin_gripper(action[None])[0]


def load_vla_adapter(
    checkpoint: Union[str, Path], platform: str, device: int = 0, calvin_schedule: str = "reference"
) -> VLAAdapterPolicy:
    """Load a VLA-Adapter checkpoint dir with VLA-Adapter's own loaders on cuda:<device>."""
    import tensorflow as tf

    # TF only resizes images here; keep it off the GPUs the policy and renderer use.
    tf.config.set_visible_devices([], "GPU")
    from experiments.robot import openvla_utils, robot_utils

    # VLA-Adapter hardcodes cuda:0 as a module global; point it at this worker's GPU.
    openvla_utils.DEVICE = robot_utils.DEVICE = torch.device(f"cuda:{device}")
    checkpoint = str(checkpoint)
    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=checkpoint,
        load_in_8bit=False,
        load_in_4bit=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        use_l1_regression=True,
        # get_action_head infers this from "Pro" in the path; made explicit here.
        use_pro_version="Pro" in Path(checkpoint).name,
        center_crop=True,
        num_open_loop_steps=CHUNK,
        save_version="",
        unnorm_key=None,
    )
    vla = robot_utils.get_model(cfg)
    proprio_projector = openvla_utils.get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    action_head = openvla_utils.get_action_head(cfg, vla.llm_dim)
    processor = openvla_utils.get_processor(cfg)
    cfg.unnorm_key = select_unnorm_key(vla.norm_stats, platform)

    def predict_chunk(obs: Dict[str, np.ndarray], instruction: str) -> np.ndarray:
        actions = robot_utils.get_action(
            cfg,
            vla,
            dict(obs),
            instruction,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            use_film=False,
            use_minivlm=True,
        )
        return np.stack(actions)

    return VLAAdapterPolicy(predict_chunk, platform, calvin_schedule)


def make_calvin_env(dataset_path: Union[str, Path], device_id: int):
    """Raw calvin_env PlayTableSimEnv on the validation split (no FlowerVLA transforms), as
    VLA-Adapter's calvin_env_wrapper; rendering on the EGL device matching cuda:<device_id>."""
    from calvin_env.envs.play_table_env import get_env

    from flower.wrappers.hulc_wrapper import HulcWrapper

    HulcWrapper.set_egl_device(torch.device(f"cuda:{device_id}"))
    obs_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
    return get_env(Path(dataset_path) / "validation", obs_space=obs_space, show_gui=False)
