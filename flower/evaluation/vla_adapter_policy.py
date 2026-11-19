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

Both platforms resize to 224 with VLA-Adapter's JPEG + lanczos resize and center-crop.

Inputs can be withheld the way the harnesses withhold them from FlowerVLA (forward_chunk;
with all inputs it reproduces VLA-Adapter's predict_action exactly):
- removed (eval_modality_mask / eval_proprio_mask=False): a camera's patch tokens or the
  whole prompt (all but its first token, which anchors the layout like FlowerVLA's <Flow>)
  are left out, kept tokens keep their full-layout positions; proprio's token is left out
  of the action head's adapter attention.
- masked: a camera's normalized pixels replaced per channel by their mean
  (eval_mean_pixel), the instruction blanked inside the prompt template
  (eval_blank_instruction), proprio replaced by its training-mean state from the
  checkpoint's dataset_statistics.json (eval_proprio_fill, via training_mean_proprio).
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

CHUNK = 8
PLATFORMS = ("libero", "calvin")
CALVIN_SCHEDULES = ("reference", "chunk")
CALVIN_PERIOD = 10
NUM_ACTION_QUERIES = 64  # prismatic.vla.constants.NUM_TOKENS
PATCHES_PER_IMAGE = 256
PROMPT = (
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nWhat action should the robot take to {}?<|im_end|>\n<|im_start|>assistant\n"
)  # experiments/robot/openvla_utils.get_vla_action, use_minivlm=True


@dataclass(frozen=True)
class Modalities:
    """Which inputs reach VLA-Adapter, and how: a camera or language is present or removed,
    a present camera may be mean-pixel masked, the instruction blanked, and proprio is
    "state" (the robot's), "mean" (training mean) or "off" (removed)."""

    static: bool = True
    wrist: bool = True
    lang: bool = True
    blank_instruction: bool = False
    mean_static: bool = False
    mean_wrist: bool = False
    proprio: str = "state"

    @property
    def complete(self) -> bool:
        return self.static and self.wrist and self.lang


PredictChunk = Callable[[Dict[str, np.ndarray], str, Modalities], np.ndarray]


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


def layout_positions(
    n_prompt: int, static: bool, wrist: bool, lang: bool, n_patches: int = PATCHES_PER_IMAGE
) -> List[int]:
    """Position ids of the kept tokens in the full sequence layout
    [tok0, static patches, wrist patches, n_prompt prompt tokens, action queries, stop]:
    a removed input's tokens are left out without shifting the others (FlowerVLA's
    position_correct)."""
    static_ids = list(range(1, 1 + n_patches))
    wrist_ids = list(range(1 + n_patches, 1 + 2 * n_patches))
    prompt_start = 1 + 2 * n_patches
    prompt_ids = list(range(prompt_start, prompt_start + n_prompt))
    tail_start = prompt_start + n_prompt
    tail_ids = list(range(tail_start, tail_start + NUM_ACTION_QUERIES + 1))
    return [0] + static_ids * static + wrist_ids * wrist + prompt_ids * lang + tail_ids


def mean_pixel(pixel_values: torch.Tensor, masked: Sequence[bool]) -> torch.Tensor:
    """Replace each masked image's channels by their spatial mean. pixel_values stacks the
    kept images along dim 1 (each with its own normalized channels, 6 for DINO + SigLIP)."""
    per_image = pixel_values.shape[1] // len(masked)
    out = pixel_values.clone()
    for k, mask in enumerate(masked):
        if mask:
            block = out[:, k * per_image : (k + 1) * per_image]
            out[:, k * per_image : (k + 1) * per_image] = (
                block.float().mean(dim=(-2, -1), keepdim=True).expand_as(block).to(block.dtype)
            )
    return out


class VLAAdapterPolicy:
    """Rollout interface of FLOWERVLA (step / step_batch / reset / rollout_step_counter /
    multistep / eval_* attributes) around a `predict_chunk(obs, instruction) -> (8, 7)`
    callable returning VLA-Adapter's un-normalized chunk (gripper in [0, 1], 1 = open)."""

    def __init__(
        self,
        predict_chunk: PredictChunk,
        platform: str,
        calvin_schedule: str = "reference",
        proprio_mean: Optional[np.ndarray] = None,
    ):
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
        self.proprio_mean = proprio_mean
        # Set by the harnesses, as on FLOWERVLA; read into a Modalities per query.
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

    def training_mean_proprio(self) -> np.ndarray:
        """The masked-proprio fill (eval_records.apply_mean_proprio): the training-mean
        state from the checkpoint's dataset statistics."""
        if self.proprio_mean is None:
            raise ValueError("This VLAAdapterPolicy was built without a training-mean proprio state")
        return self.proprio_mean

    def modalities(self) -> Modalities:
        keep = self.eval_modality_mask or (True, True, True)
        mean_static, mean_wrist = tuple(self.eval_mean_pixel) or (False, False)
        if self.eval_proprio_fill is not None:
            proprio = "mean"
        elif not self.eval_proprio_mask:
            proprio = "off"
        else:
            proprio = "state"
        return Modalities(
            static=bool(keep[0]),
            wrist=bool(keep[1]),
            lang=bool(keep[2]),
            blank_instruction=bool(self.eval_blank_instruction),
            mean_static=bool(mean_static),
            mean_wrist=bool(mean_wrist),
            proprio=proprio,
        )

    # --- LIBERO (batched harness) ---
    def process_raw_obs(self, obs_list: Sequence[Dict[str, np.ndarray]], lang_text: Union[str, Sequence[str]]):
        """Used by EvaluateLibero.process_env_obs_batch in place of FlowerVLA's transforms."""
        texts = [lang_text] * len(obs_list) if isinstance(lang_text, str) else list(lang_text)
        return [libero_observation(o) for o in obs_list], texts

    def step_batch(self, data, goal) -> torch.Tensor:
        t = self.rollout_step_counter % self.multistep
        if t == 0:
            mods = self.modalities()
            self._chunks = [libero_gripper(self.predict_chunk(obs, text, mods)) for obs, text in zip(data, goal)]
        self.rollout_step_counter += 1
        return torch.from_numpy(np.stack([chunk[t] for chunk in self._chunks]))

    # --- CALVIN (sequential harness, raw calvin_env observations) ---
    def _query_calvin(self, obs: Dict[str, Any], instruction: str) -> np.ndarray:
        chunk = np.array(self.predict_chunk(calvin_observation(obs), instruction, self.modalities()), dtype=np.float64)
        chunk[:, -1] = 1.0 - chunk[:, -1]
        return chunk

    def step(self, obs: Dict[str, Any], goal: Dict[str, str]) -> np.ndarray:
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
    parts = SimpleNamespace(
        cfg=cfg, vla=vla, processor=processor, action_head=action_head, proprio_projector=proprio_projector
    )

    def predict_chunk(obs: Dict[str, np.ndarray], instruction: str, mods: Modalities) -> np.ndarray:
        return forward_chunk(parts, obs, instruction, mods)

    proprio_mean = np.asarray(vla.norm_stats[cfg.unnorm_key]["proprio"]["mean"], dtype=np.float64)
    policy = VLAAdapterPolicy(predict_chunk, platform, calvin_schedule, proprio_mean=proprio_mean)
    policy.parts = parts  # the loaded components, for tests/test_vla_adapter_equivalence.py
    return policy


@contextmanager
def rope_covering(language_model: torch.nn.Module, n_positions: int):
    """Let Qwen2's rotary tables cover position ids up to n_positions - 1. transformers
    4.40 slices them to the sequence length, which a sequence with tokens left out (but
    keeping their full-layout positions) undercuts; the values are the same table, longer."""
    rotaries = [m for m in language_model.modules() if type(m).__name__.endswith("RotaryEmbedding")]
    originals = [m.forward for m in rotaries]
    for module, original in zip(rotaries, originals):
        module.forward = lambda x, seq_len=None, _f=original: _f(x, seq_len=max(seq_len, n_positions))
    try:
        yield
    finally:
        for module, original in zip(rotaries, originals):
            module.forward = original


@torch.inference_mode()
def forward_chunk(parts: SimpleNamespace, obs: Dict[str, np.ndarray], instruction: str, mods: Modalities) -> np.ndarray:
    """VLA-Adapter's predict_action (L1-regression head, B=1) with inputs removed or masked
    per `mods`; returns the un-normalized (8, 7) chunk. Mirrors
    OpenVLAForActionPrediction.predict_action / _regression_or_discrete_prediction and
    L1RegressionActionHead.predict_action step for step, including which hidden states they
    slice, so that Modalities() reproduces them exactly (tests/test_vla_adapter_equivalence.py)."""
    from experiments.robot import openvla_utils
    from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK

    cfg, vla, processor, action_head = parts.cfg, parts.vla, parts.processor, parts.action_head
    device = openvla_utils.DEVICE
    views = [(obs["full_image"], mods.static, mods.mean_static), (obs["wrist_image"], mods.wrist, mods.mean_wrist)]
    kept = [(image, masked) for image, keep, masked in views if keep]
    if not kept:
        raise ValueError("At least one camera must be kept")
    images = openvla_utils.prepare_images_for_vla([image for image, _ in kept], cfg)
    prompt = PROMPT.format(("" if mods.blank_instruction else instruction).lower())
    inputs = processor(prompt, images[0]).to(device, dtype=torch.bfloat16)
    pixel_values = torch.cat(
        [inputs["pixel_values"]]
        + [processor(prompt, image).to(device, dtype=torch.bfloat16)["pixel_values"] for image in images[1:]],
        dim=1,
    )
    if any(masked for _, masked in kept):
        pixel_values = mean_pixel(pixel_values, [masked for _, masked in kept])

    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    n_prompt_full = input_ids.shape[-1] - 1
    if not mods.lang:
        input_ids, attention_mask = input_ids[:, :1], attention_mask[:, :1]
    n_prompt = input_ids.shape[-1] - 1
    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX
    input_ids, attention_mask = vla._prepare_input_for_action_prediction(input_ids, attention_mask)
    labels = vla._prepare_labels_for_action_prediction(labels, input_ids)
    input_embeddings = vla.get_input_embeddings()(input_ids)
    all_actions_mask = vla._process_action_masks(labels)

    vla.vision_backbone.set_num_images_in_input(len(kept))
    patch_embeddings = vla._process_vision_features(pixel_values, None, False)
    n_patches = vla.vision_backbone.get_num_patches() * len(kept)

    batch = input_embeddings.shape[0]
    queries = vla.action_queries.weight
    queries = queries.view(1, queries.shape[0], queries.shape[1]).repeat(batch, 1, 1)
    input_embeddings = vla._replace_input_embeddings(input_embeddings.clone(), all_actions_mask, queries)
    embeddings, multimodal_mask = vla._build_multimodal_attention(input_embeddings, patch_embeddings, attention_mask)
    position_ids, n_positions = None, embeddings.shape[1]
    if not mods.complete:
        positions = layout_positions(
            n_prompt_full, mods.static, mods.wrist, mods.lang, vla.vision_backbone.get_num_patches()
        )
        position_ids, n_positions = torch.tensor([positions], device=embeddings.device), positions[-1] + 1
    with rope_covering(vla.language_model, n_positions):
        output = vla.language_model(
            input_ids=None,
            attention_mask=multimodal_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
    start = n_patches + n_prompt
    layers = []
    for hidden in output.hidden_states:
        actions = hidden[:, start : start + NUM_ACTION_QUERIES, :].reshape(1, 1, NUM_ACTION_QUERIES, -1)
        task = hidden[:, :n_patches].reshape(batch, 1, n_patches, -1)
        layers.append(torch.cat((task, actions.to(torch.bfloat16)), 2))
    hidden_states = torch.cat(layers, dim=1)
    task_states, action_states = hidden_states[:, :, :n_patches, :], hidden_states[:, :, n_patches:, :]

    if mods.proprio == "off":
        proprio_features = action_states.new_zeros((batch, 0, action_states.shape[-1]))
    else:
        stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
        state = np.asarray(stats["mean"]) if mods.proprio == "mean" else obs["state"]
        proprio = torch.Tensor(openvla_utils.normalize_proprio(state, stats))
        proprio = proprio.to(patch_embeddings.device, dtype=patch_embeddings.dtype)
        proprio = proprio.reshape(batch, -1).to(torch.bfloat16)
        proprio_features = parts.proprio_projector(proprio).unsqueeze(dim=1)
    cond = torch.zeros(
        (batch, action_head.action_dim * NUM_ACTIONS_CHUNK, action_head.hidden_dim),
        device=action_states.device,
        dtype=action_states.dtype,
    )
    normalized = action_head.model(
        cond.reshape(batch, NUM_ACTIONS_CHUNK, -1), h_a=action_states, p=proprio_features, h_t=task_states
    )
    normalized = normalized.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM).float().cpu().detach().numpy()
    return vla._unnormalize_actions(normalized, cfg.unnorm_key)


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
