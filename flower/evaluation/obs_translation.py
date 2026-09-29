"""Robosuite-observation -> FLOWER-model-input translation, shared by every eval that
steps a robosuite/robomimic env (LIBERO, LIBERO-Plus, MimicGen): mapping the env's
`agentview_image` / `robot0_eye_in_hand_image` / `robot0_joint_pos` /
`robot0_gripper_qpos` keys to the model's `rgb_static` / `rgb_gripper` / `robot_obs`,
applying the eval-time image transforms, and batching B such observations for a
vectorized env step.

Extracted from flower/evaluation/flower_eval_libero.py's EvaluateLibero, which now
delegates to these free functions; nothing here is LIBERO-specific.
"""

import numpy as np
import torch

_printed_transforms = False


def translate_obs_space(obs_space):
    """Convert a robosuite/robomimic env observation dict to the model's input keys."""
    translated_dict = {}
    translated_dict['rgb_obs'] = {}

    # Map environment camera observations to expected keys
    # The environment uses 'agentview_image' but model expects 'rgb_static'
    if 'agentview_image' in obs_space:
        translated_dict['rgb_obs']['rgb_static'] = obs_space['agentview_image']
    # The environment uses 'robot0_eye_in_hand_image' but model expects 'rgb_gripper'
    if 'robot0_eye_in_hand_image' in obs_space:
        translated_dict['rgb_obs']['rgb_gripper'] = obs_space['robot0_eye_in_hand_image']

    # Map robot state observations. Match training's proprio layout
    # (libero_data_module.py / mimicgen_data_module.py): joint positions + gripper
    # state, concatenated.
    if 'robot0_joint_pos' in obs_space and 'robot0_gripper_qpos' in obs_space:
        translated_dict['robot_obs'] = np.concatenate(
            [obs_space['robot0_joint_pos'], obs_space['robot0_gripper_qpos']], axis=-1
        )

    # Empty dict for depth since not used
    translated_dict['depth_obs'] = {}

    return translated_dict


def apply_transforms(data, transforms, device, train=False):
    """Apply the eval-time image transforms ('val' unless train=True) to one sample."""
    global _printed_transforms
    transform_set = 'train' if train else 'val'

    if not _printed_transforms:
        print(f"Transform structure: {type(transforms)}")
        if hasattr(transforms, 'keys'):
            print(f"Top-level transform keys: {list(transforms.keys())}")
            if transform_set in transforms:
                print(f"{transform_set} transform keys: {list(transforms[transform_set].keys())}")
        _printed_transforms = True

    if transform_set in transforms:
        transforms_to_use = transforms[transform_set]
    else:
        print(f"Warning: '{transform_set}' not found in transforms. Available keys: {list(transforms.keys())}")
        transforms_to_use = transforms  # Fall back to top level

    for key in data['rgb_obs']:
        x = data['rgb_obs'][key]
        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x).byte().permute(0, 3, 1, 2)

        transform_found = False
        if key in transforms_to_use:
            for transform in transforms_to_use[key]:
                x = transform(x)
            transform_found = True
        else:
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
            x = x.float() / 255.0

        data['rgb_obs'][key] = x.unsqueeze(0).to(device)

    if 'robot_obs' in data and not isinstance(data['robot_obs'], torch.Tensor):
        data['robot_obs'] = torch.tensor(data['robot_obs'], dtype=torch.float32).unsqueeze(0).to(device)

    return data


def process_env_obs(env_obs, lang_embed, lang_text, transforms, device):
    return_obs = translate_obs_space(env_obs)
    return_obs = apply_transforms(return_obs, transforms, device)

    goal = {'lang_text': lang_text, 'lang': lang_embed}
    return return_obs, goal


def process_env_obs_batch(obs_list, lang_embed, lang_text, transforms, device):
    """Process a list of B obs dicts (from a vector env) into a batched model input.

    Vectorized equivalent of calling process_env_obs B times and concatenating:
    one H2D copy and one transform-chain pass per camera instead of B of each.
    The val transform chain (Resize -> ScaleImageTensor -> Normalize) is applied
    per-sample/per-pixel, so batching it is numerically identical to the loop --
    see tests/test_batched_eval.py::test_process_env_obs_batch_values_match_single.

    lang_text may be a single string, broadcast across the batch (every slot is
    the same task), or a list of B strings (one task per slot, cross-task
    batching) -- mirrors the dispatch in FLOWERVLA.forward().

    Returns data with rgb_obs tensors of shape [B, 1, C, H, W] and goal with
    lang_text as a list of B strings (handled by model.forward).
    """
    translated = [translate_obs_space(obs) for obs in obs_list]
    transforms_to_use = transforms['val'] if 'val' in transforms else transforms

    rgb_obs = {}
    for key in translated[0]['rgb_obs']:
        imgs = np.stack([t['rgb_obs'][key] for t in translated])  # [B, H, W, C]
        x = torch.from_numpy(imgs).byte().permute(0, 3, 1, 2)  # [B, C, H, W]
        for transform in transforms_to_use[key]:
            x = transform(x)
        rgb_obs[key] = x.unsqueeze(1).to(device)  # [B, 1, C, H, W]
    batch_data = {'rgb_obs': rgb_obs}

    if 'robot_obs' in translated[0]:
        robot_obs = np.stack([t['robot_obs'] for t in translated])  # [B, D]
        batch_data['robot_obs'] = torch.from_numpy(robot_obs).float().to(device)

    lang_text_list = [lang_text] * len(obs_list) if isinstance(lang_text, str) else list(lang_text)
    goal = {
        'lang_text': lang_text_list,
        'lang': lang_embed,
    }
    return batch_data, goal
