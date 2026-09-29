import hydra
import numpy as np
import torch
from torch.utils.data import Dataset

# Source-hdf5 key names for the four fields this dataset reads off a robomimic
# SequenceDataset sample. LIBERO's hdf5s use these; a MimicGen-rendered hdf5 (via
# robomimic's dataset_states_to_obs.py) uses the robosuite-native names instead
# (agentview_image / robot0_eye_in_hand_image / robot0_joint_pos / robot0_gripper_qpos).
DEFAULT_KEY_MAP = {
    "static": "agentview_rgb",
    "wrist": "eye_in_hand_rgb",
    "joints": "joint_states",
    "gripper": "gripper_states",
}


class TranslatedSequenceVLDataset(Dataset):
    def __init__(
        self,
        sequence_dataset,
        task_emb,
        task_description,
        obs_seq_len: int = 1,
        act_seq_len: int = 1,
        transforms=None,
        key_map: dict = None,
    ):
        self.obs_seq_len = obs_seq_len
        self.act_seq_len = act_seq_len
        self.transforms = hydra.utils.instantiate(transforms)
        self.sequence_dataset = sequence_dataset
        # add goal mode to the sequence dataset
        self.sequence_dataset.goal_mode = "last"
        self.task_emb = task_emb
        self.task_description = task_description
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences
        self.key_map = dict(DEFAULT_KEY_MAP if key_map is None else key_map)

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        main_dict = {}
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_emb"] = self.task_emb
        return_dict["lang_text"] = self.task_description
        return_dict = self.get_des_act_obs_sequence(return_dict)
        main_dict["lang"] = self.translation_dict(return_dict)
        # main_dict['modality'] = 'lang'

        # Apply transforms
        if self.transforms:
            main_dict = self.apply_transforms(main_dict["lang"])
        main_dict['idx'] = idx
        return main_dict

    def apply_transforms(self, data, train=True):
        # Assuming data contains images in 'rgb_static' and 'rgb_gripper'
        if train:
            transforms = self.transforms['train']
        for key in data['rgb_obs']:
            x = data['rgb_obs'][key]
            x = torch.from_numpy(x).byte().permute(0, 3, 1, 2)
            for transform in transforms[key]:
                x = transform(x)
            data['rgb_obs'][key] = x
            # data['rgb_obs'][key] = transforms[key](data['rgb_obs'][key])

        return data

    def get_des_act_obs_sequence(self, return_dict):

        for key in return_dict['obs']:
            return_dict['obs'][key] = return_dict['obs'][key][:self.obs_seq_len]
        return_dict['actions'] = return_dict['actions'][:self.act_seq_len]

        return_dict['robot_obs'] = return_dict['obs'][self.key_map['joints']][:self.obs_seq_len]
        return_dict['gripper_states'] = return_dict['obs'][self.key_map['gripper']][:self.obs_seq_len]

        return return_dict

    def translation_dict(self, dict):
        translated_dict = {}
        # dict['obs'] = self.combine_goal_obs_with_obs(dict['obs'], dict['goal_obs'])
        if 'obs' in dict.keys():
            translated_dict['rgb_obs'] = {}
            translated_dict["rgb_obs"]['rgb_static'] = dict['obs'][self.key_map['static']]
            translated_dict["rgb_obs"]['rgb_gripper'] = dict['obs'][self.key_map['wrist']]
            translated_dict['robot_obs'] = dict['obs'][self.key_map['joints']]
            # translated_dict['gripper_states'] = dict['obs']['gripper_states']

        translated_dict['lang_text'] = dict['lang_text']
        translated_dict['depth_obs'] = {}
        translated_dict['actions'] = dict['actions']
        # translated_dict['robot_obs'] = dict['robot_obs']
        translated_dict['robot_obs'] = np.concatenate([dict['robot_obs'], np.expand_dims(dict['obs'][self.key_map['gripper']][0], 0)], axis=-1)
        return translated_dict

    def combine_goal_obs_with_obs(self, obs, goal_obs):
        combined_obs = {}
        for key in obs:
            if key in ('actions'):
                combined_obs[key] = obs[key]
            else:
                combined_obs[key] = np.concatenate([obs[key], np.expand_dims(goal_obs[key], axis=0)],axis=0)
        return combined_obs
