import os

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader

from flower.datasets import mimicgen_tasks
from flower.datasets.translated_dataset import TranslatedSequenceVLDataset
from flower.datasets.utils.libero_utils import get_dataset

# Robosuite-native observation keys a MimicGen hdf5 (rendered via robomimic's
# dataset_states_to_obs.py) uses, as opposed to LIBERO's agentview_rgb/eye_in_hand_rgb/
# joint_states/gripper_states. See TranslatedSequenceVLDataset.DEFAULT_KEY_MAP.
KEY_MAP = {
    "static": "agentview_image",
    "wrist": "robot0_eye_in_hand_image",
    "joints": "robot0_joint_pos",
    "gripper": "robot0_gripper_qpos",
}


class MimicgenDataModule(pl.LightningDataModule):
    """Multi-task MimicGen datamodule: one TranslatedSequenceVLDataset per dataset in
    mimicgen_tasks.CORE_DATASETS, concatenated into a single training set -- structurally
    the same shape LiberoDataModule builds (flower/datasets/libero_data_module.py), minus
    the LIBERO benchmark registry and the (discarded) CLIP task-embedding pass.
    """

    def __init__(
        self,
        datasets: DictConfig,
        observation_space: DictConfig,
        num_workers: int = 8,
        transforms: DictConfig = None,
        shuffle_val: bool = False,
        data_dir: str = None,
        split_ratio: float = 0.0,
        dataset_names=None,
        pad_seq_length: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.datasets_cfg = datasets
        self.num_workers = num_workers
        self.transforms = transforms
        self.shuffle_val = shuffle_val
        self.data_dir = data_dir
        self.split_ratio = split_ratio
        # Restricts which datasets this datamodule loads -- e.g. for a one-dataset smoke
        # test before the full 26-dataset download finishes. Defaults to every dataset in
        # the registry, matching the eval side's identical `datasets` override
        # (conf/eval_mimicgen.yaml).
        self.dataset_names = list(dataset_names) if dataset_names else list(mimicgen_tasks.CORE_DATASETS)
        self.pad_seq_length = pad_seq_length
        self.train_datasets = []
        self.val_datasets = []
        self.modalities = []

    def translate_obs_space(self, obs_space):
        self.mimicgen_observation_space = {
            "rgb": obs_space["rgb_obs"],
            "low_dim": obs_space["state_obs"],
        }

    def _initialize_datasets(self, datasets_cfg):
        self.translate_obs_space(datasets_cfg.lang_dataset.obs_space)

        data_dir = self.data_dir or datasets_cfg.get("custom_data_path")
        train_datasets = []

        for i, dataset_name in enumerate(self.dataset_names):
            dataset_path = os.path.join(data_dir, f"{dataset_name}.hdf5")
            task_dataset, _shape_meta = get_dataset(
                dataset_path=dataset_path,
                obs_modality=self.mimicgen_observation_space,
                initialize_obs_utils=(i == 0),
                seq_len=datasets_cfg.lang_dataset.action_seq_len,
                pad_seq_length=self.pad_seq_length,
            )
            vl_dataset = TranslatedSequenceVLDataset(
                task_dataset,
                task_emb=None,
                task_description=mimicgen_tasks.language(dataset_name),
                act_seq_len=datasets_cfg.lang_dataset.action_seq_len,
                obs_seq_len=datasets_cfg.lang_dataset.obs_seq_len,
                transforms=self.transforms,
                key_map=KEY_MAP,
            )
            train_datasets.append(vl_dataset)

        concat_train_datasets = ConcatDataset(train_datasets)
        datasets = {"lang": concat_train_datasets}
        # Same as LiberoDataModule: no held-out split, validation runs on training data
        # (see conf/config_libero.yaml's limit_val_batches, which bounds how much of it is used).
        val_datasets = {"lang": concat_train_datasets}
        return datasets, val_datasets

    def setup(self, stage=None):
        self.train_datasets, self.val_datasets = self._initialize_datasets(self.datasets_cfg)
        self.modalities.append("lang")

    def train_dataloader(self):
        return {
            key: DataLoader(
                dataset,
                batch_size=self.datasets_cfg.lang_dataset.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=True,
            )
            for key, dataset in self.train_datasets.items()
        }

    def val_dataloader(self):
        val_dataloaders = {
            key: DataLoader(
                dataset,
                batch_size=self.datasets_cfg.lang_dataset.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
            )
            for key, dataset in self.val_datasets.items()
        }
        return list(val_dataloaders.values())
