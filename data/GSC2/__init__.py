"""
All files in this folder are a modified copy of previous work done for a MLPerf Tiny submission
for the FINN project.
Exact link is here: https://github.com/mlcommons/tiny_results_v0.7/tree/main/open/hls4ml-finn/code/kws/KWS-W3A3/training/data
At commit: 8813eded7403f4167867d305f8d997597f06181a
"""

from pathlib import Path

import tensorflow_datasets as tfds
import yaml
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

import data.GSC2.input_data_tf as input_data_tf
from data.GSC2.gsc_dataset import KWSDatasetPy, KWSDatasetTf
from data.GSC2.preprocessing_module import PYPreprocessTransform


class GSC2_TF():
    name = "GSC2_TF"
    dims = (1, 10, 49)
    has_test_dataset = True

    def __init__(self, batch_size=128, num_workers=2, data_root="/tmp/datasets/", normalize=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = str(Path(data_root).expanduser())
        self.normalize = normalize

        # Load default settings
        with open('data/GSC2/config.yaml') as f:
            self.default_data_config = yaml.safe_load(f)

        # Set paths externally
        # For the TF dataset
        self.default_data_config['tf_data_dir'] = str(Path(self.data_root) / Path('TF/'))

        # For the PyTroch dataset and TF preprocessing
        self.default_data_config['docker_dataset_dir'] = self.data_root
        self.default_data_config['tf_bg_path'] = str(Path(self.data_root) / Path('SpeechCommands/speech_commands/0.0.2/_background_noise_'))


    @property
    def num_classes(self) -> int:
        """
        Return:
            12
        """
        return 12

    def prepare_data(self) -> None:
        # Download the PyTorch version to make use the explicitly extracted _background_noise_ folder
        SPEECHCOMMANDS(root=self.data_root, url=self.default_data_config['speech_commands_version'], download=True)
        # Download the TF version
        splits = ['train', 'test', 'validation']
        _ = tfds.load('speech_commands', split=splits, data_dir=self.default_data_config['tf_data_dir'],
                      with_info=True, download=True)


    def setup(self) -> None:
        # Load TF data
        self._ds_train, self._ds_test, self._ds_val = input_data_tf.get_training_data(self.default_data_config)
        # Immediately convert to datasets
        self._ds_train = KWSDatasetTf(self._ds_train, self.default_data_config)
        self._ds_val = KWSDatasetTf(self._ds_val, self.default_data_config)
        self._ds_test = KWSDatasetTf(self._ds_test, self.default_data_config)

    def train_dataloader(self):
        train_loader = DataLoader(
            self._ds_train, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self._ds_val, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self._ds_test, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers)
        return test_loader


class GSC2_PyTorch():
    name = "GSC2_PyTorch"
    dims = (1, 10, 49)
    has_test_dataset = True

    def __init__(self, batch_size=128, num_workers=2, data_root="/tmp/datasets/", normalize=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = str(Path(data_root).expanduser())
        self.normalize = normalize

        # Load default settings
        with open('data/GSC2/config.yaml') as f:
            self.default_data_config = yaml.safe_load(f)
        # Set the dataset root externally
        self.default_data_config['docker_dataset_dir'] = self.data_root


    @property
    def num_classes(self) -> int:
        """
        Return:
            12
        """
        return 12

    def prepare_data(self) -> None:
        SPEECHCOMMANDS(root=self.data_root, url=self.default_data_config['speech_commands_version'], download=True)

    def setup(self) -> None:
        # Set the fitting audio pre-processor
        self.audio_processor = PYPreprocessTransform(self.default_data_config)
        self.train_set = KWSDatasetPy(self.audio_processor, 'training')
        self.val_set = KWSDatasetPy(self.audio_processor, 'validation')
        self.test_set = KWSDatasetPy(self.audio_processor, 'testing')

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_set, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_set, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers)
        return test_loader
