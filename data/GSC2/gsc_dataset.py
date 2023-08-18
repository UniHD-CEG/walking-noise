import numpy as np
import torch
from torch.utils.data import Dataset
import functools


class KWSDatasetPy(Dataset):
    def __init__(self, audio_processor, mode, cache_data=False):
        self.audio_processor = audio_processor
        self.mode = mode
        self.cache_data = cache_data

    def __getitem__(self, index):
        if self.cache_data:
            return self._cached_call(index)
        else:
            return self._pure_call(index)

    @functools.cache
    def _cached_call(self, index):
        """
        Cached version of the pure call.
        This is technically not okay, since the get_data function is not pure.
        Because it calls np.random to vary the injection of background noise.
        """
        return self._pure_call(index)

    def _pure_call(self, index):
        data, label = self.audio_processor(index, self.mode)
        data = torch.tensor(data, dtype=torch.float).squeeze(0)
        label = torch.tensor(label[0], dtype=torch.long)
        return data, label

    def __len__(self):
        return self.audio_processor.len_dataset(self.mode)


class KWSDatasetTf(Dataset):
    def __init__(self, ds_numpy, data_config):
        self.data_config = data_config
        ds_list = []
        for elm in ds_numpy:
            ds_list.append(elm)
        self._data_array = np.asarray(np.asarray([d[0] for d in ds_list]))
        self._label_array = np.asarray(np.asarray([d[1] for d in ds_list]))

    def __getitem__(self, index):
        data, label = self._data_array[index], self._label_array[index]
        # For some reason the normal TF data is transposed compared to the one from torchaudio,
        # so we adjust for this here.
        if self.data_config["tf_feature_type"] == "td_samples":
            data = torch.tensor(data.flatten().reshape((1, -1)), dtype=torch.float)
        else:
            data = torch.tensor(data.T, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return data, label

    def __len__(self):
        return len(self._data_array)
