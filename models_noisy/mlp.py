'''Abitrary linear MLP in Pytorch.'''
import numpy as np
import torch
import torch.nn as nn

from sacred.config.custom_containers import ReadOnlyList

from noise_operator.config import NoNoiseConfig
from noise_operator.factory import NoiseOperatorFactory
from noise_operator.operators import NoiseOperator

cfg = {
    'mlp1br': [2048,'BN','ReLU',512,'BN','ReLU'],
    'mlp1r': [2048,'ReLU',512,'ReLU'],
    'RL_TFC': [64, 'BN', 'ReLU', 64, 'BN', 'ReLU', 64, 'BN', 'ReLU'],
    'RL_TFC-BN-noNoise': [64, 'BN-NN', 'ReLU', 64, 'BN-NN', 'ReLU', 64, 'BN-NN', 'ReLU'],
    'RL_TFC-noBN': [64, 'ReLU', 64, 'ReLU', 64, 'ReLU'],
    'RL_TFC_32': [32, 'BN', 'ReLU', 32, 'BN', 'ReLU', 32, 'BN', 'ReLU'],
    'RL_TFC-noBN_32': [32, 'ReLU', 32, 'ReLU', 32, 'ReLU'],
    'RL_TFC_16': [16, 'BN', 'ReLU', 16, 'BN', 'ReLU', 16, 'BN', 'ReLU'],
    'RL_TFC-noBN_16': [16, 'ReLU', 16, 'ReLU', 16, 'ReLU'],
    'RL_TFC_8': [8, 'BN', 'ReLU', 8, 'BN', 'ReLU', 8, 'BN', 'ReLU'],
    'RL_TFC-noBN_8': [8, 'ReLU', 8, 'ReLU', 8, 'ReLU'],
    'RL_SFC': [256, 'BN', 'ReLU', 256, 'BN', 'ReLU', 256, 'BN', 'ReLU'],
    'RL_SFC-BN-noNoise': [256, 'BN-NN', 'ReLU', 256, 'BN-NN', 'ReLU', 256, 'BN-NN', 'ReLU'],
    'RL_SFC-noBN': [256, 'ReLU', 256, 'ReLU', 256, 'ReLU'],
    'RL_LFC': [1024, 'BN', 'ReLU', 1024, 'BN', 'ReLU', 1024, 'BN', 'ReLU'],
    'RL_LFC-BN-noNoise': [1024, 'BN-NN', 'ReLU', 1024, 'BN-NN', 'ReLU', 1024, 'BN-NN', 'ReLU'],
    'RL_LFC-noBN': [1024, 'ReLU', 1024, 'ReLU', 1024, 'ReLU'],
}


class MLP(nn.Module):

    def __init__(self,
                 conf_name='RL_TFC',
                 input_shape=(1, 28, 28),
                 default_noise_config=NoNoiseConfig(),
                 layer_wise_noise_config=dict(),
                 repetition_config=dict(),
                 num_classes=10,
                 ):
        # :)
        super().__init__()
        
        # Makes some noise xD
        self._noise_factory = NoiseOperatorFactory(default_noise_config, layer_wise_config=layer_wise_noise_config)
        num_input_features = np.prod(input_shape)
        self.features, output_channels = self._make_layers(cfg[conf_name], num_input_features)
        self.features.extend([
            nn.Flatten(),
            nn.Linear(output_channels, num_classes),
            self._noise_factory.get_noise_operator(),
            ])

        num_noise_layers = self._noise_factory._layer_counter
        print(f"Created the following number of noise layers / operators: {num_noise_layers}")
        # Check for out of bounds layer noise configurations
        if self._noise_factory.check_for_unused_configs():
            raise ValueError(f"A noise setting for a layer not contained in the network was requested. "
                             f"This is likely due to an incorrect configuration. "
                             f"The built network has {num_noise_layers} noise layers, "
                             f"but layer wise configurations were requested for the following layer indices: "
                             f"{layer_wise_noise_config.keys()}")

        self.repetition_map = self._build_repetition_map(repetition_config, num_noise_layers)
        print(f'Repetition map set as: {self.repetition_map}')


    def forward(self, x):
        num_noise_op = 0
        for mod in self.features:
            # If the layer is a noise op, we potentially execute it multiple times
            if isinstance(mod, NoiseOperator):
                summed_x_out = torch.zeros_like(x)
                num_rep = self.repetition_map[num_noise_op]
                for _ in range(num_rep):
                    internal_const_x = x.clone()
                    summed_x_out += mod(internal_const_x)
                x = summed_x_out/num_rep
                num_noise_op += 1
            else:
                x = mod(x)
        return x

    def _build_repetition_map(self, repetition_config, num_noise_layers):
        '''
        Builds a map of how many times a layer + noise should be executed. Per default this is 1.
        But can be set on a global and per-layer level.
        ToDo: Classes should just sub-class from a parent, which contains this function and the forward path.
        '''
        # Set globally first
        try:
            global_rep = repetition_config['global']
        except KeyError:
            global_rep = 1
        rep_list = [global_rep] * num_noise_layers
        # Then set locally
        try:
            layer_wise_mapped = repetition_config['layer_wise_mapped']
        except KeyError:
            # No layer wise config found
            return rep_list
        if isinstance(layer_wise_mapped, ReadOnlyList):
            # Convert to dict
            assert len(layer_wise_mapped) == num_noise_layers, 'When layer_wise_mapped is represented as a list, then the list must be as long as there are noise layers in the model.'
            inter_dict = {}
            for i in range(len(layer_wise_mapped)):
                inter_dict[i] = layer_wise_mapped[i]
            layer_wise_mapped = inter_dict
        for key in layer_wise_mapped.keys():
            int_key = int(key)
            assert type(layer_wise_mapped[key]) == type(int()), 'layer_wise_mapped elements must be integer.'
            assert layer_wise_mapped[key] > 0, 'Noise layer must be executed at least once.'
            rep_list[int_key] = layer_wise_mapped[key]
        return rep_list

    def _make_layers(self, config, num_input_features):
        layers = []
        layers += [
            nn.Flatten(),
            self._noise_factory.get_noise_operator(),
        ]
        in_channels = num_input_features
        for x in config:
            if x == 'BN':
                layers += [
                    nn.BatchNorm1d(in_channels),
                    self._noise_factory.get_noise_operator(),
                ]
            elif x == 'BN-NN':
                layers += [
                    nn.BatchNorm1d(in_channels),
                ]
            elif x == 'ReLU':
                layers += [
                    nn.ReLU(inplace=True),
                    self._noise_factory.get_noise_operator(),
                ]
            elif type(x) == int:
                layers += [
                    nn.Linear(in_channels, x),
                    self._noise_factory.get_noise_operator(),
                ]
                in_channels = x
            else:
                raise NotImplementedError
        return nn.ModuleList(layers), in_channels

