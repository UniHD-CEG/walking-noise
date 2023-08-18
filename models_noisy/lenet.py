import torch
from torch import nn

from sacred.config.custom_containers import ReadOnlyList

from noise_operator.config import NoNoiseConfig
from noise_operator.factory import NoiseOperatorFactory
from noise_operator.operators import NoiseOperator

cfg = {
    'LeNet5': ['C6@5', 'ReLU', 'M2', 'C16@5', 'ReLU', 'M2', 'Flatten', 'FC120', 'ReLU', 'FC84', 'ReLU', ],
    'LeNet5-BN': ['C6@5', 'BN2d', 'ReLU', 'M2', 'C16@5', 'BN2d', 'ReLU', 'M2', 'Flatten', 'FC120', 'BN1d',
                  'ReLU', 'FC84', 'BN1d', 'ReLU', ],
    'LeNet5-BN-noNoise': ['C6@5', 'BN2d-NN', 'ReLU', 'M2', 'C16@5', 'BN2d-NN', 'ReLU', 'M2', 'Flatten', 'FC120',
                          'BN1d-NN',
                          'ReLU', 'FC84', 'BN1d-NN', 'ReLU', ],
}


class LeNet(nn.Module):
    """
    LeNet network, implemented as described in the original paper on page 7: https://www.researchgate.net/profile/Yann-Lecun/publication/2985446_Gradient-Based_Learning_Applied_to_Document_Recognition/links/0deec519dfa1983fc2000000/Gradient-Based-Learning-Applied-to-Document-Recognition.pdf?origin=publication_detail
    Also inspired by: https://github.com/ChawDoe/LeNet5-MNIST-PyTorch/blob/master/model.py
    """
    def __init__(self,
                 conf_name='LeNet5',
                 input_shape=(1, 28, 28),
                 default_noise_config=NoNoiseConfig(),
                 layer_wise_noise_config=dict(),
                 repetition_config=dict(),
                 num_classes=10,
                 ):
        super().__init__()
        # Makes some noise xD
        self._noise_factory = NoiseOperatorFactory(default_noise_config, layer_wise_config=layer_wise_noise_config)
        self.features, output_channels = self._make_layers(cfg[conf_name], input_shape)
        self.features.extend([
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
            # If the layer is a noise op, we might need to execute it multiple times
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
    
    def _make_layers(self, config, input_shape):
        layers = [
            self._noise_factory.get_noise_operator(),
        ]
        in_channels = input_shape[0]
        for x in config:
            if x.startswith('M'):
                kernel_size = int(x.split('M')[-1])
                layers += [
                    nn.MaxPool2d(kernel_size),
                    self._noise_factory.get_noise_operator()
                ]
            elif x == 'ReLU':
                layers += [
                    nn.ReLU(),
                    self._noise_factory.get_noise_operator(),
                ]
            elif x.startswith('FC'):
                num_ch = int(x.split('FC')[-1])
                layers += [
                    nn.LazyLinear(num_ch),
                    self._noise_factory.get_noise_operator(),
                ]
                in_channels = num_ch
            elif x.startswith('C'):
                num_ch = int(x.split('C')[-1].split('@')[0])
                kernel_size = int(x.split('C')[-1].split('@')[-1])
                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=num_ch, kernel_size=kernel_size),
                    self._noise_factory.get_noise_operator(),
                ]
                in_channels = num_ch
            elif x == 'BN2d':
                layers += [
                    nn.BatchNorm2d(in_channels),
                    self._noise_factory.get_noise_operator(),
                ]
            elif x == 'BN1d':
                layers += [
                    nn.BatchNorm1d(in_channels),
                    self._noise_factory.get_noise_operator(),
                ]
            elif x == 'BN2d-NN':
                layers += [
                    nn.BatchNorm2d(in_channels),
                ]
            elif x == 'BN1d-NN':
                layers += [
                    nn.BatchNorm1d(in_channels),
                ]
            elif x == 'Flatten':
                layers += [
                    nn.Flatten(),
                ]
            else:
                raise NotImplementedError
        return nn.ModuleList(layers), in_channels
