'''VGG11/13/16/19 in Pytorch.'''
import torch.nn as nn

from noise_operator.config import NoNoiseConfig
from noise_operator.factory import NoiseOperatorFactory

cfg = {
    'VGG11': [64, 'BN2d', 'ReLU', 'M', 128, 'BN2d', 'ReLU', 'M', 256, 'BN2d', 'ReLU', 256, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 'M'],
    'VGG11-BN-noNoise': [64, 'BN2d-NN', 'ReLU', 'M', 128, 'BN2d-NN', 'ReLU', 'M', 256, 'BN2d-NN', 'ReLU', 256, 'BN2d-NN', 'ReLU', 'M', 512, 'BN2d-NN', 'ReLU', 512, 'BN2d-NN', 'ReLU', 'M', 512, 'BN2d-NN', 'ReLU', 512, 'BN2d-NN', 'ReLU', 'M'],
    'VGG11-noBN': [64, 'ReLU', 'M', 128, 'ReLU', 'M', 256, 'ReLU', 256, 'ReLU', 'M', 512, 'ReLU', 512, 'ReLU', 'M', 512, 'ReLU', 512, 'ReLU', 'M'],
    'VGG11_mnist': [64, 'BN2d', 'ReLU', 'M', 128, 'BN2d', 'ReLU', 'M', 256, 'BN2d', 'ReLU', 256, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512],
    'VGG11_mnist-BN-noNoise': [64, 'BN2d-NN', 'ReLU', 'M', 128, 'BN2d-NN', 'ReLU', 'M', 256, 'BN2d-NN', 'ReLU', 256, 'BN2d-NN', 'ReLU', 'M', 512, 'BN2d-NN', 'ReLU', 512, 'BN2d-NN', 'ReLU', 'M', 512, 'BN2d-NN', 'ReLU', 512],
    'VGG11_mnist-noBN': [64, 'ReLU', 'M', 128, 'ReLU', 'M', 256, 'ReLU', 256, 'ReLU', 'M', 512, 'ReLU', 512, 'ReLU', 'M', 512, 'ReLU', 512],
    'VGG13': [64, 'BN2d', 'ReLU', 64, 'BN2d', 'ReLU', 'M', 128, 'BN2d', 'ReLU', 128, 'BN2d', 'ReLU', 'M', 256, 'BN2d', 'ReLU', 256, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 'M'],
    'VGG16': [64, 'BN2d', 'ReLU', 64, 'BN2d', 'ReLU', 'M', 128, 'BN2d', 'ReLU', 128, 'BN2d', 'ReLU', 'M', 256, 'BN2d', 'ReLU', 256, 'BN2d', 'ReLU', 256, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 'M'],
    'VGG19': [64, 'BN2d', 'ReLU', 64, 'BN2d', 'ReLU', 'M', 128, 'BN2d', 'ReLU', 128, 'BN2d', 'ReLU', 'M', 256, 'BN2d', 'ReLU', 256, 'BN2d', 'ReLU', 256, 'BN2d', 'ReLU', 256, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 'M'],
}


class VGG(nn.Module):
    def __init__(self,
                 conf_name='VGG11',
                 input_shape=(1, 28, 28),
                 default_noise_config=NoNoiseConfig(),
                 layer_wise_noise_config=dict(),
                 num_classes=10,
                 ):
        super().__init__()
        # Makes some noise xD
        self._noise_factory = NoiseOperatorFactory(default_noise_config, layer_wise_config=layer_wise_noise_config)
        self.features, output_channels = self._make_layers(cfg[conf_name], input_shape)
        self.classifier = nn.Sequential(
            nn.Linear(output_channels, num_classes),
            self._noise_factory.get_noise_operator(),
        )

        print(f"Created the following number of noise layers / operators: {self._noise_factory._layer_counter}")
        # Check for out of bounds layer noise configurations
        if self._noise_factory.check_for_unused_configs():
            raise ValueError(f"A noise setting for a layer not contained in the network was requested. "
                             f"This is likely due to an incorrect configuration. "
                             f"The built network has {self._noise_factory._layer_counter} noise layers, "
                             f"but layer wise configurations were requested for the following layer indices: "
                             f"{layer_wise_noise_config.keys()}")

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, config, input_shape):
        layers = [
            self._noise_factory.get_noise_operator(),
        ]
        in_channels = input_shape[0]
        for x in config:
            if x == 'M':
                layers += [
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    self._noise_factory.get_noise_operator()
                ]
            elif type(x) == int:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    self._noise_factory.get_noise_operator(),
                ]
                in_channels = x
            elif x == 'BN2d':
                layers += [
                    nn.BatchNorm2d(in_channels),
                    self._noise_factory.get_noise_operator(),
                ]
            elif x == 'BN2d-NN':
                layers += [
                    nn.BatchNorm2d(in_channels),
                ]
            elif x == 'ReLU':
                layers += [
                    nn.ReLU(),
                    self._noise_factory.get_noise_operator(),
                ]
            else:
                raise NotImplementedError
        layers += [
            nn.AvgPool2d(kernel_size=1, stride=1),
            self._noise_factory.get_noise_operator(),
        ]
        return nn.Sequential(*layers), in_channels




























