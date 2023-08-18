# Noise operator factory to insert different configs into networks

from .operators import NoiseOperator
from copy import deepcopy


class NoiseOperatorFactory():
    """Factory class for injecting noise into a network in a programmatic way.
    Noise can be injected on a global scale with the default configuration and on a per-layer basis with the
    layer wise configuration."""
    def __init__(self, default_config, layer_wise_config=dict()):
        self._default_config = default_config
        self._layer_wise_config = layer_wise_config
        self._layer_counter = 0

    def _generate_op_from_index(self, index):
        if index in self._layer_wise_config.keys():
            config = self._layer_wise_config[index]
        else:
            config = self._default_config

        # Pass along the op/layer-index
        local_conf = deepcopy(config)
        local_conf.index_of_operator = index
        return NoiseOperator(local_conf)

    def get_noise_operator(self):
        """Generate a noise operator according to the configuration defined before"""
        noise_op = self._generate_op_from_index(self._layer_counter)
        self._layer_counter += 1
        return noise_op

    def check_for_unused_configs(self):
        """Checks if there are layer wise configurations, which have not yet been inserted.
        Left over layer wise configurations are an indicator for a configuration issues,
        where noise would be injected into a non-existent layer."""
        configs_left = False
        for index in self._layer_wise_config.keys():
            if index >= self._layer_counter:
                configs_left = True
        return configs_left
