from torch.nn.modules.batchnorm import _NormBase


class WeightClamper:
    """
    Class for clamping the weights of a given model.
    Inspired by: https://stackoverflow.com/a/70330290
    """
    def __init__(self, min_clamp=None, max_clamp=None):
        self._min = min_clamp
        self._max = max_clamp

    def __call__(self, module):
        # Only continue if something is to be done
        if (self._max is None) and (self._min is None):
            return
        # Only consider layer, which have weights
        if hasattr(module, 'weight'):
            # Skip Batchnorm layers
            if not issubclass(type(module), _NormBase):
                # Clamp weights
                w = module.weight.data
                w = w.clamp(self._min, self._max)
                module.weight.data = w
