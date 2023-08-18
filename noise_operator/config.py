import warnings
from dataclasses import dataclass
from inspect import signature

# General noise settings for all implementations


@dataclass
class BaseNoiseConfig:
    #: Enables noise in training
    enable_in_training: bool = True
    #: Enables noise in evaluation
    enable_in_eval: bool = True
    #: Enables debug output for this particular noise operator
    print_debug_output: bool = False
    #: Index of the operator, when created by the NoiseOperatorFactory
    index_of_operator: int = -1
    #: Weather the noise should be normalized to the number of parameters at the input tensor
    normalize_to_input_size: bool = False


# Specific settings for each type of noise
@dataclass
class DropoutConfig(BaseNoiseConfig):
    _name = "Dropout"
    #: Dropout probability
    p: float = 0.1
    #: Compute dropout inplace
    inplace: bool = False


@dataclass
class GaussAddConfig(BaseNoiseConfig):
    _name = "GaussAdd"
    #: Gauss mean value
    GaussMean: float = 0.0
    #: Gauss standard-deviation
    GaussStd: float = 1.0


@dataclass
class GaussMulConfig(BaseNoiseConfig):
    _name = "GaussMul"
    #: Gauss mean value
    GaussMean: float = 1.0
    #: Gauss standard-deviation
    GaussStd: float = 1.0


@dataclass
class GaussCombinedConfig(BaseNoiseConfig):
    _name = "GaussCombined"
    #: Order of operations
    FirstMulThenAdd: bool = True
    #: Gauss mean value for the multiplier part
    GaussMeanMul: float = 1.0
    #: Gauss mean value for the adder part
    GaussMeanAdd: float = 0.0
    #: Amplitude for the standard deviations of both distributions
    StdAmplitude: float = 1.0
    #: Ratio between the two amplitudes of the standard deviations, must be between 0 and 1.
    StdRatio: float = 0.5


@dataclass
class GaussCombinedDirectConfig(BaseNoiseConfig):
    _name = "GaussCombined_direct"
    #: Order of operations
    FirstMulThenAdd: bool = True
    #: Gauss mean value for the multiplier part
    GaussMeanMul: float = 1.0
    #: Gauss mean value for the adder part
    GaussMeanAdd: float = 0.0
    #: Gauss std value for the multiplier part
    GaussStdMul: float = 1.0
    #: Gauss std value for the adder part
    GaussStdAdd: float = 1.0


@dataclass
class NoNoiseConfig(BaseNoiseConfig):
    _name = "NoNoise"


def resolve_config_from_name(name: str, **kwarg_settings):
    # Find out which configs are available
    available_cfgs = {cls._name: cls for cls in BaseNoiseConfig.__subclasses__()}
    # Find out which type of noise was selected
    for noise_key in available_cfgs.keys():
        if name == noise_key:
            # Find out which parameters we can transfer from the kwargs to the noise configuration
            selected_cfg = available_cfgs[noise_key]
            available_parameters = list(signature(available_cfgs[noise_key]).parameters.keys())
            matching_parameters = {}
            for p_name in available_parameters:
                if p_name in kwarg_settings.keys():
                    matching_parameters[p_name] = kwarg_settings[p_name]
                else:
                    warnings.warn(f"The keyword argument '{p_name}' is defined by the noise configuration, "
                                  f"but wasn't passed to this function as a keyword argument. "
                                  f"This means that the noise operator "
                                  f"will be created with the default value of the keyword argument, "
                                  f"as defined by the noise configuraiton. "
                                  f"This could lead to incorrect noise configurations.")
            # Do reverse matching and check if any parameters were left over
            for p_name in kwarg_settings.keys():
                if p_name not in matching_parameters.keys():
                    warnings.warn(f"The keyword argument '{p_name}' passed to this function was not matched to "
                                  f"a keyword argument, accepatble by the noise operator configuration. "
                                  f"This can likely be a typo and could lead to incorrect noise configurations.")
            # Instantiate the noise config
            noise_params = selected_cfg(**matching_parameters)
            break
    else:  # No break executed
        raise ValueError(f"{name} is not a supported noise configuration. "
                         f"Supported configurations are: {available_cfgs.keys()}")

    return noise_params
