'''Copyright The Microsoft DeepSpeed Team'''
"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from pydantic import Field
from deepspeed.runtime.config_utils import DeepSpeedConfigModel


def get_flops_profiler_config(param_dict):
    flops_profiler_config_dict = param_dict.get("flops_profiler", {})
    return DeepSpeedFlopsProfilerConfig(**flops_profiler_config_dict)


class DeepSpeedFlopsProfilerConfig(DeepSpeedConfigModel):
    """ Sets parameters for the flops profiler. """

    enabled: bool = False
    """ Enables the flops profiler. This also enables wall_clock_breakdown. """

    profile_step: int = Field(1, ge=1)
    """
    The global training step at which to profile. Note that warm up steps are
    needed for accurate time measurement.
    """

    module_depth: int = -1
    """
    The depth of the model at which to print the aggregated module information.
    When set to `-1`, it prints information from the top module to the
    innermost modules (the maximum depth).
    """

    top_modules: int = 1
    """ Limits the aggregated profile output to the number of top modules specified. """

    detailed: bool = True
    """ Whether to print the detailed model profile. """

    output_file: str = None
    """ Path to the output file. If None, the profiler prints to stdout. """
