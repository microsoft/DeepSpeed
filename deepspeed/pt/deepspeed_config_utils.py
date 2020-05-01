"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""
"""
Collection of DeepSpeed configuration utilities
"""


def get_scalar_param(param_dict, param_name, param_default_value):
    if param_name in param_dict.keys():
        return param_dict[param_name]
    else:
        return param_default_value
