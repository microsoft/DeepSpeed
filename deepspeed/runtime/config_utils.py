"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""
"""
Collection of DeepSpeed configuration utilities
"""
import json
import collections
import collections.abc
from pydantic import BaseModel


class DeepSpeedConfigModel(BaseModel):
    def __init__(self, **data):
        super().__init__(**data)
        self.deprecated_fields_check(self)

    @classmethod
    def deprecated_fields_check(self, pydantic_config):
        fields = pydantic_config.__fields__
        fields_set = pydantic_config.__fields_set__
        for field in fields.values():
            kwargs = field.field_info.extra
            if kwargs.get("deprecated", False):
                dep_param = field.name
                new_param = kwargs.get("new_param", "")
                if new_param:
                    assert (
                        new_param not in fields_set
                    ), f"Cannot provide deprecated parameter '{dep_param}' and replacing parameter '{new_param}' together"
                    try:
                        setattr(
                            pydantic_config,
                            new_param,
                            getattr(pydantic_config,
                                    dep_param),
                        )
                    except Exception as e:
                        logger.Error(
                            f"Error: tried setting value for '{new_param}' with value from deprecated '{dep_param}'"
                        )
                        raise e

    class Config:
        validate_all = True
        validate_assignment = True
        use_enum_values = True
        extra = "forbid"


# adapted from https://stackoverflow.com/a/50701137/9201239
class ScientificNotationEncoder(json.JSONEncoder):
    """
    This class overrides ``json.dumps`` default formatter.

    This version keeps everything as normal except formats numbers bigger than 1e3 using scientific notation.

    Just pass ``cls=ScientificNotationEncoder`` to ``json.dumps`` to activate it

    """
    def iterencode(self, o, _one_shot=False, level=0):
        indent = self.indent if self.indent is not None else 4
        prefix_close = " " * level * indent
        level += 1
        prefix = " " * level * indent
        if isinstance(o, bool):
            return "true" if o else "false"
        elif isinstance(o, float) or isinstance(o, int):
            if o > 1e3:
                return f"{o:e}"
            else:
                return f"{o}"
        elif isinstance(o, collections.abc.Mapping):
            x = [
                f'\n{prefix}"{k}": {self.iterencode(v, level=level)}' for k,
                v in o.items()
            ]
            return "{" + ', '.join(x) + f"\n{prefix_close}" + "}"
        elif isinstance(o, collections.abc.Sequence) and not isinstance(o, str):
            return f"[{ f', '.join(map(self.iterencode, o)) }]"
        return "\n, ".join(super().iterencode(o, _one_shot))


class DeepSpeedConfigObject(object):
    """
    For json serialization
    """
    def repr(self):
        return self.__dict__

    def __repr__(self):
        return json.dumps(
            self.__dict__,
            sort_keys=True,
            indent=4,
            cls=ScientificNotationEncoder,
        )


def get_scalar_param(param_dict, param_name, param_default_value):
    return param_dict.get(param_name, param_default_value)


def get_list_param(param_dict, param_name, param_default_value):
    return param_dict.get(param_name, param_default_value)


def get_dict_param(param_dict, param_name, param_default_value):
    return param_dict.get(param_name, param_default_value)


def dict_raise_error_on_duplicate_keys(ordered_pairs):
    """Reject duplicate keys."""
    d = dict((k, v) for k, v in ordered_pairs)
    if len(d) != len(ordered_pairs):
        counter = collections.Counter([pair[0] for pair in ordered_pairs])
        keys = [key for key, value in counter.items() if value > 1]
        raise ValueError("Duplicate keys in DeepSpeed config: {}".format(keys))
    return d
