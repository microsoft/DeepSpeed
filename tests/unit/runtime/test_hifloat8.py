# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from deepspeed.runtime.config import DeepSpeedConfigError, get_hifloat8_config, get_low_precision_config
from deepspeed.runtime.engine import DeepSpeedEngine

# Native tests require the public API; a missing extension must fail there.
# Gate only device availability, without importing TorchAO during collection.
requires_npu = pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="NPU unavailable")


def test_hifloat8_config_defaults_disabled():
    assert get_hifloat8_config({}) == {
        "enabled": False,
        "backend": "torchao",
        "module_name_patterns": (),
        "min_numel": 0,
        "expected_module_count": None,
        "config": None,
    }


@pytest.mark.parametrize(
    "config",
    [
        {
            "hifloat8": {
                "enabled": True
            }
        },
        {
            "hifloat8": {
                "enabled": "yes",
                "module_name_patterns": ["*"]
            }
        },
        {
            "hifloat8": {
                "enabled": True,
                "module_name_patterns": [""],
                "min_numel": 0
            }
        },
        {
            "hifloat8": {
                "enabled": True,
                "module_name_patterns": ["*"],
                "min_numel": -1
            }
        },
        {
            "hifloat8": {
                "enabled": True,
                "module_name_patterns": ["*"],
                "probe_kernel": False
            }
        },
        {
            "hifloat8": {
                "enabled": True,
                "module_name_patterns": ["*"],
                "unknown": 1
            }
        },
    ],
)
def test_hifloat8_config_rejects_invalid_values(config):
    with pytest.raises(DeepSpeedConfigError):
        get_hifloat8_config(config)


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "84"])
def test_hifloat8_config_rejects_invalid_expected_count(count):
    with pytest.raises(DeepSpeedConfigError):
        get_hifloat8_config({"hifloat8": {"expected_module_count": count}})


def test_hifloat8_config_accepts_dense_selection_contract():
    # Existing Swift Dense configurations require exactly 28 * 3 projections.
    config = get_hifloat8_config({"hifloat8": {"expected_module_count": 84}})
    assert config["expected_module_count"] == 84


@pytest.mark.parametrize("backend", ["", "unknown", None, 1])
def test_hifloat8_config_rejects_unknown_backend(backend):
    with pytest.raises(DeepSpeedConfigError, match="hifloat8.backend"):
        get_hifloat8_config({"hifloat8": {"backend": backend}})


def test_hifloat8_config_selects_torchao_npu():
    assert get_hifloat8_config({"hifloat8": {"backend": "torchao"}})["backend"] == "torchao"


@requires_npu
@pytest.mark.parametrize("policy", [
    None, {}, {
        "input_dst_type_max": 15,
        "weight_dst_type_max": 15,
        "grad_dst_type_max": 224,
        "scale_policy": "pertensor",
        "compute_dtype": "bfloat16"
    }, {
        "input_dst_type_max": 31,
        "weight_dst_type_max": 31,
        "grad_dst_type_max": 127,
        "compute_dtype": "bfloat16"
    }
])
def test_torchao_backend_preserves_dense_and_grouped_parameters(policy):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    from deepspeed.moe.ep_experts import GroupedExperts
    from torchao.hifloat8 import HiFloat8Linear

    model = nn.ModuleDict({
        "dense": nn.Linear(16, 16, bias=False),
        "experts": GroupedExperts(16, 32, 2, use_grouped_mm=True),
    }).to("npu").bfloat16()
    parameters_before = dict(model.named_parameters())
    keys_before = tuple(model.state_dict())
    engine = _make_engine(model, ["dense", "experts"])
    engine.device = torch.device("npu:0")
    engine._config.low_precision_config["backend"] = "torchao"
    engine._config.hifloat8_config["config"] = policy
    engine._configure_low_precision()

    assert isinstance(model["dense"], HiFloat8Linear)
    assert model["experts"].low_precision_provider is not None
    assert model["dense"].config is model["experts"].low_precision_config
    assert model["dense"].config.grad_dst_type_max == (policy.get("grad_dst_type_max", 224) if policy else 224)
    assert tuple(model.state_dict()) == keys_before
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in parameters_before.items())


class _ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.mlp = nn.ModuleDict({
            "gate_proj": nn.Linear(4, 8, bias=False),
            "up_proj": nn.Linear(4, 8, bias=False),
            "down_proj": nn.Linear(8, 4, bias=False),
        })
        self.model.self_attn = nn.ModuleDict({"q_proj": nn.Linear(4, 4, bias=False)})


def _make_engine(model, patterns, min_numel=0):
    engine = object.__new__(DeepSpeedEngine)
    nn.Module.__init__(engine)
    engine._config = SimpleNamespace(
        hifloat8_config={
            "enabled": True,
            "module_name_patterns": tuple(patterns),
            "min_numel": min_numel,
        },
        bfloat16_config=SimpleNamespace(enabled=True),
        tensor_parallel_config=SimpleNamespace(autotp_size=1),
        zero_optimization_stage=2,
    )
    engine._config.low_precision_config = get_low_precision_config({"hifloat8": engine._config.hifloat8_config})
    engine._config.hifloat8_config = engine._config.low_precision_config
    engine._config.float16_config = SimpleNamespace(enabled=False)
    engine.pipeline_parallelism = False
    engine.has_moe_layers = False
    engine.mpu = None
    engine.device = torch.device("cpu")
    engine._set_client_model(model)
    return engine


@requires_npu
def test_engine_selects_routed_experts_without_replacing_parameters(monkeypatch):
    if not hasattr(torch, "npu"):
        pytest.skip("NPU accelerator required")
    # Catches silently selecting the router or orphaning packed expert weights.
    from deepspeed.moe.ep_experts import GroupedExperts
    import torchao.hifloat8 as implementation

    model = nn.ModuleDict({
        "experts": GroupedExperts(8, 16, 2, use_grouped_mm=False),
        "router": nn.Linear(8, 2, bias=False),
    })
    model.experts.w1.requires_grad_(False)
    parameters = dict(model.named_parameters())
    keys = tuple(model.state_dict())
    monkeypatch.setattr(implementation, "assert_hifloat8_training_available", lambda **kwargs: None)
    engine = _make_engine(model, ["experts"])
    engine._configure_low_precision()
    assert engine.hifloat8_grouped_module_names == ("experts", )
    assert engine.hifloat8_converted_module_names == ()
    assert model.experts.low_precision_provider is not None
    assert type(model.router) is nn.Linear
    assert not model.experts.w1.requires_grad
    assert tuple(model.state_dict()) == keys
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in parameters.items())


@requires_npu
def test_engine_converts_only_selected_modules_and_preserves_parameters(monkeypatch):
    if not hasattr(torch, "npu"):
        pytest.skip("NPU accelerator required")
    model = _ToyModel()
    before = dict(model.named_parameters())
    probe_calls = []
    import torchao.hifloat8 as implementation

    HiFloat8Linear = implementation.HiFloat8Linear

    monkeypatch.setattr(
        implementation,
        "assert_hifloat8_training_available",
        lambda **kwargs: probe_calls.append(kwargs),
    )
    engine = _make_engine(model, ["*.mlp.gate_proj", "*.mlp.up_proj", "*.mlp.down_proj"])
    engine._config.hifloat8_config["expected_module_count"] = 3
    engine._configure_low_precision()

    assert len(probe_calls) == 1
    assert probe_calls[0]["device"] == torch.device("cpu")
    assert probe_calls[0]["linear"] and not probe_calls[0]["grouped"]
    assert probe_calls[0]["config"] is model.model.mlp["gate_proj"].config
    assert engine.hifloat8_converted_module_names == (
        "model.mlp.gate_proj",
        "model.mlp.up_proj",
        "model.mlp.down_proj",
    )
    assert all(isinstance(model.model.mlp[name], HiFloat8Linear) for name in model.model.mlp)
    assert type(model.model.self_attn["q_proj"]) is nn.Linear
    after = dict(model.named_parameters())
    assert before.keys() == after.keys()
    assert all(after[name] is parameter for name, parameter in before.items())


def test_engine_fails_when_module_selection_is_empty():
    engine = _make_engine(_ToyModel(), ["*.does_not_exist"])
    with pytest.raises(RuntimeError, match="matched no nn.Linear"):
        engine._configure_low_precision()


def test_engine_rejects_partially_unmatched_patterns_before_conversion():
    # A typo must not silently leave part of the requested model in BF16.
    model = _ToyModel()
    engine = _make_engine(model, ["*.mlp.gate_proj", "*.missing_projection"])
    with pytest.raises(RuntimeError, match="patterns matched no eligible modules"):
        engine._configure_low_precision()
    assert type(model.model.mlp["gate_proj"]) is nn.Linear


def test_engine_rejects_wrong_module_count_before_kernel_probe(monkeypatch):
    # A partially selected model must fail before native initialization or mutation.
    from deepspeed.accelerator import get_accelerator

    def unexpected_provider(backend, format):
        pytest.fail("module-count mismatch must fail before resolving the optional backend")

    monkeypatch.setattr(get_accelerator(), "get_low_precision_provider", unexpected_provider)
    model = _ToyModel()
    engine = _make_engine(model, ["*.mlp.gate_proj"])
    engine._config.hifloat8_config["expected_module_count"] = 3
    with pytest.raises(RuntimeError, match="expected_module_count"):
        engine._configure_low_precision()
    assert type(model.model.mlp["gate_proj"]) is nn.Linear


def test_hifloat8_validation_rejects_autotp():
    engine = _make_engine(_ToyModel(), ["*"])
    engine._config.tensor_parallel_config.autotp_size = 2

    with pytest.raises(RuntimeError, match="AutoTP"):
        engine._validate_low_precision_configuration()


def test_missing_optional_backend_reports_selection_before_model_mutation(monkeypatch):
    # Provider policy parsing may report a missing optional dependency on any
    # accelerator; the engine must retain its diagnostic and parameter contract.
    from deepspeed.accelerator import get_accelerator

    error = ImportError("optional low-precision backend is not installed")

    def missing_policy(values):
        raise error

    provider = SimpleNamespace(config_from_dict=missing_policy)
    monkeypatch.setattr(get_accelerator(), "get_low_precision_provider", lambda backend, format: provider)
    model = _ToyModel()
    parameters = dict(model.named_parameters())
    engine = _make_engine(model, ["*.mlp.gate_proj"])
    with pytest.raises(RuntimeError,
                       match=r"selected 1 Linear modules \(32 matrix elements\).+before conversion") as caught:
        engine._configure_low_precision()
    assert caught.value.__cause__ is error
    assert type(model.model.mlp["gate_proj"]) is nn.Linear
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in parameters.items())


def test_hifloat8_validation_rejects_custom_model_parallel_unit():
    engine = _make_engine(_ToyModel(), ["*"])
    engine.mpu = object()

    with pytest.raises(RuntimeError, match="model-parallel unit"):
        engine._validate_low_precision_configuration()


@requires_npu
@pytest.mark.parametrize("error", [RuntimeError("unsupported device"), ImportError("missing torchao_npu backend")])
def test_engine_kernel_failure_reports_selection_and_does_not_mutate(monkeypatch, error):
    if not hasattr(torch, "npu"):
        pytest.skip("NPU accelerator required")
    import torchao.hifloat8 as implementation

    model = _ToyModel()
    before = dict(model.named_parameters())

    def fail_probe(**_kwargs):
        raise error

    monkeypatch.setattr(implementation, "assert_hifloat8_training_available", fail_probe)
    engine = _make_engine(model, ["*.mlp.gate_proj"])

    with pytest.raises(RuntimeError, match=r"selected 1 Linear modules \(32 matrix elements\).+before conversion"):
        engine._configure_low_precision()

    assert type(model.model.mlp["gate_proj"]) is nn.Linear
    after = dict(model.named_parameters())
    assert before.keys() == after.keys()
    assert all(after[name] is parameter for name, parameter in before.items())


@pytest.mark.parametrize("policy", [True, 1, "bf16", []])
def test_hifloat8_config_rejects_non_object_policy(policy):
    with pytest.raises(DeepSpeedConfigError, match="hifloat8.config"):
        get_hifloat8_config({"hifloat8": {"backend": "torchao_npu", "config": policy}})


def test_hifloat8_legacy_backend_is_rejected_with_migration_instruction():
    with pytest.raises(DeepSpeedConfigError, match="legacy torch_npu helper was removed"):
        get_hifloat8_config({"hifloat8": {"backend": "torch_npu"}})
    policy = {"grad_dst_type_max": 15, "compute_dtype": "bfloat16"}
    assert get_hifloat8_config({"hifloat8": {"config": policy}})["config"] == policy


def test_minimal_hifloat8_json_requires_no_numerical_fields():
    config = get_hifloat8_config({
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2
        },
        "hifloat8": {
            "enabled": True,
            "backend": "torchao_npu",
            "module_name_patterns": ["*.experts", "*.shared_experts.*_proj"],
            "min_numel": 65536,
        },
    })
    assert config["enabled"] and config["config"] is None
    assert config["backend"] == "torchao"
    assert config["min_numel"] == 65536


def test_generic_low_precision_config_preserves_backend_policy():
    # The framework must accept an opaque policy instead of parsing device fields.
    from deepspeed.runtime.config import DeepSpeedConfig

    policy = {"backend_specific_option": 7}
    config = DeepSpeedConfig({
        "train_batch_size": 1,
        "low_precision": {
            "enabled": True,
            "backend": "torchao",
            "format": "hifloat8",
            "module_name_patterns": ["*.mlp.*_proj"],
            "config": policy,
        },
    })
    assert config.low_precision_config["enabled"]
    assert config.low_precision_config["backend"] == "torchao"
    assert config.low_precision_config["format"] == "hifloat8"
    assert config.low_precision_config["config"] == policy


def test_legacy_and_generic_low_precision_sections_select_the_same_provider():
    from deepspeed.runtime.config import get_low_precision_config

    selection = {"enabled": True, "module_name_patterns": ["*.experts"], "config": {"grad_dst_type_max": 15}}
    legacy = get_low_precision_config({"hifloat8": dict(selection, backend="torchao_npu")})
    generic = get_low_precision_config({"low_precision": dict(selection, backend="torchao", format="hifloat8")})
    assert legacy == generic


def test_low_precision_rejects_ambiguous_legacy_and_generic_sections():
    from deepspeed.runtime.config import get_low_precision_config

    with pytest.raises(DeepSpeedConfigError, match="both"):
        get_low_precision_config({"low_precision": {}, "hifloat8": {}})


def test_unsupported_accelerator_provider_fails_before_conversion():
    from deepspeed.runtime.config import get_low_precision_config

    model = _ToyModel()
    parameters = dict(model.named_parameters())
    engine = _make_engine(model, ["*.mlp.gate_proj"])
    engine._config.low_precision_config = get_low_precision_config({
        "low_precision": {
            "enabled": True,
            "backend": "unavailable_backend",
            "format": "unavailable_format",
            "module_name_patterns": ["*.mlp.gate_proj"],
        },
    })
    with pytest.raises(RuntimeError, match="provider"):
        engine._configure_low_precision()
    assert type(model.model.mlp["gate_proj"]) is nn.Linear
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in parameters.items())


def test_engine_delegates_policy_probe_and_selection_to_accelerator(monkeypatch):
    # A backend-owned policy must not be interpreted by the generic engine.
    from deepspeed.accelerator import get_accelerator

    model = _ToyModel()
    parameters = dict(model.named_parameters())
    values = {"backend_specific_option": 7}
    policy = object()
    calls = []

    def parse(actual):
        assert actual == values
        calls.append("parse")
        return policy

    def probe(**kwargs):
        assert kwargs["config"] is policy
        assert kwargs["linear"] and not kwargs["grouped"]
        assert kwargs["model_dtype"] == torch.bfloat16
        assert all(not hasattr(module, "backend_policy") for module in model.modules())
        calls.append("probe")

    def convert(actual_model, config, *, filter_fn):
        assert actual_model is model and config is policy
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and filter_fn(module, name):
                module.backend_policy = policy
        calls.append("convert")
        return model

    provider = SimpleNamespace(config_from_dict=parse, assert_training_available=probe, convert_to_training=convert)
    requested = []

    def resolve(backend, format):
        requested.append((backend, format))
        return provider

    monkeypatch.setattr(get_accelerator(), "get_low_precision_provider", resolve)
    engine = _make_engine(model, ["*.mlp.gate_proj", "*.mlp.up_proj", "*.mlp.down_proj"])
    engine._config.low_precision_config.update(backend="example", format="example8", config=values)
    engine._configure_low_precision()
    assert requested == [("example", "example8")]
    assert calls == ["parse", "probe", "convert"]
    assert all(module.backend_policy is policy for module in model.model.mlp.values())
    assert not hasattr(model.model.self_attn["q_proj"], "backend_policy")
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in parameters.items())


def test_npu_provider_missing_dependency_has_install_hint(monkeypatch):
    from importlib.metadata import PackageNotFoundError
    from deepspeed.accelerator.npu import low_precision

    def missing_version(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(low_precision, "version", missing_version)
    provider = low_precision.TorchAOHiFloat8Provider()
    with pytest.raises(ImportError, match="Install torchao"):
        provider.config_from_dict(None)
