'''
Copyright 2021 The Microsoft DeepSpeed Team
'''
import torch
import time
import os

from deepspeed import comm as dist
from deepspeed.utils.logging import log_dist

from torch.nn.modules import Module
from packaging import version as pkg_version
from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
from deepspeed.utils.timer import SynchronizedWallClockTimer

from ..runtime.state_dict_factory import SDLoaderFactory
from ..runtime.weight_quantizer import WeightQuantization
from ..module_inject.replace_module import replace_transformer_layer, generic_injection
from ..comm.comm import init_distributed
from ..pipe import PipelineModule
from ..moe.utils import has_moe_layers
from ..runtime.zero import GatheredParameters
from ..module_inject import LinearAllreduce, LinearLayer, Normalize, ReplaceWithTensorSlicing
from ..module_inject.replace_policy import DSPolicy

DS_INFERENCE_ENABLED = False
from torch import nn

INFERENCE_MODEL_TIMER = "model-forward-inference"


class InferenceEngine(Module):
    inference_mp_group = None
    inference_ep_group = None
    expert_mp_group = None

    def __init__(self, model, config):
        """
        Args:
            model: torch.nn.Module
            config: DeepSpeedInferenceConfig
        """
        global DS_INFERENCE_ENABLED
        DS_INFERENCE_ENABLED = True

        super().__init__()

        self.module = model
        self._config = config

        self._get_model_config_generate(config)  # keep for weird backward compatibility

        if hasattr(self.module, "config"):
            DSPolicy.hf_model_config = self.module.config

        # todo: keep this self.injection_dict because we don't use to change config.injection_policy API
        # todo: this will get changed when Molly's PR on auto injection dict is merged
        self.injection_dict = config.injection_policy

        # todo: refactor the mp_group and mp_size related in the next refactor
        self.mp_group = config.tensor_parallel.tp_group
        self.mpu = config.tensor_parallel.mpu

        #self._validate_args(self.mpu, config.replace_with_kernel_inject)
        self.quantize_merge_count = 1
        self.quantization_scales = None

        # these are not needed in the config as we are creating them ourselves in the inference engine
        self.ep_group = None  # config.moe.ep_group
        self.expert_mp_group = None  # config.moe.ep_mp_group

        self.cuda_graph_created = False
        self.checkpoint_engine = TorchCheckpointEngine()
        quantization_setting = None
        self._init_quantization_setting(
            quantization_setting
        )  # todo: update with the new quant config for weight quant
        self.model_profile_enabled = False
        self._model_times = []

        # This is a hack to remove the prepare_mask function on HF side for BLOOM architecture
        self.remove_mask_prepare_for_bloom()

        if config.enable_cuda_graph:
            assert pkg_version.parse(torch.__version__) >= pkg_version.parse("1.10"), \
                "If you want to use cuda graph, please upgrade torch to at least v1.10"

        if config.checkpoint and not config.replace_with_kernel_inject:
            self._load_checkpoint(config.checkpoint)

        # convert model to intended dtype
        if config.dtype:
            self._convert_to_dtype(config)

        if self.mpu:
            config.tensor_parallel.tp_size = dist.get_world_size(
                group=self.mpu.get_model_parallel_group())
            self.mp_group = self.mpu.get_model_parallel_group()
        elif config.tensor_parallel.tp_size > 1:
            self._create_model_parallel_group(config)
            config.tensor_parallel.tp_group = self.mp_group

        if isinstance(self.module, torch.nn.Module):
            moe, _ = has_moe_layers(self.module)
        else:
            moe = False

        if moe and dist.get_world_size() > 1:
            self._create_ep_parallel_group(config.moe.moe_experts)

        # retain this from the old conditional argument being passed to apply_injection_policy()
        if not config.replace_with_kernel_inject:
            config.checkpoint = None

        if self.injection_dict:
            for client_module, injection_policy in self.injection_dict.items():
                # construct the tuple and pass that instead of a string or dict.
                if isinstance(injection_policy, str):
                    config.injection_policy_tuple = (injection_policy, )
                else:
                    config.injection_policy_tuple = injection_policy
                self._apply_injection_policy(config, client_module)
        elif config.replace_method == 'auto':
            self._apply_injection_policy(config)

        device = torch.cuda.current_device()
        #self.module.to(device)

        if config.tensor_parallel.tp_size > 1:
            _rng_state = torch.cuda.get_rng_state().to(torch.cuda.current_device())
            dist.broadcast(_rng_state, 0)
            torch.cuda.set_rng_state(_rng_state.cpu())

        if config.tensor_parallel.tp_size > 1:
            assert not config.enable_cuda_graph, "Cuda graph is not supported for model parallelism"

    def profile_model_time(self, use_cuda_events=True):
        if not self.model_profile_enabled and not self._config.enable_cuda_graph:
            self.module.register_forward_pre_hook(self._pre_forward_hook)
            self.module.register_forward_hook(self._post_forward_hook)
        self.model_profile_enabled = True
        self.use_cuda_events = use_cuda_events
        if self.use_cuda_events:
            self.timers = SynchronizedWallClockTimer()

    # todo: remove this once all the config dicts are centralized from top level pydantic config
    def _get_model_config_generate(self, config):
        # this is being passed to replace_transformer_layer(config=self.user_model_config_dict)
        self.config = getattr(self.module,
                              'config',
                              None) if config.config is None else config.config
        # todo: clarify with Reza if this gets used anywhere
        self.generate = getattr(self.module, 'generate', None)

    def remove_mask_prepare_for_bloom(self):
        if hasattr(self.module, 'transformer'):
            if hasattr(self.module.transformer, '_prepare_attn_mask'):
                self.module.transformer._prepare_attn_mask = lambda attention_mask, *args, **kwargs: attention_mask

    def _pre_forward_hook(self, module, *inputs, **kwargs):
        if self.use_cuda_events:
            self.timers(INFERENCE_MODEL_TIMER).start()
        else:
            torch.cuda.synchronize()
            self._start = time.time()

    def _post_forward_hook(self, module, input, output):
        if self.use_cuda_events:
            self.timers(INFERENCE_MODEL_TIMER).stop()
            elapsed_time = self.timers(INFERENCE_MODEL_TIMER).elapsed(reset=True)
        else:
            torch.cuda.synchronize()
            self._end = time.time()
            elapsed_time = self._end - self._start
        self._model_times.append(elapsed_time)

    def _create_model_parallel_group(self, config):
        # Call the init process
        if InferenceEngine.inference_mp_group is None:
            init_distributed()
            local_rank = int(os.getenv('LOCAL_RANK', '0'))
            torch.cuda.set_device(local_rank)

            ranks = [i for i in range(config.tensor_parallel.tp_size)]
            self.mp_group = dist.new_group(ranks)
            InferenceEngine.inference_mp_group = self.mp_group
        else:
            self.mp_group = InferenceEngine.inference_mp_group

    def _create_ep_parallel_group(self, moe_experts):
        # Call the init process
        self.ep_group = {}
        self.expert_mp_group = {}
        moe_experts = moe_experts if type(moe_experts) is list else [moe_experts]
        for e in moe_experts:
            self.ep_group.update({e: None})
            self.expert_mp_group.update({e: None})
        for moe_ep_size in self.ep_group.keys():
            num_ep_groups = dist.get_world_size() // moe_ep_size
            for i in range(num_ep_groups):
                ep_cnt = i * moe_ep_size
                size = dist.get_world_size(
                ) if moe_ep_size > dist.get_world_size() else moe_ep_size
                ranks = list(range(ep_cnt, ep_cnt + size))
                _ep_group = dist.new_group(ranks)
                if dist.get_rank() in ranks:
                    self.ep_group.update({moe_ep_size: _ep_group})

            if dist.get_world_size() > moe_ep_size:
                num_expert_mp_groups = dist.get_world_size() // num_ep_groups
                expert_mp_size = dist.get_world_size() // moe_ep_size
                for i in range(num_expert_mp_groups):
                    expert_mp_comm_ranks = [
                        i + nr * moe_ep_size for nr in range(expert_mp_size)
                    ]
                    _expert_mp_group = dist.new_group(expert_mp_comm_ranks)
                    if dist.get_rank() in expert_mp_comm_ranks:
                        self.expert_mp_group.update({moe_ep_size: _expert_mp_group})

    def _init_quantization_setting(self, quantization_setting):
        self.quantize_bits = 8
        self.mlp_extra_grouping = False
        self.quantize_groups = 1
        if type(quantization_setting) is tuple:
            self.mlp_extra_grouping, \
            self.quantize_groups = quantization_setting
        elif quantization_setting is not None:
            self.quantize_groups = quantization_setting
        log_dist(
            f"quantize_bits = {self.quantize_bits} "
            f"mlp_extra_grouping = {self.mlp_extra_grouping}, "
            f"quantize_groups = {self.quantize_groups}",
            [0])

    # TODO: remove this function and add this functionality to pydantic config checking
    def _validate_args(self, mpu, replace_with_kernel_inject):
        # TODO: to support SD pipeline we need to avoid this check for now
        if replace_with_kernel_inject and not isinstance(self.module, Module):
            raise ValueError(f"model must be a torch.nn.Module, got {type(self.module)}")
        if not isinstance(self._config.tensor_parallel.tp_size,
                          int) or self._config.tensor_parallel.tp_size < 1:
            raise ValueError(
                f"mp_size must be an int >= 1, got {self._config.tensor_parallel.tp_size}"
            )

        if mpu:
            methods = ["get_model_parallel_group", "get_data_parallel_group"]
            for method in methods:
                if not hasattr(mpu, method):
                    raise ValueError(f"mpu is missing {method}")
        if self._config.checkpoint is not None and not isinstance(
                self._config.checkpoint,
            (str,
             dict)):
            raise ValueError(
                f"checkpoint must be None, str or dict, got {type(self._config.checkpoint)}"
            )

        supported_dtypes = [None, torch.half, torch.int8, torch.float]
        if self._config.dtype not in supported_dtypes:
            raise ValueError(
                f"{self._config.dtype} not supported, valid dtype: {supported_dtypes}")

        if self.injection_dict is not None and not isinstance(self.injection_dict, dict):
            raise ValueError(
                f"injection_dict must be None or a dict, got: {self.injection_dict}")

    def load_model_with_checkpoint(self, r_module):
        self.mp_replace = ReplaceWithTensorSlicing(
            mp_group=self.mp_group,
            mp_size=self._config.tensor_parallel.tp_size)  #, out_dim=0, in_dim=1)
        error_msgs = []

        def load(module, state_dict, prefix):
            args = (state_dict, prefix, {}, True, [], [], error_msgs)
            if len(list(module.parameters())) > 0 and list(
                    module.parameters())[0].numel() == 0:
                with GatheredParameters(list(module.parameters(recurse=False)),
                                        modifier_rank=0):
                    if dist.get_rank() == 0:
                        module._load_from_state_dict(*args)
            else:
                if hasattr(module, 'weight'):
                    if 'query_key_value' in prefix:
                        module.weight = self.mp_replace.qkv_copy(
                            module.weight.data,
                            state_dict[prefix + 'weight'])
                    else:
                        module.weight = self.mp_replace.copy(
                            module.weight.data,
                            state_dict[prefix + 'weight'])
                else:
                    module.norm.weight = self.mp_replace.copy(
                        module.norm.weight.data,
                        state_dict[prefix + 'weight'])
                if prefix + 'bias' in self.key_list:
                    if hasattr(module, 'norm'):
                        module.norm.bias = self.mp_replace.copy(
                            module.norm.bias,
                            state_dict[prefix + 'bias'])
                    else:
                        data = state_dict[prefix + 'bias']
                        data = data.to(torch.cuda.current_device())
                        module.bias = self.mp_replace.copy(module.bias, data)

        layer_policies = {
            nn.Linear: load,
            nn.Embedding: load,
            nn.LayerNorm: load,
            LinearLayer: load,
            LinearAllreduce: load
        }

        def load_module_recursive(module, prefix='', level=0):
            for name, child in module.named_children():
                if child.__class__ in layer_policies:
                    checking_key = prefix + name + '.'
                    if not any(checking_key in item for item in self.key_list):
                        continue
                    if len(list(child.parameters())) > 0 and list(
                            child.parameters())[0].numel() == 0:
                        if len(child.weight.ds_shape) == 1:
                            child = Normalize(dim=child.weight.ds_shape[-1],
                                              dtype=child.weight.dtype,
                                              eps=child.eps)
                            setattr(module, name, child)
                    load(child, self.sd, prefix + name + '.')
                else:
                    load_module_recursive(child,
                                          prefix if level == 0 else prefix + name + '.',
                                          level + 1)

        load_module_recursive(r_module)

    def _apply_injection_policy(self, config, client_module=None):
        # client_module is only passed when using the injection_dict method.
        checkpoint_dir = config.checkpoint
        checkpoint = SDLoaderFactory.get_sd_loader_json(
            checkpoint_dir,
            self.checkpoint_engine) if checkpoint_dir is not None else None

        generic_injection(self.module,
                          fp16=(config.dtype == torch.half)
                          or (config.dtype == torch.int8),
                          enable_cuda_graph=config.enable_cuda_graph)

        if isinstance(self.module, torch.nn.Module):
            # config is our DeepSpeedInferenceConfig and self.config is the HF model config
            replace_transformer_layer(client_module,
                                      self.module,
                                      checkpoint,
                                      config,
                                      self.config)

    def _get_all_ckpt_names(self, checkpoints_path, tag):
        ckpt_file_pattern = self._get_ckpt_name(checkpoints_path,
                                                tag,
                                                mp_placeholder="*")
        import glob

        ckpt_files = glob.glob(ckpt_file_pattern)
        ckpt_files.sort()
        return ckpt_files

    def _get_ckpt_name(self, checkpoints_path, tag, mp_placeholder=None):
        if mp_placeholder is not None:
            mp_rank_str = mp_placeholder
        else:
            mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
            mp_rank_str = "{:02d}".format(mp_rank)

        ckpt_name = os.path.join(
            checkpoints_path,
            "mp_rank_" + mp_rank_str + "_model_states.pt",
        )
        return ckpt_name

    def _load_checkpoint(self, load_dir, load_module_strict=True, tag=None):
        is_pipe_parallel = isinstance(self.module, PipelineModule)
        if is_pipe_parallel:
            raise RuntimeError(
                'pipeline parallelism is currently not supported in inference.')
        if not isinstance(load_dir, dict) and os.path.isdir(load_dir):
            if tag is None:
                latest_path = os.path.join(load_dir, "latest")
                if os.path.isfile(latest_path):
                    with open(latest_path, "r") as fd:
                        tag = fd.read().strip()

            ckpt_list = self._get_all_ckpt_names(load_dir, tag)
            sd_loader = SDLoaderFactory.get_sd_loader(ckpt_list, self.checkpoint_engine)
        else:
            sd_loader = SDLoaderFactory.get_sd_loader_json(load_dir,
                                                           self.checkpoint_engine)

        if type(sd_loader) is list:
            self.sd = torch.load(sd_loader[0], map_location='cpu')
            self.key_list = list(self.sd.keys())

            self.load_model_with_checkpoint(self.module)

            for i in range(1, len(sd_loader)):
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"loading checkpoint ({i})")
                self.sd = torch.load(sd_loader[i], map_location='cuda')
                self.key_list = list(self.sd.keys())
                self.load_model_with_checkpoint(self.module)
        else:
            mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()

            load_path, checkpoint, quantize_config = sd_loader.load(self._config.tensor_parallel.tp_size,
                                                    mp_rank,
                                                    is_pipe_parallel=is_pipe_parallel,
                                                    quantize=(self._config.dtype is torch.int8),
                                                    quantize_groups=self.quantize_groups,
                                                    mlp_extra_grouping=self.mlp_extra_grouping)

            self.quantization_scales, self.quantize_merge_count = quantize_config

            moe, _ = has_moe_layers(self.module)
            if moe:
                from deepspeed.runtime.engine import DeepSpeedEngine
                old_moe_load = False
                if not isinstance(checkpoint['num_experts'], list):
                    old_moe_load = True
                DeepSpeedEngine.load_moe_state_dict(
                    load_dir,
                    tag,
                    state_dict=checkpoint[self._choose_module_key(checkpoint)],
                    old_moe_load=old_moe_load,
                    model=self.module,
                    mpu=self.mpu,
                    checkpoint_engine=self.checkpoint_engine)

            self.module.load_state_dict(
                state_dict=checkpoint[self._choose_module_key(checkpoint)],
                strict=load_module_strict)

    def _choose_module_key(self, sd):
        assert not ('module' in sd and 'model' in sd), "checkpoint has both 'model' and 'module' keys, not sure how to proceed"
        assert 'module' in sd or 'model' in sd, "checkpoint contains neither 'model' or 'module' keys, not sure how to proceed"
        if 'module' in sd:
            return 'module'
        elif 'model' in sd:
            return 'model'

    def _convert_to_dtype(self, config):
        if not isinstance(self.module, torch.nn.Module):
            return

        if False:  #config.dtype is torch.int8 and self.quantization_scales is None:
            quantizer = WeightQuantization(mlp_extra_grouping=self.mlp_extra_grouping)
            model, self.quantization_scales = quantizer.model_quantize(self.module,
                                                                        self.injection_dict,
                                                                        self.quantize_bits,
                                                                        self.quantize_groups)
        elif config.dtype == torch.half:
            self.module.half()
        elif config.dtype == torch.bfloat16:
            self.module.bfloat16()
        elif config.dtype == torch.float:
            self.module.float()

    def _create_cuda_graph(self, *inputs, **kwargs):
        # warmup to create the workspace and cublas handle
        cuda_stream = torch.cuda.Stream()
        cuda_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cuda_stream):
            for i in range(3):
                ret = self.module(*inputs, **kwargs)
        torch.cuda.current_stream().wait_stream(cuda_stream)

        # create cuda_graph and assign static_inputs and static_outputs
        self._cuda_graphs = torch.cuda.CUDAGraph()
        self.static_inputs = inputs
        self.static_kwargs = kwargs

        with torch.cuda.graph(self._cuda_graphs):
            self.static_output = self.module(*self.static_inputs, **self.static_kwargs)

        self.cuda_graph_created = True

    def _graph_replay(self, *inputs, **kwargs):
        for i in range(len(inputs)):
            if torch.is_tensor(inputs[i]):
                self.static_inputs[i].copy_(inputs[i])
        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                self.static_kwargs[k].copy_(kwargs[k])
        self._cuda_graphs.replay()
        return self.static_output

    def model_times(self):
        assert self.model_profile_enabled, "model profiling is not enabled"
        model_times = self._model_times
        if self._config.enable_cuda_graph and len(self._model_times) == 0:
            raise ValueError(
                "Model times are empty and cuda graph is enabled. If "
                "this is a GPT-style model this combo is not supported. If this is a "
                "BERT-style model this is a bug, please report it. "
                f"Model type is: {type(self.module)}")
        self._model_times = []
        return model_times

    def forward(self, *inputs, **kwargs):
        """Execute forward propagation

        Arguments:
            *inputs: Variable length input list
            **kwargs: variable length keyword arguments
        """
        start = None
        if self.model_profile_enabled and self._config.enable_cuda_graph:
            torch.cuda.synchronize()
            start = time.time()

        if self._config.enable_cuda_graph:
            if self.cuda_graph_created:
                outputs = self._graph_replay(*inputs, **kwargs)
            else:
                self._create_cuda_graph(*inputs, **kwargs)
                outputs = self._graph_replay(*inputs, **kwargs)
        else:
            outputs = self.module(*inputs, **kwargs)

        if self.model_profile_enabled and self._config.enable_cuda_graph:
            torch.cuda.synchronize()
            duration = time.time() - start
            self._model_times.append(duration)

        return outputs
