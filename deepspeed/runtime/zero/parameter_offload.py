# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import sys
import torch
from collections import OrderedDict
from deepspeed.utils import z3_leaf_module, set_z3_leaf_modules
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.runtime.zero.utils import apply_to_tensors_only, is_zero_param
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.zero.partition_parameters import _init_external_params
from deepspeed.runtime.zero.partition_parameters import *
from deepspeed.runtime.zero.partitioned_param_coordinator import PartitionedParameterCoordinator, InflightParamRegistry, iter_params
from deepspeed.accelerator import get_accelerator

FWD_MODULE_STACK = list()


#for each tensor in outputs run the forward_function and register backward_function as hook
def _apply_forward_and_backward_to_tensors_only(module, forward_function, backward_function, outputs):
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_forward_and_backward_to_tensors_only(module, forward_function, backward_function,
                                                                         output)
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        forward_function(outputs)
        if outputs.requires_grad:
            outputs.register_hook(backward_function)
        return outputs
    else:
        return outputs


class ZeROOrderedDict(OrderedDict):

    def __init__(self, parent_module, *args, **kwargs):
        """A replacement for ``collections.OrderedDict`` to detect external ZeRO params.

        Args:
            parent_module (``collections.OrderedDict``): the collection to replace
        """

        super().__init__(*args, **kwargs)
        self._parent_module = parent_module
        self._in_forward = False

    def __reduce__(self):
        r0, _, *r2 = super().__reduce__()
        return (r0, (self._parent_module, )) + r2

    def __getitem__(self, key):
        param = super().__getitem__(key)

        # Params can be registered as None (e.g., bias)
        if param is None:
            return param

        # TODO: only weaken this check during compilation
        if hasattr(param, "ds_status") and param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if self._parent_module._parameters._in_forward:
                register_external_parameter(FWD_MODULE_STACK[-1], param)
                param.all_gather()
                print_rank_0(f'Registering external parameter from getter {key} ds_id = {param.ds_id}', force=False)

        return param


def _inject_parameters(module, cls):
    for module in module.modules():
        if cls == ZeROOrderedDict:
            new_param = cls(parent_module=module)
        else:
            new_param = cls()

        for key, param in module._parameters.items():
            new_param[key] = param
        module._parameters = new_param


class DeepSpeedZeRoOffload(object):

    def __init__(
        self,
        module,
        timers,
        ds_config,
        overlap_comm=True,
        prefetch_bucket_size=50000000,
        max_reuse_distance=1000000000,
        max_live_parameters=1000000000,
        param_persistence_threshold=100000,
        model_persistence_threshold=sys.maxsize,
        dp_process_group=None,
        offload_param_config=None,
        mpu=None,
        zero_param_parallel_group=None,
        zero_quantized_weights=False,
        zero_quantized_nontrainable_weights=False,
        zero_force_coalesced_fetch_layers=None,
    ):

        see_memory_usage("DeepSpeedZeRoOffload initialize [begin]", force=True)

        print_rank_0(f"initialized {__class__.__name__} with args: {locals()}", force=False)

        self.module = module
        self.timers = timers
        self.dtype = list(module.parameters())[0].dtype
        self.dp_process_group = dp_process_group
        self.offload_device = None
        self.offload_param_pin_memory = False
        self.zero_param_parallel_group = zero_param_parallel_group
        self.zero_quantized_weights = zero_quantized_weights
        self.zero_quantized_nontrainable_weights = zero_quantized_nontrainable_weights

        if offload_param_config is not None and offload_param_config.device != OffloadDeviceEnum.none:
            self.offload_device = offload_param_config.device
            self.offload_param_pin_memory = offload_param_config.pin_memory

        self._convert_to_zero_parameters(ds_config, module, mpu)

        for m in module.modules():
            _init_external_params(m)

        _inject_parameters(module, ZeROOrderedDict)

        self.param_numel_persistence_threshold = int(param_persistence_threshold)
        self.model_persistence_threshold = int(model_persistence_threshold)
        self.persistent_parameters = self.mark_persistent_parameters(self.param_numel_persistence_threshold,
                                                                     self.model_persistence_threshold)

        self.param_coordinators = {}
        self._prefetch_bucket_sz = int(prefetch_bucket_size)
        self._max_reuse_distance_in_numel = int(max_reuse_distance)
        self._max_available_parameters_in_numel = int(max_live_parameters)
        self.__allgather_stream = None if get_accelerator().is_synchronized_device() else get_accelerator().Stream(
        ) if overlap_comm else get_accelerator().default_stream()

        if not hasattr(module, "ds_inflight_param_registry"):
            module.ds_inflight_param_registry = dict()
            # we need two registries, one for training and one for eval. They will be used when creating PartitionedParameterCoordinator
            module.ds_inflight_param_registry[True] = InflightParamRegistry()
            module.ds_inflight_param_registry[False] = InflightParamRegistry()
        self.__inflight_param_registry = module.ds_inflight_param_registry

        self.forward_hooks = []
        self.backward_hooks = []
        if zero_force_coalesced_fetch_layers is not None and len(zero_force_coalesced_fetch_layers) > 0:
            set_z3_leaf_modules(module, zero_force_coalesced_fetch_layers)

        self.setup_zero_stage3_hooks()
        print_rank_0(
            f'Created module hooks: forward = {len(self.forward_hooks)}, backward = {len(self.backward_hooks)}',
            force=False)

        see_memory_usage("DeepSpeedZeRoOffload initialize [end]", force=True)

    @instrument_w_nvtx
    def partition_all_parameters(self):
        """Partitioning Parameters that were not partitioned usually if parameters
        of modules whose input parameters do not require grad computation do not
        trigger post call and will therefore will remain unpartitioned"""
        self.get_param_coordinator(training=self.module.training).release_and_reset_all(self.module)
        for param in iter_params(self.module, recurse=True):
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(f"{param.ds_summary()} expected to be released")

    def get_param_coordinator(self, training):
        if not training in self.param_coordinators:
            self.param_coordinators[training] = PartitionedParameterCoordinator(
                prefetch_bucket_sz=self._prefetch_bucket_sz,
                max_reuse_distance_in_numel=self._max_reuse_distance_in_numel,
                max_available_parameters_in_numel=self._max_available_parameters_in_numel,
                allgather_stream=self.__allgather_stream,
                inflight_param_registry=self.__inflight_param_registry[training],
                prefetch_nvme=self.offload_device == OffloadDeviceEnum.nvme,
                timers=self.timers,
                zero_quantized_weights=self.zero_quantized_weights,
                zero_quantized_nontrainable_weights=self.zero_quantized_nontrainable_weights,
            )

        return self.param_coordinators[training]

    def empty_partition_cache(self):
        self.partition_all_parameters()

    def _convert_to_zero_parameters(self, ds_config, module, mpu):
        non_zero_params = [p for p in module.parameters() if not is_zero_param(p)]
        if non_zero_params:
            zero_params = [p for p in module.parameters() if is_zero_param(p)]
            if zero_params:
                zero_params[0].convert_to_zero_parameters(param_list=non_zero_params)
            else:
                group = None
                if mpu:
                    group = mpu.get_data_parallel_group()

                Init(module=module,
                     data_parallel_group=group,
                     dtype=self.dtype,
                     config_dict_or_path=ds_config,
                     remote_device=self.offload_device,
                     pin_memory=self.offload_param_pin_memory,
                     mpu=mpu,
                     zero_param_parallel_group=self.zero_param_parallel_group,
                     zero_quantized_weights=self.zero_quantized_weights,
                     zero_quantized_nontrainable_weights=self.zero_quantized_nontrainable_weights)

    def destroy(self):
        self._remove_module_hooks()

    def _remove_module_hooks(self):
        num_forward_hooks = len(self.forward_hooks)
        num_backward_hooks = len(self.backward_hooks)

        for hook in self.forward_hooks:
            hook.remove()

        for hook in self.backward_hooks:
            hook.remove()

        print_rank_0(f'Deleted module hooks: forward = {num_forward_hooks}, backward = {num_backward_hooks}',
                     force=False)

    def setup_zero_stage3_hooks(self):
        self.hierarchy = 0

        #reset step if in inference mode
        @instrument_w_nvtx
        def _end_of_forward_hook(module, *args):

            if not torch._C.is_grad_enabled():
                self.get_param_coordinator(training=False).reset_step()

        #likely one of them should be enough but just to be safe
        self._register_hooks_recursively(self.module)
        self.module.register_forward_hook(_end_of_forward_hook)

        # Add top module to stack trace
        global FWD_MODULE_STACK
        FWD_MODULE_STACK.append(self.module)

    def mark_persistent_parameters(self, param_threshold, model_threshold):
        persistent_params = []
        total_persistent_parameters = 0
        params_count = 0
        for name, param in self.module.named_parameters(recurse=True):
            if param.ds_numel + total_persistent_parameters > model_threshold:
                continue

            if param.ds_numel <= param_threshold:
                params_count += 1
                param.ds_persist = True
                persistent_params.append(param)
                total_persistent_parameters += param.ds_numel

        print_rank_0(
            f"Parameter Offload: Total persistent parameters: {total_persistent_parameters} in {params_count} params",
            force=True)

        return persistent_params

    def _register_hooks_recursively(self, module, count=[0]):
        my_count = count[0]
        module.id = my_count

        #print(f"{module.__class__} : {module.id}")

        if z3_leaf_module(module):
            for param in module.parameters():
                param.ds_z3_leaf_module = module
        else:
            for child in module.children():
                count[0] = count[0] + 1
                self._register_hooks_recursively(child, count=count)

        @instrument_w_nvtx
        def _pre_forward_module_hook(module, *args):
            self.pre_sub_module_forward_function(module)

        @instrument_w_nvtx
        def _post_forward_module_hook(module, input, output):

            global FWD_MODULE_STACK
            FWD_MODULE_STACK.pop()
            if output is None:
                output = []
            elif not isinstance(output, (list, tuple)):
                if torch.is_tensor(output):
                    output = [output]
                else:
                    #print(f'got UNKNOWN type {type(output)}')
                    outputs = []
                    output = output if isinstance(output, dict) else vars(output)
                    for name, val in output.items():
                        if not name.startswith('__') and torch.is_tensor(val):
                            outputs.append(val)
                    output = outputs

            for item in filter(lambda item: is_zero_param(item) or hasattr(item, 'ds_param_alias'), output):
                key = id(item) if hasattr(item, 'ds_id') else id(item.ds_param_alias)
                actual_external_param = item if hasattr(item, 'ds_id') else item.ds_param_alias

                if not any(key in m._external_params for m in FWD_MODULE_STACK):
                    actual_external_param.is_external_param = True
                    module_to_register = FWD_MODULE_STACK[-1]
                    register_external_parameter(module_to_register, actual_external_param)
                    print_rank_0(
                        f'Registering dangling parameter for module {module_to_register.__class__.__name__}, ds_id = {actual_external_param.ds_id}.',
                        force=False)

                    # It's possible that the parameter was already external to the completed module. If so, remove it the
                    # registration as it will be covered by the outer module instead.
                    if key in module._external_params:
                        print_rank_0(
                            f'  Unregistering nested dangling parameter from module {module.__class__.__name__}, ds_id = {actual_external_param.ds_id}',
                            force=False)
                        unregister_external_parameter(module, actual_external_param)

                    actual_external_param.all_gather()

            self.post_sub_module_forward_function(module)

        def _bwd_hook_unexpected_inputs_msg(value):
            return f"A module has unknown inputs or outputs type ({type(value)}) and the tensors embedded in it cannot be detected. " \
                "The ZeRO-3 hooks designed to trigger before or after backward pass of the module relies on knowing the input and " \
                "output tensors and therefore may not get triggered properly."

        def _pre_backward_module_hook(module, inputs, output):

            if not hasattr(module, "pre_bwd_fn"):

                @instrument_w_nvtx
                def _run_before_backward_function(sub_module):
                    # some models (e.g. Albert) may run multiple forwards on the same layer in a loop
                    # before doing backwards, so each backward will need a pre-fetch - using reference
                    # counting to support this scenario
                    #print(f"COUNTER before: {sub_module.applied_pre_backward_ref_cnt}")
                    if sub_module.applied_pre_backward_ref_cnt > 0:
                        self.pre_sub_module_backward_function(sub_module)
                        sub_module.applied_pre_backward_ref_cnt -= 1
                    #print(f"COUNTER after: {sub_module.applied_pre_backward_ref_cnt}")

                class PreBackwardFunctionForModule(torch.autograd.Function):

                    @staticmethod
                    def forward(ctx, outputs):
                        # Capture `module` and _run_before_backward_function
                        ctx.module = module
                        ctx.pre_backward_function = _run_before_backward_function
                        if not hasattr(ctx.module, "applied_pre_backward_ref_cnt"):
                            ctx.module.applied_pre_backward_ref_cnt = 0
                        ctx.module.applied_pre_backward_ref_cnt += 1
                        outputs = outputs.detach()
                        return outputs

                    @staticmethod
                    def backward(ctx, *args):
                        ctx.pre_backward_function(ctx.module)
                        return args

                module.pre_bwd_fn = PreBackwardFunctionForModule

            return apply_to_tensors_only(module.pre_bwd_fn.apply,
                                         output,
                                         warning_msg_fn=_bwd_hook_unexpected_inputs_msg)

        #This is an alternate to doing _post_backward_module_hook
        #it uses tensor.register_hook instead of using torch.autograd.Function
        def _alternate_post_backward_module_hook(module, inputs):
            module.ds_grads_remaining = 0

            #print(f"Before Forward {module.__class__.__name__}")

            def _run_after_backward_hook(*unused):
                module.ds_grads_remaining = module.ds_grads_remaining - 1
                if module.ds_grads_remaining == 0:
                    #print(f"After backward {module.__class__.__name__}")
                    self.post_sub_module_backward_function(module)

            def _run_before_forward_function(input):
                if input.requires_grad:
                    module.ds_grads_remaining += 1

            return _apply_forward_and_backward_to_tensors_only(module, _run_before_forward_function,
                                                               _run_after_backward_hook, inputs)

        def _post_backward_module_hook(module, inputs):
            module.ds_grads_remaining = 0

            if not hasattr(module, "post_bwd_fn"):

                @instrument_w_nvtx
                def _run_after_backward_function(sub_module):
                    if sub_module.ds_grads_remaining == 0:
                        self.post_sub_module_backward_function(sub_module)

                class PostBackwardFunctionModule(torch.autograd.Function):

                    @staticmethod
                    def forward(ctx, output):
                        ctx.module = module
                        if output.requires_grad:
                            #TODO SOME TIMES post backward does not seem to be triggered debug in detail
                            #Should only cause increase in memory not correctness issue
                            #if output.grad_fn.__class__.__name__ == 'ViewBackward':
                            #    ctx.view=True
                            #    print(f"Warning view tensor for input to module : {module.__class__.__name__}. Backward hooks may not trigger properly")
                            #assert len(module.parameters(recurse=False)), "The input tensor to the module is a view, and autograd Function or register_hook is not triggered with view tensors."
                            #if module.ds_grads_remaining == 0:
                            #    print(f"Before Forward: {ctx.module.__class__.__name__}")
                            module.ds_grads_remaining += 1
                            ctx.post_backward_function = _run_after_backward_function
                        output = output.detach()
                        return output

                    @staticmethod
                    def backward(ctx, *args):
                        ctx.module.ds_grads_remaining = ctx.module.ds_grads_remaining - 1
                        if ctx.module.ds_grads_remaining == 0:
                            ctx.post_backward_function(ctx.module)
                        return args

                module.post_bwd_fn = PostBackwardFunctionModule

            return apply_to_tensors_only(module.post_bwd_fn.apply,
                                         inputs,
                                         warning_msg_fn=_bwd_hook_unexpected_inputs_msg)

        # Pre forward hook
        self.forward_hooks.append(module.register_forward_pre_hook(_pre_forward_module_hook))

        # Post forward hook
        self.forward_hooks.append(module.register_forward_hook(_post_forward_module_hook))

        # Pre backward hook
        self.backward_hooks.append(module.register_forward_hook(_pre_backward_module_hook))

        # post backward hook
        self.backward_hooks.append(module.register_forward_pre_hook(_post_backward_module_hook))

    @torch.no_grad()
    def pre_sub_module_forward_function(self, sub_module):
        see_memory_usage(f"Before sub module function {sub_module.__class__.__name__}", force=False)

        global FWD_MODULE_STACK
        FWD_MODULE_STACK.append(sub_module)

        param_coordinator = self.get_param_coordinator(training=sub_module.training)
        param_coordinator.trace_prologue(sub_module)
        if param_coordinator.is_record_trace():
            param_coordinator.record_module(sub_module)
        param_coordinator.fetch_sub_module(sub_module, forward=True)

        see_memory_usage(f"Before sub module function {sub_module.__class__.__name__} after fetch", force=False)

    @torch.no_grad()
    def post_sub_module_forward_function(self, sub_module):
        see_memory_usage(f"After sub module function {sub_module.__class__.__name__} {sub_module.id} before release",
                         force=False)

        param_coordinator = self.get_param_coordinator(training=sub_module.training)
        param_coordinator.release_sub_module(sub_module)

        see_memory_usage(f"After sub module function {sub_module.__class__.__name__}  {sub_module.id} after release",
                         force=False)

    @torch.no_grad()
    def pre_sub_module_backward_function(self, sub_module):
        assert sub_module.training, "backward pass is invalid for module in evaluation mode"
        param_coordinator = self.get_param_coordinator(training=True)
        param_coordinator.trace_prologue(sub_module)
        if param_coordinator.is_record_trace():
            param_coordinator.record_module(sub_module)
        param_coordinator.fetch_sub_module(sub_module, forward=False)

    @torch.no_grad()
    def post_sub_module_backward_function(self, sub_module):
        assert sub_module.training, "backward pass is invalid for module in evaluation mode"
        see_memory_usage(
            f"After sub module backward function {sub_module.__class__.__name__} {sub_module.id} before release",
            force=False)

        self.get_param_coordinator(training=True).release_sub_module(sub_module)

        see_memory_usage(
            f"After sub module backward function {sub_module.__class__.__name__} {sub_module.id} after release",
            force=False)
