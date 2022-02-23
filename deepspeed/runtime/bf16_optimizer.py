import torch
import torch.distributed as dist
from deepspeed.runtime.constants import PIPE_REPLICATED
from deepspeed.ops.op_builder import UtilsBuilder

from deepspeed.runtime.utils import (get_global_norm_of_tensors,
                                     clip_tensors_by_global_norm,
                                     get_grad_norm,
                                     clip_gradients,
                                     align_dense_tensors)


class BF16_Optimizer:
    def __init__(self,
                 init_optimizer,
                 mpu=None,
                 clip_grad=0.0,
                 norm_type=2,
                 dp_process_group=None,
                 timers=None):
        super().__init__()
        self.timers = timers
        self.optimizer = init_optimizer
        self.clip_grad = clip_grad
        self.norm_type = norm_type
        self.mpu = mpu
        self.dp_process_group = dp_process_group
        self.dp_rank = dist.get_rank(group=self.dp_process_group)

        # Load pre-built or JIT compile (un)flatten ops
        util_ops = UtilsBuilder().load()
        self.flatten = util_ops.flatten
        self.unflatten = util_ops.unflatten

        #align nccl all-gather send buffers to 4-bye boundary
        self.nccl_start_alignment_factor = 2  # 4-byte alignment/sizeof(fp16) = 2

        # Build BF16/FP32 groups
        self.bf16_groups = []
        self.bf16_groups_flat = []
        # TODO: Need to only track fp32 params of this partition
        self.fp32_groups = []
        self.fp32_groups_flat = []
        self.single_partition_of_fp32_groups = []
        self.fp32_groups_gradients = []
        self.fp32_groups_gradients_flat = []

        dp_world_size = dist.get_world_size(group=self.dp_process_group)

        for i, param_group in enumerate(self.optimizer.param_groups):
            # grab the original list
            self.bf16_groups.append(param_group['params'])

            # create flat bf16 params
            self.bf16_groups_flat.append(
                self._flatten_dense_tensors_aligned(
                    self.bf16_groups[i],
                    self.nccl_start_alignment_factor * dp_world_size))

            # Make bf16 params point to flat tensor storage
            self._update_storage_to_flattened_tensor(
                tensor_list=self.bf16_groups[i],
                flat_tensor=self.bf16_groups_flat[i])

            # create flat fp32 params
            self.fp32_groups_flat.append(
                self.bf16_groups_flat[i].clone().float().detach())
            self.fp32_groups_flat[i].requires_grad = True

            num_elem_list = [t.numel() for t in self.bf16_groups[i]]

            # create fp32 params using flat tensor storage
            fp32_group_params = self._split_flat_tensor(
                flat_tensor=self.fp32_groups_flat[i],
                num_elem_list=num_elem_list)
            self._propagate_attributes(src_tensor_list=self.bf16_groups[i],
                                       dst_tensor_list=fp32_group_params)
            self.fp32_groups.append(fp32_group_params)

            # create fp32 gradients
            self.fp32_groups_gradients_flat.append(
                torch.zeros_like(self.fp32_groups_flat[i]))
            fp32_gradients = self._split_flat_tensor(
                flat_tensor=self.fp32_groups_gradients_flat[i],
                num_elem_list=num_elem_list)
            self.fp32_groups_gradients.append(fp32_gradients)

            # create fp32 partition from flat tensor storage
            assert self.fp32_groups_flat[i].numel() % dp_world_size == 0, \
            f'group {i} flat tensor size {self.fp32_groups_flat[i].numel()} not divisible by DP world size {dp_world_size}'

            partition_size = self.fp32_groups_flat[i].numel() // dp_world_size
            self.single_partition_of_fp32_groups.append(
                torch.narrow(self.fp32_groups_flat[i],
                             0,
                             self.dp_rank * partition_size,
                             partition_size))
            param_group['params'] = [self.single_partition_of_fp32_groups[i]]

        self.initialize_optimizer_states()
        self._init_hp_grads()

    def initialize_optimizer_states(self):
        """Take an optimizer step with zero-valued gradients to allocate internal
        optimizer state.

        This helps prevent memory fragmentation by allocating optimizer state at the
        beginning of training instead of after activations have been allocated.
        """
        for i, single_partition in enumerate(self.single_partition_of_fp32_groups):
            single_partition.grad = self.fp32_groups_gradients_flat[i]

        self.optimizer.step()
        self.clear_hp_grads()

    def _propagate_attributes(self, src_tensor_list, dst_tensor_list):
        for src_tensor, dst_tensor in zip(src_tensor_list, dst_tensor_list):
            if hasattr(src_tensor, 'model_parallel'):
                dst_tensor.model_parallel = src_tensor.model_parallel
            if hasattr(src_tensor, PIPE_REPLICATED):
                dst_tensor.ds_pipe_replicated = src_tensor.ds_pipe_replicated

    def _split_flat_tensor(self, flat_tensor, num_elem_list):
        assert sum(num_elem_list) <= flat_tensor.numel()
        tensor_list = []
        offset = 0
        for num_elem in num_elem_list:
            dense_tensor = torch.narrow(flat_tensor, 0, offset, num_elem)
            tensor_list.append(dense_tensor)
            offset += num_elem

        return tensor_list

    def _update_storage_to_flattened_tensor(self, tensor_list, flat_tensor):
        updated_params = self.unflatten(flat_tensor, tensor_list)
        for p, q in zip(tensor_list, updated_params):
            p.data = q.data

    def _flatten_dense_tensors_aligned(self, tensor_list, alignment):
        return self.flatten(align_dense_tensors(tensor_list, alignment))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError(f'{self.__class__} does not support closure.')

        params = self.get_fp32_params(filter_nograd=True)
        all_groups_norm = get_grad_norm(parameters=params,
                                        mpu=self.mpu,
                                        norm_type=self.norm_type)
        self._global_grad_norm = all_groups_norm

        assert all_groups_norm > 0.
        if self.clip_grad > 0.:
            clip_gradients(parameters=params,
                           max_norm=self.clip_grad,
                           mpu=self.mpu,
                           global_grad_norm=all_groups_norm)

        self.optimizer.step()

        self.clear_hp_grads()
        self.update_lp_params()

    def get_fp32_params(self, filter_nograd=False):
        params = []
        for group in self.fp32_groups:
            for param in group:
                if filter_nograd and param.grad is not None:
                    params.append(param)
        return params

    def backward(self, loss, update_hp_grads=True, clear_lp_grads=False, **bwd_kwargs):
        """Perform a backward pass and copy the low-precision gradients to the
        high-precision copy.

        We copy/accumulate to the high-precision grads now to prevent accumulating in the
        bf16 grads after successive backward() calls (i.e., grad accumulation steps > 1)

        The low-precision grads are deallocated during this procedure.
        """
        self.clear_lp_grads()
        loss.backward(**bwd_kwargs)

        if update_hp_grads:
            self.update_hp_grads(clear_lp_grads=clear_lp_grads)

    @torch.no_grad()
    def update_hp_grads(self, clear_lp_grads=False):
        for i, group in enumerate(self.bf16_groups):
            for j, (lp, hp) in enumerate(zip(group, self.fp32_groups[i])):
                if lp.grad is None:
                    continue

                assert hp.grad is not None, \
                    f'high precision param has no gradient, param_id = {id(hp)} group_info = [{i}][{j}]'

                hp.grad.data.add_(lp.grad.data.to(hp.dtype).view(hp.shape))
                lp._hp_grad = hp.grad

                # clear gradients
                if clear_lp_grads:
                    lp.grad = None

    @torch.no_grad()
    def get_grads_for_reduction(self):
        return self.fp32_groups_gradients_flat

    @torch.no_grad()
    def update_lp_params(self):
        for i, group in enumerate(self.bf16_groups):
            for lp, hp in zip(group, self.fp32_groups[i]):
                lp.data.copy_(hp.data.to(lp.dtype).view(lp.shape))

    @torch.no_grad()
    def _init_hp_grads(self):
        for i, group in enumerate(self.bf16_groups):
            for j, (lp, hp) in enumerate(zip(group, self.fp32_groups[i])):
                hp.grad = self.fp32_groups_gradients[i][j]

    def clear_hp_grads(self):
        for flat_gradients in self.fp32_groups_gradients_flat:
            flat_gradients.zero_()

    def clear_lp_grads(self):
        for group in self.bf16_groups:
            for param in group:
                param.grad = None

    def state_dict(self):
        state_dict = {}
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict['fp32_groups'] = self.fp32_groups
        state_dict['clip_grad'] = self.clip_grad
        return state_dict

    def load_state_dict(self, state_dict, load_optimizer_states=True):
        if load_optimizer_states:
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.clip_grad = state_dict['clip_grad']

        for i in range(len(self.fp32_groups)):
            for current, saved in zip(self.fp32_groups[i], state_dict['fp32_groups'][i]):
                current.data.copy_(saved.data)

    @property
    def param_groups(self):
        """Forward the wrapped optimizer's parameters."""
        return self.optimizer.param_groups
