import torch
import torch.distributed as dist
from deepspeed.runtime.constants import PIPE_REPLICATED
from deepspeed.ops.op_builder import UtilsBuilder

from deepspeed.runtime.utils import (get_global_norm_of_tensors,
                                     clip_tensors_by_global_norm,
                                     get_grad_norm,
                                     clip_gradients,
                                     align_dense_tensors,
                                     all_gather_dp_groups,
                                     see_memory_usage)


class BF16_Optimizer:
    def __init__(self,
                 init_optimizer,
                 mpu=None,
                 clip_grad=0.0,
                 norm_type=2,
                 allgather_bucket_size=5000000000,
                 dp_process_group=None,
                 timers=None):
        super().__init__()
        see_memory_usage('begin bf16_optimizer', force=True)
        self.timers = timers
        self.optimizer = init_optimizer
        self.clip_grad = clip_grad
        self.norm_type = norm_type
        self.mpu = mpu
        self.allgather_bucket_size = int(allgather_bucket_size)
        self.dp_process_group = dp_process_group
        self.dp_rank = dist.get_rank(group=self.dp_process_group)
        self.real_dp_process_group = [
            dp_process_group for i in range(len(self.optimizer.param_groups))
        ]

        # Load pre-built or JIT compile (un)flatten ops
        util_ops = UtilsBuilder().load()
        self.flatten = util_ops.flatten
        self.unflatten = util_ops.unflatten

        #align nccl all-gather send buffers to 4-bye boundary
        self.nccl_start_alignment_factor = 2  # 4-byte alignment/sizeof(fp16) = 2

        # Build BF16/FP32 groups
        self.bf16_groups = []
        self.bf16_groups_flat = []
        self.bf16_partitioned_groups = []

        # TODO: Need to only track fp32 params of this partition
        self.fp32_groups = []
        self.fp32_groups_flat = []
        self.fp32_groups_flat_partition = []

        # Maintain different fp32 gradients views for convenience
        self.fp32_groups_gradients = []
        self.fp32_groups_gradients_flat = []
        self.fp32_groups_actual_gradients_flat = []
        self.fp32_groups_gradient_flat_partition = []

        dp_world_size = dist.get_world_size(group=self.dp_process_group)

        for i, param_group in enumerate(self.optimizer.param_groups):
            see_memory_usage(f'before initializing group {i}', force=True)

            partition_id = dist.get_rank(group=self.real_dp_process_group[i])

            # grab the original list
            self.bf16_groups.append(param_group['params'])

            # create flat bf16 params
            self.bf16_groups_flat.append(
                self._flatten_dense_tensors_aligned(
                    self.bf16_groups[i],
                    self.nccl_start_alignment_factor * dp_world_size))

            # divide flat weights into equal sized partitions
            partition_size = self.bf16_groups_flat[i].numel() // dp_world_size
            bf16_dp_partitions = [
                self.bf16_groups_flat[i].narrow(0,
                                                dp_index * partition_size,
                                                partition_size)
                for dp_index in range(dp_world_size)
            ]
            self.bf16_partitioned_groups.append(bf16_dp_partitions)

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

            # flat tensor corresponding to actual fp32 gradients
            length_without_padding = sum(num_elem_list)
            self.fp32_groups_actual_gradients_flat.append(
                torch.narrow(self.fp32_groups_gradients_flat[i],
                             0,
                             0,
                             length_without_padding))

            # flat tensor corresponding to gradient partition
            self.fp32_groups_gradient_flat_partition.append(
                torch.narrow(self.fp32_groups_gradients_flat[i],
                             0,
                             partition_id * partition_size,
                             partition_size))

            # create fp32 partition from flat tensor storage
            assert self.fp32_groups_flat[i].numel() % dp_world_size == 0, \
            f'group {i} flat tensor size {self.fp32_groups_flat[i].numel()} not divisible by DP world size {dp_world_size}'

            self.fp32_groups_flat_partition.append(
                torch.narrow(self.fp32_groups_flat[i],
                             0,
                             self.dp_rank * partition_size,
                             partition_size))

            param_group['params'] = [self.fp32_groups_flat_partition[i]]
            see_memory_usage(f'after initializing group {i}', force=True)

        see_memory_usage('before initialize_optimizer', force=True)
        self.initialize_optimizer_states()
        see_memory_usage('end initialize_optimizer', force=True)

        see_memory_usage('end bf16_optimizer', force=True)

    def initialize_optimizer_states(self):
        """Take an optimizer step with zero-valued gradients to allocate internal
        optimizer state.

        This helps prevent memory fragmentation by allocating optimizer state at the
        beginning of training instead of after activations have been allocated.
        """
        for param_partition, grad_partition in zip(self.fp32_groups_flat_partition, self.fp32_groups_gradient_flat_partition):
            param_partition.grad = grad_partition

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

        all_groups_norm = get_global_norm_of_tensors(
            input_tensors=self.get_grads_for_norm(),
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

        self.update_lp_params()

        all_gather_dp_groups(partitioned_param_groups=self.bf16_partitioned_groups,
                             dp_process_group=self.real_dp_process_group,
                             start_alignment_factor=self.nccl_start_alignment_factor,
                             allgather_bucket_size=self.allgather_bucket_size)

        self.clear_hp_grads()

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
            for j, lp in enumerate(group):
                if lp.grad is None:
                    continue

                hp_grad = self.fp32_groups_gradients[i][j]
                assert hp_grad is not None, \
                    f'high precision param has no gradient, lp param_id = {id(lp)} group_info = [{i}][{j}]'

                hp_grad.data.add_(lp.grad.data.to(hp_grad.dtype).view(hp_grad.shape))
                lp._hp_grad = hp_grad

                # clear gradients
                if clear_lp_grads:
                    lp.grad = None

    @torch.no_grad()
    def get_grads_for_reduction(self):
        return self.fp32_groups_gradients_flat

    @torch.no_grad()
    def get_grads_for_norm(self):
        return self.fp32_groups_actual_gradients_flat

    @torch.no_grad()
    def update_lp_params(self):
        for i, group in enumerate(self.bf16_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            for bf16_partitions, fp32_partition in zip(self.bf16_partitioned_groups, self.fp32_groups_flat_partition):
                bf16_partitions[partition_id].data.copy_(fp32_partition.data)

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
