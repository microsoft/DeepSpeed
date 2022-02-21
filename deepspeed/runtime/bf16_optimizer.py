import torch
import torch.distributed as dist
from deepspeed.ops.op_builder import UtilsBuilder

from deepspeed.runtime.utils import (get_grad_norm, clip_gradients, align_dense_tensors)


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
        self.fp32_groups = []
        self.single_partition_of_fp32_groups = []
        data_parallel_world_size = dist.get_world_size(group=self.dp_process_group)

        for i, param_group in enumerate(self.optimizer.param_groups):
            # grab the original list
            self.bf16_groups.append(param_group['params'])
            self.bf16_groups_flat.append(
                self._flatten_dense_tensors_aligned(
                    self.bf16_groups[i],
                    self.nccl_start_alignment_factor *
                    dist.get_world_size(group=self.real_dp_process_group[i])))

            #            self.single_partition_of_fp32_groups.append(self.bf16_groups_flat[i].clone().float().detach())
            #            self.single_partition_of_fp32_groups[i].requires_grad = True

            #            param_group['params'] = [self.single_partition_of_fp32_groups[i]]
            #            self._unflatten_dense_tensors(self.bf16_groups[i], self.bf16_groups_flat[i])

            fp32_group = [p.clone().float().detach() for p in param_group['params']]
            for p in fp32_group:
                p.requires_grad = True

            # Ensure model parallel attributes are carried over
            for lp, hp in zip(param_group['params'], fp32_group):
                if hasattr(lp, 'model_parallel'):
                    hp.model_parallel = lp.model_parallel
                if hasattr(lp, '_pipe_replicated'):
                    hp._pipe_replicated = lp._pipe_replicated

            self.fp32_groups.append(fp32_group)
            param_group['params'] = self.fp32_groups[i]

        self.initialize_optimizer_states()

    def initialize_optimizer_states(self):
        """Take an optimizer step with zero-valued gradients to allocate internal
        optimizer state.

        This helps prevent memory fragmentation by allocating optimizer state at the
        beginning of training instead of after activations have been allocated.
        """
        for group in self.fp32_groups:
            for param in group:
                param.grad = torch.zeros(param.size(),
                                         device=param.device,
                                         dtype=param.dtype)

        self.optimizer.step()
        self.clear_hp_grads()

    def _unflatten_dense_tensors(self, tensor_list, flattened_tensors):
        updated_params = self.unflatten(flattened_tensors, tensor_list)
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
            for lp, hp in zip(group, self.fp32_groups[i]):
                if lp.grad is None:
                    continue

                data_type = hp.dtype

                if hp.grad is None:
                    hp.grad = lp.grad.to(data_type)
                    # give the model parameter access to the hp grad as well
                    lp._hp_grad = hp.grad
                else:
                    hp.grad.data.add_(lp.grad.data.to(data_type))

                # clear gradients
                if clear_lp_grads:
                    lp.grad = None

    def update_lp_params(self):
        for i, group in enumerate(self.bf16_groups):
            for lp, hp in zip(group, self.fp32_groups[i]):
                lp.data.copy_(hp.data.to(lp.dtype))

    def clear_hp_grads(self):
        for group in self.fp32_groups:
            for param in group:
                param.grad = None

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
