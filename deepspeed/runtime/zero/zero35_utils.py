import os

from deepspeed.accelerator import get_accelerator
from deepspeed import comm as dist
from deepspeed.utils import logger
from deepspeed.runtime.swap_tensor.partitioned_param_swapper import PartitionedParamStatus

import torch

global_zero35_manager = None


def zero35_debug(msg, rank=None, force=True):
    if force:
        msg = f"Rank: {os.environ['SLURM_PROCID']}, " + msg
        if rank is None:
            logger.info(msg)
        elif os.environ['SLURM_PROCID'] == str(rank):
            logger.info(msg)


def zero35_g_p_reduce_scatter_coalesced(tensor_list, dp_comm_group, param_comm_group, partition_type):
    # reshape 的逻辑
    # [0, 1, 2, 3, 4, 5, 6, 7]  -> [0, 4, 1, 5, 2, 6, 3, 7]
    # [0, 1, 2, 3, 4, 5, 6, 7]  -> [0, 4, 1, 5, 2, 6, 3, 7]
    # [0, 1, 2, 3, 4, 5, 6, 7]  -> [0, 4, 1, 5, 2, 6, 3, 7]
    # [0, 1, 2, 3, 4, 5, 6, 7]  -> [0, 4, 1, 5, 2, 6, 3, 7]

    do_reshape = partition_type == "grad" or partition_type == "param"
    dtype = tensor_list[0].dtype

    if do_reshape:
        scatter_comm_group = param_comm_group

        dp_world_size = dist.get_world_size(dp_comm_group)
        param_world_size = dist.get_world_size(param_comm_group)

        new_tensor_list = []
        _undo_indexs_for_per_tensor = []
        for grad in tensor_list:
            assert grad.numel() % dp_world_size == 0
            assert grad.numel() % param_world_size == 0

            dp_partition_size = int(grad.numel() / dp_world_size)  # 按照dp范围划分的最小part大小
            param_partition_size = int(grad.numel() / param_world_size)
            assert param_partition_size % dp_partition_size == 0
            param_partition_num = int(param_partition_size / dp_partition_size)  # 每个节点内包含的 dp_partition_size 的数量
            grad = grad.reshape(-1, dp_partition_size)
            indexs = []
            for idx in range(param_world_size):
                for jdx in range(param_partition_num):
                    indexs.append(idx + jdx * param_world_size)

            zero35_debug(f"scatter index : {indexs}")

            indexs=torch.tensor(indexs).to(get_accelerator().device_name())
            _, undo_indices = torch.sort(indexs, dim=0, descending=False)
            _undo_indexs_for_per_tensor.append(undo_indices)

            reshape_grad = torch.index_select(grad, 0, indexs)
            assert reshape_grad.is_contiguous()
            new_tensor_list.append(reshape_grad.view(-1))
        tensor_list = new_tensor_list
    else:
        scatter_comm_group = dp_comm_group

    # if do_reshape:
    #     new_tensor_list = []
    #     for i, grad in enumerate(tensor_list):
    #         new_tensor_list.append(torch.index_select(grad, 0, _undo_indexs_for_per_tensor[i]))
    #     tensor_list = new_tensor_list

    return tensor_list, scatter_comm_group

def zero35_g_p_all_gather_coalesced(tensor_list, dp_comm_group, param_comm_group, partition_type=None):
    pass
    # reshape 的逻辑
    # [0, 4, 1, 5, 2, 6, 3, 7]  -> [0, 1, 2, 3, 4, 5, 6, 7]  
    # [0, 4, 1, 5, 2, 6, 3, 7]  -> [0, 1, 2, 3, 4, 5, 6, 7] 
    # [0, 4, 1, 5, 2, 6, 3, 7]  -> [0, 1, 2, 3, 4, 5, 6, 7] 
    # [0, 4, 1, 5, 2, 6, 3, 7]  -> [0, 1, 2, 3, 4, 5, 6, 7]
    # 
    # [0, , 1, ] 
    # do_reshape = partition_type == "grad" or partition_type == "param"
    dtype = tensor_list[0].dtype

    # if do_reshape:
    all_gather_comm_group = param_comm_group

    dp_world_size = dist.get_world_size(dp_comm_group)
    param_world_size = dist.get_world_size(param_comm_group)

    new_tensor_list = []

    for t_data in tensor_list:
        param_full_tensor = t_data.data
        indexs = []

        partition_unit_size = t_data.numel() // dp_world_size  # 按照dp范围划分的最小part大小

        zero35_debug(f"param_full_tensor.numel() :{t_data.numel()}, dp_world_size:{dp_world_size},partition_unit_size:{partition_unit_size}", flush=True)

        partition_unit_num = t_data.numel() // partition_unit_size

        partition_unit_size_per_rank = t_data.numel() // param_world_size  
        partition_unit_num_per_rank = partition_unit_size_per_rank // partition_unit_size # 2
        partition_unit_num_per_node = partition_unit_num_per_rank * param_world_size    # 2 * 4 -> 8

        param_full_tensor = param_full_tensor.reshape(-1, partition_unit_size)

        for idx in range(partition_unit_num_per_rank): # 8
            indexs.extend([idx + jdx * partition_unit_num_per_rank for jdx in range(param_world_size)])

        zero35_debug(f"gather index : {indexs}")
        # indexs = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]
        indexs=torch.tensor(indexs).to(get_accelerator().device_name())
        reshape_t_data = torch.index_select(param_full_tensor, 0, indexs)
        reshape_t_data = reshape_t_data.view(-1)
        assert reshape_t_data.is_contiguous()

        # param_full_tensor.ds_tensor = reshape_t_data
        t_data.data = reshape_t_data

        new_tensor_list.append(reshape_t_data)
    tensor_list = new_tensor_list
    # else:
    #     all_gather_comm_group = dp_comm_group

    return tensor_list, all_gather_comm_group


class GlobalZero35GroupManager:
    def __init__(self, enable_zero35, mpu=None) -> None:
        # TODO, zero35 不能和tp或pp一起使用
        self.enable_zero35 = enable_zero35
        self._dp_process_group = dist.get_world_group()
        self.world_size = dist.get_world_size(self._dp_process_group)

        if mpu is not None:
            self.model_parallel_group = mpu.get_model_parallel_group()
            self.model_parallel_rank = mpu.get_model_parallel_rank()
            self.mode_parallel_size = dist.get_world_size(self.model_parallel_group)
            self.data_parallel_size = self.world_size // self.mode_parallel_size
        else:
            self.mode_parallel_size = 1
            self.model_parallel_group = None
            self.model_parallel_rank = 0
            self.data_parallel_size = 1

        assert self.mode_parallel_size == 1, "zero35现在不支持模型并行"

        self.zero35_parallel_size = 8 # TODO: 将这个变成可配置参数
        self.rank_num_per_dp_group = self.world_size // self.data_parallel_size
        self.num_zero35_parallel_group = self.data_parallel_size // self.zero35_parallel_size

        if self.enable_zero35:
            self.zero35_group = self.init_zero35_process_group()
            self._grad_process_group = self.zero35_group
            self._param_process_group = self.zero35_group

            self._param_rank = dist.get_rank(group=self._param_process_group)
            self._param_world_size = dist.get_world_size(group=self._param_process_group)

            self._grad_rank = dist.get_rank(group=self._grad_process_group)
            self._grad_world_size = dist.get_world_size(group=self._grad_process_group)

        else:
            self._grad_process_group = self._dp_process_group
            self._param_process_group = self._dp_process_group
            
            self._param_rank = 0
            self._param_world_size = 1
            self._grad_rank = 0
            self._grad_world_size = 1


    def get_partition_dp_group(self, param, partition_type):
        """ Return the communication group with all data-parallel ranks """
        if partition_type == "os":
            return param.dp_process_group
        elif partition_type == "grad":
            return param.grad_process_group
        elif partition_type == "param":
            return param.param_process_group
        else:
            assert False, f"unkonwn partition_type: {partition_type}"

    def get_partition_rank(self, partition_type):
        """subclass can overload to specify different relative rank in
        parameter partition group"""
        if partition_type == "os":
            return self._os_rank
        elif partition_type == "grad":
            return self._grad_rank
        elif partition_type == "param":
            return self._param_rank
        else:
            assert False, f"unkonwn partition_type: {partition_type}"

    # @property
    def get_partition_count(self, partition_type):
        return dist.get_world_size(group=self.get_dp_process_group(partition_type))

    def get_dp_process_group(self, partition_type):
        """ Return the communication group with all data-parallel ranks """
        if partition_type == "os":
            return self._dp_process_group
        elif partition_type == "grad":
            return self._grad_process_group
        elif partition_type == "param":
            return self._param_process_group
        else:
            assert False, f"unkonwn partition_type: {partition_type}"

    def get_rank_in_group(self, partition_type):
        if partition_type == "os":
            return self._os_rank
        elif partition_type == "grad":
            return self._grad_rank
        elif partition_type == "param":
            return self._param_rank
        else:
            assert False, f"unkonwn partition_type: {partition_type}"

    def get_world_size(self, partition_type):
        return dist.get_world_size(self.get_dp_process_group(partition_type))

    def get_partition_unit_size(self, param):
        dp_partition_count = self.get_partition_count("os")    
        assert param.ds_numel % dp_partition_count == 0
        return param.ds_numel // dp_partition_count

    def init_zero35_process_group(self):
        my_zero35_group = None
        for i in range(self.mode_parallel_size):
            for j in range(self.num_zero35_parallel_group):
                ranks = [
                    i + (j * self.zero35_parallel_size + k) * self.mode_parallel_size
                    for k in range(self.zero35_parallel_size)
                ]
                group = dist.new_group(ranks)

                if dist.get_rank() in ranks:
                    my_zero35_group = group
                    zero35_debug(ranks, force=True)
        return my_zero35_group

    def get_sub_p_g_parition(self, ds_param, grad=None):
        assert hasattr(ds_param, 'ds_numel'), 'get_sub_p_g_parition input must be ds_param'
        if self._enable_zero35:
            zero35_rank = dist.get_rank() // self.zero35_parallel_size  # TODO, remove hard code
            partition_unit_size = self.get_partition_unit_size(ds_param.ds_numel)
            if grad is not None:
                reshape_grad = grad.reshape(-1, partition_unit_size)[zero35_rank]
                assert reshape_grad.storage().data_ptr() == grad.storage().data_ptr()
                return reshape_grad
            else:
                reshape_param = ds_param.ds_tensor.reshape(-1, partition_unit_size)[zero35_rank]
                assert reshape_param.storage().data_ptr() == ds_param.ds_tensor.storage().data_ptr()
                return reshape_param
        else:
            if grad is not None:
                return grad
            else:
                return ds_param.ds_tensor
            
    def zero35_judge_gahter_boundary(self, mico_step, forward):
        return mico_step == 0 and forward
            
    def zero35_hack_allgahter_ds_tensor(self, param, mico_step, forward):

        if self.zero35_judge_gahter_boundary(mico_step, forward):
            # gather boundary
            partition_type = "os"

            assert hasattr(param, 'ds_numel'), 'zero35_hack_allgahter_ds_tensor input must be ds_param'
            parition_num = self.get_world_size(partition_type)

            node_id =  dist.get_rank() // self.zero35_parallel_size
            partition_unit_size = param.ds_numel // parition_num
            param_ds_tensor = param.ds_tensor.view(-1, partition_unit_size)

            # backup ds_tensor
            param.ds_numel_backup = param.ds_tensor.ds_numel
            param.ds_tensor_backup = param.ds_tensor.data

            # hack ds_tensor
            param.ds_tensor = param_ds_tensor[node_id]
            assert param.ds_tensor.storage().data_ptr() == param_ds_tensor.storage().data_ptr()

            param.ds_tensor.ds_numel = partition_unit_size
            param.ds_tensor.status = PartitionedParamStatus.AVAILABLE
            param.ds_tensor.final_location = None
            param.ds_tensor.is_first_fwd_all_gahter = True

            zero35_debug(f"zero35_hack_allgahter_ds_tensor DEBUG: mico_step: {mico_step}, forward:{forward}, param.ds_numel : {param.ds_numel}, get : {param_ds_tensor}, partition_type:{partition_type}, partition_unit_size:{partition_unit_size}", flush=True)
        else:
            partition_type = "param"
            partition_unit_size = param.ds_tensor.ds_numel
            zero35_debug(f"zero35_hack_allgahter_ds_tensor DEBUG: mico_step: {mico_step}, forward:{forward}, SKIP hack, partition_unit_size:{partition_unit_size}", flush=True)

        return partition_unit_size
