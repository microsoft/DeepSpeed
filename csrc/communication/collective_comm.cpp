#include <torch/extension.h>

#include <nccl.h>
#include <c10d/ProcessGroup.hpp>
#include <c10d/ProcessGroupNCCL.hpp>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

int debug_flag = std::getenv("DS_DEBUG") ? std::stoi(std::getenv("DS_DEBUG")) : 0;

// recording created ncclComm_t
// using processGroup Name as key
std::unordered_map<std::string, ncclComm_t> group_communicators;

// NCCL type typing
// copied from pytorch source code
std::map<at::ScalarType, ncclDataType_t> ncclDataType = {
    {at::kChar, ncclInt8},
    {at::kByte, ncclUint8},
    {at::kFloat, ncclFloat},
    {at::kDouble, ncclDouble},
    {at::kInt, ncclInt32},
    {at::kLong, ncclInt64},
    {at::kHalf, ncclHalf},
    {at::kBool, ncclUint8},
#if defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 301
    {at::kBFloat16, ncclBfloat16},
#endif
};

// Helper function that gets the data type and issues error if not supported
// from pytorch source code
ncclDataType_t getNcclDataType(at::ScalarType type)
{
    auto it = ncclDataType.find(type);
    TORCH_CHECK(it != ncclDataType.end(),
                "Input tensor data type is not supported for NCCL process group: ",
                type);
    return it->second;
}

void check_tensors(std::vector<at::Tensor>& output_tensors,
                   std::vector<at::Tensor>& input_tensors,
                   int world_size)
{
    if (input_tensors.size() == 0 || output_tensors.size() == 0) {
        TORCH_CHECK(false, "output/input tensor list must be nonempty");
    }
    if (output_tensors.size() != input_tensors.size()) {
        TORCH_CHECK(false, "output and input tensors must have same size");
    }

    for (size_t i = 0; i < input_tensors.size(); ++i) {
        auto out = output_tensors[i];
        auto in = input_tensors[i];
        if (out.numel() != in.numel() * world_size) {
            std::stringstream ss;
            ss << "output tensor numel != input tensor numel * world_size at" << i;
            TORCH_CHECK(false, ss.str());
        }
    }
}

// rank0 create the ncclUniqueId
// broadcast using old ProcessGroupNCCL
// ncclCommInitRank with ncclUniqueId and same rank and world size from current
// ProcessGroupNCCL
//
// Note: reason for creating new ncclComm_t, ::c10d::ProcessGroupNCCL didn't expose
// APIs for getting communicator
ncclComm_t create_communicator(std::vector<at::Tensor>& input_tensors,
                               std::string& pg_name,
                               ::c10d::ProcessGroupNCCL& pg)
{
    int rank = pg.getRank();
    int world_size = pg.getSize();
    at::Tensor& first_tensor = input_tensors[0];
    auto device_idx = first_tensor.get_device();
    if (debug_flag) printf("creating new communicator at device %ld\n", device_idx);

    //
    ncclUniqueId nccl_id;
    ncclComm_t nccl_comm;

    auto id_tensor_option = torch::TensorOptions()
                                .dtype(torch::kUInt8)
                                .layout(torch::kStrided)  // dense tensor
                                .requires_grad(false);

    std::vector<at::Tensor> bcast_tensor;
    if (rank == 0) {
        auto _result = ncclGetUniqueId(&nccl_id);
        if (_result != ncclSuccess) {
            TORCH_CHECK(false, "Getting nccl unique id failed");
            // it suppose to exit
        }
        id_tensor_option.device(torch::kCPU);
        at::Tensor cpu_tensor = torch::empty(sizeof(ncclUniqueId), id_tensor_option).zero_();
        memcpy(cpu_tensor.data_ptr(), &nccl_id, sizeof(ncclUniqueId));

        at::Tensor id_tensor = cpu_tensor.to(first_tensor.device());
        bcast_tensor.push_back(std::move(id_tensor));
    } else {
        at::Tensor id_tensor =
            torch::empty(sizeof(ncclUniqueId), id_tensor_option).zero_().to(first_tensor.device());
        bcast_tensor.push_back(std::move(id_tensor));
    }
    if (debug_flag)
        printf("rank %d, created tensor holder, device %ld, is_cuda %d \n",
               rank,
               device_idx,
               bcast_tensor[0].is_cuda());

    // bcast
    {
        at::cuda::CUDAGuard gpuGuard(device_idx);
        // make sure the allocated tensors are ready
        AT_CUDA_CHECK(cudaDeviceSynchronize());
        auto work = pg.broadcast(bcast_tensor);
        // make sure the broadcast finished
        AT_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // if rank != 0
    // then need to copy ncclUniqueId from bcast_tensor
    if (rank != 0) {
        auto cpu_tensor = bcast_tensor[0].to(at::kCPU);
        std::memcpy(&nccl_id, cpu_tensor.data_ptr(), cpu_tensor.nbytes());
    }

    {
        at::cuda::CUDAGuard gpuGuard(device_idx);
        // init communicator and save
        ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank);
        group_communicators[pg_name] = nccl_comm;

        if (debug_flag) printf("nccl_comm initialized at rank %d, device %ld\n", rank, device_idx);
    }

    return nccl_comm;
}

// get communicator from global map
// if not found, create a new one
ncclComm_t get_communicator(std::vector<at::Tensor>& input_tensors,
                            std::string& pg_name,
                            ::c10d::ProcessGroupNCCL& pg)
{
    auto found = group_communicators.find(pg_name);
    if (found == group_communicators.end()) {
        return create_communicator(input_tensors, pg_name, pg);
    } else {
        return found->second;
    }
}

int launch_nccl_allgather(std::vector<at::Tensor>& output_tensors,
                          std::vector<at::Tensor>& input_tensors,
                          ncclComm_t comm)
{
    auto& first_input = input_tensors[0];
    auto device_idx = first_input.get_device();
    if (debug_flag)
        printf("launching allgather op with number of tensors %lu, at device %ld \n",
               input_tensors.size(),
               device_idx);

    // this suppose to get the cuda stream specified by `with torch.cuda.stream(comm_stream): ...`
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(device_idx);

    ncclGroupStart();
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        at::Tensor& input = input_tensors[i];
        at::Tensor& output = output_tensors[i];
        ncclAllGather(input.data_ptr(),
                      output.data_ptr(),
                      input.numel(),
                      getNcclDataType(input.scalar_type()),
                      comm,
                      stream.stream());
    }
    ncclGroupEnd();

    return 0;
}

int inplaceAllgather(std::vector<at::Tensor>& output_tensors,
                     std::vector<at::Tensor>& input_tensors,
                     ::c10d::ProcessGroupNCCL& pg,
                     std::string pg_name)
{
    // ::c10d::ProcessGroup& p_pg = pg;
    if (debug_flag)
        printf("inplaceAllgather:: process group rank %d, size %d, pg_name %s \n",
               pg.getRank(),
               pg.getSize(),
               pg_name.c_str());

    check_tensors(output_tensors, input_tensors, pg.getSize());

    auto nccl_comm = get_communicator(input_tensors, pg_name, pg);

    int res = launch_nccl_allgather(output_tensors, input_tensors, nccl_comm);

    return res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("_inplace_allgather", &inplaceAllgather, "inplace all-gather (without memcpy)");
}
