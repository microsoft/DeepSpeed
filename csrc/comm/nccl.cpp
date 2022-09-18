#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <mpi.h>
#include <nccl.h>
#include <pybind11/embed.h>
#include <torch/extension.h>
#include <chrono>
namespace py = pybind11;

#include <c10/util/irange.h>

#include <iostream>
#include <string>

#include <comm.h>

// TODO: remove
#include <stdio.h>

namespace nccl {

void create_comm_group(std::vector<int> comm_ranks, int rank, int comm_id, int color);
ncclComm_t _get_comm_from_group(py::object group);

cudaStream_t s;
ncclComm_t _world_nccl_comm;

// py::module_ dist = py::module_::import("deepspeed.comm");

std::vector<MPI_Comm> global_mpi_comms;
std::vector<ncclComm_t> global_nccl_comms;
std::vector<cudaStream_t> global_streams;

// REZA+AMMAR CODE
// curandGenerator_t _gen;
// cublasHandle_t _cublasHandle;
cudaEvent_t _comp_event;
cudaEvent_t _comm_event;
void* _workspace;
uint64_t _seed;
uint64_t _curr_offset;
size_t _workSpaceSize;
unsigned _token_length;
unsigned _num_tokens;
std::vector<std::array<int, 3>> _gemm_algos;
cudaStream_t _comp_stream = at::cuda::getDefaultCUDAStream();
cudaStream_t _comm_stream;
MPI_Group _group;
std::unordered_map<int, ncclComm_t> _nccl_comms;
std::unordered_map<int, int> _world_sizes;
std::set<int> _comm_ids;
std::set<int> _colors;
std::unordered_map<int, int> _color_map;
MPI_Comm _comm;
bool _comm_created;
// py::object ProcessGroup = py::module_::import("deepspeed.comm").attr("ProcessGroup");
// py::object world_group;

int get_rank(int group = 0)
{
    int world_rank;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    return world_rank;
}

int get_world_size(int group = 0)
{
    int world_size;
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    return world_size;
}

// Given a ncclUniqueId, convert it to a string representation that can be put
// in the store.
std::string buildNcclUniqueIdStr(const ncclUniqueId& ncclID)
{
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&ncclID);
    std::ostringstream oss;
    for (const auto i : c10::irange(NCCL_UNIQUE_ID_BYTES)) {
        oss << std::hex << static_cast<int>(bytes[i]);
    }
    return oss.str();
}

std::string getNcclId()
{
    ncclUniqueId ncclID;
    NCCLCHECK(ncclGetUniqueId(&ncclID));
    return buildNcclUniqueIdStr(ncclID);

    // std::string id = "hello";
    // for (int i=0; i<128; i++)
    //     std::cout << "ncclID =" << ncclID[i];
    // std::cout<< std::endl;
    // return id;
}

void barrier() { MPICHECK(MPI_Barrier(MPI_COMM_WORLD)); }

void create_comms()
{
    ncclUniqueId ncclID;
    int world_rank = get_rank(0);
    int world_size = get_world_size(0);
    int ngpus;

    CUDACHECK(cudaGetDeviceCount(&ngpus));

    CUDACHECK(cudaSetDevice(world_rank % ngpus));
    // CUDACHECK(cudaStreamCreate(&s));
    CUDACHECK(cudaStreamCreateWithPriority(&_comm_stream, cudaStreamNonBlocking, -1));
    // std::vector<int> ranks(world_size);
    // std::iota(ranks.begin(), ranks.end(), 0);
    if (world_rank == 0) { ncclGetUniqueId(&ncclID); }
    MPICHECK(MPI_Bcast(&ncclID, sizeof(ncclID), MPI_BYTE, 0, MPI_COMM_WORLD));

    NCCLCHECK(ncclCommInitRank(&_world_nccl_comm, world_size, ncclID, world_rank));
    _nccl_comms[0] = _world_nccl_comm;
    // Create the world group
    // py::object ProcessGroup = py::module_::import("deepspeed.comm").attr("ProcessGroup");
    // py::object world_group;
    // world_group = py::none();
    // world_group = ProcessGroup(0, ranks);
    // std::cout << "RANK: " << get_rank() << " COMM_ID: " << py::int_(world_group.attr("comm_id"))
    // << std::endl; world_group.attr("ranks") = ranks;
    // NCCLCHECK(ncclCommDestroy(_world_nccl_comm));
}

py::object get_world_group()
{
    int world_size = get_world_size(0);
    std::vector<int> ranks(world_size);
    std::iota(ranks.begin(), ranks.end(), 0);
    py::object ProcessGroup = py::module_::import("deepspeed.comm").attr("ProcessGroup");
    return ProcessGroup(0, ranks);
}

void _print_comm_number() { std::cout << "Number of Sub-Comms:" << _nccl_comms.size() + 1 << "\n"; }

void initialize(int rank, int size)
{
    create_comms();
    cudaEventCreate(&_comp_event, (cudaEventDisableTiming | cudaEventBlockingSync));
    cudaEventCreate(&_comm_event, (cudaEventDisableTiming | cudaEventBlockingSync));
}

inline void SynchComp()
{
    cudaEventRecord(_comp_event, _comp_stream);
    cudaStreamWaitEvent(_comm_stream, _comp_event, 0);
}
inline void SynchComm()
{
    cudaEventRecord(_comm_event, _comm_stream);
    cudaStreamWaitEvent(_comp_stream, _comm_event, 0);
}

cudaStream_t GetCommStream(bool async_op = false)
{
    return _comm_stream;
    // if (!_comm_stream)
    //     _comm_stream = async_op ? at::cuda::getStreamFromPool(true)
    //                             : at::cuda::getCurrentCUDAStream();
    // return _comm_stream;
}

void finalize() { NCCLCHECK(ncclCommDestroy(_world_nccl_comm)); }

ncclDataType_t get_nccl_datatype(c10::ScalarType type)
{
    ncclDataType_t nccl_type;
    switch (type) {
        case c10::ScalarType::Int: nccl_type = ncclInt; break;
        case c10::ScalarType::Float: nccl_type = ncclFloat; break;
        case c10::ScalarType::Double: nccl_type = ncclDouble; break;
        default: nccl_type = ncclChar;
    }
    return nccl_type;
}

ncclRedOp_t get_nccl_reduce_op(py::object op, at::Tensor& input)
{
    py::object ReduceOp = py::module_::import("deepspeed.comm").attr("ReduceOp");
    if (!py::isinstance(op, ReduceOp)) {
        throw std::runtime_error("Error: Op must be of type ReduceOp");
    }

    int op_val = py::int_(op.attr("value"));
    ncclRedOp_t nccl_op;

    if (input.scalar_type() == at::kBool) {
        if (op_val == (int)py::int_(ReduceOp.attr("SUM").attr("value"))) {
            // For bool tensors, map sum to max, which both represent a bitwise or.
            // This is to prevent overflow issues with sum, since we use uint8 to
            // represent a bool (see ncclDataType mapping).
            nccl_op = ncclMax;
        } else if (op_val == (int)py::int_(ReduceOp.attr("AVG").attr("value"))) {
            throw std::runtime_error("Error: For bool tensors, op must be of type ReduceOp");
        }
    }

    if (op_val == (int)py::int_(ReduceOp.attr("SUM").attr("value"))) {
        nccl_op = ncclSum;
    } else if (op_val == (int)py::int_(ReduceOp.attr("MIN").attr("value"))) {
        nccl_op = ncclMin;
    } else if (op_val == (int)py::int_(ReduceOp.attr("MAX").attr("value"))) {
        nccl_op = ncclMax;
    } else if (op_val == (int)py::int_(ReduceOp.attr("PRODUCT").attr("value"))) {
        nccl_op = ncclProd;
        //} else if (op_val == (int)py::int_(ReduceOp.attr("AVERAGE").attr("value"))) {
        //    nccl_op = ncclAvg;
    } else {
        throw std::runtime_error("Error: Unrecognized ReduceOp type");
    }
    return nccl_op;
}

void send(torch::Tensor data, int rank, int tag, bool block, py::object group, bool async_op)
{
    ncclComm_t comm = _get_comm_from_group(group);
    NCCLCHECK(ncclSend(data.data_ptr(),
                       data.numel(),
                       get_nccl_datatype(data.scalar_type()),
                       rank,
                       comm,
                       GetCommStream(async_op)));
    if (block) { CUDACHECK(cudaStreamSynchronize(GetCommStream(async_op))); }
    if (async_op) { SynchComm(); }
}

void recv(torch::Tensor data, int rank, int tag, bool block, py::object group, bool async_op)
{
    ncclComm_t comm = _get_comm_from_group(group);
    NCCLCHECK(ncclRecv(data.data_ptr(),
                       data.numel(),
                       get_nccl_datatype(data.scalar_type()),
                       rank,
                       comm,
                       GetCommStream(async_op)));
    if (block) { CUDACHECK(cudaStreamSynchronize(GetCommStream(async_op))); }
    if (async_op) { SynchComm(); }
}

void all_reduce(torch::Tensor& data, py::object op, bool block, py::object group, bool async_op)
{
    ncclComm_t comm = _get_comm_from_group(group);
    NCCLCHECK(ncclAllReduce(data.data_ptr(),
                            data.data_ptr(),
                            data.numel(),
                            get_nccl_datatype(data.scalar_type()),
                            get_nccl_reduce_op(op, data),
                            comm,
                            GetCommStream(async_op)));
    if (block) { CUDACHECK(cudaStreamSynchronize(GetCommStream(async_op))); }
    if (!async_op) { SynchComm(); }
}

inline ncclComm_t GetNCCLComm(int comm_id = 0) { return _nccl_comms[comm_id]; }

void create_comm_group(std::vector<int> comm_ranks, int rank, int comm_id, int color)
{
    // printf("creating comm : size: %d , comm_id: %d , color: %d\n", comm_ranks.size(),
    // comm_id, color);

    // If we have a global communicator, destroy it
    // if (rank == 0 && !_comm_created) {
    //    NCCLCHECK(ncclCommDestroy(ncclcomm));
    //}
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    ncclComm_t _nccl_comm;
    MPI_Comm _comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &_comm);
    MPI_Comm_group(_comm, &_group);
    unsigned num_ranks = comm_ranks.size();
    MPI_Comm _newcomm;
    // printf("*************** number of ranks: %d, world size: %d ****************\n",
    // num_ranks, world_size);
    if (num_ranks < world_size) {
        auto total_group = _group;
        MPI_Group_incl(total_group, num_ranks, comm_ranks.data(), &_group);
        MPI_Comm_split(_comm, color, 0, &_newcomm);
        int local_world_rank, local_world_size;
        MPI_Comm_rank(_newcomm, &local_world_rank);
        MPI_Comm_size(_newcomm, &local_world_size);
        // printf("************ CPP %d , %d, \t %d, %d **************\n",
        // local_world_rank,local_world_size, world_rank, world_size);
        // MPI_Group_free(&total_group);
    } else if (num_ranks > world_size) {
        auto message =
            std::string("Fail to create comm group (number of ranks is higher than world_size).");
        std::cerr << message << std::endl;
        throw std::runtime_error(message);
    }
    ncclUniqueId _nccl_uid;
    if (rank == comm_ranks[0]) { ncclGetUniqueId(&_nccl_uid); }
    MPI_Bcast((void*)&_nccl_uid,
              sizeof(ncclUniqueId),
              MPI_BYTE,
              comm_ranks[0],
              num_ranks < world_size ? _newcomm : _comm);
    if (std::find(comm_ranks.begin(), comm_ranks.end(), rank) != comm_ranks.end()) {
        ncclCommInitRank(&_nccl_comm, num_ranks, _nccl_uid, rank % num_ranks);
    }
    // std::cout << "nccl comm = " << _nccl_comm << std::endl;
    _comm_created = true;
    _world_sizes[comm_id] = num_ranks;
    _nccl_comms[comm_id] = _nccl_comm;
    _color_map[comm_id] = color;
    _comm_ids.insert(comm_id);
    _colors.insert(color);
}

// Find the next ordered, unique value to a set. E.g. <0,1,2,7> --> 3
int next_unique_val(std::set<int> s)
{
    // std::cout << "GETTING CALLED" << std::endl;
    std::set<int>::iterator itr;
    // Base case. Add 0 to start of set.
    if (s.empty() || *s.begin() != 0) {
        return 0;
        // second base case where s = {0} (the case of s = {n != 0} is caught above)
    } else if (s.size() == 1) {
        return 1;
    } else {
        int prev_val = *s.begin();
        for (itr = std::next(s.begin()); itr != s.end(); itr++) {
            if (*itr != prev_val + 1) { return prev_val + 1; }
            prev_val = *itr;
        }
        return *(s.end()) + 1;
    }
}

void test_set()
{
    std::set<int> val1 = {6, 5, 10, 1};
    std::set<int> val2 = {};
    std::set<int> val3 = {0};
    std::set<int> val4 = {0, 1, 2, 3, 6, 4};
    // std::cout << next_unique_val(val1) << " " << next_unique_val(val2) << " " <<
    // next_unique_val(val3) << " " << next_unique_val(val4) << std::endl;
    if (get_rank() == 0) { std::cout << next_unique_val(val4) << std::endl; }
}

py::object new_group(std::vector<int> ranks)
{
    // std::cout << "RANK: " << get_rank() << " COMM_ID: " << comm_id << " COLOR: " << color <<
    // std::endl;
    int comm_id = next_unique_val(_comm_ids);
    int color = next_unique_val(_colors);
    create_comm_group(ranks, get_rank(), comm_id, color);
    py::object ProcessGroup = py::module_::import("deepspeed.comm").attr("ProcessGroup");
    py::object newPG = ProcessGroup(comm_id, ranks);
    return newPG;
}

ncclComm_t _get_comm_from_group(py::object group)
{
    ncclComm_t comm;
    if (group == Py_None) {
        comm = _nccl_comms[0];
    } else {
        py::object ProcessGroup = py::module_::import("deepspeed.comm").attr("ProcessGroup");
        if (!py::isinstance(group, ProcessGroup)) {
            throw std::runtime_error("Error: group must be of type ProcessGroup");
        }
        comm = GetNCCLComm(py::int_(group.attr("comm_id")));
    }
    return comm;
    // return _nccl_comms[0];
}

void all_gather_base(torch::Tensor& output,
                     torch::Tensor& input,
                     bool block,
                     py::object group,
                     bool async_op)
{
    // void* sendbuff = data.data_ptr();
    // torch::Tensor recvbuf = torch::empty_like(data);
    ncclComm_t comm = _get_comm_from_group(group);
    // ncclComm_t comm = _world_nccl_comm;
    NCCLCHECK(ncclAllGather(input.data_ptr(),
                            output.data_ptr(),
                            input.numel(),
                            get_nccl_datatype(input.scalar_type()),
                            _world_nccl_comm,
                            GetCommStream(async_op)));
    if (block) { CUDACHECK(cudaStreamSynchronize(GetCommStream(async_op))); }
    if (async_op) { SynchComm(); }
}

inline at::Tensor newLikeFlat(std::vector<std::vector<at::Tensor>>& tensors, size_t deviceIdx)
{
    if (tensors.size() == 0 || tensors[0].size() == 0) {
        throw std::runtime_error("Received an empty list");
    }
    if (deviceIdx >= tensors.size()) { throw std::runtime_error("Invalid device index"); }
    auto& t = tensors[deviceIdx][0];
    auto device = t.device();
    for (const auto i : c10::irange(1, tensors[deviceIdx].size())) {
        if (tensors[deviceIdx][i].device() != device) {
            throw std::runtime_error("Expecting all tensors on the same device");
        }
    }
    at::DeviceGuard gpuGuard(device);
    std::vector<int64_t> sizes{static_cast<int64_t>(tensors[deviceIdx].size())};
    std::vector<int64_t> strides{static_cast<int64_t>(t.numel())};
    sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
    strides.insert(strides.end(), t.strides().begin(), t.strides().end());
    return at::empty_strided(sizes, strides, t.options().memory_format(c10::nullopt));
}

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> flatten_for_scatter_gather(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other,
    size_t world_size)
{
    if (tensor_lists.size() != other.size()) {
        throw std::runtime_error(
            "Tensor list operands to scatter/gather must have the same length");
    }
    const auto num_devices = tensor_lists.size();

    std::vector<at::Tensor> flattened;
    flattened.resize(num_devices);

    for (const auto i : c10::irange(size_t{}, num_devices)) {
        if (tensor_lists[i].size() != world_size * num_devices) {
            throw std::runtime_error(
                "Tensor list input to scatter/gather must match number of collective"
                " participants");
        }

        // Only check device match for the first tensor in the list; the call to
        // newLikeFlat() below will check the rest.
        if (tensor_lists[i].front().get_device() != other[i].get_device()) {
            throw std::runtime_error(
                "Corresponding input/output tensors to scatter/gather must all reside"
                " on the same device");
        }

        for (const auto& t : tensor_lists[i]) {
            if (t.numel() != other[i].numel()) {
                throw std::runtime_error(
                    "All tensor operands to scatter/gather must have the same number of elements");
            }
        }
        // Flatten the tensors (from all ranks) into a single big tensor.
        flattened[i] = newLikeFlat(tensor_lists, i);
    }
    return flattened;
}

void all_gather(std::vector<std::vector<torch::Tensor>>& outputTensors,
                std::vector<torch::Tensor>& inputTensors,
                bool block,
                py::object group,
                bool async_op)
{
    auto outputFlattened =
        flatten_for_scatter_gather(outputTensors, inputTensors, get_world_size(0));
    ncclComm_t comm = _get_comm_from_group(group);

    NCCLCHECK(ncclGroupStart());

    for (size_t i = 0; i < inputTensors.size(); ++i) {
        NCCLCHECK(ncclAllGather(inputTensors[i].data_ptr(),
                                outputFlattened[i].data_ptr(),
                                inputTensors[i].numel(),
                                get_nccl_datatype(inputTensors[i].scalar_type()),
                                comm,
                                GetCommStream(async_op)));
    }

    NCCLCHECK(ncclGroupEnd());

    if (block) { CUDACHECK(cudaStreamSynchronize(GetCommStream(async_op))); }
    if (async_op) { SynchComm(); }

    for (const auto i : c10::irange(outputTensors.size())) {
        // at::cuda::CUDAStreamGuard guard(ncclStreams[i]);
        for (const auto j : c10::irange(outputTensors[0].size())) {
            outputTensors[i][j].copy_(outputFlattened[i][j], true);
        }
    }
}

void reduce(torch::Tensor& data,
            int root,
            py::object op,
            bool block,
            py::object group,
            bool async_op)
{
    // void* sendbuff = data.data_ptr();
    // torch::Tensor recvbuf = torch::empty_like(data);
    ncclComm_t comm = _get_comm_from_group(group);
    NCCLCHECK(ncclReduce(data.data_ptr(),
                         data.data_ptr(),
                         data.numel(),
                         get_nccl_datatype(data.scalar_type()),
                         get_nccl_reduce_op(op, data),
                         root,
                         comm,
                         GetCommStream(async_op)));
    if (block) { CUDACHECK(cudaStreamSynchronize(GetCommStream(async_op))); }
    if (async_op) { SynchComm(); }
}

void reduce_scatter(torch::Tensor& data, py::object op, bool block, py::object group, bool async_op)
{
    // void* sendbuff = data.data_ptr();
    // torch::Tensor recvbuf = torch::empty_like(data);
    ncclComm_t comm = _get_comm_from_group(group);
    NCCLCHECK(ncclReduceScatter(data.data_ptr(),
                                data.data_ptr(),
                                data.numel(),
                                get_nccl_datatype(data.scalar_type()),
                                get_nccl_reduce_op(op, data),
                                comm,
                                GetCommStream(async_op)));
    if (block) { CUDACHECK(cudaStreamSynchronize(GetCommStream(async_op))); }
    if (async_op) { SynchComm(); }
}

void broadcast(torch::Tensor& data, int src, bool block, py::object group, bool async_op)
{
    ncclComm_t comm = _get_comm_from_group(group);
    NCCLCHECK(ncclBroadcast(data.data_ptr(),
                            data.data_ptr(),
                            data.numel(),
                            get_nccl_datatype(data.scalar_type()),
                            src,
                            comm,
                            GetCommStream(async_op)));
    if (block) { CUDACHECK(cudaStreamSynchronize(GetCommStream(async_op))); }
    if (async_op) { SynchComm(); }
}

void all_to_all_single(torch::Tensor outputTensor,
                       torch::Tensor inputTensor,
                       bool block,
                       py::object group,
                       bool async_op)
{
    // std::chrono::steady_clock::time_point begin, end;
    const auto* sendbuff = reinterpret_cast<char*>(inputTensor.data_ptr());
    auto* recvbuff = reinterpret_cast<char*>(outputTensor.data_ptr());
    int nRanks;
    ncclComm_t comm = _get_comm_from_group(group);
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    size_t rankdiff = inputTensor.nbytes() / nRanks;
    // if (is_prof) { begin = std::chrono::steady_clock::now(); }
    NCCLCHECK(ncclGroupStart());
    int count = inputTensor.numel() / nRanks;
    ncclDataType_t type = get_nccl_datatype(inputTensor.scalar_type());
    for (int r = 0; r < nRanks; r++) {
        if (count != 0) {
            NCCLCHECK(
                ncclSend(sendbuff + r * rankdiff, count, type, r, comm, GetCommStream(async_op)));
            NCCLCHECK(
                ncclRecv(recvbuff + r * rankdiff, count, type, r, comm, GetCommStream(async_op)));
        }
    }
    NCCLCHECK(ncclGroupEnd());
    if (block) { CUDACHECK(cudaStreamSynchronize(GetCommStream(async_op))); }
    if (async_op) { SynchComm(); }
    // CUDACHECK(cudaStreamSynchronize(s));
}

void all_to_all(std::vector<torch::Tensor>& inputTensors,
                std::vector<torch::Tensor>& outputTensors,
                bool block,
                py::object group,
                bool async_op)
{
    ncclComm_t comm = _get_comm_from_group(group);
    NCCLCHECK(ncclGroupStart());
    for (int t = 0; t < inputTensors.size(); t++) {
        torch::Tensor& input = inputTensors[t];
        torch::Tensor& output = outputTensors[t];
        if (input.numel() != 0) {
            NCCLCHECK(ncclSend(input.data_ptr(),
                               input.numel(),
                               get_nccl_datatype(input.scalar_type()),
                               t,
                               comm,
                               GetCommStream(async_op)));
        }
        if (output.numel() != 0) {
            NCCLCHECK(ncclRecv(output.data_ptr(),
                               output.numel(),
                               get_nccl_datatype(output.scalar_type()),
                               t,
                               comm,
                               GetCommStream(async_op)));
        }
    }
    NCCLCHECK(ncclGroupEnd());
    if (block) { CUDACHECK(cudaStreamSynchronize(GetCommStream(async_op))); }
    if (async_op) { SynchComm(); }
}

void synchronize() { CUDACHECK(cudaDeviceSynchronize()); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("send", &send, "nccl send");
    m.def("recv", &recv, "nccl recv");
    m.def("all_reduce", &all_reduce, "nccl all_reduce");
    m.def("broadcast", &broadcast, "nccl broadcast");
    m.def("all_to_all_single", &all_to_all_single, "nccl alltoall");
    m.def("all_toall_list", &all_to_all, "nccl alltoall list");
    m.def("all_gather_base", &all_gather_base, "nccl all_gather_base");
    m.def("all_gather", &all_gather, "nccl all_gather");
    m.def("reduce", &reduce, "nccl reduce");
    m.def("reduce_scatter", &reduce_scatter, "nccl reduce scatter");
    m.def("initialize", &initialize, "nccl initialize");
    m.def("finalize", &finalize, "nccl finalize");
    m.def("getNcclId", &getNcclId, "Get Unique NCCL ID");
    m.def("get_rank", &get_rank, "get rank");
    m.def("barrier", &barrier, "barrier");
    m.def("synchronize", &synchronize, "synchronize CUDA device");
    m.def("get_world_size", &get_world_size, "get world size");
    // m.def("create_comms", &create_comms, "nccl create comms");
    m.def("create_comm_group", &create_comm_group, "manually create comm group");
    m.def("test_set", &test_set, "manually create comm group");
    m.def("new_group", &new_group, "automatically create comm group");
    m.def("get_world_group", &get_world_group, "Returns the WORLD process group");
}

}  // namespace nccl
