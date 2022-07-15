#include <cuda.h>
#include <cuda_runtime_api.h>
#include <mpi.h>
#include <nccl.h>
#include <torch/extension.h>
#include <chrono>
#include <pybind11/embed.h>
namespace py = pybind11;

#include <c10/util/irange.h>

#include <iostream>
#include <string>

#include <comm.h>

// TODO: remove
#include <stdio.h>

#define MPICHECK(cmd)                                                        \
    do {                                                                     \
        int e = cmd;                                                         \
        if (e != MPI_SUCCESS) {                                              \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CUDACHECK(cmd)                                                                            \
    do {                                                                                          \
        cudaError_t e = cmd;                                                                      \
        if (e != cudaSuccess) {                                                                   \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

#define NCCLCHECK(cmd)                                                                           \
    do {                                                                                         \
        ncclResult_t ret = cmd;                                                                  \
        if (ret != ncclSuccess) {                                                                \
            printf(                                                                              \
                "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(ret)); \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    } while (0)

#define CUDA_STREAM_SYNCHRONIZE(_nccl_stream)                                            \
    do {                                                                                 \
        cudaError_t err = cudaErrorNotReady;                                             \
        int flag;                                                                        \
        while (err == cudaErrorNotReady) {                                               \
            err = cudaStreamQuery(_nccl_stream);                                         \
            MPICHECK(MPI_Iprobe(                                                         \
                MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE)); \
        }                                                                                \
        CUDACHECK(err);                                                                  \
    } while (0)

namespace nccl {

int counter = 0;
cudaStream_t s;
ncclComm_t ncclcomm;

//py::module_ dist = py::module_::import("deepspeed.comm");

std::vector<MPI_Comm> global_mpi_comms;
std::vector<ncclComm_t> global_nccl_comms;
std::vector<cudaStream_t> global_streams;


//REZA+AMMAR CODE
//curandGenerator_t _gen;
//cublasHandle_t _cublasHandle; 
cudaEvent_t _comp_event;
cudaEvent_t _comm_event;    
void* _workspace;
uint64_t _seed;
uint64_t _curr_offset;
size_t _workSpaceSize;
unsigned _token_length;
unsigned _num_tokens;
std::vector<std::array<int, 3>> _gemm_algos;    
cudaStream_t _comp_stream;
cudaStream_t _comm_stream;  
MPI_Group _group;
std::unordered_map<int, ncclComm_t> _nccl_comms;
std::unordered_map<int, int> _world_sizes;
MPI_Comm _comm;
bool _comm_created;

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

void create_comms(int number = 1)
{
    ncclUniqueId ncclID;
    int world_rank = get_rank(0);
    int world_size = get_world_size(0);
    int ngpus;

    CUDACHECK(cudaGetDeviceCount(&ngpus));

    CUDACHECK(cudaSetDevice(world_rank % ngpus));
    CUDACHECK(cudaStreamCreate(&s));
    if (world_rank == 0) { ncclGetUniqueId(&ncclID); }
    MPICHECK(MPI_Bcast(&ncclID, sizeof(ncclID), MPI_BYTE, 0, MPI_COMM_WORLD));

    NCCLCHECK(ncclCommInitRank(&ncclcomm, world_size, ncclID, world_rank));
}

void print_comm_number() { std::cout << "Number of Comms:" << global_mpi_comms.size() << "\n"; }

void increase_counter() { counter++; }

void decrease_counter() { counter--; }

void print_counter() { std::cout << "Counter is:" << counter << "\n"; }

void initialize(int rank, int size)
{
    //initialize_mpi();
    create_comms();
}

void finalize()
{
    NCCLCHECK(ncclCommDestroy(ncclcomm));
    //finalize_mpi();
}


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
        throw std::runtime_error ("Error: Op must be of type ReduceOp");
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
            throw std::runtime_error ("Error: For bool tensors, op must be of type ReduceOp");
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
        throw std::runtime_error ("Error: Unrecognized ReduceOp type");
    }
    return nccl_op;
}

void send(torch::Tensor data, int rank, int tag, bool block)
{
    NCCLCHECK(ncclSend(
        data.data_ptr(), data.numel(), get_nccl_datatype(data.scalar_type()), rank, ncclcomm, s));
    if (block) { CUDACHECK(cudaStreamSynchronize(s)); }
}

void recv(torch::Tensor data, int rank, int tag, bool block)
{
    NCCLCHECK(ncclRecv(
        data.data_ptr(), data.numel(), get_nccl_datatype(data.scalar_type()), rank, ncclcomm, s));
    if (block) { CUDACHECK(cudaStreamSynchronize(s)); }
}


//TODO: implement torch's async_op behavior, document it.
void all_reduce(torch::Tensor& data, py::object op, bool block)
{

    // std::chrono::steady_clock::time_point begin, end;
    // void* sendbuff = data.data_ptr();
    // torch::Tensor recvbuf = torch::empty_like(data);
    // if (is_prof) { begin = std::chrono::steady_clock::now(); }
    NCCLCHECK(ncclAllReduce(data.data_ptr(),
                            data.data_ptr(),
                            data.numel(),
                            get_nccl_datatype(data.scalar_type()),
                            get_nccl_reduce_op(op, data),
                            ncclcomm,
                            s));
    if (!block) { CUDACHECK(cudaStreamSynchronize(s)); }
    // if (is_prof) {
    //    end = std::chrono::steady_clock::now();
    //    if (get_rank(0) == 0) {
    //        std::cout << "NCCL allreduce time = "
    //                  << std::chrono::duration_cast<std::chrono::microseconds>(end -
    //                  begin).count()
    //                  << "us, Size = " << data.numel() * data.element_size() << " B"
    //                  << "\n";
    //    }
    //}
}

inline ncclComm_t GetNCCLComm(int comm_id=0) { return _nccl_comms[comm_id]; }

void create_comm_group(std::vector<int> comm_ranks, int rank, int comm_id, int color)
{
    printf("creating comm : size: %d , comm_id: %d , color: %d\n", comm_ranks.size(), comm_id, color);
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
        auto message = std::string(
            "Fail to create comm group (number of ranks is higher than world_size).");
        std::cerr << message << std::endl;
        throw std::runtime_error(message);
    }
    ncclUniqueId _nccl_uid;
    ncclGetUniqueId(&_nccl_uid);
    MPI_Bcast((void*)&_nccl_uid,
              sizeof(ncclUniqueId),
              MPI_BYTE,
              0,
              num_ranks < world_size ? _newcomm : _comm);
    ncclCommInitRank(&_nccl_comm, num_ranks, _nccl_uid, rank);
    std::cout << "nccl comm = " << _nccl_comm << std::endl;
    _comm_created = true;
    _world_sizes[comm_id] = num_ranks;
    _nccl_comms[comm_id] = _nccl_comm;
}

//TODO: implement torch's async_op behavior, document it.
void all_gather_base(torch::Tensor& output, torch::Tensor& input, bool block, int comm_id)
{
    // std::chrono::steady_clock::time_point begin, end;
    // void* sendbuff = data.data_ptr();
    // torch::Tensor recvbuf = torch::empty_like(data);
    // if (is_prof) { begin = std::chrono::steady_clock::now(); }
    std::cout << "Number of NCCL comms: " << _nccl_comms.size() << "\n";
    NCCLCHECK(ncclAllGather(input.data_ptr(),
                            output.data_ptr(),
                            input.numel(),
                            get_nccl_datatype(input.scalar_type()),
                            GetNCCLComm(comm_id),
                            s));
    if (block) { CUDACHECK(cudaStreamSynchronize(s)); }
    // if (is_prof) {
    //    end = std::chrono::steady_clock::now();
    //    if (get_rank(0) == 0) {
    //        std::cout << "NCCL allreduce time = "
    //                  << std::chrono::duration_cast<std::chrono::microseconds>(end -
    //                  begin).count()
    //                  << "us, Size = " << data.numel() * data.element_size() << " B"
    //                  << "\n";
    //    }
    //}
}

inline at::Tensor newLikeFlat(
    std::vector<std::vector<at::Tensor>>& tensors,
    size_t deviceIdx) {
  if (tensors.size() == 0 || tensors[0].size() == 0) {
    throw std::runtime_error ("Received an empty list");
  }
  if (deviceIdx >= tensors.size()) {
    throw std::runtime_error ("Invalid device index");
  }
  auto& t = tensors[deviceIdx][0];
  auto device = t.device();
  for (const auto i : c10::irange(1, tensors[deviceIdx].size())) {
    if (tensors[deviceIdx][i].device() != device) {
      throw std::runtime_error ("Expecting all tensors on the same device");
    }
  }
  at::DeviceGuard gpuGuard(device);
  std::vector<int64_t> sizes{static_cast<int64_t>(tensors[deviceIdx].size())};
  std::vector<int64_t> strides{static_cast<int64_t>(t.numel())};
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  strides.insert(strides.end(), t.strides().begin(), t.strides().end());
  return at::empty_strided(
      sizes, strides, t.options().memory_format(c10::nullopt));
}

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
//std::vector<at::Tensor> flatten_for_scatter_gather(
//    std::vector<std::vector<at::Tensor>>& tensor_lists,
//    size_t world_size) {
//  const auto num_devices = tensor_lists.size();
//
//  std::vector<at::Tensor> flattened;
//  flattened.resize(num_devices);
//
//  for (const auto i : c10::irange(size_t{}, num_devices)) {
//    // Flatten the tensors (from all ranks) into a single big tensor.
//    flattened[i] = newLikeFlat(tensor_lists, i);
//  }
//  return flattened;
//}

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> flatten_for_scatter_gather(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other,
    size_t world_size) {
  if (tensor_lists.size() != other.size()) {
    throw std::runtime_error ("Tensor list operands to scatter/gather must have the same length");
  }
  const auto num_devices = tensor_lists.size();

  std::vector<at::Tensor> flattened;
  flattened.resize(num_devices);

  for (const auto i : c10::irange(size_t{}, num_devices)) {
    if (tensor_lists[i].size() != world_size * num_devices) {
      throw std::runtime_error ("Tensor list input to scatter/gather must match number of collective"
          " participants");
    }

    // Only check device match for the first tensor in the list; the call to
    // newLikeFlat() below will check the rest.
    if (tensor_lists[i].front().get_device() != other[i].get_device()) {
      throw std::runtime_error ("Corresponding input/output tensors to scatter/gather must all reside"
          " on the same device");
    }

    for (const auto& t : tensor_lists[i]) {
      if (t.numel() != other[i].numel()) {
        throw std::runtime_error ("All tensor operands to scatter/gather must have the same number of elements");
      }
    }
    // Flatten the tensors (from all ranks) into a single big tensor.
    flattened[i] = newLikeFlat(tensor_lists, i);
  }
  return flattened;
}


//void coll_

//TODO: implement torch's async_op behavior, document it.
void all_gather(std::vector<std::vector<torch::Tensor>>& outputTensors, std::vector<torch::Tensor>& inputTensors, bool block)
{
    //std::vector<at::Tensor> flattenOutputTensors;
    //flattenOutputTensors.resize(outputTensors.size());
//
    //for (size_t i = 0; i < outputTensors.size(); ++i) {
    //  // Flatten the output tensors (from all ranks) to a single big tensor
    //  flattenOutputTensors[i] = newLikeFlat(outputTensors);
    //}

      auto outputFlattened = flatten_for_scatter_gather(outputTensors, inputTensors, get_world_size(0));

    
    NCCLCHECK(ncclGroupStart());

    for (size_t i = 0; i < inputTensors.size(); ++i) {

        NCCLCHECK(ncclAllGather(
            inputTensors[i].data_ptr(),
            outputFlattened[i].data_ptr(),
            inputTensors[i].numel(),
            get_nccl_datatype(inputTensors[i].scalar_type()),
            ncclcomm,
            s));
        }

    NCCLCHECK(ncclGroupEnd());




    // std::chrono::steady_clock::time_point begin, end;
    // void* sendbuff = data.data_ptr();
    // torch::Tensor recvbuf = torch::empty_like(data);
    // if (is_prof) { begin = std::chrono::steady_clock::now(); }
    //NCCLCHECK(ncclAllGather(input.data_ptr(),
    //                        output.data_ptr(),
    //                        input.numel(),
    //                        get_nccl_datatype(data.scalar_type()),
    //                        ncclcomm,
    //                        s));
    if (block) { CUDACHECK(cudaStreamSynchronize(s)); }
    // if (is_prof) {
    //    end = std::chrono::steady_clock::now();
    //    if (get_rank(0) == 0) {
    //        std::cout << "NCCL allreduce time = "
    //                  << std::chrono::duration_cast<std::chrono::microseconds>(end -
    //                  begin).count()
    //                  << "us, Size = " << data.numel() * data.element_size() << " B"
    //                  << "\n";
    //    }
    //}

    for (const auto i : c10::irange(outputTensors.size())) {
          //at::cuda::CUDAStreamGuard guard(ncclStreams[i]);
          for (const auto j : c10::irange(outputTensors[0].size())) {
            outputTensors[i][j].copy_(outputFlattened[i][j], true);
          }
        }
}

//TODO: implement torch's async_op behavior, document it.
void reduce(torch::Tensor& data, int root, py::object op, bool block)
{

    // std::chrono::steady_clock::time_point begin, end;
    // void* sendbuff = data.data_ptr();
    // torch::Tensor recvbuf = torch::empty_like(data);
    // if (is_prof) { begin = std::chrono::steady_clock::now(); }
    NCCLCHECK(ncclReduce(data.data_ptr(),
                         data.data_ptr(),
                         data.numel(),
                         get_nccl_datatype(data.scalar_type()),
                         get_nccl_reduce_op(op, data),
                         root,
                         ncclcomm,
                         s));
    if (block) { CUDACHECK(cudaStreamSynchronize(s)); }
    // if (is_prof) {
    //    end = std::chrono::steady_clock::now();
    //    if (get_rank(0) == 0) {
    //        std::cout << "NCCL allreduce time = "
    //                  << std::chrono::duration_cast<std::chrono::microseconds>(end -
    //                  begin).count()
    //                  << "us, Size = " << data.numel() * data.element_size() << " B"
    //                  << "\n";
    //    }
    //}
}

//TODO: implement torch's async_op behavior, document it.
void reduce_scatter(torch::Tensor& data, py::object op, bool block)
{
    // std::chrono::steady_clock::time_point begin, end;
    // void* sendbuff = data.data_ptr();
    // torch::Tensor recvbuf = torch::empty_like(data);
    // if (is_prof) { begin = std::chrono::steady_clock::now(); }
    NCCLCHECK(ncclReduceScatter(data.data_ptr(),
                                data.data_ptr(),
                                data.numel(),
                                get_nccl_datatype(data.scalar_type()),
                                get_nccl_reduce_op(op, data),
                                ncclcomm,
                                s));
    if (block) { CUDACHECK(cudaStreamSynchronize(s)); }
    // if (is_prof) {
    //    end = std::chrono::steady_clock::now();
    //    if (get_rank(0) == 0) {
    //        std::cout << "NCCL allreduce time = "
    //                  << std::chrono::duration_cast<std::chrono::microseconds>(end -
    //                  begin).count()
    //                  << "us, Size = " << data.numel() * data.element_size() << " B"
    //                  << "\n";
    //    }
    //}
}

void broadcast(torch::Tensor& data, int src, bool block)
{
    NCCLCHECK(ncclBroadcast(data.data_ptr(),
                            data.data_ptr(),
                            data.numel(),
                            get_nccl_datatype(data.scalar_type()),
                            src,
                            ncclcomm,
                            s));
    if (block) { CUDACHECK(cudaStreamSynchronize(s)); }
}

void alltoall(torch::Tensor outputTensor, torch::Tensor inputTensor, bool block)
{
    //std::chrono::steady_clock::time_point begin, end;
    const auto* sendbuff = reinterpret_cast<char*>(inputTensor.data_ptr());
    auto* recvbuff = reinterpret_cast<char*>(outputTensor.data_ptr());
    int nRanks;
    NCCLCHECK(ncclCommCount(ncclcomm, &nRanks));
    size_t rankdiff = inputTensor.nbytes() / nRanks;
    //if (is_prof) { begin = std::chrono::steady_clock::now(); }
    NCCLCHECK(ncclGroupStart());
    int count = inputTensor.numel() / nRanks;
    ncclDataType_t type = get_nccl_datatype(inputTensor.scalar_type());
    for (int r = 0; r < nRanks; r++) {
        if (count != 0) {
            NCCLCHECK(ncclSend(sendbuff + r * rankdiff, count, type, r, ncclcomm, s));
            NCCLCHECK(ncclRecv(recvbuff + r * rankdiff, count, type, r, ncclcomm, s));
        }
    }
    NCCLCHECK(ncclGroupEnd());
    if (block) { CUDACHECK(cudaStreamSynchronize(s)); }
    // CUDACHECK(cudaStreamSynchronize(s));
    //if (is_prof) {
    //    end = std::chrono::steady_clock::now();
    //    if (get_rank(0) == 0) {
    //        std::cout << "NCCL alltoall time = "
    //                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
    //                  << " us, Size = " << inputTensor.numel() * inputTensor.element_size() << " B"
    //                  << "\n";
    //    }
    //}
}

void alltoall_list(std::vector<torch::Tensor>& inputTensors,
                        std::vector<torch::Tensor>& outputTensors)
{
    NCCLCHECK(ncclGroupStart());
    for (int t = 0; t < inputTensors.size(); t++) {
        torch::Tensor& input = inputTensors[t];
        torch::Tensor& output = outputTensors[t];
        if (input.numel() != 0) {
            NCCLCHECK(ncclSend(input.data_ptr(),
                               input.numel(),
                               get_nccl_datatype(input.scalar_type()),
                               t,
                               ncclcomm,
                               s));
        }
        if (output.numel() != 0) {
            NCCLCHECK(ncclRecv(output.data_ptr(),
                               output.numel(),
                               get_nccl_datatype(output.scalar_type()),
                               t,
                               ncclcomm,
                               s));
        }
    }
    NCCLCHECK(ncclGroupEnd());
    CUDACHECK(cudaStreamSynchronize(s));
}

void synchronize() {
    CUDACHECK(cudaDeviceSynchronize());
}
    

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("send", &send, "nccl send");
    m.def("recv", &recv, "nccl recv");
    m.def("all_reduce", &all_reduce, "nccl all_reduce");
    m.def("broadcast", &broadcast, "nccl broadcast");
    m.def("alltoall", &alltoall, "nccl alltoall");
    m.def("alltoall_list", &alltoall_list, "nccl alltoall list");
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
    m.def("increase_counter", &increase_counter, "mpi increase counter");
    m.def("decrease_counter", &decrease_counter, "mpi decrease counter");
    m.def("print_counter", &print_counter, "mpi print counter");
    // m.def("create_comms", &create_comms, "nccl create comms");
    m.def("print_comm_number", &print_comm_number, "mpi print comm number");
    m.def("create_comm_group", &create_comm_group, "create comm group");
}

} // namespace nccl