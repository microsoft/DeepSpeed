#include <cuda.h>
#include <cuda_runtime_api.h>
#include <mpi.h>
#include <nccl.h>
#include <torch/extension.h>
#include <chrono>

#include <c10/util/irange.h>

#include <iostream>
#include <string>

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

int counter = 0;
cudaStream_t s;
ncclComm_t ncclcomm;

std::vector<MPI_Comm> global_mpi_comms;
std::vector<ncclComm_t> global_nccl_comms;
std::vector<cudaStream_t> global_streams;

void create_mpi_comms(int number = 1)
{
    int size = global_mpi_comms.size();
    global_mpi_comms.resize(size + number);

    for (int i = 0; i < number; ++i) {
        MPICHECK(MPI_Comm_dup(MPI_COMM_WORLD, &global_mpi_comms[size + i]));
    }
}

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

void create_nccl_comms(int number = 1)
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

    CUDACHECK(cudaStreamCreate(&s));
}

void print_comm_number() { std::cout << "Number of Comms:" << global_mpi_comms.size() << "\n"; }

void increase_counter() { counter++; }

void decrease_counter() { counter--; }

void print_counter() { std::cout << "Counter is:" << counter << "\n"; }

void initialize_mpi()
{
    int flag;
    MPICHECK(MPI_Initialized(&flag));
    if (!flag) MPICHECK(MPI_Init(NULL, NULL));
    create_mpi_comms();
}

void initialize_nccl(int rank, int size)
{
    initialize_mpi();
    create_nccl_comms();
}

void finalize_mpi() { MPICHECK(MPI_Finalize()); }

void finalize_nccl()
{
    NCCLCHECK(ncclCommDestroy(ncclcomm));
    finalize_mpi();
}

MPI_Datatype get_mpi_datatype(c10::ScalarType type)
{
    MPI_Datatype mpi_type;
    switch (type) {
        case c10::ScalarType::Int: mpi_type = MPI_INT; break;

        case c10::ScalarType::Long: mpi_type = MPI_LONG; break;

        case c10::ScalarType::Float: mpi_type = MPI_FLOAT; break;

        case c10::ScalarType::Double: mpi_type = MPI_DOUBLE; break;

        default: mpi_type = MPI_BYTE;
    }

    return mpi_type;
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

void mpi_send(torch::Tensor data, int rank, int tag)
{
    MPICHECK(MPI_Send(data.data_ptr(),
                      data.numel(),
                      get_mpi_datatype(data.scalar_type()),
                      rank,
                      tag,
                      MPI_COMM_WORLD));
}

void barrier() { MPICHECK(MPI_Barrier(MPI_COMM_WORLD)); }

void nccl_send(torch::Tensor data, int rank, int tag)
{
    NCCLCHECK(ncclSend(
        data.data_ptr(), data.numel(), get_nccl_datatype(data.scalar_type()), rank, ncclcomm, s));
    CUDACHECK(cudaStreamSynchronize(s));
}

void nccl_recv(torch::Tensor data, int rank, int tag)
{
    NCCLCHECK(ncclRecv(
        data.data_ptr(), data.numel(), get_nccl_datatype(data.scalar_type()), rank, ncclcomm, s));
    CUDACHECK(cudaStreamSynchronize(s));
}

void mpi_recv(torch::Tensor data, int rank, int tag)
{
    MPICHECK(MPI_Recv(data.data_ptr(),
                      data.numel(),
                      get_mpi_datatype(data.scalar_type()),
                      rank,
                      tag,
                      MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE));
}

size_t mpi_isend(torch::Tensor data, int rank, int tag, int comm = 0)
{
    MPI_Request req;
    MPICHECK(MPI_Isend(data.data_ptr(),
                       data.numel(),
                       get_mpi_datatype(data.scalar_type()),
                       rank,
                       tag,
                       global_mpi_comms[comm],
                       &req));
    size_t casted_req = size_t(req);
    return casted_req;
}

size_t mpi_irecv(torch::Tensor data, int rank, int tag, int comm = 0)
{
    MPI_Request req;
    MPICHECK(MPI_Irecv(data.data_ptr(),
                       data.numel(),
                       get_mpi_datatype(data.scalar_type()),
                       rank,
                       tag,
                       global_mpi_comms[comm],
                       &req));
    size_t casted_req = size_t(req);
    return casted_req;
}

void mpi_allreduce(torch::Tensor data, int comm, bool is_prof)
{
    std::chrono::steady_clock::time_point begin, end;
    torch::Tensor recvbuf = torch::empty_like(data);
    if (is_prof) { begin = std::chrono::steady_clock::now(); }
    MPICHECK(MPI_Allreduce(MPI_IN_PLACE,
                           data.data_ptr(),
                           data.numel(),
                           get_mpi_datatype(data.scalar_type()),
                           MPI_SUM,
                           global_mpi_comms[comm]));
    if (is_prof) {
        end = std::chrono::steady_clock::now();
        if (get_rank() == 0) {
            std::cout << "MPI allreduce time = "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                      << "us, Size = " << data.numel() * data.element_size() << " B"
                      << "\n";
        }
    }
}

void nccl_allreduce(torch::Tensor& data, bool is_prof)
{
    std::chrono::steady_clock::time_point begin, end;
    void* sendbuff = data.data_ptr();
    torch::Tensor recvbuf = torch::empty_like(data);
    if (is_prof) { begin = std::chrono::steady_clock::now(); }
    NCCLCHECK(ncclAllReduce(data.data_ptr(),
                            data.data_ptr(),
                            data.numel(),
                            get_nccl_datatype(data.scalar_type()),
                            ncclSum,
                            ncclcomm,
                            s));
    // CUDACHECK(cudaStreamSynchronize(s));
    if (is_prof) {
        end = std::chrono::steady_clock::now();
        if (get_rank(0) == 0) {
            std::cout << "NCCL allreduce time = "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                      << "us, Size = " << data.numel() * data.element_size() << " B"
                      << "\n";
        }
    }
}

void nccl_bcast(torch::Tensor& data, int src)
{
    NCCLCHECK(ncclBroadcast(data.data_ptr(),
                            data.data_ptr(),
                            data.numel(),
                            get_nccl_datatype(data.scalar_type()),
                            src,
                            ncclcomm,
                            s));
}

void nccl_alltoall(torch::Tensor outputTensor, torch::Tensor inputTensor, bool is_prof)
{
    std::chrono::steady_clock::time_point begin, end;
    const auto* sendbuff = reinterpret_cast<char*>(inputTensor.data_ptr());
    auto* recvbuff = reinterpret_cast<char*>(outputTensor.data_ptr());
    int nRanks;
    NCCLCHECK(ncclCommCount(ncclcomm, &nRanks));
    size_t rankdiff = inputTensor.nbytes() / nRanks;
    if (is_prof) { begin = std::chrono::steady_clock::now(); }
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
    // CUDACHECK(cudaStreamSynchronize(s));
    if (is_prof) {
        end = std::chrono::steady_clock::now();
        if (get_rank(0) == 0) {
            std::cout << "NCCL alltoall time = "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                      << " us, Size = " << inputTensor.numel() * inputTensor.element_size() << " B"
                      << "\n";
        }
    }
}

void nccl_alltoall_list(std::vector<torch::Tensor>& inputTensors,
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

void mpi_allgather(torch::Tensor data, int comm = 0)
{
    torch::Tensor recvbuf = torch::empty_like(data);
    MPICHECK(MPI_Allgather(data.data_ptr(),
                           data.numel(),
                           get_mpi_datatype(data.scalar_type()),
                           recvbuf.data_ptr(),
                           data.numel(),
                           get_mpi_datatype(data.scalar_type()),
                           global_mpi_comms[comm]));
}

void mpi_gather(torch::Tensor data, int root_rank, int comm = 0)
{
    torch::Tensor recvbuf = torch::empty_like(data);
    MPICHECK(MPI_Gather(data.data_ptr(),
                        data.numel(),
                        get_mpi_datatype(data.scalar_type()),
                        recvbuf.data_ptr(),
                        data.numel(),
                        get_mpi_datatype(data.scalar_type()),
                        root_rank,
                        global_mpi_comms[comm]));
}

void mpi_scatter(torch::Tensor data, int root_rank, int comm = 0)
{
    torch::Tensor recvbuf = torch::empty_like(data);
    MPICHECK(MPI_Scatter(data.data_ptr(),
                         data.numel(),
                         get_mpi_datatype(data.scalar_type()),
                         recvbuf.data_ptr(),
                         data.numel(),
                         get_mpi_datatype(data.scalar_type()),
                         root_rank,
                         global_mpi_comms[comm]));
}
/*
void mpi_reduce_scatter(torch::Tensor data, int comm=0)
{
  torch::Tensor recvbuf = torch::empty_like(data);
  MPI_Reduce_scatter( data.data_ptr(),
            recvbuf.data_ptr(),
            data.numel(),
            get_mpi_datatype(data.scalar_type()),
            MPI_SUM,
            global_mpi_comms[comm]);
}
*/
void mpi_reduce(torch::Tensor data, int root_rank, int comm = 0)
{
    torch::Tensor recvbuf = torch::empty_like(data);
    MPICHECK(MPI_Reduce(data.data_ptr(),
                        recvbuf.data_ptr(),
                        data.numel(),
                        get_mpi_datatype(data.scalar_type()),
                        MPI_SUM,
                        root_rank,
                        global_mpi_comms[comm]));
}

void mpi_bcast(torch::Tensor data, int root_rank, int comm = 0)
{
    MPICHECK(MPI_Bcast(data.data_ptr(),
                       data.numel(),
                       get_mpi_datatype(data.scalar_type()),
                       root_rank,
                       global_mpi_comms[comm]));
}

void mpi_alltoall(torch::Tensor outputTensor, torch::Tensor inputTensor, int comm, bool is_prof)
{
    std::chrono::steady_clock::time_point begin, end;
    if (is_prof) { begin = std::chrono::steady_clock::now(); }
    MPICHECK(MPI_Alltoall(inputTensor.data_ptr(),
                          inputTensor.numel() / get_world_size(0),
                          get_mpi_datatype(inputTensor.scalar_type()),
                          outputTensor.data_ptr(),
                          outputTensor.numel() / get_world_size(0),
                          get_mpi_datatype(outputTensor.scalar_type()),
                          global_mpi_comms[comm]));
    if (is_prof) {
        end = std::chrono::steady_clock::now();
        if (get_rank(0) == 0) {
            std::cout << "MPI alltoall time = "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                      << "us, Size = " << inputTensor.numel() * inputTensor.element_size() << " B"
                      << "\n";
        }
    }
}

void mpi_alltoall_list(std::vector<torch::Tensor>& outputTensors,
                       std::vector<torch::Tensor>& inputTensors,
                       int comm = 0)
{
    for (int t = 0; t < inputTensors.size(); t++) {
        torch::Tensor& input = inputTensors[t];
        torch::Tensor& output = outputTensors[t];
        MPICHECK(MPI_Alltoall(input.data_ptr(),
                              input.numel(),
                              get_mpi_datatype(input.scalar_type()),
                              output.data_ptr(),
                              output.numel(),
                              get_mpi_datatype(output.scalar_type()),
                              global_mpi_comms[comm]));
    }
}

void mpi_wait(size_t casted_req)
{
    MPI_Request req = (MPI_Request)casted_req;
    MPI_Status status;

    MPICHECK(MPI_Wait(&req, &status));
}

void mpi_device_sync() { CUDACHECK(cudaDeviceSynchronize()); }

torch::Tensor mpi_wait_multi(size_t casted_req, torch::Tensor data)
{
    MPI_Request req = (MPI_Request)casted_req;
    MPI_Status status;

    MPICHECK(MPI_Wait(&req, &status));
    return data;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("nccl_send", &nccl_send, "nccl send");
    m.def("nccl_recv", &nccl_recv, "nccl recv");
    m.def("nccl_allreduce", &nccl_allreduce, "nccl allreduce");
    m.def("nccl_bcast", &nccl_bcast, "nccl broadcast");
    m.def("nccl_alltoall", &nccl_alltoall, "nccl alltoall");
    m.def("nccl_alltoall_list", &nccl_alltoall_list, "nccl alltoall list");
    m.def("mpi_send", &mpi_send, "mpi send");
    m.def("mpi_recv", &mpi_recv, "mpi recv");
    m.def("mpi_isend", &mpi_isend, "mpi isend");
    m.def("mpi_irecv", &mpi_irecv, "mpi irecv");
    m.def("mpi_allreduce", &mpi_allreduce, "mpi allreduce");
    m.def("mpi_allgather", &mpi_allgather, "mpi allgather");
    m.def("mpi_gather", &mpi_gather, "mpi gather");
    m.def("mpi_scatter", &mpi_scatter, "mpi scatter");
    // m.def("mpi_reduce_scatter", &mpi_reduce_scatter, "mpi reduce-scatter");
    m.def("mpi_reduce", &mpi_reduce, "mpi reduce");
    m.def("mpi_bcast", &mpi_bcast, "mpi bcast");
    m.def("mpi_alltoall", &mpi_alltoall, "mpi alltoall");
    m.def("mpi_alltoall_list", &mpi_alltoall_list, "mpi alltoall list");
    m.def("mpi_wait", &mpi_wait, "mpi wait");
    m.def("mpi_device_sync", &mpi_device_sync, "mpi device sync");
    m.def("mpi_wait_multi", &mpi_wait_multi, "mpi wait multi");
    m.def("initialize_mpi", &initialize_mpi, "mpi initialize");
    m.def("initialize_nccl", &initialize_nccl, "nccl initialize");
    m.def("finalize_mpi", &finalize_mpi, "mpi finalize");
    m.def("finalize_nccl", &finalize_nccl, "nccl finalize");
    m.def("get_rank", &get_rank, "get rank");
    m.def("barrier", &barrier, "barrier");
    m.def("get_world_size", &get_world_size, "get world size");
    m.def("increase_counter", &increase_counter, "mpi increase counter");
    m.def("decrease_counter", &decrease_counter, "mpi decrease counter");
    m.def("print_counter", &print_counter, "mpi print counter");
    m.def("create_mpi_comms", &create_mpi_comms, "mpi create comms");
    // m.def("create_nccl_comms", &create_nccl_comms, "nccl create comms");
    m.def("print_comm_number", &print_comm_number, "mpi print comm number");
    m.def("getNcclId", &getNcclId, "Get Unique NCCL ID");
}
