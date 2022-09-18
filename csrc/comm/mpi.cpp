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

namespace mpi {

int counter = 0;
ncclComm_t ncclcomm;

// py::module_ dist = py::module_::import("deepspeed.comm");

std::vector<MPI_Comm> global_mpi_comms;

void create_comms(int number = 1)
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

void print_comm_number() { std::cout << "Number of Comms:" << global_mpi_comms.size() << "\n"; }

void increase_counter() { counter++; }

void decrease_counter() { counter--; }

void print_counter() { std::cout << "Counter is:" << counter << "\n"; }

void initialize()
{
    int flag;
    MPICHECK(MPI_Initialized(&flag));
    if (!flag) MPICHECK(MPI_Init(NULL, NULL));
    create_comms();
}

void finalize() { MPICHECK(MPI_Finalize()); }

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

void send(torch::Tensor data, int rank, int tag)
{
    MPICHECK(MPI_Send(data.data_ptr(),
                      data.numel(),
                      get_mpi_datatype(data.scalar_type()),
                      rank,
                      tag,
                      MPI_COMM_WORLD));
}

void barrier() { MPICHECK(MPI_Barrier(MPI_COMM_WORLD)); }

void recv(torch::Tensor data, int rank, int tag)
{
    MPICHECK(MPI_Recv(data.data_ptr(),
                      data.numel(),
                      get_mpi_datatype(data.scalar_type()),
                      rank,
                      tag,
                      MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE));
}

size_t isend(torch::Tensor data, int rank, int tag, int comm = 0)
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

size_t irecv(torch::Tensor data, int rank, int tag, int comm = 0)
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

void allreduce(torch::Tensor data, int comm, bool is_prof)
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

void allgather(torch::Tensor data, int comm = 0)
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

void gather(torch::Tensor data, int root_rank, int comm = 0)
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

void scatter(torch::Tensor data, int root_rank, int comm = 0)
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
void reduce_scatter(torch::Tensor data, int comm=0)
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
void reduce(torch::Tensor data, int root_rank, int comm = 0)
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

void bcast(torch::Tensor data, int root_rank, int comm = 0)
{
    MPICHECK(MPI_Bcast(data.data_ptr(),
                       data.numel(),
                       get_mpi_datatype(data.scalar_type()),
                       root_rank,
                       global_mpi_comms[comm]));
}

void alltoall(torch::Tensor outputTensor, torch::Tensor inputTensor, int comm, bool is_prof)
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

void alltoall_list(std::vector<torch::Tensor>& outputTensors,
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

void wait(size_t casted_req)
{
    MPI_Request req = (MPI_Request)casted_req;
    MPI_Status status;

    MPICHECK(MPI_Wait(&req, &status));
}

void device_sync() { CUDACHECK(cudaDeviceSynchronize()); }

torch::Tensor wait_multi(size_t casted_req, torch::Tensor data)
{
    MPI_Request req = (MPI_Request)casted_req;
    MPI_Status status;

    MPICHECK(MPI_Wait(&req, &status));
    return data;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("send", &send, "mpi send");
    m.def("recv", &recv, "mpi recv");
    m.def("isend", &isend, "mpi isend");
    m.def("irecv", &irecv, "mpi irecv");
    m.def("allreduce", &allreduce, "mpi allreduce");
    m.def("allgather", &allgather, "mpi allgather");
    m.def("gather", &gather, "mpi gather");
    m.def("scatter", &scatter, "mpi scatter");
    // m.def("reduce_scatter", &reduce_scatter, "mpi reduce-scatter");
    m.def("reduce", &reduce, "mpi reduce");
    m.def("bcast", &bcast, "mpi bcast");
    m.def("alltoall", &alltoall, "mpi alltoall");
    m.def("alltoall_list", &alltoall_list, "mpi alltoall list");
    m.def("wait", &wait, "mpi wait");
    m.def("device_sync", &device_sync, "mpi device sync");
    m.def("wait_multi", &wait_multi, "mpi wait multi");
    m.def("initialize", &initialize, "mpi initialize");
    m.def("finalize", &finalize, "mpi finalize");
    m.def("get_rank", &get_rank, "get rank");
    m.def("barrier", &barrier, "barrier");
    m.def("get_world_size", &get_world_size, "get world size");
    m.def("increase_counter", &increase_counter, "mpi increase counter");
    m.def("decrease_counter", &decrease_counter, "mpi decrease counter");
    m.def("print_counter", &print_counter, "mpi print counter");
    m.def("create_comms", &create_comms, "mpi create comms");
    m.def("print_comm_number", &print_comm_number, "mpi print comm number");
}

}  // namespace mpi
