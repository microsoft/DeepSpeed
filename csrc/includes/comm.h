//enum class ReduceOp {
//    SUM = 0,
//    AVG,
//    PRODUCT,
//    MIN,
//    MAX,
//    BAND,  // Bitwise AND
//    BOR,   // Bitwise OR
//    BXOR,  // Bitwise XOR
//    UNUSED,
//};

//void create_comm_group(std::vector<int> comm_ranks, int rank, int comm_id, int color);

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