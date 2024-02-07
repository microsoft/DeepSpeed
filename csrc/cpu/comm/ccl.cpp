// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>

#include <fcntl.h>
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <oneapi/ccl.hpp>

// states for collectives
enum coll_state {
    coll_begin = 0,
    // coll states for naive allreduce
    coll_allreduce_naive__copy_in_done,   // this state is for rank != 0
    coll_allreduce_naive__reduce_done,    // this state is for rank == 0
    coll_allreduce_naive__copy_out_done,  // this state is for rank != 0
    //coll_allreduce_ring__copy_in_done,
    //coll_allreduce_ring__copy_out_done,
    //coll_allreduce_ring__reduce_step_start,
};

// SHM building blocks
struct SharedData {
    const char* name;
    int descriptor;
    void* bytes;
    size_t nbytes;
};

void shared_open(SharedData* data, const char* name, size_t nbytes)
{
    int d = shm_open(name, O_RDWR, S_IRUSR | S_IWUSR);
    if (d != -1) {
        void* bytes = mmap(NULL, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, d, 0);
        data->name = name;
        data->descriptor = d;
        data->bytes = bytes;
        data->nbytes = nbytes;
    } else {
        printf("shared_open %s failed\n", name);
        data->descriptor = -1;
    }
}

void shared_create(SharedData* data, const char* name, void* bytes, size_t nbytes)
{
    int d = shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (d != -1) {
        if (nbytes = write(d, bytes, nbytes)) { shared_open(data, name, nbytes); }
    } else {
        printf("shared_create %s failed\n", name);
    }
}

void shared_close(SharedData* data)
{
    if (data->descriptor != -1) {
        munmap(data->bytes, data->nbytes);
        shm_unlink(data->name);
    }
}

// SHM based allreduce helper functions
// buffer that holds shm name
#define NAME_BUF_SIZE 1000
#define MAX_BUF_SIZE 1048576
#define SHM_BUFFER_NAME "deepspeed_allreduce_buffer"
SharedData allreduce_buffer;
struct allreduce_workspace {
    enum coll_state state;
    char buffer[MAX_BUF_SIZE];
};
struct allreduce_workspace** workspace;

void wait_buffer_state_until(int index, enum coll_state state)
{
    volatile enum coll_state* state_ptr = &(workspace[index]->state);

    while (*state_ptr != state)
        ;
}

void wait_buffer_state_until_leq(int index, enum coll_state state)
{
    volatile enum coll_state* state_ptr = &(workspace[index]->state);

    while (*state_ptr > state)
        ;
}

void wait_buffer_state_until_beq(int index, enum coll_state state)
{
    volatile enum coll_state* state_ptr = &(workspace[index]->state);

    while (*state_ptr < state)
        ;
}

void wait_buffer_state_until_not(int index, enum coll_state state)
{
    volatile enum coll_state* state_ptr = &(workspace[index]->state);

    while (*state_ptr == state)
        ;
}

__m512 cvt_bf16_to_fp32(const __m256i src) __attribute__((target("avx512bw")));
inline __m512 cvt_bf16_to_fp32(const __m256i src)
{
    auto y = _mm512_cvtepu16_epi32(src);
    return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
}

inline __m256i cvt_fp32_to_bf16(const __m512 src) __attribute__((target("avx512bw")));
inline __m256i cvt_fp32_to_bf16(const __m512 src)
{
    __m512i value = _mm512_castps_si512(src);
    __m512i nan = _mm512_set1_epi32(0xffff);
    auto mask_value = _mm512_cmp_ps_mask(src, src, _CMP_ORD_Q);
    __m512i ones = _mm512_set1_epi32(0x1);
    __m512i vec_bias = _mm512_set1_epi32(0x7fff);
    // uint32_t lsb = (input >> 16) & 1;
    auto t_value = _mm512_and_si512(_mm512_srli_epi32(value, 16), ones);
    // uint32_t rounding_bias = 0x7fff + lsb;
    t_value = _mm512_add_epi32(t_value, vec_bias);
    // input += rounding_bias;
    t_value = _mm512_add_epi32(t_value, value);
    // input = input >> 16;
    t_value = _mm512_srli_epi32(t_value, 16);
    // Check NaN before converting back to bf16
    t_value = _mm512_mask_blend_epi32(mask_value, nan, t_value);
    return _mm512_cvtusepi32_epi16(t_value);
}

void reduce_2_bf16_buffers_io(int num_elements, void* in_out, void* in)
    __attribute__((target("avx512bw")));

void reduce_2_bf16_buffers_iio(int num_elements, void* in0, void* in1, void* out)
    __attribute__((target("avx512bw")));

void reduce_bf16_buffers(int num_elements, int num_buffers, struct allreduce_workspace** workspace)
    __attribute__((target("avx512bw")));

void reduce_2_fp32_buffers_io(int num_elements, void* in_out, void* in)
    __attribute__((target("avx512bw")));

void reduce_2_fp32_buffers_iio(int num_elements, void* in0, void* in1, void* out)
    __attribute__((target("avx512bw")));

void reduce_fp32_buffers(int num_elements, int num_buffers, struct allreduce_workspace** workspace)
    __attribute__((target("avx512bw")));

// N_REDUCE_LIMIT is the number of buffers that can be reduced together in one shot.
// Compared with do N-1 2-reduces which needs 2*(N-1) read and N-1 write,
// N-reduce only needs N read and 1 write, this saves 2/3 memory bandwidth.
// When increase N_REDUCE_LIMIT to a bigger number, do the following steps
// 1. Extend REPEAT_<X> macros list down below
// 2. Extend switch cases which call "REPEAT(X, ...)" down below
#define N_REDUCE_LIMIT 8

void reduce_all_buffers(struct allreduce_workspace** workspace,
                        int num_elements,
                        c10::ScalarType scalar_type,
                        int num_buffers)
{
    switch (scalar_type) {
        case c10::ScalarType::BFloat16:
            if (num_buffers > 2 && num_buffers <= N_REDUCE_LIMIT) {
                reduce_bf16_buffers(num_elements, num_buffers, workspace);
            } else {
                for (int i = 1; i < num_buffers; i++) {
                    reduce_2_bf16_buffers_io(num_elements, workspace[0]->buffer, workspace[i]->buffer);
                }
            }
            break;
        case c10::ScalarType::Float:
            if (num_buffers > 2 && num_buffers <= N_REDUCE_LIMIT) {
                reduce_fp32_buffers(num_elements, num_buffers, workspace);
            } else {
                for (int i = 1; i < num_buffers; i++) {
                    reduce_2_fp32_buffers_io(num_elements, workspace[0]->buffer, workspace[i]->buffer);
                }
            }
            break;
        default: assert(!"Should not get here");
    }
}

#define REPEAT(N, x) REPEAT_##N(x)
#define REPEAT_1(x) x(1)
#define REPEAT_2(x) \
    REPEAT_1(x);    \
    x(2)
#define REPEAT_3(x) \
    REPEAT_2(x);    \
    x(3)
#define REPEAT_4(x) \
    REPEAT_3(x);    \
    x(4)
#define REPEAT_5(x) \
    REPEAT_4(x);    \
    x(5)
#define REPEAT_6(x) \
    REPEAT_5(x);    \
    x(6)
#define REPEAT_7(x) \
    REPEAT_6(x);    \
    x(7)

#define CVT_ADD_BF16(x)                                                                \
    do {                                                                               \
        auto in##x##_val =                                                             \
            cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(workspace[x]->buffer + i))); \
        inout_val = _mm512_add_ps(inout_val, in##x##_val);                             \
    } while (0)

// Reduce functions down below use vectorized algorithm, the number of bytes processed each
// iteration depends on vector length.  256bit vector ==> 32 bytes, 512bit vector ==> 64 bytes
// If you change implementation of reduce_2_bf16_buffers_io or reduce_2_fp32_buffers_io, check
// whether this number needs to be changed
#define VECTOR_LENGTH_IN_BYTES 32

// num_elements must be divisible by 16 (caller check)
void reduce_bf16_buffers(int num_elements, int num_buffers, struct allreduce_workspace** workspace)
{
#pragma omp parallel for
    for (int i = 0; i < num_elements * 2; i += VECTOR_LENGTH_IN_BYTES) {
        auto inout_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(workspace[0]->buffer + i)));
        switch (num_buffers) {
            case 8: REPEAT(7, CVT_ADD_BF16); break;
            case 7: REPEAT(6, CVT_ADD_BF16); break;
            case 6: REPEAT(5, CVT_ADD_BF16); break;
            case 5: REPEAT(4, CVT_ADD_BF16); break;
            case 4: REPEAT(3, CVT_ADD_BF16); break;
            case 3: REPEAT(2, CVT_ADD_BF16); break;
            default: assert(!"Should not get here.");
        }
        _mm256_storeu_si256((__m256i*)(workspace[0]->buffer + i), cvt_fp32_to_bf16(inout_val));
    }
}

void reduce_2_bf16_buffers_io(int num_elements, void* in_out, void* in1)
{
#pragma omp parallel for
    for (int i = 0; i < num_elements * 2; i += VECTOR_LENGTH_IN_BYTES) {
        auto inout_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)((char*)in_out + i)));
        auto in1_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)((char*)in1 + i)));
        inout_val = _mm512_add_ps(inout_val, in1_val);
        _mm256_storeu_si256((__m256i*)((char*)in_out + i), cvt_fp32_to_bf16(inout_val));
    }
}

void reduce_2_bf16_buffers_iio(int num_elements, void* in0, void* in1, void* out)
{
#pragma omp parallel for
    for (int i = 0; i < num_elements * 2; i += VECTOR_LENGTH_IN_BYTES) {
        auto in0_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)((char*)in0 + i)));
        auto in1_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)((char*)in1 + i)));
        auto out_val = _mm512_add_ps(in0_val, in1_val);
        _mm256_storeu_si256((__m256i*)((char*)out + i), cvt_fp32_to_bf16(out_val));
    }
}

#define CVT_ADD_F32(x)                                                         \
    do {                                                                       \
        auto in##x##_val = _mm256_loadu_ps((float*)(workspace[x]->buffer + i)); \
        inout_val = _mm256_add_ps(inout_val, in##x##_val);                     \
    } while (0)

// num_elements must be divisible by 16 (caller check)
void reduce_fp32_buffers(int num_elements, int num_buffers, struct allreduce_workspace** workspace)
{
#pragma omp parallel for
    for (int i = 0; i < num_elements * 4; i += VECTOR_LENGTH_IN_BYTES) {
        auto inout_val = _mm256_loadu_ps((float*)(workspace[0]->buffer + i));
        switch (num_buffers) {
            case 8: REPEAT(7, CVT_ADD_F32); break;
            case 7: REPEAT(6, CVT_ADD_F32); break;
            case 6: REPEAT(5, CVT_ADD_F32); break;
            case 5: REPEAT(4, CVT_ADD_F32); break;
            case 4: REPEAT(3, CVT_ADD_F32); break;
            case 3: REPEAT(2, CVT_ADD_F32); break;
            default: assert(!"Should not get here.");
        }
        _mm256_storeu_ps((float*)(workspace[0]->buffer + i), inout_val);
    }
}

void reduce_2_fp32_buffers_io(int num_elements, void* in_out, void* in1)
{
#pragma omp parallel for
    for (int i = 0; i < num_elements * 4; i += VECTOR_LENGTH_IN_BYTES) {
        auto inout_val = _mm256_loadu_ps((float*)((char*)in_out + i));
        auto in1_val = _mm256_loadu_ps((float*)((char*)in1 + i));
        inout_val = _mm256_add_ps(inout_val, in1_val);
        _mm256_storeu_ps((float*)((char*)in_out + i), inout_val);
    }
}

void reduce_2_fp32_buffers_iio(int num_elements, void* in0, void* in1, void* out)
{
#pragma omp parallel for
    for (int i = 0; i < num_elements * 4; i += VECTOR_LENGTH_IN_BYTES) {
        auto in0_val = _mm256_loadu_ps((float*)((char*)in0 + i));
        auto in1_val = _mm256_loadu_ps((float*)((char*)in1 + i));
        auto out_val = _mm256_add_ps(in0_val, in1_val);
        _mm256_storeu_ps((float*)((char*)out + i), out_val);
    }
}

// Communicatiooon settings
int world_rank = -1;
int world_size = -1;

std::set<int> _comm_ids;
std::set<int> _colors;
std::vector<ccl::communicator> _ccl_comms;
ccl::shared_ptr_class<ccl::kvs> sub_kvs;
std::map<std::vector<int>, int> group_to_comm_id;

ccl::communicator& _get_comm_from_group() { return _ccl_comms[0]; }
ccl::communicator& _get_comm_from_group(py::object group) { return _ccl_comms[0]; }
ccl::communicator& _get_comm_from_group(std::vector<int> ranks)
{
    if (group_to_comm_id.find(ranks) != group_to_comm_id.end()) {
        auto id = group_to_comm_id.find(ranks);
        return _ccl_comms[id->second];
    }
    return _ccl_comms[0];
}

#define CCLCHECK(cmd) \
    do {              \
        cmd;          \
    } while (0)

#define KVS_CREATE_SUCCESS 0
#define KVS_CREATE_FAILURE -1

bool is_initialized = 0;

ccl::shared_ptr_class<ccl::kvs> kvs;

bool all_ranks_local_p = false;

void initialize(int size, int rank, torch::Tensor& kvs_data)
{
    if (is_initialized) return;

    // Check whether all ranks is on the same physical machine.
    // If true, we will use an SHM based low latency allreduce

    auto ls_string = std::getenv("LOCAL_SIZE");
    int ls = 0;
    if (ls_string != NULL) { ls = std::stoi(std::getenv("LOCAL_SIZE")); }

    if (size >= 1 && size == ls) { all_ranks_local_p = true; }

    world_size = size;
    world_rank = rank;
    is_initialized = 1;

    ccl::kvs::address_type main_addr;

    if (rank != 0) {
        memcpy(main_addr.data(), kvs_data.data_ptr(), main_addr.size());
        kvs = ccl::create_kvs(main_addr);
    }

    _ccl_comms.emplace_back(ccl::create_communicator(size, rank, kvs));

    auto addr_string = std::getenv("MASTER_ADDR");
    if (addr_string == NULL) { addr_string = ""; }
    auto port_string = std::getenv("MASTER_PORT");
    if (port_string == NULL) { port_string = ""; }
    char shm_name[NAME_BUF_SIZE];
    snprintf(shm_name,
             NAME_BUF_SIZE,
             "%s_%d_%s_%s",
             SHM_BUFFER_NAME,
             getuid(),
             addr_string,
             port_string);
    // create shared workspace for SHM based allreduce
    if (all_ranks_local_p) {
        struct allreduce_workspace *workspace_buf;
        if (rank == 0) {
            workspace_buf =
                (struct allreduce_workspace*)malloc(size * sizeof(struct allreduce_workspace));
            shared_create(
                &allreduce_buffer, shm_name, workspace_buf, size * sizeof(struct allreduce_workspace));
            workspace_buf = (struct allreduce_workspace*)allreduce_buffer.bytes;
            for (int i = 0; i < size; i++) { workspace_buf[i].state = coll_begin; }
        }
        CCLCHECK(ccl::barrier(_get_comm_from_group()).wait());
        if (rank != 0) {
            shared_open(&allreduce_buffer, shm_name, size * sizeof(struct allreduce_workspace));
        }
        workspace_buf = (struct allreduce_workspace*)allreduce_buffer.bytes;
        workspace = (struct allreduce_workspace**)malloc(size * sizeof(struct allreduce_workspace*));
        for (int i = 0; i < size; i++) {
            workspace[i] = workspace_buf + i;
        }
    }
}

/*
    rank == 0: create main kvs and return its address
    rank == else: return an empty address
*/
std::vector<uint8_t> get_kvs_addr(int rank)
{
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        ccl::kvs::address_type main_addr = kvs->get_address();
        auto ccl_kvs_addr = std::vector<uint8_t>(main_addr.begin(), main_addr.end());
        return ccl_kvs_addr;
    } else {
        ccl::kvs::address_type main_addr;
        auto ccl_kvs_addr = std::vector<uint8_t>(main_addr.begin(), main_addr.end());
        return ccl_kvs_addr;
    }
}

int get_rank(int group = 0) { return world_rank; }

int get_world_size(int group = 0) { return world_size; }

// Find the next ordered, unique value to a set. E.g. <0,1,2,7> --> 3
int next_unique_val(std::set<int> s)
{
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

std::vector<uint8_t> get_sub_kvs_addr(bool first)
{
    if (first) {
        sub_kvs = ccl::create_main_kvs();
        ccl::kvs::address_type main_addr = sub_kvs->get_address();
        auto ccl_kvs_addr = std::vector<uint8_t>(main_addr.begin(), main_addr.end());
        return ccl_kvs_addr;
    } else {
        ccl::kvs::address_type main_addr;
        auto ccl_kvs_addr = std::vector<uint8_t>(main_addr.begin(), main_addr.end());
        return ccl_kvs_addr;
    }
}

void initialize_sub_comm(int size, int rank, torch::Tensor& kvs_data, std::vector<int> ranks)
{
    ccl::kvs::address_type main_addr;
    if (rank != 0) {
        memcpy(main_addr.data(), kvs_data.data_ptr(), main_addr.size());
        sub_kvs = ccl::create_kvs(main_addr);
    }
    _ccl_comms.push_back(ccl::create_communicator(size, rank, sub_kvs));
    group_to_comm_id[ranks] = _ccl_comms.size() - 1;
}

ccl::datatype get_ccl_datatype(c10::ScalarType type)
{
    ccl::datatype ccl_type;
    switch (type) {
        case c10::ScalarType::Int: ccl_type = ccl::datatype::int32; break;
        case c10::ScalarType::Long: ccl_type = ccl::datatype::int64; break;
        case c10::ScalarType::Float: ccl_type = ccl::datatype::float32; break;
        case c10::ScalarType::Double: ccl_type = ccl::datatype::float64; break;
        case c10::ScalarType::BFloat16: ccl_type = ccl::datatype::bfloat16; break;
        case c10::ScalarType::Half: ccl_type = ccl::datatype::float16; break;
        default: ccl_type = ccl::datatype::int8;
    }
    return ccl_type;
}

ccl::reduction get_ccl_reduce_op(py::object op, at::Tensor& input)
{
    py::object ReduceOp = py::module_::import("deepspeed.comm").attr("ReduceOp");
    if (!py::isinstance(op, ReduceOp)) {
        throw std::runtime_error("Error: Op must be of type ReduceOp");
    }

    int op_val = py::int_(op.attr("value"));
    ccl::reduction ccl_op;

    if (input.scalar_type() == at::kBool) {
        if (op_val == (int)py::int_(ReduceOp.attr("SUM").attr("value"))) {
            // For bool tensors, map sum to max, which both represent a bitwise or.
            // This is to prevent overflow issues with sum, since we use uint8 to
            // represent a bool (see cclDataType mapping).
            ccl_op = ccl::reduction::max;
        } else if (op_val == (int)py::int_(ReduceOp.attr("AVG").attr("value"))) {
            throw std::runtime_error("Error: For bool tensors, op must be of type ReduceOp");
        }
    }

    if (op_val == (int)py::int_(ReduceOp.attr("SUM").attr("value"))) {
        ccl_op = ccl::reduction::sum;
    } else if (op_val == (int)py::int_(ReduceOp.attr("MIN").attr("value"))) {
        ccl_op = ccl::reduction::min;
    } else if (op_val == (int)py::int_(ReduceOp.attr("MAX").attr("value"))) {
        ccl_op = ccl::reduction::max;
    } else if (op_val == (int)py::int_(ReduceOp.attr("PRODUCT").attr("value"))) {
        ccl_op = ccl::reduction::prod;
    } else {
        throw std::runtime_error("Error: Unrecognized ReduceOp type");
    }
    return ccl_op;
}

void broadcast(torch::Tensor& data, int src, std::vector<int> group, bool async_op)
{
    CCLCHECK(ccl::broadcast(data.data_ptr(),
                            data.numel(),
                            get_ccl_datatype(data.scalar_type()),
                            src,
                            _get_comm_from_group(group))
                 .wait());
}

// TODO: implement torch's async_op behavior, document it.
void all_reduce(torch::Tensor& data, py::object op, std::vector<int> group, bool async_op)
{
    CCLCHECK(ccl::allreduce(data.data_ptr(),
                            data.data_ptr(),
                            data.numel(),
                            get_ccl_datatype(data.scalar_type()),
                            get_ccl_reduce_op(op, data),
                            _get_comm_from_group(group))
                 .wait());
}

void all_reduce_caching(torch::Tensor& data,
                        py::object op,
                        std::string match_id,
                        std::vector<int> group,
                        bool async_op)
{
    ccl::allreduce_attr attr = ccl::default_allreduce_attr;
    auto match_str = ccl::v1::string(match_id);
    attr.template set<ccl::operation_attr_id::to_cache>(true);
    attr.template set<ccl::operation_attr_id::match_id>(match_str);
    // To control this, use operation attribute and set true value for to_cache field and unique
    // string (for example, tensor name) for match_id field. Note that:
    //   match_id should be the same for a specific communication operation across all ranks.
    //   If the same tensor is a part of different communication operations, match_id should have
    //   different values for each of these operations.
    CCLCHECK(ccl::allreduce(data.data_ptr(),
                            data.data_ptr(),
                            data.numel(),
                            get_ccl_datatype(data.scalar_type()),
                            get_ccl_reduce_op(op, data),
                            _get_comm_from_group(group),
                            attr)
                 .wait());
}

static void parallel_memcpy(void* to, void* from, size_t n_bytes)
    __attribute__((target("avx512bw")));
static void parallel_memcpy(void* to, void* from, size_t n_bytes)
{
#pragma omp parallel for
    for (int i = 0; i < n_bytes; i += VECTOR_LENGTH_IN_BYTES) {
        auto val = _mm256_loadu_si256((__m256i*)((char*)from + i));
        _mm256_storeu_si256((__m256i*)((char*)to + i), val);
    }
}

size_t ring_slice_size(size_t chunk_el, int el_size, int slice_idx)
{
    slice_idx = ((slice_idx%world_size)+world_size)%world_size;
    size_t slice_size = chunk_el / world_size;
    return el_size * (slice_idx == world_size-1 ? slice_size+(chunk_el%world_size) : slice_size);
}

char* ring_slice_data(char* data_ptr, size_t chunk_el, int el_size, int slice_idx)
{
    slice_idx = ((slice_idx%world_size)+world_size)%world_size;
    size_t slice_size = chunk_el / world_size;
    size_t el_offset = slice_size * slice_idx;
    return data_ptr + el_offset*el_size;
}

void ring_all_reduce(char* data_ptr, c10::ScalarType scalar_type, size_t chunk_size, size_t chunk_el)
{
    int data_size = chunk_size/chunk_el;
    parallel_memcpy(ring_slice_data(workspace[world_rank]->buffer, chunk_el, data_size, world_rank),
                    ring_slice_data(data_ptr, chunk_el, data_size, world_rank),
                    ring_slice_size(chunk_el, data_size, world_rank));
    std::atomic_thread_fence(std::memory_order_release);

    int prev_rank = (world_rank+world_size-1) % world_size;
    int next_rank = (world_rank+1) % world_size;

    workspace[world_rank]->state = (enum coll_state)(workspace[world_rank]->state+1);
    wait_buffer_state_until_beq(prev_rank, workspace[world_rank]->state);

    int step;

    for (step=0; step<world_size-1; step++) {
        reduce_2_bf16_buffers_iio(ring_slice_size(chunk_el, data_size, world_rank-1-step)/data_size,
                                  ring_slice_data(workspace[prev_rank]->buffer, chunk_el, data_size, world_rank-1-step),
                                  ring_slice_data(data_ptr, chunk_el, data_size, world_rank-1-step),
                                  ring_slice_data(workspace[world_rank]->buffer, chunk_el, data_size, world_rank-1-step));
        std::atomic_thread_fence(std::memory_order_release);
        workspace[world_rank]->state = (enum coll_state)(workspace[world_rank]->state+1);
        wait_buffer_state_until_beq(prev_rank, workspace[world_rank]->state);
    }

    for (step=0; step<world_size; step++) {
        parallel_memcpy(ring_slice_data(data_ptr, chunk_el, data_size, world_rank+step),
                        ring_slice_data(workspace[(world_rank+step-1+world_size)%world_size]->buffer, chunk_el, data_size, world_rank+step),
                        ring_slice_size(chunk_el, data_size, world_rank+step));
    }
    std::atomic_thread_fence(std::memory_order_release);
    workspace[world_rank]->state = coll_begin;
    for (int rank=0; rank<world_size; rank++) {
        // there could only be two possible state: coll_begin if the other rank didn't spin too fast
        // coll_begin+1 if the other rank spin too fast
        // all other state mean the other rank didn't reach this point
        wait_buffer_state_until_leq(rank, (enum coll_state)(coll_begin+1));
    }
}

void naive_all_reduce(char* data_ptr, c10::ScalarType scalar_type, size_t chunk_size, size_t chunk_el)
{
    parallel_memcpy(workspace[world_rank]->buffer, data_ptr, chunk_size);
    std::atomic_thread_fence(std::memory_order_release);
    workspace[world_rank]->state = coll_allreduce_naive__copy_in_done;

    if (world_rank == 0) {
        // compute allreduce result on rank 0
        for (int i = 1; i < world_size; i++) {
            // wait until the other rank copy the buffer
            wait_buffer_state_until(i, coll_allreduce_naive__copy_in_done);
        }
        reduce_all_buffers(workspace, chunk_el, scalar_type, world_size);
        std::atomic_thread_fence(std::memory_order_release);
        workspace[world_rank]->state = coll_allreduce_naive__reduce_done;
        parallel_memcpy(data_ptr, workspace[0]->buffer, chunk_size);
    }
    if (world_rank != 0) {
        wait_buffer_state_until(0, coll_allreduce_naive__reduce_done);
        parallel_memcpy(data_ptr, workspace[0]->buffer, chunk_size);
        std::atomic_thread_fence(std::memory_order_release);
        workspace[world_rank]->state = coll_allreduce_naive__copy_out_done;
    }
    if (world_rank == 0) {
        for (int i = 1; i < world_size; i++) {
            wait_buffer_state_until(i, coll_allreduce_naive__copy_out_done);
        }
        std::atomic_thread_fence(std::memory_order_release);
        workspace[world_rank]->state = coll_begin;
    }
    if (world_rank != 0) {
        // if rank 0 spin too fast it could be in state 1 of next allreduce
        // in this case wait_buffer_state_until(0, 0) may cause deadlock
        // what we are certain is when rank 0 finishes the state won't be 2
        wait_buffer_state_until_not(0, coll_allreduce_naive__reduce_done);
        workspace[world_rank]->state = coll_begin;
    }
}

void all_reduce_outer_loop(torch::Tensor& data, size_t numel, int data_size)
{
    for (int offset = 0; offset < data_size; offset += MAX_BUF_SIZE) {
        auto data_ptr = ((char*)(data.data_ptr()) + offset);
        size_t chunk_size = data_size - offset > MAX_BUF_SIZE ? MAX_BUF_SIZE : data_size - offset;
        size_t chunk_el = chunk_size / (data_size / numel);
        //naive_all_reduce(data_ptr, data.scalar_type(), chunk_size, chunk_el);
        ring_all_reduce(data_ptr, data.scalar_type(), chunk_size, chunk_el);
    }
}

void inference_all_reduce(torch::Tensor& data, py::object op, bool async_op)
{
    static py::object ReduceOp = py::module_::import("deepspeed.comm").attr("ReduceOp");
    static auto ReduceOpSum = (int)py::int_(ReduceOp.attr("SUM").attr("value"));

    assert(py::int_(op.attr("value")) == ReduceOpSum);

    auto numel = data.numel();

    int data_size = 0;
    bool data_type_fallback = false;

    switch (data.scalar_type()) {
        case c10::ScalarType::BFloat16: data_size = numel * 2; break;
        case c10::ScalarType::Float: data_size = numel * 4; break;
        default: data_type_fallback = true;
    }

    if (data_type_fallback || (data_size % VECTOR_LENGTH_IN_BYTES) != 0 || !all_ranks_local_p) {
        // fallback to oneccl allreduce
        CCLCHECK(ccl::allreduce(data.data_ptr(),
                                data.data_ptr(),
                                data.numel(),
                                get_ccl_datatype(data.scalar_type()),
                                get_ccl_reduce_op(op, data),
                                _get_comm_from_group())
                     .wait());
        return;
    }

    all_reduce_outer_loop(data, numel, data_size);
}

void barrier(std::vector<int> group, bool async_op)
{
    CCLCHECK(ccl::barrier(_get_comm_from_group(group)).wait());
}

std::vector<std::string> get_available_coll()
{
    std::vector<std::string> colls{
        "broadcast", "all_reduce", "inference_all_reduce", "all_reduce_caching", "barrier"};
    return colls;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_kvs_addr", &get_kvs_addr, "create and get main kvs addr");
    m.def("initialize", &initialize, "ccl initialize");
    m.def("get_rank", &get_rank, "get rank");
    m.def("get_world_size", &get_world_size, "get world size");
    m.def("broadcast", &broadcast, "ccl broadcast");
    m.def("all_reduce", &all_reduce, "ccl all_reduce");
    m.def("inference_all_reduce", &inference_all_reduce, "low latency all_reduce implementation");
    m.def("all_reduce_caching", &all_reduce_caching, "ccl all_reduce with caching");
    m.def("barrier", &barrier, "barrier");
    m.def("initialize_sub_comm", &initialize_sub_comm, "initialize_sub_comm");
    m.def("get_sub_kvs_addr", &get_sub_kvs_addr, "get_sub_kvs_addr");
    m.def("get_available_coll", &get_available_coll, "get_available_coll");
}
