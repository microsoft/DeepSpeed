/* Copyright 2020 The Microsoft DeepSpeed Team
   Copyright NVIDIA/apex
*/

#include <assert.h>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include "multi_tensor_apply.dp.hpp"
#include "type_shim.hpp"

#define BLOCK_SIZE 512
#define ILP 4

typedef enum {
    ADAM_MODE_0 = 0,  // L2 regularization mode
    ADAM_MODE_1 = 1   // Decoupled weight decay mode(AdamW)
} adamMode_t;

using MATH_T = float;

template <typename T>
void AdamFunctor(sycl::nd_item<1> item_ct1,
                 int chunk_size,
                 int* noop_gmem,
                 const int tensor_loc,
                 const int chunk_idx,
                 int n,
                 T* g,
                 T* p,
                 T* m,
                 T* v,
                 const float beta1,
                 const float beta2,
                 const float beta1_correction,
                 const float beta2_correction,
                 const float epsilon,
                 const float lr,
                 const int mode,
                 const float decay)
{
    g += chunk_idx * chunk_size;

    p += chunk_idx * chunk_size;

    m += chunk_idx * chunk_size;

    v += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for (int i_start = 0; i_start < n && i_start < chunk_size;
         i_start += item_ct1.get_local_range(0) * ILP) {
        MATH_T r_g[ILP];
        MATH_T r_p[ILP];
        MATH_T r_m[ILP];
        MATH_T r_v[ILP];
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
            int i = i_start + item_ct1.get_local_id(0) + ii * item_ct1.get_local_range(0);
            if (i < n && i < chunk_size) {
                r_g[ii] = g[i];
                r_p[ii] = p[i];
                r_m[ii] = m[i];
                r_v[ii] = v[i];
            } else {
                r_g[ii] = MATH_T(0);
                r_p[ii] = MATH_T(0);
                r_m[ii] = MATH_T(0);
                r_v[ii] = MATH_T(0);
            }
        }

#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
            if (mode == ADAM_MODE_0) {  // L2
                r_g[ii] = r_g[ii] + (decay * r_p[ii]);
                r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
                r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
                MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
                MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
                MATH_T denom = sycl::sqrt((float)next_v_unbiased) + epsilon;
                MATH_T update = next_m_unbiased / denom;
                r_p[ii] = r_p[ii] - (lr * update);
            } else {  // weight decay
                r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
                r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
                MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
                MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
                MATH_T denom = sycl::sqrt((float)next_v_unbiased) + epsilon;
                MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
                r_p[ii] = r_p[ii] - (lr * update);
            }
        }
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
            int i = i_start + item_ct1.get_local_id(0) + ii * item_ct1.get_local_range(0);
            if (i < n && i < chunk_size) {
                p[i] = r_p[ii];
                m[i] = r_m[ii];
                v[i] = r_v[ii];
            }
        }
    }
}

void test_queue_with_accessor(void)
{
    printf("Test queue with accessor\n");
    auto type_ = c10::DeviceType::XPU;
    c10::impl::VirtualGuardImpl impl(type_);
    auto device_ = c10::Device(type_);
    c10::Stream dpcpp_stream = impl.getStream(device_);
    sycl::queue* stream = &(xpu::get_queue_from_stream(dpcpp_stream));
    sycl::default_selector d_selector;
    static auto exception_handler = [](sycl::exception_list e_list) {
        for (std::exception_ptr const& e : e_list) {
            try {
                std::rethrow_exception(e);
            } catch (std::exception const& e) {
                std::cout << "Failure" << std::endl;
                std::terminate();
            }
        }
    };
    sycl::queue dq(d_selector,
                   exception_handler,
                   {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    struct {
        unsigned char block_to_tensor[320];
        int block_to_chunk[320];
        void* addresses[4][36];
        int sizes[36];
    } tll;
    sycl::buffer<unsigned char, 1> block_to_tensor_buf(&(tll.block_to_tensor[0]), {320});
    sycl::buffer<int, 1> block_to_chunk_buf(&(tll.block_to_chunk[0]), {320});
    sycl::buffer<void*, 2> addresses_buf(&(tll.addresses[0][0]), {4, 36});
    sycl::buffer<int, 1> sizes_buf(&(tll.sizes[0]), {36});
    printf("submit dq without accessor ");
    dq.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(320 * 512, 512), [=](sycl::nd_item<1> item_ct1) {});
    });
    dq.wait();
    printf("done\n");
    printf("submit dq with accessor ");
    dq.submit([&](sycl::handler& cgh) {
        sycl::accessor tl_block_to_tensor(block_to_tensor_buf, cgh, sycl::read_only);
        sycl::accessor tl_block_to_chunk(block_to_chunk_buf, cgh, sycl::read_only);
        sycl::accessor tl_addresses(addresses_buf, cgh, sycl::read_only);
        sycl::accessor tl_sizes(sizes_buf, cgh, sycl::read_only);
        cgh.parallel_for(sycl::nd_range<1>(320 * 512, 512), [=](sycl::nd_item<1> item_ct1) {});
    });
    dq.wait();
    printf("done\n");
    printf("submit xpu::stream without accessor ");
    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(320 * 512, 512), [=](sycl::nd_item<1> item_ct1) {});
    });
    stream->wait();
    printf("done\n");
    printf("submit xpu::stream with accessor ");
    stream->submit([&](sycl::handler& cgh) {
        sycl::accessor tl_block_to_tensor(block_to_tensor_buf, cgh, sycl::read_only);
        sycl::accessor tl_block_to_chunk(block_to_chunk_buf, cgh, sycl::read_only);
        sycl::accessor tl_addresses(addresses_buf, cgh, sycl::read_only);
        sycl::accessor tl_sizes(sizes_buf, cgh, sycl::read_only);
        cgh.parallel_for(sycl::nd_range<1>(320 * 512, 512), [=](sycl::nd_item<1> item_ct1) {});
    });
    stream->wait();
    printf("done\n");
}

void multi_tensor_test(void)
{
    printf("inside multi_tensor_test\n");
    test_queue_with_accessor();
}

void multi_tensor_adam_sycl(int chunk_size,
                            at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists,
                            const float lr,
                            const float beta1,
                            const float beta2,
                            const float epsilon,
                            const int step,
                            const int mode,
                            const int bias_correction,
                            const float weight_decay)
{
    using namespace at;

    // Handle bias correction mode
    float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
    if (bias_correction == 1) {
        bias_correction1 = 1 - std::pow(beta1, step);
        bias_correction2 = 1 - std::pow(beta2, step);
    }
    // Assume single type across p,g,m1,m2
    DISPATCH_DOUBLE_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(),
                                   0,
                                   "adam",
                                   multi_tensor_apply<4, scalar_t_0>(BLOCK_SIZE,
                                                                     chunk_size,
                                                                     noop_flag,
                                                                     tensor_lists,
                                                                     beta1,
                                                                     beta2,
                                                                     bias_correction1,
                                                                     bias_correction2,
                                                                     epsilon,
                                                                     lr,
                                                                     mode,
                                                                     weight_decay))
}
