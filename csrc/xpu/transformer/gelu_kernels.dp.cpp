#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
using namespace sycl;
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
using namespace cl::sycl;
#else
#error "Unsupported compiler"
#endif
#include <ext/oneapi/bfloat16.hpp>

using bf16 = sycl::ext::oneapi::bfloat16;

inline float gelu(const float x)
{
    const float sqrt_param = 0.79788456080286535587989211986876f;
    const float mul_param = 0.044715;
    return x * 0.5f * (1.0f + tanh(sqrt_param * (x + mul_param * x * x * x)));
}

inline float d_gelu(const float x)
{
    const float sqrt_param = 0.79788456080286535587989211986876f;
    const float mul_param = 0.044715;

    float x2mul = x * x * mul_param;
    float tan_h = tanh(sqrt_param * (x + x * x2mul));
    float dg1 = 0.5f * (1.0f + tan_h);
    float dg2 = x * 0.5f * sqrt_param * (1 - tan_h * tan_h);
    float dg3 = dg2 * 3 * x2mul;
    return (dg1 + dg2 + dg3);
}

/*
  Fused bias add with GELU

  Loads a vector of 4 elements each iteration, for stride
  iterations. It was written with the intention to launch 256 thread
  threadblocks, so to launch for bert-large, we would set ITERATIONS
  to 4. This is currently done automatically as a heuristic, setting
  the number of iterations as blocks of 1024.

  For FP16, the values are loaded from memory as half, but converted
  to FP32 for the arithmetic itself, to prevent numerous overflow on
  the intermediate hyperbolic tangent, since there's no intrinsic
  that computes it directly.
*/

void gelu_kernel(const float* input,
                 float* vals,
                 int row_stride,
                 int iterations,
                 nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    int loop_stride = item_ct1.get_local_range(2);

    const float4* input_cast = reinterpret_cast<const float4*>(input);
    float4* vals_cast = reinterpret_cast<float4*>(vals);

    for (int i = 0; i < iterations; i++) {
        if (i * loop_stride + id < row_stride) {
            float4 data = input_cast[row * row_stride + i * loop_stride + id];

            data.x() = gelu(data.x());
            data.y() = gelu(data.y());
            data.z() = gelu(data.z());
            data.w() = gelu(data.w());

            vals_cast[row * row_stride + i * loop_stride + id] = data;
        }
    }
}

void gelu_kernel(const half* input, half* vals, int row_stride, int iterations, nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    int loop_stride = item_ct1.get_local_range(2);

    const float2* input_cast = reinterpret_cast<const float2*>(input);
    float2* vals_cast = reinterpret_cast<float2*>(vals);

    for (int i = 0; i < iterations; i++) {
        if (i * loop_stride + id < row_stride) {
            float2 vals_vec = input_cast[row * row_stride + i * loop_stride + id];

            half2* vals_half = reinterpret_cast<half2*>(&vals_vec);

            float2 low_data = vals_half[0].convert<float>();   // __half22float2(vals_half[0]);
            float2 high_data = vals_half[1].convert<float>();  // __half22float2(vals_half[1]);

            low_data.x() = gelu(low_data.x());
            low_data.y() = gelu(low_data.y());
            high_data.x() = gelu(high_data.x());
            high_data.y() = gelu(high_data.y());

            vals_half[0] = low_data.convert<half>();   // __float22half2_rn(low_data);
            vals_half[1] = high_data.convert<half>();  // __float22half2_rn(high_data);

            vals_cast[row * row_stride + i * loop_stride + id] = vals_vec;
        }
    }
}

void fused_bias_gelu(const float* input,
                     const float* bias,
                     float* vals,
                     int row_stride,
                     int iterations,
                     nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    int loop_stride = item_ct1.get_local_range(2);

    const float4* input_cast = reinterpret_cast<const float4*>(input);
    float4* vals_cast = reinterpret_cast<float4*>(vals);
    const float4* bias_cast = reinterpret_cast<const float4*>(bias);

    for (int i = 0; i < iterations; i++) {
        if (i * loop_stride + id < row_stride) {
            float4 data = input_cast[row * row_stride + i * loop_stride + id];
            float4 bias_data = bias_cast[i * loop_stride + id];

            data.x() += bias_data.x();
            data.y() += bias_data.y();
            data.z() += bias_data.z();
            data.w() += bias_data.w();

            data.x() = gelu(data.x());
            data.y() = gelu(data.y());
            data.z() = gelu(data.z());
            data.w() = gelu(data.w());

            vals_cast[row * row_stride + i * loop_stride + id] = data;
        }
    }
}

void fused_bias_gelu(const bf16* input,
                     const bf16* bias,
                     bf16* vals,
                     int row_stride,
                     int iterations,
                     nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    int loop_stride = item_ct1.get_local_range(2);

    const ushort4* input_cast = reinterpret_cast<const ushort4*>(input);
    ushort4* vals_cast = reinterpret_cast<ushort4*>(vals);
    const ushort4* bias_cast = reinterpret_cast<const ushort4*>(bias);

    for (int i = 0; i < iterations; i++) {
        if (i * loop_stride + id < row_stride) {
            ushort4 vals_vec = input_cast[row * row_stride + i * loop_stride + id];
            ushort4 bias_vec = bias_cast[i * loop_stride + id];

            float4 data = {float(vals_vec.x()),
                           float(vals_vec.y()),
                           float(vals_vec.z()),
                           float(vals_vec.w())};
            float4 bias = {float(bias_vec.x()),
                           float(bias_vec.y()),
                           float(bias_vec.z()),
                           float(bias_vec.w())};

            data += bias;

            data.x() = gelu(data.x());
            data.y() = gelu(data.y());
            data.z() = gelu(data.z());
            data.w() = gelu(data.w());

            vals_vec.x() = bf16(data.x());
            vals_vec.y() = bf16(data.y());
            vals_vec.z() = bf16(data.z());
            vals_vec.w() = bf16(data.w());

            vals_cast[row * row_stride + i * loop_stride + id] = vals_vec;
        }
    }
}

void fused_bias_gelu(const half* input,
                     const half* bias,
                     half* vals,
                     int row_stride,
                     int iterations,
                     nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    int loop_stride = item_ct1.get_local_range(2);

    const float2* input_cast = reinterpret_cast<const float2*>(input);
    float2* vals_cast = reinterpret_cast<float2*>(vals);
    const float2* bias_cast = reinterpret_cast<const float2*>(bias);

    for (int i = 0; i < iterations; i++) {
        if (i * loop_stride + id < row_stride) {
            float2 vals_vec = input_cast[row * row_stride + i * loop_stride + id];
            float2 bias_vec = bias_cast[i * loop_stride + id];

            half2* vals_half = reinterpret_cast<half2*>(&vals_vec);
            half2* bias_half = reinterpret_cast<half2*>(&bias_vec);

            float2 low_data = vals_half[0].convert<float>();   // __half22float2(vals_half[0]);
            float2 high_data = vals_half[1].convert<float>();  // __half22float2(vals_half[1]);

            float2 low_bias = bias_half[0].convert<float>();   // __half22float2(bias_half[0]);
            float2 high_bias = bias_half[1].convert<float>();  // __half22float2(bias_half[1]);

            low_data.x() += low_bias.x();
            low_data.y() += low_bias.y();
            high_data.x() += high_bias.x();
            high_data.y() += high_bias.y();

            low_data.x() = gelu(low_data.x());
            low_data.y() = gelu(low_data.y());
            high_data.x() = gelu(high_data.x());
            high_data.y() = gelu(high_data.y());

            vals_half[0] = low_data.convert<half>();   // __float22half2_rn(low_data);
            vals_half[1] = high_data.convert<half>();  // __float22half2_rn(high_data);

            vals_cast[row * row_stride + i * loop_stride + id] = vals_vec;
        }
    }
}

void d_gelu_func(float* d_output,
                 const float* gelu_input,
                 const float* bias,
                 int row_stride,
                 int iterations,
                 nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    int loop_stride = item_ct1.get_local_range(2);

    float4* d_output_cast = reinterpret_cast<float4*>(d_output);
    const float4* gelu_input_cast = reinterpret_cast<const float4*>(gelu_input);
    const float4* bias_cast = reinterpret_cast<const float4*>(bias);

    for (int i = 0; i < iterations; i++) {
        if (i * loop_stride + id < row_stride) {
            float4 output_data = d_output_cast[row * row_stride + i * loop_stride + id];
            float4 gelu_input_data = gelu_input_cast[row * row_stride + i * loop_stride + id];
            float4 bias_data = bias_cast[i * loop_stride + id];

            gelu_input_data.x() += bias_data.x();
            gelu_input_data.y() += bias_data.y();
            gelu_input_data.z() += bias_data.z();
            gelu_input_data.w() += bias_data.w();

            output_data.x() *= d_gelu(gelu_input_data.x());
            output_data.y() *= d_gelu(gelu_input_data.y());
            output_data.z() *= d_gelu(gelu_input_data.z());
            output_data.w() *= d_gelu(gelu_input_data.w());

            d_output_cast[row * row_stride + i * loop_stride + id] = output_data;
        }
    }
}

void d_gelu_func(bf16* d_output,
                 const bf16* gelu_input,
                 const bf16* bias,
                 int row_stride,
                 int iterations,
                 nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    int loop_stride = item_ct1.get_local_range(2);

    ushort4* d_output_cast = reinterpret_cast<ushort4*>(d_output);
    const ushort4* gelu_input_cast = reinterpret_cast<const ushort4*>(gelu_input);
    const ushort4* bias_cast = reinterpret_cast<const ushort4*>(bias);

    for (int i = 0; i < iterations; i++) {
        if (i * loop_stride + id < row_stride) {
            ushort4 output_vec = d_output_cast[row * row_stride + i * loop_stride + id];
            ushort4 gelu_input_vec = gelu_input_cast[row * row_stride + i * loop_stride + id];
            ushort4 bias_vec = bias_cast[i * loop_stride + id];

            float4 gelu_input_data = {float(gelu_input_vec.x()),
                                      float(gelu_input_vec.y()),
                                      float(gelu_input_vec.z()),
                                      float(gelu_input_vec.w())};
            float4 bias_data = {
                float(bias_vec.x()),
                float(bias_vec.y()),
                float(bias_vec.z()),
                float(bias_vec.w()),
            };
            float4 output_data = {
                float(output_vec.x()),
                float(output_vec.y()),
                float(output_vec.z()),
                float(output_vec.w()),
            };

            gelu_input_data.x() += bias_data.x();
            gelu_input_data.y() += bias_data.y();
            gelu_input_data.z() += bias_data.z();
            gelu_input_data.w() += bias_data.w();

            output_data.x() *= d_gelu(gelu_input_data.x());
            output_data.y() *= d_gelu(gelu_input_data.y());
            output_data.z() *= d_gelu(gelu_input_data.z());
            output_data.w() *= d_gelu(gelu_input_data.w());

            output_vec.x() = bf16(output_data.x());
            output_vec.y() = bf16(output_data.y());
            output_vec.z() = bf16(output_data.z());
            output_vec.w() = bf16(output_data.w());
            d_output_cast[row * row_stride + i * loop_stride + id] = output_vec;
        }
    }
}

void d_gelu_func(half* d_output,
                 const half* gelu_input,
                 const half* bias,
                 int row_stride,
                 int iterations,
                 nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    int loop_stride = item_ct1.get_local_range(2);

    float2* d_output_cast = reinterpret_cast<float2*>(d_output);
    const float2* gelu_input_cast = reinterpret_cast<const float2*>(gelu_input);
    const float2* bias_cast = reinterpret_cast<const float2*>(bias);

#pragma unroll
    for (int i = 0; i < iterations; i++) {
        if (i * loop_stride + id < row_stride) {
            float2 output_data = d_output_cast[row * row_stride + i * loop_stride + id];
            float2 gelu_input_data = gelu_input_cast[row * row_stride + i * loop_stride + id];
            float2 bias_vec = bias_cast[i * loop_stride + id];

            half2* output_data_half = reinterpret_cast<half2*>(&output_data);
            half2* gelu_input_data_half = reinterpret_cast<half2*>(&gelu_input_data);
            half2* bias_half = reinterpret_cast<half2*>(&bias_vec);

            float2 output_half_0 =
                output_data_half[0].convert<float>();  // __half22float2(output_data_half[0]);
            float2 output_half_1 =
                output_data_half[1].convert<float>();  // __half22float2(output_data_half[1]);

            float2 gelu_input_half_0 =
                gelu_input_data_half[0]
                    .convert<float>();  // __half22float2(gelu_input_data_half[0]);
            float2 gelu_input_half_1 =
                gelu_input_data_half[1]
                    .convert<float>();  // __half22float2(gelu_input_data_half[1]);

            float2 bias_half_0 = bias_half[0].convert<float>();  // __half22float2(bias_half[0]);
            float2 bias_half_1 = bias_half[1].convert<float>();  // __half22float2(bias_half[1]);

            gelu_input_half_0.x() += bias_half_0.x();
            gelu_input_half_0.y() += bias_half_0.y();
            gelu_input_half_1.x() += bias_half_1.x();
            gelu_input_half_1.y() += bias_half_1.y();

            output_half_0.x() *= d_gelu(gelu_input_half_0.x());
            output_half_0.y() *= d_gelu(gelu_input_half_0.y());
            output_half_1.x() *= d_gelu(gelu_input_half_1.x());
            output_half_1.y() *= d_gelu(gelu_input_half_1.y());

            float2 result;
            half2* result_half2 = reinterpret_cast<half2*>(&result);

            result_half2[0] = output_half_0.convert<half>();  // __float22half2_rn(output_half_0);
            result_half2[1] = output_half_1.convert<half>();  // __float22half2_rn(output_half_1);

            d_output_cast[row * row_stride + i * loop_stride + id] = result;
        }
    }
}

template <typename T>
void launch_bias_gelu(const T* input,
                      const T* bias,
                      T* output,
                      int intermediate_size,
                      int batch_size,
                      queue* stream)
{
    int iterations = (intermediate_size + 1023) / 1024;
    int threads = (intermediate_size - 1) / (iterations * 4) + 1;
    range<3> block_dims(1, 1, threads);
    range<3> grid_dims(1, 1, batch_size);

    stream->submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dims * block_dims, block_dims), [=](nd_item<3> item_ct1) {
            fused_bias_gelu(input, bias, output, intermediate_size / 4, iterations, item_ct1);
        });
    });
}

template <typename T>
void launch_gelu(const T* input, T* output, int intermediate_size, int batch_size, queue* stream)
{
    int iterations = (intermediate_size + 1023) / 1024;
    int threads = (intermediate_size - 1) / (iterations * 4) + 1;
    range<3> block_dims(1, 1, threads);
    range<3> grid_dims(1, 1, batch_size);

    stream->submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dims * block_dims, block_dims), [=](nd_item<3> item_ct1) {
            gelu_kernel(input, output, intermediate_size / 4, iterations, item_ct1);
        });
    });
}

template void launch_bias_gelu<float>(const float*, const float*, float*, int, int, queue*);
template void launch_bias_gelu<half>(const half*, const half*, half*, int, int, queue*);
template void launch_bias_gelu<bf16>(const bf16*, const bf16*, bf16*, int, int, queue*);

template void launch_gelu<float>(const float*, float*, int, int, queue*);
template void launch_gelu<half>(const half*, half*, int, int, queue*);

template <typename T>
void launch_d_gelu(T* d_output,
                   const T* input,
                   const T* bias,
                   int intermediate_size,
                   int batch_size,
                   queue* stream)
{
    int iterations = (intermediate_size + 1023) / 1024;
    int threads = (intermediate_size - 1) / (iterations * 4) + 1;
    range<3> block_dims(1, 1, threads);
    range<3> grid_dims(1, 1, batch_size);

    stream->submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dims * block_dims, block_dims), [=](nd_item<3> item_ct1) {
            d_gelu_func(d_output, input, bias, intermediate_size / 4, iterations, item_ct1);
        });
    });
}

template void launch_d_gelu<float>(float*, const float*, const float*, int, int, queue*);
template void launch_d_gelu<half>(half*, const half*, const half*, int, int, queue*);
template void launch_d_gelu<bf16>(bf16*, const bf16*, const bf16*, int, int, queue*);
