#include "common.hpp"
#include "context.hpp"
#include "strided_batch_gemm.hpp"

template <typename T>
std::vector<torch::Tensor> stridedbatchgemm_forward(const int batchSize,
                                                    const int m,
                                                    const int n,
                                                    const int k,
                                                    const float alpha,
                                                    const float beta,
                                                    const torch::Tensor& matA,
                                                    const torch::Tensor& matB)
{
    CHECK_INPUT(matA);
    CHECK_INPUT(matB);

    auto options = torch::TensorOptions()
                       .dtype(matA.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    StridedBatchGemm<T> _sbgemm =
        StridedBatchGemm<T>(typename StridedBatchGemm<T>::Config(batchSize,
                                                                 m,
                                                                 n,
                                                                 k,
                                                                 alpha,
                                                                 beta,
                                                                 oneapi::mkl::transpose::trans,
                                                                 oneapi::mkl::transpose::nontrans,
                                                                 {0, 0, 0}));

    const T* matA_ptr = (const T*)matA.data_ptr();
    const T* matB_ptr = (const T*)matB.data_ptr();

    auto matC = torch::empty({batchSize, n, m}, options);

    T* matC_ptr = (T*)matC.data_ptr();

    sycl::queue* q = ::SyclContext::Instance().GetCurrentStream();

    _sbgemm.Forward(batchSize, matC_ptr, matA_ptr, matB_ptr, q);
    return {matC};
}

template <typename T>
std::vector<torch::Tensor> stridedbatchgemm_backward(const int batchSize,
                                                     const int m,
                                                     const int n,
                                                     const int k,
                                                     const float alpha,
                                                     const float beta,
                                                     const torch::Tensor& grad_matC,
                                                     const torch::Tensor& matA,
                                                     const torch::Tensor& matB)
{
    CHECK_INPUT(grad_matC);
    CHECK_INPUT(matA);
    CHECK_INPUT(matB);

    auto options = torch::TensorOptions()
                       .dtype(matA.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    StridedBatchGemm<T> _sbgemm =
        StridedBatchGemm<T>(typename StridedBatchGemm<T>::Config(batchSize,
                                                                 m,
                                                                 n,
                                                                 k,
                                                                 alpha,
                                                                 beta,
                                                                 oneapi::mkl::transpose::trans,
                                                                 oneapi::mkl::transpose::nontrans,
                                                                 {0, 0, 0}));

    const T* grad_c_ptr = (const T*)grad_matC.data_ptr();
    const T* matA_ptr = (const T*)matA.data_ptr();
    const T* matB_ptr = (const T*)matB.data_ptr();

    auto grad_matA = torch::empty(matA.sizes(), options);
    auto grad_matB = torch::empty(matB.sizes(), options);
    CHECK_INPUT(grad_matA);
    CHECK_INPUT(grad_matB);

    T* grad_a_ptr = (T*)grad_matA.data_ptr();
    T* grad_b_ptr = (T*)grad_matB.data_ptr();
    sycl::queue* q = ::SyclContext::Instance().GetCurrentStream();

    _sbgemm.Backward(batchSize, grad_c_ptr, matA_ptr, matB_ptr, q, grad_a_ptr, grad_b_ptr);
    return {grad_matA, grad_matB};
}
