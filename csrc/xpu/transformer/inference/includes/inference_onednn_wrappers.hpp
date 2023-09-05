#pragma once
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/sycl.hpp>

template <typename T>
int onednn_matmul_ex(sycl::queue handle,
                     bool trans_src,
                     bool trans_wgt,
                     int m,
                     int n,
                     int k,
                     const float alpha,
                     const float beta,
                     const T* src_ptr,
                     const T* wgt_ptr,
                     T* dst_ptr);

template <typename T>
int onednn_batchgemm(sycl::queue handle,
                     int m,
                     int n,
                     int k,
                     const float alpha,
                     const float beta,
                     const T* src_ptr,
                     const T* wgt_ptr,
                     T* dst_ptr,
                     bool trans_src,
                     bool trans_wgt,
                     int batch);
