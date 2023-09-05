#include <ATen/ATen.h>
#include "inference_onednn_wrappers.hpp"
#include "inference_sycl_layers.hpp"

struct hash_pair {
  static size_t hash_combine( size_t lhs, size_t rhs ) {
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
  }

  template <class T1, class T2>
  std::size_t operator() (const std::pair<T1, T2> &pair) const {
    return hash_combine(std::hash<T1>()(pair.first), std::hash<T2>()(pair.second));
  }
};

// DNNL engine and stream should be permnantly existing and binding to sycl queue
static std::pair<dnnl::engine, dnnl::stream> get_dnnl_engine_stream(sycl::queue queue) {
  static std::unordered_map<sycl::queue, dnnl::stream> dnnl_streams;
  auto it_stream = dnnl_streams.find(queue);

  static std::unordered_map<std::pair<sycl::device, sycl::context>, dnnl::engine, hash_pair> dnnl_engines;
  auto context = std::make_pair(queue.get_device(), queue.get_context());
  // if hit, we know both engine and queue are preserved
  if (it_stream != dnnl_streams.end()) {
    return std::make_pair(dnnl_engines[context], it_stream->second);
  }

  auto it = dnnl_engines.find(context);

  dnnl::engine engine;
  if (it != dnnl_engines.end()) {
    engine = it->second;
  } else {
    engine = dnnl::sycl_interop::make_engine(context.first, context.second);
    dnnl_engines.emplace(std::make_pair(context, engine));
  }

  dnnl::stream stream = dnnl::sycl_interop::make_stream(engine, queue);
  dnnl_streams.emplace(std::make_pair(queue, stream));

  return std::make_pair(engine, stream);
}

template <typename T, bool bmm>
inline int onednn_matmul(sycl::queue handle,
                         bool trans_src,
                         bool trans_wgt,
                         int m,
                         int n,
                         int k,
                         const float alpha,
                         const float beta,
                         const T* src_ptr,
                         const T* wgt_ptr,
                         T* dst_ptr,
                         int batch)
{
    /*
     * src, [m, k], m: batch, k: in_feature
     * wgt, [k, n], n: k: in_features, out_feature
     * dst, [m, n], m: batch, n: out_features
     */
    auto engine_stream = get_dnnl_engine_stream(handle);
    auto engine = engine_stream.first;
    auto stream = engine_stream.second;

    dnnl::memory::dims src_dims, wgt_dims, dst_dims;
    constexpr auto dnnl_dtype_16 = std::is_same<T, fp16>::value ? dnnl::memory::data_type::f16
                                                                : dnnl::memory::data_type::bf16;
    if constexpr (bmm) {
        src_dims = {batch, m, k};
        wgt_dims = {batch, k, n};
        dst_dims = {batch, m, n};
    } else {
        src_dims = {m, k};
        wgt_dims = {k, n};
        dst_dims = {m, n};
    }

    dnnl::memory::desc src_md, wgt_md, dst_md;

    if constexpr (bmm) {
        src_md = dnnl::memory::desc(
            src_dims,
            dnnl_dtype_16,
            trans_src ? dnnl::memory::format_tag::acb : dnnl::memory::format_tag::abc);
        wgt_md = dnnl::memory::desc(
            wgt_dims,
            dnnl_dtype_16,
            trans_wgt ? dnnl::memory::format_tag::acb : dnnl::memory::format_tag::abc);
        dst_md = dnnl::memory::desc(dst_dims, dnnl_dtype_16, dnnl::memory::format_tag::abc);
    } else {
        src_md = dnnl::memory::desc(
            src_dims,
            dnnl_dtype_16,
            trans_src ? dnnl::memory::format_tag::ba : dnnl::memory::format_tag::ab);
        wgt_md = dnnl::memory::desc(
            wgt_dims,
            dnnl_dtype_16,
            trans_wgt ? dnnl::memory::format_tag::ba : dnnl::memory::format_tag::ab);
        dst_md = dnnl::memory::desc(dst_dims, dnnl_dtype_16, dnnl::memory::format_tag::ab);
    }

    auto src_mem = dnnl::memory(src_md, engine, (void*)src_ptr);
    auto wgt_mem = dnnl::memory(wgt_md, engine, (void*)wgt_ptr);
    auto dst_mem = dnnl::memory(dst_md, engine, (void*)dst_ptr);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    std::unordered_map<int, dnnl::memory> matmul_args;
    if (alpha != 1.0f) {
        float alpha_v(alpha);
        attr.set_scales_mask(DNNL_ARG_DST, /* mask */ 0);
        dnnl::memory alpha_mem({{1}, dnnl::memory::data_type::f32, {1}}, engine, &alpha_v);
        matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, alpha_mem});
    }
    if (beta != 0.0f) {
        dnnl::post_ops po;
        po.append_sum(beta);
        attr.set_post_ops(po);
    }

    auto matmul_pd = dnnl::matmul::primitive_desc(engine, src_md, wgt_md, dst_md, attr);

    auto matmul_prim = dnnl::matmul(matmul_pd);
    dnnl::memory::desc scratchpad_md = matmul_pd.scratchpad_desc();
    auto options = at::TensorOptions()
                       .dtype(at::kByte)
                       .layout(at::kStrided)
                       .device(at::kXPU)
                       .requires_grad(false);
    auto scratchpad_tensor = at::empty({(int64_t)scratchpad_md.get_size()}, options);
    dnnl::memory scratchpad(scratchpad_md, engine, scratchpad_tensor.data_ptr());

    matmul_args.insert({DNNL_ARG_SRC, src_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, wgt_mem});
    matmul_args.insert({DNNL_ARG_DST, dst_mem});
    matmul_args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});

    matmul_prim.execute(stream, matmul_args);

    return 0;
}

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
                     T* dst_ptr)
{
    return onednn_matmul<T, false>(
        handle, trans_src, trans_wgt, m, n, k, alpha, beta, src_ptr, wgt_ptr, dst_ptr, 1);
}

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
                     int batch)
{
    return onednn_matmul<T, true>(
        handle, trans_src, trans_wgt, m, n, k, alpha, beta, src_ptr, wgt_ptr, dst_ptr, batch);
}

template int onednn_matmul_ex(sycl::queue handle,
                              bool trans_src,
                              bool trans_wgt,
                              int m,
                              int n,
                              int k,
                              const float alpha,
                              const float beta,
                              const float* src_ptr,
                              const float* wgt_ptr,
                              float* dst_ptr);

template int onednn_matmul_ex(sycl::queue handle,
                              bool trans_src,
                              bool trans_wgt,
                              int m,
                              int n,
                              int k,
                              const float alpha,
                              const float beta,
                              const bf16* src_ptr,
                              const bf16* wgt_ptr,
                              bf16* dst_ptr);

template int onednn_matmul_ex(sycl::queue handle,
                              bool trans_src,
                              bool trans_wgt,
                              int m,
                              int n,
                              int k,
                              const float alpha,
                              const float beta,
                              const fp16* src_ptr,
                              const fp16* wgt_ptr,
                              fp16* dst_ptr);

template int onednn_batchgemm(sycl::queue handle,
                              int m,
                              int n,
                              int k,
                              const float alpha,
                              const float beta,
                              const float* src_ptr,
                              const float* wgt_ptr,
                              float* dst_ptr,
                              bool trans_src,
                              bool trans_wgt,
                              int batch);

template int onednn_batchgemm(sycl::queue handle,
                              int m,
                              int n,
                              int k,
                              const float alpha,
                              const float beta,
                              const bf16* src_ptr,
                              const bf16* wgt_ptr,
                              bf16* dst_ptr,
                              bool trans_src,
                              bool trans_wgt,
                              int batch);

template int onednn_batchgemm(sycl::queue handle,
                              int m,
                              int n,
                              int k,
                              const float alpha,
                              const float beta,
                              const fp16* src_ptr,
                              const fp16* wgt_ptr,
                              fp16* dst_ptr,
                              bool trans_src,
                              bool trans_wgt,
                              int batch);
