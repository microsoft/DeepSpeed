#ifndef DS_GPU_LIB_HPP
#define DS_GPU_LIB_HPP

#ifdef __HIP_PLATFORM_HCC__
#include "gpu_lib/rocm.hpp"
namespace gpu_lib {
using namespace rocm;
}
#else
#include "gpu_lib/cuda.hpp"
namespace gpu_lib {
using namespace cuda;
}
#endif  // #ifdef __HIP_PLATFORM_HCC__
#endif  // #ifndef DS_GPU_LIB_HPP
