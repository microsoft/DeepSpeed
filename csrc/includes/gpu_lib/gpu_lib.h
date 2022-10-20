#ifdef __HIP_PLATFORM_HCC__
#include "rocm.hpp"
namespace gpu_lib {
using namespace rocm;
}
#else
#include "cuda.hpp"
namespace gpu_lib {
using namespace cuda;
}
#endif  // #ifdef __HIP_PLATFORM_HCC__
