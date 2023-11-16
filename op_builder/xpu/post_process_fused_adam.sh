# delete including useless cuda headers
find ./deepspeed/third-party/csrc -name "multi_tensor_adam.dp.cpp" -exec sed -Ei "s:#include <ATen/cuda/CUDAContext.h>:// \0:g" {} +
find ./deepspeed/third-party/csrc -name "multi_tensor_adam.dp.cpp" -exec sed -Ei "s:#include <ATen/cuda/Exceptions.h>:// \0:g" {} +

# delete AT_CUDA_CHECK
find ./deepspeed/third-party/csrc/ -type f -exec sed -Ei "s:AT_CUDA_CHECK:// \0:g" {} +

# replace multi_tensor_apply_dp.hpp
cp ./op_builder/xpu/multi_tensor_apply.dp.hpp ./deepspeed/third-party/csrc/adam
