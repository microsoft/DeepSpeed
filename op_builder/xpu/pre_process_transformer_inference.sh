
# disable PTX_AVAILABLE
find ./build/csrc -name "*.h" -exec sed -Ei "s:#define.*PTX_AVAILABLE:// \0:g" {} +

# fix inference_context.h to make it could be migrate
patch ./build/csrc/transformer/inference/includes/inference_context.h << 'DIFF___'
@@ -5,14 +5,31 @@

 #pragma once

-#include <c10/cuda/CUDAStream.h>
+// #include <c10/cuda/CUDAStream.h>
 #include <cuda_runtime_api.h>
 #include <cassert>
 #include <iostream>
 #include <vector>
 #include "cublas_v2.h"
 #include "cuda.h"
+#include <array>
+#include <unordered_map>
+namespace at {
+  namespace cuda {
+    dpct::queue_ptr getCurrentCUDAStream() {
+      auto device_type = c10::DeviceType::XPU;
+      c10::impl::VirtualGuardImpl impl(device_type);
+      c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
+      auto& queue = xpu::get_queue_from_stream(c10_stream);
+      return &queue;
+    }

+    dpct::queue_ptr getStreamFromPool(bool) {
+      // not implemented
+      return nullptr;
+    }
+  }
+}
 #define MEGABYTE (1024 * 1024)
 #define GIGABYTE (1024 * 1024 * 1024)

DIFF___

# fix narrow cast error in pt_binding.cpp
find ./build/csrc/ -type f -exec sed -i "s/inline size_t GetMaxTokenLength()/inline int GetMaxTokenLength()/g" {} +
find ./build/csrc/ -type f -exec sed -i "s/const size_t mlp_1_out_neurons/const int mlp_1_out_neurons/g" {} +

# fix #include <c10/cuda/CUDAStream.h>
find ./build/csrc/ -type f -exec sed -Ei "s:#include <c10/cuda/CUDAStream.h>:// \0:g" {} +
