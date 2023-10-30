
# fix cg::thread_block_tile<threadsPerHead> to auto
find ./deepspeed/third-party/ -type f -exec sed -Ei "s/cg::\S*/auto/g" {} +

# migrate thread_rank() to get_local_linear_id()
find ./deepspeed/third-party/ -type f -exec sed -i "s/thread_rank()/get_local_linear_id()/g" {} +

# migrate shfl to shuffle
find ./deepspeed/third-party/ -type f -exec sed -Ei "s/\.shfl/\.shuffle/g" {} +

# fix __half to sycl::half
find ./deepspeed/third-party/ -type f -exec sed -Ei "s/__half/sycl::half/g" {} +

# fix half2_raw to half2
find ./deepspeed/third-party/ -type f -exec sed -Ei "s/half2_raw/half2/g" {} +

# migrate meta_group_size to get_group_range().size()
find ./deepspeed/third-party/ -type f -exec sed -Ei "s/meta_group_size[(][)]/get_group_range().size()/g" {} +

# add #include <ipex.h>
find ./deepspeed/third-party/ -type f -exec sed -Ei "s:#include <c10/cuda/CUDAStream.h>:&\n#include <ipex.h>:g" {} +

# fix _free_memory_size is 0 error, give it 20G.
find ./deepspeed/third-party -type f -exec sed -i "s/if (\!_free_memory_size/_free_memory_size = 21474836480\;\n&/g" {} +

# change group_local_memory to group_local_memory_for_overwrite
find ./deepspeed/third-party -type f -exec sed -i "s/group_local_memory</group_local_memory_for_overwrite</g" {} +

# fix attn_softmax_v2 lacking of iterations
find ./deepspeed/third-party/ -type f -exec sed -i "s/attn_softmax_v2<T>/attn_softmax_v2<T, iterations>/g" {} +

# fix device at::kCUDA to at::kXPU
find ./deepspeed/third-party/ -type f -exec sed -i "s/at::kCUDA/at::kXPU/g" {} +

# fix __nv_bfloat16 error
find ./deepspeed/third-party -type f -exec sed -i "s/(__nv_bfloat16)/(sycl::ext::oneapi::bfloat16)/g" {} +
