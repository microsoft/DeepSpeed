/*
Copyright 2023 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality for managing pinned CPU memory.
*/

#include <map>
#include "deepspeed_py_aio.h"

struct deepspeed_pin_tensor_t {

    deepspeed_pin_tensor_t();

    ~deepspeed_pin_tensor_t();

    torch::Tensor alloc(const size_t num_elem, const at::ScalarType& elem_type);
    
    bool free(torch::Tensor& locked_tensor);       
};
