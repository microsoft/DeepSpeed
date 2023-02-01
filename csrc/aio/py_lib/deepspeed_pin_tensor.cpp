/*
Copyright 2023 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality for managing pinned CPU memory.
*/

#include "deepspeed_pin_tensor.h"

using namespace std;

deepspeed_pin_tensor_t::deepspeed_pin_tensor_t() {}

deepspeed_pin_tensor_t::~deepspeed_pin_tensor_t() {}

torch::Tensor deepspeed_pin_tensor_t::alloc(const size_t num_elem, const at::ScalarType& elem_type)
{
    const auto num_bytes = num_elem * elementSize(elem_type);
    auto pinned_buffer = ds_page_aligned_alloc(num_bytes, true);
    assert(nullptr != pinned_buffer);

    auto options = torch::TensorOptions().dtype(elem_type).device(torch::kCPU);

    return at::from_blob(pinned_buffer, static_cast<long int>(num_bytes), options);
}

bool deepspeed_pin_tensor_t::free(torch::Tensor& locked_tensor) { return true; }
