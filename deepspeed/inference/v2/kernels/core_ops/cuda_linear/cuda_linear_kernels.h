// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "ds_kernel_utils.h"

void cuda_wf6af16_linear(torch::Tensor& output,
                         torch::Tensor& hidden_states,
                         torch::Tensor& weights);
