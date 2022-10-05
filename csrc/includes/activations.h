/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#pragma once

namespace activation {

/*
Namespace wrapping an enum of activation types we can support in kernels.

Currently isolated from the implementations of the interface since it is reasonable
to want to include this file for build sequences that may not require CUDA (or won't
compile easily in that environment).
*/

enum Type {
    // For pass through
    Identity = 0,

    // NewGelu: OpenAI style GELU calculation
    GELU = 1,

    // Gelu in original BERT repo
    OldGELU = 2,

    // ReLU
    ReLU = 3,

    // Sigmoid activation
    Sigmoid = 4,

    // Sigmoid linear unit, also known as swish
    SiLU = 5,
};

}  // namespace activation
