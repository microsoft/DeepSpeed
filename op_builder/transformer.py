import torch
from .builder import OpBuilder


class TransformerBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER"
    OP_NAME = "transformer_op"

    def __init__(self, name_prefix='', stochastic_mode=False):
        name = self.OP_NAME + "_stochastic" if stochastic_mode else self.OP_NAME
        super().__init__(name=name, name_prefix=name_prefix)
        self.stochastic_mode = stochastic_mode

    def sources(self):
        return [
            'csrc/transformer/ds_transformer_cuda.cpp',
            'csrc/transformer/cublas_wrappers.cu',
            'csrc/transformer/transform_kernels.cu',
            'csrc/transformer/gelu_kernels.cu',
            'csrc/transformer/dropout_kernels.cu',
            'csrc/transformer/normalize_kernels.cu',
            'csrc/transformer/softmax_kernels.cu',
            'csrc/transformer/general_kernels.cu'
        ]

    def include_paths(self):
        return ['csrc/includes']

    def nvcc_args(self):
        args = [
            '-O3',
            '--use_fast_math',
            '-std=c++14',
            '-U__CUDA_NO_HALF_OPERATORS__',
            '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__'
        ]

        if self.jit_mode:
            # Compile for underlying architecture since we know it at runtime
            CC_MAJOR, CC_MINOR = torch.cuda.get_device_capability()
            compute_capability = f"{CC_MAJOR}{CC_MINOR}"
            args.append('-gencode')
            args.append(
                f'arch=compute_{compute_capability},code=compute_{compute_capability}')
        else:
            # Cross-compile mode, compile for various architectures
            for compute_capability in ['60', '61', '70']:
                args.append('-gencode')
                args.append(
                    f'arch=compute_{compute_capability},code=compute_{compute_capability}'
                )

        if self.stochastic_mode:
            args.append('-D__STOCHASTIC_MODE__')

        return args

    def cxx_args(self):
        return ['-O3', '-std=c++14', '-g', '-Wno-reorder']
