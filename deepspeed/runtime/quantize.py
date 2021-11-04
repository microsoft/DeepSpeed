import torch
import math
from deepspeed.utils import log_dist
from deepspeed.utils import logger
from deepspeed.ops.quantizer import ds_quantizer

# number of 2-dimensional parameters in a layer
# this is set for transformer-based models
TWO_D_PARAMS = 6


class Quantizer(object):
    def __init__(self,
                 q_target_bits=8,
                 q_start_bits=16,
                 q_period=100,
                 q_offset=100,
                 q_groups=1,
                 q_mixed_fp16=False,
                 q_change_ratio=0.01,
                 q_type=0,
                 q_rounding=0,
                 q_verbose=False,
                 q_eigenvalue=False,
                 use_quantizer_kernel=False,
                 layer_num=0):

        self.q_target_bits = q_target_bits

        self.q_start_bits = [q_start_bits] * (layer_num if layer_num != 0 else 1)
        self.q_period = [q_period] * (layer_num if layer_num != 0 else 1)
        self.q_offset = q_offset
        self.q_groups = q_groups
        self.q_mixed_fp16 = q_mixed_fp16
        self.q_change_ratio = q_change_ratio
        self.q_type = q_type
        self.qsteps = 0
        self.q_init_period = q_period
        self.quantize_real_ratio = 1.000
        self.q_verbose = q_verbose
        self.q_eigenvalue = q_eigenvalue
        self.use_quantizer_kernel = use_quantizer_kernel
        self.q_rounding = q_rounding
        self.layer_num = layer_num

    def any_precision_switch(self):
        if self.layer_num == 0:
            return True
        result = False
        for index in range(self.layer_num):
            if self.q_start_bits[index] != self.q_target_bits:
                next_step = self.qsteps + (
                    TWO_D_PARAMS * (self.layer_num if self.layer_num != 0 else 1))
                if next_step >= self.q_period[index]:
                    result = True
        return result

    def quantize(self,
                 parameter_group,
                 overflow,
                 eigenvalue_enabled,
                 block_eigenvalue={}):

        if overflow and not eigenvalue_enabled:
            return

        self.step()

        self.update_fp16_ratio()

        for i in range(len(parameter_group)):
            for p in parameter_group[i]:
                if len(p.size()) > 1:
                    param_id = id(p)
                    eigenvalue, layer_id = block_eigenvalue[param_id] if param_id in block_eigenvalue else (None, 0)
                    if eigenvalue is not None:
                        factor = 1 + math.floor(eigenvalue * 4)
                        p.data = self.compute_quantization(p.data, layer_id, factor)
                    else:
                        p.data = self.compute_quantization(p.data, layer_id)

    def step(self):
        self.qsteps += (TWO_D_PARAMS * (self.layer_num if self.layer_num != 0 else 1))

    def sr_quantize(self, input_g, scale, q_range, zero_point=0):
        # Random number generator (Uniform)
        p = input_g.new(input_g.shape).uniform_(-0.5, 0.5)
        if self.q_type == 0:
            input_flat = (input_g / scale + p).round().clamp(-(q_range >> 1),
                                                        (q_range >> 1) - 1) * scale
        else:
           input_flat = ((input_g - zero_point) / scale + p).round().clamp(0,
                                                            (q_range - 1)) * scale + zero_point
        return input_flat

    def mixed_fp16_quantize(self, input, input_q, index):
        if self.q_mixed_fp16 and self.q_start_bits[index] >= (self.q_target_bits - 1):
            input_q = input * self.quantize_real_ratio + (
                1 - self.quantize_real_ratio) * input_q
            return input_q
        return input_q

    def compute_quantization(self, input, index=0, factor=1):
        # fixing the quantization bits based on the training steps
        # when reducing 1 bit at each period, we increase the period
        # to go slowly toward the target quantization bits
        # the period and starting bit can be configured
        if self.q_offset > 0:
            if self.qsteps >= self.q_offset:
                self.q_offset = 0
                self.qsteps = 0
            else:
                return input

        if self.q_start_bits[index] != self.q_target_bits:
            if self.qsteps >= self.q_period[index]:
                self.quantize_real_ratio = 1.0
                if self.q_eigenvalue:
                    self.q_period[index] <<= 1
                    self.q_period[index] *= factor
                    self.q_start_bits[index] -= 1
                else:
                    for i in range(len(self.q_start_bits)):
                        self.q_start_bits[i] -= 1
                        self.q_period[i] <<= 1
                if self.q_verbose:
                    logger.info(
                        f'Quantization settings: current bit-precision = {self.q_start_bits[index]}, step = {self.qsteps}, quantization period = {self.q_period[index]}, index = {index}'
                    )
        assert (self.q_start_bits[index] >= self.q_target_bits), \
            'Quantization bit is lower than target precision bits!'

        # quantize the weights base on the selected bits and the value-range
        if not self.use_quantizer_kernel:
            q_range = 2**self.q_start_bits[index]
            # To aviod the overflow, we use the following formula to convert
            # the input to float32
            input_flat = input.float().view(-1)  
            input_g =input_flat.view(self.q_groups, -1) # split as q_groups
        if self.q_type == 0:  #symmetric
            if self.use_quantizer_kernel:
                if self.q_rounding == 0: # Nearest value rounding
                    input_q = ds_quantizer(input.clone(),
                                        self.q_groups,
                                        self.q_start_bits[index])
                else: # Stochastic Rounding
                    input_q = ds_quantizer(input.clone(),
                                               self.q_groups,
                                               self.q_start_bits[index],
                                               sr=True)
            else:
                scale = ( 2 * torch.max( 
                    input_g.amax(dim=-1, keepdim=True), input_g.amin(dim=-1, keepdim=True).abs() )
                    ) / q_range
                if self.q_rounding == 0:  # Nearest value rounding
                    input_flat = (input_g / scale).round().clamp(-(q_range >> 1),
                                                        (q_range >> 1) - 1) * scale
                else:
                    input_flat = self.sr_quantize(input_g, scale, q_range)

        else:  #asymmetric
            if self.use_quantizer_kernel:
                if self.q_rounding == 0: # Nearest value rounding
                    input_q = ds_quantizer(input.clone(),
                                        self.q_groups,
                                        self.q_start_bits[index],
                                        asym=True)
                else:
                    input_q = ds_quantizer(input.clone(),
                                    self.q_groups,
                                    self.q_start_bits[index],
                                    asym=True,
                                    sr=True)
            else:
                g_min = input_g.amin(dim=-1, keepdim=True)
                scale = (input_g.amax(dim=-1, keepdim=True) - g_min ) / q_range
                # make sure the zero point is mapped to integer 
                # aka zero --> some int num --> zero
                zero_point = (g_min / scale).round() * scale 
                if self.q_rounding == 0: # Nearest value rounding
                    input_flat = ((input_g - zero_point) / scale).round().clamp(0,
                                                        (q_range - 1)) * scale + zero_point
                else:
                    input_flat = self.sr_quantize(input_g, scale, q_range, zero_point)

        if self.use_quantizer_kernel:
            return self.mixed_fp16_quantize(input, input_q, index)
        else:
            if self.q_mixed_fp16 and self.q_start_bits[index] >= (self.q_target_bits -
                                                                  1):
                input_flat = self.quantize_real_ratio * input_g + \
                              (1 - self.quantize_real_ratio) * input_flat
            input_q = input_flat.reshape(input.size())
            # alway change the type of input_q back to input type 
            return input_q.type(input.type())

    def update_fp16_ratio(self):
        if self.q_mixed_fp16:
            if self.quantize_real_ratio > 0:
                self.quantize_real_ratio -= self.q_change_ratio
            else:
                self.quantize_real_ratio = 0.000
