from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, NamedTuple, Union

import torch

from .orthogonalization import orthogonalize
from .utils import allreduce_average, pack, unpack, is_distributed


class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Aggregates gradients across workers into an (approximate) average gradient.
        This method also changes its input gradients. It either sets them to zero if there is no compression,
        or to the compression errors, for error feedback.
        """
        pass


class AllReduce(Aggregator):
    def aggregate(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(gradients) == 0:
            return []
        buffer, shapes = pack(gradients)
        allreduce_average(buffer)
        out = unpack(buffer, shapes)
        for g in gradients:
            g.zero_()
        return out


class Config(NamedTuple):
    rank: int  # lower rank => more aggressive compression
    min_compression_rate: float = 2  # skip compression on some gradients
    num_iters_per_step: int = 1  # lower number => more aggressive compression
    start_compressing_after_num_steps: int = 100


class PowerSGD(Aggregator):
    """
    Applies PowerSGD only after a configurable number of steps,
    and only on parameters with strong compression.
    """

    def __init__(self, params: List[torch.Tensor], config: Config):
        self.config = config
        self.device = list(params)[0].device
        if torch.distributed.get_rank() == 0:
            print("Power SGD init:")
            for p in params:
                print(p.shape)
                print(self._should_compress(p.shape))
        self.is_compressed_mask = [self._should_compress(p.shape) for p in params]

        self.step_counter = 0

        compressed_params, _ = self._split(params)
        self._powersgd = BasicPowerSGD(
            compressed_params,
            config=BasicConfig(
                rank=config.rank,
                num_iters_per_step=config.num_iters_per_step,
            ),
        )
        self._allreduce = AllReduce()

    def aggregate(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        self.step_counter += 1

        if self.step_counter <= self.config.start_compressing_after_num_steps:
            return self._allreduce.aggregate(gradients)

        if torch.distributed.get_rank() == 0:
            print("In powersgd aggregate (before):", self.step_counter)

        compressed_grads, uncompressed_grads = self._split(gradients)
        
        avg_grad_compressed = self._powersgd.aggregate(compressed_grads)
        if torch.distributed.get_rank() == 0:
            print("In powersgd aggregate (after powersgd aggregate):", self.step_counter)
        avg_grad_uncompressed = self._allreduce.aggregate(uncompressed_grads)
        if torch.distributed.get_rank() == 0:
            print("In powersgd aggregate (after uncompressed aggregate):", self.step_counter)
        
        avg_grads = self._merge(avg_grad_compressed, avg_grad_uncompressed)
        # avg_grads = self._merge(
        #     self._powersgd.aggregate(compressed_grads),
        #     self._allreduce.aggregate(uncompressed_grads),
        # )

        if torch.distributed.get_rank() == 0:
            print("In powersgd aggregate (after):", self.step_counter)

        return avg_grads

    def _split(self, params: List[torch.Tensor]):
        compressed_params = []
        uncompressed_params = []
        for param, is_compressed in zip(params, self.is_compressed_mask):
            if is_compressed:
                compressed_params.append(param)
            else:
                uncompressed_params.append(param)
        return compressed_params, uncompressed_params

    def _merge(
        self, compressed: List[torch.Tensor], uncompressed: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        assert len(compressed) + len(uncompressed) == len(self.is_compressed_mask)
        compressed_iter = iter(compressed)
        uncompressed_iter = iter(uncompressed)
        merged_list = []
        for is_compressed in self.is_compressed_mask:
            if is_compressed:
                merged_list.append(next(compressed_iter))
            else:
                merged_list.append(next(uncompressed_iter))

        return merged_list

    def _should_compress(self, shape: torch.Size) -> bool:
        return (
            shape.numel() / avg_compressed_size(shape, self.config)
            > self.config.min_compression_rate
        )


class BasicConfig(NamedTuple):
    rank: int  # lower rank => more aggressive compression
    num_iters_per_step: int = 1  # lower number => more aggressive compression


class BasicPowerSGD(Aggregator):
    def __init__(self, params: List[torch.Tensor], config: BasicConfig):
        # Configuration
        self.config = config
        self.params = list(params)
        self.device = self.params[0].device
        self.dtype = self.params[0].dtype
        self.params_per_shape = self._matrices_per_shape(self.params)

        # State
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        self.step_counter = 0

        # Initilize and allocate the low rank approximation matrices p and q.
        # _ps_buffer and _qs_buffer are contiguous memory that can be easily all-reduced, and
        # _ps and _qs are pointers into this memory.
        # _ps and _qs represent batches p/q for all tensors of the same shape.
        self._ps_buffer, ps_shapes = pack(
            [
                self._init_p_batch(shape, params)
                for shape, params in self.params_per_shape.items()
            ]
        )
        self._ps = unpack(self._ps_buffer, ps_shapes)

        self._qs_buffer, qs_shapes = pack(
            [
                self._init_q_batch(shape, params)
                for shape, params in self.params_per_shape.items()
            ]
        )
        self._qs = unpack(self._qs_buffer, qs_shapes)

    def aggregate(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Create a low-rank approximation of the average gradients by communicating with other workers.
        Modifies its inputs so that they contain the 'approximation error', used for the error feedback
        mechanism.
        """
        # Allocate memory for the return value of this function
        output_tensors = [torch.empty_like(g) for g in gradients]

        # if torch.distributed.get_rank() == 0:
        #     print(">> In aggregate:")
        #     for p in gradients:
        #         print(p.shape) 

        # Group the gradients per shape, and view them as matrices (2D tensors)
        gradients_per_shape = self._matrices_per_shape(gradients)
        outputs_per_shape = self._matrices_per_shape(output_tensors)
        shape_groups = [
            dict(
                shape=shape,
                grads=matrices,
                outputs=outputs_per_shape[shape],
                grad_batch=torch.stack(matrices),
                approximation=torch.zeros(
                    size=(len(matrices), *shape), device=self.device, dtype=self.dtype
                ),
            )
            for shape, matrices in list(gradients_per_shape.items())
        ]

        if torch.distributed.get_rank() == 0:
            print("aggregate: step 1:")

        num_iters_per_step = self.config.num_iters_per_step
        for it in range(num_iters_per_step):
            # Alternate between left and right matrix multiplications
            iter_is_even = (self.step_counter * num_iters_per_step + it) % 2 == 0
            if iter_is_even:
                maybe_transpose = lambda g: g
                out_batches, in_batches = self._qs, self._ps
                out_buffer = self._qs_buffer
            else:
                maybe_transpose = batch_transpose
                out_batches, in_batches = self._ps, self._qs
                out_buffer = self._ps_buffer

            if torch.distributed.get_rank() == 0:
                print("aggregate: step 2:")
            # Matrix multiplication
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                if torch.distributed.get_rank() == 0:
                    print("aggregate: step 2a: in_batch ", in_batch.shape)
                orthogonalize(in_batch)
                if torch.distributed.get_rank() == 0:
                    print("aggregate: step 2b: in_batch ", in_batch.shape)
                # if torch.distributed.get_rank() == 0:
                #     print(">> In aggregator:")
                #     print("Group:", group)
                #     print("Transposed Group:", batch_transpose(maybe_transpose(group["grad_batch"])).shape)
                #     print("in_batch:", in_batch.shape)
                #     print("out_batch:", out_batch.shape)
                torch.bmm(
                    batch_transpose(maybe_transpose(group["grad_batch"])), 
                    in_batch, 
                    out=out_batch
                )
                if torch.distributed.get_rank() == 0:
                    print("aggregate: step 2c: group ", batch_transpose(maybe_transpose(group["grad_batch"])).shape)

            if torch.distributed.get_rank() == 0:
                print("aggregate: step 3:")

            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                maybe_transpose(group["grad_batch"]).baddbmm_(
                    in_batch, 
                    batch_transpose(out_batch), 
                    alpha=-1
                )

            if torch.distributed.get_rank() == 0:
                print("Before aggregate all_reduce:")
            # Average across workers
            if is_distributed():
                num_workers = torch.distributed.get_world_size()
                torch.distributed.all_reduce(out_buffer)
            else:
                num_workers = 1
            if torch.distributed.get_rank() == 0:
                print("After aggregate all_reduce:")

            # Construct low-rank reconstruction and update the approximation and error buffer
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                maybe_transpose(group["approximation"]).baddbmm_(
                    in_batch, 
                    batch_transpose(out_batch),
                    alpha=1/num_workers
                )

        # Un-batch the approximation and error feedback, write to the output
        for group in shape_groups:
            for o, m, approx, mb in zip(
                group["outputs"],
                group["grads"],
                group["approximation"],
                group["grad_batch"],
            ):
                o.copy_(approx)
                m.copy_(mb)

        # Increment the step counter
        self.step_counter += 1

        return output_tensors

    def _init_p_batch(
        self, shape: torch.Size, params: List[torch.Tensor]
    ) -> torch.Tensor:
        rank = min(self.config.rank, min(shape))
        return torch.randn(
            [len(params), shape[0], rank], generator=self.generator, device=self.device
        )

    def _init_q_batch(
        self, shape: torch.Size, params: List[torch.Tensor]
    ) -> torch.Tensor:
        rank = min(self.config.rank, min(shape))
        return torch.randn(
            [len(params), shape[1], rank], generator=self.generator, device=self.device
        )

    @classmethod
    def _matrices_per_shape(
        cls,
        tensors: List[torch.Tensor],
    ) -> Dict[torch.Size, List[torch.Tensor]]:
        shape2tensors = defaultdict(list)
        for tensor in tensors:
            matrix = view_as_matrix(tensor)
            shape = matrix.shape
            shape2tensors[shape].append(matrix)
        return shape2tensors

    @property
    def uncompressed_num_floats(self) -> int:
        return sum(param.shape.numel() for param in self.params)

    @property
    def compressed_num_floats(self) -> float:
        return sum(avg_compressed_size(p.shape, self.config) for p in self.params)

    @property
    def compression_rate(self) -> float:
        return self.uncompressed_num_floats / self.compressed_num_floats



def batch_transpose(batch_of_matrices):
    return batch_of_matrices.permute([0, 2, 1])


def view_as_matrix(tensor: torch.Tensor):
    """
    Reshape a gradient tensor into a matrix shape, where the matrix has structure
    [output features, input features].
    For a convolutional layer, this groups all "kernel" dimensions with "input features".
    """
    return tensor.view(tensor.shape[0], -1)


def avg_compressed_size(shape: torch.Size, config: Union[Config, BasicConfig]) -> float:
    rank = min(config.rank, min(shape))
    return 0.5 * config.num_iters_per_step * rank * sum(shape)
