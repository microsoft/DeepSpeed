import torch
import torch.distributed as dist
from abc import ABC, abstractclassmethod


class MicroBatchBox:
    def __init__(self, micro_batch_idx, layer_id, block_id):
        self.micro_batch_id = micro_batch_idx
        self.layer_id = layer_id
        self.block_id = block_id
        self.prev_pos = 0
        self.curr_pos = 0
        self.can_offload = True
        self.data_block: OffloadLayerPastData = None

    def to_string(self):
        return f"prev_pos: {self.prev_pos}, curr_pos: {self.curr_pos}, micro_batch_id: {self.micro_batch_id}, layer_id: {self.layer_id}, block_id: {self.block_id}, can_offload: {self.can_offload}"


class MicroBatchBoxList:
    micro_batch_locations = None

    def __init__(self, layer_id, block_id, micro_batch_count):
        self.box_list = []
        for i in range(micro_batch_count):
            self.box_list.append(MicroBatchBox(i, layer_id, block_id))
        self.curr_micro_batch_id = 0

    def get_curr_box(self):
        return self.box_list[self.curr_micro_batch_id]

    def get_curr_pos(self):
        return self.box_list[self.curr_micro_batch_id].curr_pos

    def increment_curr_pos(self, incr):
        self.box_list[self.curr_micro_batch_id].curr_pos += incr


class GPUMemoryPool:
    def __init__(self, mem_pool_size, dtype):
        self.mem_pool_size = mem_pool_size
        self.offset = 0
        self.pool = torch.empty(mem_pool_size,
                                dtype=dtype,
                                device=torch.cuda.current_device())

    def allocate(self, size):
        if self.offset + size <= self.mem_pool_size:
            tensor = torch.narrow(self.pool, 0, self.offset, size)
            self.offset += size
        else:
            tensor = None
        return tensor

    def reset(self):
        self.offset = 0


class CPUMemoryPool:
    def __init__(self, micro_batch_count, layer_id_list, block_shape, max_tokens, dtype):
        self.key_cpu_buffers = {}
        self.value_cpu_buffers = {}

        for idx in range(micro_batch_count):
            self.key_cpu_buffers[idx] = {}
            self.value_cpu_buffers[idx] = {}

            for key in layer_id_list:
                buffer_shape = block_shape.copy()
                buffer_shape[0] = max_tokens

                key_buffer = torch.zeros(buffer_shape,
                                         dtype=dtype,
                                         device='cpu',
                                         pin_memory=True)
                value_buffer = torch.zeros(buffer_shape,
                                           dtype=dtype,
                                           device='cpu',
                                           pin_memory=True)

                self.key_cpu_buffers[idx][key] = key_buffer
                self.value_cpu_buffers[idx][key] = value_buffer

    def get_key_buffer(self, micro_batch_id, layer_id):
        return self.key_cpu_buffers[micro_batch_id][layer_id]

    def get_value_buffer(self, micro_batch_id, layer_id):
        return self.value_cpu_buffers[micro_batch_id][layer_id]


class BlockKey:
    def __init__(self, layer_id, block_id, micro_batch_id=0):
        self.layer_id = layer_id
        self.block_id = block_id
        self.micro_batch_id = micro_batch_id

    def hash(self):
        return hash(f"{self.micro_batch_id}-{self.layer_id}-{self.block_id}")


class LayerPastDataBase(ABC):
    @abstractclassmethod
    def get_data(self):
        pass

    @abstractclassmethod
    def append_data(self, new_data):
        pass

    @abstractclassmethod
    def is_empty(self):
        pass

    @abstractclassmethod
    def reset(self, box):
        pass


class HybridLayerPastData(LayerPastDataBase):
    def __init__(self, data, manager, seq_first):
        self.data = data
        self.promises = []
        self.manager = manager
        self.box = None
        self.non_blocking = True
        self.seq_first = seq_first  # [seq, batch, ...] or [batch, seq, ...]
        self.hidden_shape = None

    def get_data(self):
        if self.box.get_curr_pos() > 0:
            data = torch.narrow(self.data, 0, 0, self.box.get_curr_pos())
            if self.hidden_shape is not None:
                data = data.reshape(data.size(0), data.size(1), *self.hidden_shape)
            data = data if self.seq_first else data.transpose(0, 1)
            micro_batch_locations = self.box.micro_batch_locations[
                self.box.curr_micro_batch_id]
            return data[:, micro_batch_locations[0]:micro_batch_locations[1]]
        else:
            return None

    def append_data(self, new_data):
        assert (new_data.dim() > 2)
        assert self.seq_first
        if not self.seq_first:
            new_data = new_data.transpose(0, 1)
            new_data = new_data.view(new_data.size(0), new_data.size(1), -1)
        elif new_data.dim() == 4:
            # for data [seq, batch, h]
            self.hidden_shape = [new_data.size(2), new_data.size(3)]
            new_data = new_data.reshape(new_data.size(0), new_data.size(1), -1)
        else:
            assert False, "not supported"

        new_tokens = new_data.size(0)
        micro_batch_locations = self.box.micro_batch_locations[
            self.box.curr_micro_batch_id]

        assert new_tokens + self.box.get_curr_pos() <= self.data.size(0), f"new_tokens: {new_tokens}, max: {self.data.size(0)}, box: {self.box.get_curr_pos()}"
        assert new_data.shape == self.data[self.box.get_curr_pos():self.box.get_curr_pos()+new_tokens, micro_batch_locations[0]:micro_batch_locations[1]].shape, f"new_data: {new_data.shape}, expect data: {self.data[self.box.get_curr_pos():self.box.get_curr_pos()+new_tokens, micro_batch_locations[0]:micro_batch_locations[1]].shape}, curr: {self.box.get_curr_pos()}, new_tokens: {new_tokens}"
        assert new_tokens == 1 or self.box.get_curr_pos() == 0

        self.data[self.box.get_curr_pos():self.box.get_curr_pos() + new_tokens,
                  micro_batch_locations[0]:micro_batch_locations[1]] = new_data

        self.box.increment_curr_pos(new_tokens)
        return self.get_data()

    def is_empty(self):
        return self.box.get_curr_pos() == 0

    def reset(self, box):
        self.box = box
        self.box.data_block = self


class HybridLayerPastManager:
    gpu_memory_pool = None

    def __init__(self,
                 micro_batch_count,
                 layer_id_list,
                 block_shape,
                 rank,
                 gpu_memory_pool,
                 seq_first):
        self.boxes = dict()
        self.rank = rank
        self.is_rank0 = (rank == 0)
        self.seq_first = seq_first
        self.curr_micro_batch_id = None
        self.batch_size = block_shape[1] if seq_first else block_shape[0]
        self.gpu_memory_pool = gpu_memory_pool
        self.boxes = self._create_boxes(layer_id_list, micro_batch_count)

        assert len(block_shape) == 3

        self.blocks = []
        block_size = block_shape[0] * block_shape[1] * block_shape[2]
        for _ in range(len(self.boxes)):
            t = self.gpu_memory_pool.allocate(block_size)
            assert t is not None
            block = HybridLayerPastData(t.view(block_shape),
                                        self,
                                        seq_first=self.seq_first)
            self.blocks.append(block)

        assert len(self.blocks) >= len(self.boxes), f"not enough memory pool for prompt, {len(self.blocks)} < {len(self.boxes)}"

        MicroBatchBoxList.micro_batch_locations = self._create_micro_batches_locations(
            self.batch_size,
            micro_batch_count)
        self._schedule()

    def update_micro_batch(self, batch_size, micro_batch_count):
        assert self.batch_size == batch_size
        MicroBatchBoxList.micro_batch_locations = self._create_micro_batches_locations(
            batch_size,
            micro_batch_count)

    def get_block(self, layer_id, block_id):
        key = BlockKey(layer_id, block_id)
        b = self.boxes[key.hash()].data_block
        assert b is not None
        assert b.box.get_curr_box().layer_id == key.layer_id, f"box {b.box.get_curr_box().layer_id} has wrong layer id."
        assert b.box.get_curr_box().block_id == key.block_id, f"box {b.box.get_curr_box().block_id} has wrong block id."

        b.box.curr_micro_batch_id = self.curr_micro_batch_id
        return b

    def return_back(self, block):
        pass

    def set_curr_micro_batch_id(self, micro_batch_id):
        self.curr_micro_batch_id = micro_batch_id

    def _create_micro_batches_locations(self, total_batch_size, micro_batch_count):
        assert total_batch_size % micro_batch_count == 0
        chunk_size = total_batch_size // micro_batch_count
        # [start_index, end_index] list for each micro batch
        return [[i * chunk_size,
                 min((i + 1) * chunk_size,
                     total_batch_size)] for i in range(micro_batch_count)]

    def _create_boxes(self, layer_id_list, micro_batch_count):
        boxes = {}
        for block_id in range(2):
            for layer_id in layer_id_list:
                key = BlockKey(layer_id, block_id)
                boxes[key.hash()] = MicroBatchBoxList(layer_id,
                                                      block_id,
                                                      micro_batch_count)
        return boxes

    def _schedule(self):
        assert len(self.blocks) == len(self.boxes)
        for idx, box in enumerate(self.boxes.values()):
            box.do_offload = False
            self.blocks[idx].reset(box)


class OffloadLayerPastData(LayerPastDataBase):
    def __init__(self, data, manager, stream):
        self.data = data
        self.promises = []
        self.manager = manager
        self.box = None
        self.stream = stream
        self.non_blocking = True

    def get_data(self):
        return torch.narrow(self.data,
                            0,
                            0,
                            self.box.curr_pos) if self.box.curr_pos > 0 else None

    def append_data(self, new_data):
        new_tokens = new_data.size(0)
        assert new_tokens + self.box.curr_pos <= self.data.size(0), f"new_tokens: {new_tokens}, max: {self.data.size(0)}, box: {self.box.to_string()}"
        assert new_data.shape == self.data[self.box.curr_pos:self.box.curr_pos +
                                           new_tokens].shape
        assert new_tokens == 1 or self.box.curr_pos == 0
        self.data[self.box.curr_pos:self.box.curr_pos + new_tokens] = new_data
        self.box.curr_pos += new_tokens
        return self.get_data()

    def is_empty(self):
        return self.box.curr_pos == 0

    def reset(self, box):
        # copy the current box data to cpu
        if self.box is not None:
            if self.box.curr_pos > 0:
                self.copy_to_cpu()
            self.box.data_block = None

        self.box = box
        self.box.data_block = self

        # load the new box data from cpu
        if self.box.curr_pos > 0:
            self.copy_from_cpu()

    def copy_from_cpu(self):
        assert self.box.curr_pos > 0
        assert self.box.prev_pos == self.box.curr_pos, f"Invalid pos values in box - {self.box.to_string()}"

        cpu_buffer = self.manager.get_cpu_buffer(self.box)
        cpu_backup = torch.narrow(cpu_buffer, 0, 0, self.box.curr_pos)
        assert self.box.curr_pos <= self.data.size(0), f"curr_tokens: {self.box.curr_pos}, max: {self.data.size(0)}"
        self.copy(cpu_backup, self.data[:self.box.curr_pos], self.stream)

    def copy_to_cpu(self):
        assert self.box.curr_pos > self.box.prev_pos
        assert (self.box.curr_pos - self.box.prev_pos) == 1 or self.box.prev_pos == 0, f"Invalid pos values in box - {self.box.to_string()}"

        offload_data = self.data[self.box.prev_pos:self.box.curr_pos]

        cpu_buffer = self.manager.get_cpu_buffer(self.box)
        cpu_backup = torch.narrow(cpu_buffer,
                                  0,
                                  self.box.prev_pos,
                                  self.box.curr_pos - self.box.prev_pos)

        self.copy(offload_data, cpu_backup, self.stream)
        self.box.prev_pos = self.box.curr_pos

    def copy(self, src_tensor, dest_tensor, stream):
        class Promise:
            def __init__(self,
                         start_event: torch.cuda.Event,
                         end_event: torch.cuda.Event,
                         data_shape,
                         debug=False):
                self.start_event = start_event
                self.end_event = end_event
                self.data_shape = data_shape
                self.debug = debug

            def wait(self):
                torch.cuda.current_stream().wait_event(self.end_event)
                self.end_event.synchronize()
                if self.debug and dist.get_rank() == 0:
                    self.band_width()

            def band_width(self):
                total_size = 2
                for d in self.data_shape:
                    total_size *= d
                elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
                band_width = total_size * 1000.0 / elapsed_time_ms / 1E9

        if self.non_blocking:
            enable_timing = False
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                assert dest_tensor.shape == src_tensor.shape

                start_event = torch.cuda.Event(enable_timing=enable_timing)
                end_event = torch.cuda.Event(enable_timing=enable_timing)

                start_event.record(stream)
                dest_tensor.copy_(src_tensor, non_blocking=True)
                end_event.record(stream)

            self.promises.append(
                Promise(start_event,
                        end_event,
                        src_tensor.shape,
                        debug=enable_timing))
        else:
            dest_tensor.copy_(src_tensor)

    def wait_data_ready(self):
        [p.wait() for p in self.promises]
        self.promises.clear()


class OffloadLayerPastManager:
    copy_stream = None

    def __init__(self,
                 micro_batch_count,
                 layer_id_list,
                 block_shape,
                 rank,
                 gpu_memory_pool,
                 cpu_memory_pool):
        self.boxes = dict()
        self.offload_boxes = []
        self.curr_offload_box_idx = 0
        if self.copy_stream is None:
            self.copy_stream = torch.cuda.Stream()

        self.curr_micro_batch_id = None
        self.rank = rank
        self.is_rank0 = (rank == 0)
        self.gpu_memory_pool = gpu_memory_pool
        self.cpu_memory_pool = cpu_memory_pool

        for micro_batch_id in range(micro_batch_count):
            for layer_id in layer_id_list:
                for block_id in range(2):
                    key = BlockKey(layer_id, block_id, micro_batch_id)
                    self.boxes[key.hash()] = MicroBatchBox(micro_batch_id,
                                                           layer_id,
                                                           block_id)

        self.blocks = []
        block_size = block_shape[0] * block_shape[1] * block_shape[2] * block_shape[3]
        while True:
            t = self.gpu_memory_pool.allocate(block_size)
            if t is None:
                break
            block = OffloadLayerPastData(t.view(block_shape), self, self.copy_stream)
            self.blocks.append(block)
            if len(self.blocks) == len(self.boxes):
                break

        self._schedule()

    def get_block(self, layer_id, block_id, micro_batch_id=None):
        key = BlockKey(
            layer_id,
            block_id,
            self.curr_micro_batch_id if micro_batch_id is None else micro_batch_id)
        b = self.boxes[key.hash()].data_block

        assert b is not None
        assert b.box.layer_id == layer_id
        assert b.box.block_id == block_id

        b.wait_data_ready()
        return b

    def return_back(self, block):
        if block.box is None or block.box.can_offload:
            new_offload_box_idx = self._next_offload_box()
            block.reset(self.offload_boxes[new_offload_box_idx])

    def get_cpu_buffer(self, box):
        return self.cpu_memory_pool.get_key_buffer(
            box.micro_batch_id,
            box.layer_id
        ) if box.block_id == 0 else self.cpu_memory_pool.get_value_buffer(
            box.micro_batch_id,
            box.layer_id)

    def set_curr_micro_batch_id(self, micro_batch_id):
        self.curr_micro_batch_id = micro_batch_id

    def _schedule(self):
        if len(self.blocks) >= len(self.boxes):
            # We have enough memory blocks, no offload is needed.
            for idx, (_, box) in enumerate(self.boxes.items()):
                box.can_offload = False
                self.blocks[idx].reset(box)
        else:
            offload_blocks = 2
            pin_ratio = (len(self.blocks) - offload_blocks) / len(self.boxes)
            box_list = list(self.boxes.values())
            for idx, b in enumerate(self.blocks[:-offload_blocks]):
                box_id = round(idx / pin_ratio)
                box_list[box_id].can_offload = False
                b.reset(box_list[box_id])

            for _, box in self.boxes.items():
                if box.can_offload:
                    self.offload_boxes.append(box)

            for i in range(offload_blocks):
                self.return_back(self.blocks[-i - 1])

    def _next_offload_box(self):
        while True:
            if self.offload_boxes[
                    self.curr_offload_box_idx].can_offload and self.offload_boxes[
                        self.curr_offload_box_idx].data_block is None:
                return self.curr_offload_box_idx

            if self.curr_offload_box_idx == (len(self.offload_boxes) - 1):
                self.curr_offload_box_idx = 0
            else:
                self.curr_offload_box_idx += 1
