'''
Copyright (c) Microsoft Corporation
Licensed under the MIT license.

Use to partition the activations stored for backward propagation
Therefore reduces the memory consumption
Also implements CPU checkpointing and contigious memory checkpointing
Reduces memory consumption and memory fragmentation

Code for rng checkpointing taken from NVIDIA Megatron-LM mpu/random.py
b886b7bb972afe72bac0f5de4f42a4a7bae8ebef
'''

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch
import contextlib
import torch.distributed as dist
import torch
from torch import _C
from torch.cuda import _lazy_call, device as device_ctx_manager
from deepspeed.pt.deepspeed_timer import SynchronizedWallClockTimer as Timers
import torch.distributed as dist

#MP parameters
mp_rank = None
mp_size = None
mp_group = None

#Model Parameters
num_layers = None

#Checkpointing buffers
contigious_data_buffers = []
data_offsets = []

contigious_size_buffers = []
size_offsets = []

timers = None

#optimization flags
PARTITION_ACTIVATIONS = False
PA_TO_CPU = False
CONTIGIOUS_CHECKPOINTING = False
SYNCHRONIZE = False
PROFILE_TIME = False


def see_memory_usage(message, force=False):
    #return
    if not force:
        return
    #dist.barrier()
    if dist.get_rank() == 0:
        print(message)
        print("Memory Allocated ",
              torch.cuda.memory_allocated() / (1024 * 1024 * 1024),
              "GigaBytes")
        print("Max Memory Allocated ",
              torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),
              "GigaBytes")
        print("Cache Allocated ",
              torch.cuda.memory_cached() / (1024 * 1024 * 1024),
              "GigaBytes")
        print("Max cache Allocated ",
              torch.cuda.max_memory_cached() / (1024 * 1024 * 1024),
              "GigaBytes")
        print(" ")
        #input("Press Any Key To Continue ..")


# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'
transport_stream = None
cuda_device = None


def detach_variable(inputs, device=None):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            requires_grad = inp.requires_grad

            if device is not None:
                x = inp.to(device=device)
            else:
                x = inp

            x = x.detach()
            x.requires_grad = requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ",
            type(inputs).__name__)


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)
    else:
        # newer PyTorch
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)


class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """
    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """Set to the initial state (no tracker)."""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception('cuda rng state {} already exists'.format(name))
        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception('cuda rng state {} is not added'.format(name))
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Update the current rng state for later use.
            self.states_[name] = torch.cuda.get_rng_state()
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()


def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    return _CUDA_RNG_STATE_TRACKER


def model_parallel_cuda_manual_seed(seed):
    """Initialize model parallel cuda seed.

    This function should be called after the model parallel is
    initialized. Also, no torch.cuda.manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    Two set of RNG states are tracked:
        default state: This is for data parallelism and is the same among a
                       set of model parallel GPUs but different across
                       different model paralle groups. This is used for
                       example for dropout in the non-model-parallel regions.
        model-parallel state: This state is different among a set of model
                              parallel GPUs, but the same across data parallel
                              groups. This is used for example for dropout in
                              model parallel regions.
    """
    global mpu
    # 2718 is just for fun and any POSITIVE value will work.
    offset = seed + 2718
    model_parallel_seed = offset + mpu.get_model_parallel_rank()
    # Data parallel gets the original sedd.
    data_parallel_seed = seed

    if torch.distributed.get_rank() == 0:
        print('> initializing model parallel cuda seeds on global rank {}, '
              'model parallel rank {}, and data parallel rank {} with '
              'model parallel seed: {} and data parallel seed: {}'.format(
                  torch.distributed.get_rank(),
                  mpu.get_model_parallel_rank(),
                  mpu.get_data_parallel_rank(),
                  model_parallel_seed,
                  data_parallel_seed),
              flush=True)
    _CUDA_RNG_STATE_TRACKER.reset()
    # Set the default state.
    torch.cuda.manual_seed(data_parallel_seed)
    # and model parallel state.
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, model_parallel_seed)


def get_partition_start(item):
    global mp_rank, mp_size, mp_group
    size = item.numel()
    partition_size = size / mp_size
    start = partition_size * mp_rank
    return int(start)


def get_partition_size(item):
    global mp_rank, mp_size, mp_group
    size = item.numel()
    assert size % mp_size == 0, "Doesn't handle if partition activation if item is not divisible by mp size"
    partition_size = size / mp_size
    return int(partition_size)


def get_full_inputs(tensors, device=None):
    inputs = []
    num_args = int(len(tensors) / 2)
    for i in range(num_args - 1):

        item = tensors[2 * i]
        size = tensors[2 * i + 1]

        partition_size = item.numel()
        tensor_size = partition_size * mp_size
        if device is not None:
            flat_tensor = torch.zeros([tensor_size], dtype=item.dtype, device=device)
        else:
            flat_tensor = torch.zeros([tensor_size],
                                      dtype=item.dtype,
                                      device=item.device)
        partitions = []
        for i in range(mp_size):
            part_i = flat_tensor.narrow(0, partition_size * i, partition_size)
            if i == mp_rank:
                part_i.copy_(item)
            partitions.append(part_i)
        dist.all_gather(partitions, partitions[mp_rank], group=mp_group)
        input_tensor = flat_tensor.view(list(size.numpy()))
        item.data = input_tensor.data

        inputs.append(item)
    inputs.append(tensors[-2])

    return tuple(inputs)


class CheckpointFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
       two main changes:
           1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
           2) the states in the model parallel tracker are also properly
              tracked/set/reset.
    """
    @staticmethod
    def forward(ctx, run_function, *args):
        global mpu, timers, SYNCHRONIZE, PROFILE_TIME

        if SYNCHRONIZE:
            torch.cuda.synchronize()

        if timers is None and PROFILE_TIME:
            timers = Timers()

        if PROFILE_TIME:
            timers('forward').start()

        ctx.run_function = run_function
        global num_layers
        global mp_rank, mp_size, mp_group
        global contigious_data_buffers, contigious_size_buffers
        global data_offsets, size_offsets
        if mp_rank is None:
            mp_rank = mpu.get_model_parallel_rank()
            mp_size = mpu.get_model_parallel_world_size()
            mp_group = mpu.get_model_parallel_group()


        global cuda_device, transport_stream, PARTITION_ACTIVATIONS, buffer_0, buffer_1, buffer_0_offset, buffer_1_offset

        if cuda_device is None:
            see_memory_usage("First Forward Begining", force=True)
            if dist.get_rank() == 0:
                print(f"Activation Checkpointing Information")
                print(
                    f"----Partition Activations {PARTITION_ACTIVATIONS}, CPU CHECKPOINTING {PA_TO_CPU}"
                )
                print(
                    f"----Contigious Memory Checkpointing {CONTIGIOUS_CHECKPOINTING} with {num_layers} total layers"
                )
                print(f"----Synchronization {SYNCHRONIZE}")
                print(f"----Profiling {PROFILE_TIME}")

            cuda_device = torch.cuda.current_device()
            transport_stream = torch.cuda.Stream(device=cuda_device)

        if PARTITION_ACTIVATIONS:
            #inputs = [item.detach().contiguous().view(-1).narrow(0, get_partition_start(item), get_partition_size(item)).clone() for item in args[:-1]]
            #inputs.append(args[-1])

            inputs = []
            for i, item in enumerate(args[:-1]):
                partition_size = get_partition_size(item)
                partition = item.detach().contiguous().view(-1).narrow(
                    0,
                    get_partition_start(item),
                    partition_size).clone()

                if CONTIGIOUS_CHECKPOINTING:
                    buffer_device = torch.device(
                        'cpu') if PA_TO_CPU else partition.device

                    if i >= len(contigious_data_buffers):
                        tensor_list = [
                            torch.tensor(()).new_empty([partition_size],
                                                       dtype=partition.dtype,
                                                       device=buffer_device)
                            for i in range(num_layers)
                        ]
                        contigious_data_buffers.append(tensor_list)
                        data_offsets.append(0)
                    elif contigious_data_buffers[i] is None:
                        tensor_list = [
                            torch.tensor(()).new_empty([partition_size],
                                                       dtype=partition.dtype,
                                                       device=buffer_device)
                            for i in range(num_layers)
                        ]
                        contigious_data_buffers[i] = tensor_list
                        data_offsets[i] = 0

                    contigious_partition = contigious_data_buffers[i][
                        data_offsets[i]].data.copy_(partition.data)
                    data_offsets[i] = data_offsets[i] + 1
                    inputs.append(contigious_partition)
                else:
                    partition = partition.cpu() if PA_TO_CPU else partition
                    inputs.append(partition)

            inputs.append(args[-1])

        #just in case something funky is happening such as reuse of inputs
        inputs_cuda = [item.to(cuda_device) for item in args]

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        #ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = run_function(*inputs_cuda)

        del inputs_cuda

        #with torch.cuda.stream(transport_stream):
        #if PARTITION_ACTIVATIONS:
        #    new_args = []
        #    for arg, inp in zip(args,inputs):
        #        size= torch.tensor(arg.size())
        #        arg.data = inp.data
        #        new_args.append(arg)
        #        new_args.append(size)
        #    ctx.save_for_backward(*new_args)

        if PARTITION_ACTIVATIONS:
            new_args = []
            for i, (arg, inp) in enumerate(zip(args, inputs)):
                size = torch.tensor(arg.size())

                arg.data = inp.data
                new_args.append(arg)

                if CONTIGIOUS_CHECKPOINTING:
                    numel = size.numel()
                    if i >= len(contigious_size_buffers):
                        tmp = torch.tensor(())
                        contigious_size_buffers.append(
                            tmp.new_empty([numel * num_layers],
                                          dtype=size.dtype,
                                          device=size.device))
                        size_offsets.append(0)
                    elif contigious_size_buffers[i] is None:
                        tmp = torch.tensor(())
                        contigious_size_buffers[i] = tmp.new_empty([numel * num_layers],
                                                                   dtype=size.dtype,
                                                                   device=size.device)
                        size_offsets[i] = 0

                    contigious_size = contigious_size_buffers[i].narrow(
                        0,
                        size_offsets[i],
                        numel).data.copy_(size.data)
                    contigious_size = contigious_size.view_as(size)
                    size_offsets[i] = size_offsets[i] + numel
                    new_args.append(contigious_size)
                else:
                    new_args.append(size)
                #if dist.get_rank() == 0:
                #    print (f"The stored tensor is {contigious_size} and orginal one is {size} ")

            ctx.save_for_backward(*new_args)
        else:
            ctx.save_for_backward(*args)
        if PROFILE_TIME:
            timers('forward').stop()
            timers.log(['forward'])
        if SYNCHRONIZE:
            torch.cuda.synchronize()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        global timers
        #see_memory_usage("In backward", force=True)
        #removing pointers to the contigious buffer memory
        #so that they can be garbage collected once the checkpoints
        #have been used
        if SYNCHRONIZE:
            torch.cuda.synchronize()
        if PROFILE_TIME:
            timers('backward').start()

        if CONTIGIOUS_CHECKPOINTING:
            global data_offsets, size_offsets
            global contigious_data_buffers, contigious_size_buffers

            for buffers in contigious_data_buffers:
                buffers = []

            #frees up all the pointers to the checkpoints except for the ones
            #stored by save for backward
            contigious_data_buffers = []
            contigious_size_buffers = []
            data_offsets = []
            size_offsets = []

        #see_memory_usage("In backward checkpointing code", force=True)
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), "
                               "please use .backward() if possible")

        global cuda_device, transport_stream, PARTITION_ACTIVATIONS

        if PARTITION_ACTIVATIONS:
            #with torch.cuda.stream(transport_stream):
            inputs = get_full_inputs(ctx.saved_tensors,
                                     device=cuda_device if PA_TO_CPU else None)
            detached_inputs = detach_variable(inputs)
        else:
            inputs = ctx.saved_tensors
            detached_inputs = detach_variable(inputs)

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.cuda.get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # if PARTITION_ACTIVATIONS:
        #     current_stream=torch.cuda.current_stream()
        #     current_stream.wait_stream(transport_stream)

        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs, )
        torch.autograd.backward(outputs, args)

        if PROFILE_TIME:
            timers('backward').stop()
            timers.log(['backward'])
        if SYNCHRONIZE:
            torch.cuda.synchronize()
        return (None, ) + tuple(inp.grad for inp in detached_inputs)


def checkpoint(function, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    return CheckpointFunction.apply(function, *args)


def partition_activations_in_checkpoint(partition_activation):
    global PARTITION_ACTIVATIONS
    PARTITION_ACTIVATIONS = partition_activation
    if dist.get_rank() == 0:
        print(f"**************Partition Activations {PARTITION_ACTIVATIONS}************")


def set_num_layers(nlayers):
    global num_layers
    num_layers = nlayers


def reset():
    if CONTIGIOUS_CHECKPOINTING:
        global data_offsets, size_offsets
        global contigious_data_buffers, contigious_size_buffers

        for buffers in contigious_data_buffers:
            buffers = []

        #frees up all the pointers to the checkpoints except for the ones
        #stored by save for backward
        contigious_data_buffers = []
        contigious_size_buffers = []
        data_offsets = []
        size_offsets = []


def configure(mpu_,
              partition_activations=False,
              contigious_checkpointing=False,
              nlayers=None,
              checkpoint_in_cpu=False,
              synchronize=False,
              profile_backward=False):

    global mpu, num_layers

    global PARTITION_ACTIVATIONS, CONTIGIOUS_CHECKPOINTING, \
            PA_TO_CPU, SYNCHRONIZE, PROFILE_TIME

    num_layers = nlayers
    if checkpoint_in_cpu:
        assert partition_activations, "CPU Checkpointing is only availble with partitioned activations"
    if contigious_checkpointing:
        assert num_layers is not None, "Must specify the number of layers with contigious memory checkpointing"

    mpu = mpu_
    PARTITION_ACTIVATIONS = partition_activations
    CONTIGIOUS_CHECKPOINTING = contigious_checkpointing
    PA_TO_CPU = checkpoint_in_cpu
    SYNCHRONIZE = synchronize
    PROFILE_TIME = profile_backward
