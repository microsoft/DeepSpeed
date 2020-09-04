
from ..utils import call_to_str


class PipeSchedule:
    """Directs the execution of a pipeline engine by generating sequences of
    :class:`PipeInstruction`.

    Args:
        micro_batches (int): The number of micro-batches that comprise a training step.
        stages (int): The number of pipeline stages.
        stage_id (int): The pipe stage to execute the generated schedule.
    """
    def __init__(self, micro_batches, stages, stage_id):
        super().__init__()
        self.samples = micro_batches
        self.stages = stages
        self.stage_id = stage_id
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

    def steps(self):
        """Yields a list of ``PipelineInstruction``s for each step in the schedule.

        All PipelineSchedule objects must implement ``steps()``.
        """
        raise RuntimeError(f"{self.__class__} must implement steps()")

    def num_pipe_buffers(self):
        """The number of pipeline buffers that will be used by this stage.

        Schedules can should minimize ``num_pipe_buffers()` for memory savings at scale.

        Returns:
            int: The number of buffers for the engine to allocate.
        """
        return self.samples

    def _valid_sample(self, sample_id):
        return 0 <= sample_id < self.samples

    def _valid_stage(self, stage_id):
        return 0 <= stage_id < self.stages
    
    @property
    def stage(self):
        """Stage index used to configure this schedule."""
        return self.stage_id

    @property
    def num_stages(self):
        """The number of total pipeline stages used to configure this schedule."""
        return self.stages

    @property
    def num_samples(self):
        """The number of total samples used to configure this schedule."""
        return self.samples

    @property
    def is_first_stage(self):
        """True if the configured ``stage_id`` is the first stage in the pipeline."""
        return self.stage_id == 0

    @property
    def is_last_stage(self):
        """True if the configured ``stage_id`` is the last stage in the pipeline."""
        return self.stage_id == self.stages - 1

    def _buffer_idx(self, sample_id):
        """Map a sample index to a pipeline buffer index.

        This method uses a cyclic allocation strategy.

        Args:
            sample_id (int): The sample index relative to the beginning of the schedule.

        Returns:
            int: The index of the buffer that should store data for sample ``sample_id``
        """
        assert self._valid_sample(sample_id)
        return sample_id % self.num_pipe_buffers()
    
    def __iter__(self):
        self.it = None
        return self
    
    def __next__(self):
        if self.it is None:
            self.it = self.steps()
        return next(self.it)


class InferenceSchedule(PipeSchedule):
    """A schedule for inferencing batches using hybrid parallelism.
    """

    def steps(self):
        prev_sample_id = -1
        total_steps = self.samples + self.stages - 1
        for step_id in range(total_steps):
            cmds = []
            sample_id = step_id - self.stage_id

            # Alternate send/recv buffers
            if _is_even(self.stage_id):
                recv_buf = step_id % 2
                send_buf = (step_id + 1) % 2
            else:
                recv_buf = (step_id + 1) % 2
                send_buf = step_id % 2

            if self.is_first_stage or self.is_last_stage:
                if self._valid_sample(sample_id):
                    cmds.append(LoadSample(recv_buf))
            
            if _is_even(self.stage_id):
                if self._valid_stage(self.next_stage):
                    if self._valid_sample(sample_id - 1):
                        cmds.append(SendActivation(send_buf))
                if self._valid_stage(self.prev_stage):
                    if self._valid_sample(sample_id):
                        cmds.append(RecvActivation(recv_buf))
            else:
                if self._valid_stage(self.prev_stage):
                    if self._valid_sample(sample_id):
                        cmds.append(RecvActivation(recv_buf))

                if self._valid_stage(self.next_stage):
                    if self._valid_sample(sample_id - 1):
                        cmds.append(SendActivation(send_buf))

            if self._valid_sample(sample_id):
                cmds.append(ForwardPass(recv_buf))
            
            yield cmds
    

    def num_pipe_buffers(self):
        return 2

    
class TrainSchedule(PipeSchedule):
    """A schedule for training a batch using hybrid parallelism.

    Pipeline parallelism is extracted through gradient accumulation and thus
    convergence follows that of a data parallel approach with the same batch
    size.
    """

    def steps(self):
        prev_sample_id = -1
        total_steps = 2 * (self.samples + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the sample id and also whether it is a
            # forward or backward pass step.
            sample_id, is_forward = self._step_to_sample(step_id)

            if self._valid_sample(prev_sample_id):
                prev_buffer = self._buffer_idx(prev_sample_id)
            if self._valid_sample(sample_id):
                curr_buffer = self._buffer_idx(sample_id)

            cmds = []

            # Exchange activations
            if is_forward:
                if self._valid_sample(sample_id) and self._valid_stage(self.prev_stage):
                    cmds.append(RecvActivation(curr_buffer))
                if self._valid_sample(prev_sample_id) and self._valid_stage(
                        self.prev_stage):
                    cmds.append(SendGrad(prev_buffer))
            else:
                if self._valid_sample(prev_sample_id) and self._valid_stage(
                        self.next_stage):
                    cmds.append(SendActivation(prev_buffer))
                if self._valid_sample(sample_id) and self._valid_stage(self.next_stage):
                    cmds.append(RecvGrad(curr_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_sample(sample_id):
                    cmds.append(LoadSample(curr_buffer))

            # Computation
            if self._valid_sample(sample_id):
                if is_forward:
                    cmds.append(ForwardPass(curr_buffer))
                else:
                    cmds.append(BackwardPass(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_sample_id = sample_id
            yield cmds

    def num_pipe_buffers(self):
        """Require as many buffers as the distance from this stage to the last.

        Returns:
            int: The number of buffers required at this stage.
        """
        buffers = min(self.stages - self.stage_id + 1, self.samples)
        return max(2, buffers)
    
    def _step_to_sample(self, step_id):
        if _is_even(step_id) and _is_even(self.stage_id):
            sample_id = self._even_step_forward_id(step_id)
            is_forward = True

        elif _is_odd(step_id) and _is_odd(self.stage_id):
            sample_id = self._odd_step_forward_id(step_id)
            is_forward = True

        elif _is_even(step_id) and _is_odd(self.stage_id):
            sample_id = self._even_step_backward_id(step_id)
            is_forward = False

        elif _is_odd(step_id) and _is_even(self.stage_id):
            sample_id = self._odd_step_backward_id(step_id)
            is_forward = False
        
        else:
            assert False
        
        return sample_id, is_forward


    def _even_step_forward_id(self, step_id):
        base = step_id // 2
        sample_id = int(base - self.stage_id // 2)
        return sample_id

    def _odd_step_forward_id(self, step_id):
        base = (step_id - 1) // 2
        sample_id = int(base - self.stage_id // 2)
        return sample_id

    def _even_step_backward_id(self, step_id):
        base = step_id // 2
        sample_id = int(base - self.stages + (self.stage_id + 1) // 2)
        return sample_id

    def _odd_step_backward_id(self, step_id):
        base = ((step_id - 1) // 2) - self.stages + 1
        sample_id = int(base + self.stage_id // 2)
        return sample_id

class PipeInstruction:
    """Base class for all instructions to be executed by the pipeline engine.

    All keyword arguments are stored as members similar to a ``namedtuple``. These are
    then accessible to the :class:`PipelineEngine` during execution.

    Args:
        kwargs (optional): keyword arguments to store as members
    """
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        self.kwargs = kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        return call_to_str(self.name, **self.kwargs)


class OptimizerStep(PipeInstruction):
    """Performs one step with the optimizer and zeros gradients.

    .. note:: Should be issued after :class:`ReduceGrads` and :class:`ReduceTiedGrads`.

    .. note:: Can be a synchronization point among data-parallel ranks.
    """
    pass

class ReduceGrads(PipeInstruction):
    """Reduce the computed gradients among data-parallel processes within the stage.
    """
    pass

class ReduceTiedGrads(PipeInstruction):
    """Reduce the computed gradients of tied modules within a pipeline-parallel group.

    .. warning::
        The stages included in this synchronization point are not known until
        the model is partitioned among pipeline stages. In the worst case, it
        includes all pipeline stages. This instruction should be scheduled
        carefully to avoid deadlocks.
    """
    pass



class BufferOpInstruction(PipeInstruction):
    """A pipeline instruction that operates on pipeline buffer(s).

    Args:
        buffer_id (int): the index of the pipeline buffer() to modify.
    """
    def __init__(self, buffer_id, **kwargs):
        super().__init__(buffer_id=buffer_id, **kwargs)

# IO
class LoadSample(BufferOpInstruction):
    """Load a sample into a buffer.

    Roughly:

    .. code-block:: python

        buffers['inputs'][buffer_id] = next(data_iter)
    """
    pass

# Compute
class ForwardPass(BufferOpInstruction):
    """Compute a forward pass.
    
    Roughly:

    .. code-block:: python
    
        buffers['ouputs'][buffer_id] = forward(buffers['inputs'][buffer_id])
    """
    pass


class BackwardPass(BufferOpInstruction):
    """Compute a backward pass and accumulate gradients.
    
    Roughly:

    .. code-block:: python
    
        outputs = buffers['ouputs'][buffer_id]
        gradients = buffers['gradients'][buffer_id]
        torch.autograd.backward(tensors=outputs,
                                grad_tensors=gradients)
    """
    pass



# Communication
class SendActivation(BufferOpInstruction):
    """Send activations to the next stage in the pipeline.

    Roughly:

    .. code-block:: python

        send(buffers['outputs'][buffer_id])

    .. note::
        The communication is blocking and must be paired with a :class:`RecvActivation`
        on the next pipeline stage to avoid deadlock.
    """
    pass


class RecvActivation(BufferOpInstruction):
    """Receive activations from the previous stage in the pipeline.

    Roughly:

    .. code-block:: python

        buffers['inputs'][buffer_id] = recv()

    .. note::
        The communication is blocking and must be paired with a :class:`SendActivation`
        on the previous pipeline stage to avoid deadlock.
    """
    pass


class SendGrad(BufferOpInstruction):
    """Send computed gradients to the previous pipeline stage.
    with respect to the received activations

    .. note::
        Only received tensors with ``requires_grad==True`` will produce gradients.
        Missing gradients will be replaced with ``None`` on the receiving stage.

    .. note::
        The communication is blocking and must be paired with a :class:`RecvGrad`
        on the previous pipeline stage to avoid deadlock.
    """
    pass


class RecvGrad(BufferOpInstruction):
    """Receive computed gradients the next pipeline stage.

    .. note::
        Only activations with ``requires_grad==True`` will produce gradients.
        Missing gradients will be replaced with ``None``.

    .. note::
        The communication is blocking and must be paired with a :class:`SendGrad`
        on the next pipeline stage to avoid deadlock.
    """
    pass

def _is_even(x):
    return x % 2 == 0


def _is_odd(x):
    return x % 2 != 0
