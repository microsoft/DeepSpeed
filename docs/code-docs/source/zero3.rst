ZeRO
####

The Zero Redundancy Optimizer (ZeRO) removes the memory redundancies across
data-parallel processes by partitioning the three model states (optimizer
states, gradients, and parameters) across data-parallel processes instead of
replicating them. By doing this, it boosts memory efficiency compared to
classic data-parallelism while retaining its computational granularity and
communication efficiency.

#. **ZeRO Stage 1**: The optimizer states (e.g., for `Adam optimizer <https://arxiv.org/abs/1412.6980>`_, 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition.

#. **ZeRO Stage 2**: The reduced 16-bit gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states.

#. **ZeRO Stage 3**: The 16-bit model parameters are partitioned across the processes. ZeRO-3 will automatically collect and partition them during the forward and backward passes.

In addition, ZeRO-3 includes the *infinity offload engine* to form
ZeRO-Infinity ([paper](https://arxiv.org/abs/2104.07857)), which can offload
all model states to both CPU and NVMe memory for huge memory savings.


For a deep dive of our algorithms, please see our `papers <https://www.deepspeed.ai/#publications>`_ on `ZeRO
<https://arxiv.org/abs/1910.02054>`_, `ZeRO-Offload
<https://arxiv.org/abs/2101.06840>`_,
and `ZeRO-Infinity <https://arxiv.org/abs/2104.07857>`_.

.. note::
    DeepSpeed first included offloading capabilities with **ZeRO-Offload**, a
    system for offloading optimizer and gradient states to CPU memory within
    ZeRO-2. **ZeRO-Infinity** is the next generation of offloading
    capabilities, accessible to ZeRO-3. ZeRO-Infinity has all of the savings
    of ZeRO-Offload, plus is able to offload more the model weights and has
    more effective bandwidth utilization and overlapping of computation and
    communication.



Getting Started
---------------

If you are new to DeepSpeed, check out our `Getting Started <https://www.deepspeed.ai/getting-started/>`_ page.

Once you are training with DeepSpeed, enabling ZeRO-3 offload is as simple as enabling it
in your DeepSpeed configuration! Below are a few examples of ZeRO-3 configurations. Please see
our `config guide <https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training>`_
for a complete list of options for configuration and performance tuning.

.. note::
        ZeRO-Infinity and ZeRO-Offload work best with our heavily optimized
        :class:`deepspeed.ops.adam.DeepSpeedCPUAdam` optimizer. We recommend using
        our `optimizer config <https://www.deepspeed.ai/docs/config-json/#optimizer-parameters>`_
        to instruct :meth:`deepspeed.initialize` to build the optimizer for you.

ZeRO Configurations
===================

All the settings for DeepSpeed ZeRO are set with the `DeepSpeedZeroConfig`_.
The dictionary provided under the ``zero_optimization`` entry of the main
DeepSpeed configuration dict will be parsed and validated with this class.
Sub-configurations for parameter offload and optimizer offload settings are
parsed by `DeepSpeedZeroOffloadParamConfig`_ and
`DeepSpeedZeroOffloadOptimizerConfig`_.

.. _DeepSpeedZeroConfig:
.. autopydantic_model:: deepspeed.runtime.zero.config.DeepSpeedZeroConfig

.. _DeepSpeedZeroOffloadParamConfig:
.. autopydantic_model:: deepspeed.runtime.zero.config.DeepSpeedZeroOffloadParamConfig

.. _DeepSpeedZeroOffloadOptimizerConfig:
.. autopydantic_model:: deepspeed.runtime.zero.config.DeepSpeedZeroOffloadOptimizerConfig


Example ZeRO-3 Configurations
=============================

#. Use ZeRO to partition the optimizer states (stage 1), gradients (stage 2),
   and parameters (stage 3).

    .. code-block:: python
        :emphasize-lines: 3

        {
            "zero_optimization": {
                "stage": 3,
            },
            "fp16": {
                "enabled": true
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                "lr": 0.001,
                "betas": [
                    0.8,
                    0.999
                ],
                "eps": 1e-8,
                "weight_decay": 3e-7
                }
            },
            ...
        }


#. Additionally offload the optimizer states and computations to the CPU with ZeRO-Infinity.

    .. code-block:: python

        {
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu"
                }
            },
            ...
        }


#. Save even more memory by offloading parameters to the CPU memory.

    .. code-block:: python

        {
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu"
                }
                "offload_param": {
                    "device": "cpu"
                }
            },
            ...
        }


#. Save even MORE memory by offloading to NVMe (if available on your system):

    .. code-block:: python

        {
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "nvme",
                    "nvme_path": "/nvme_data"
                }
                "offload_param": {
                    "device": "nvme",
                    "nvme_path": "/nvme_data"
                }
            },
            ...
        }

MiCS Configurations
===================

All MiCS configurations are set with `DeepSpeedZeroConfig`. MiCS assumes ZeRO
stage 3 optimization is enabled. For now, there are two configuration fields of
MiCS `mics_shard_size` and `mics_hierarchical_params_gather`. `mics_shard_size`
controls how many devices are used for partitioning the model states.
`mics_hierarchical_params_gather` controls whether we use a two-stage
hierarchical way to gather parameters in the forward computation.
`mics_hierarchical_params_gather` is useful when model states are partitioned
across multiple nodes and the cross-node bandwidth is slow. By default this is
turned off.


Example MiCS Configurations
===========================

#. Use MiCS to partition the model states (including optimizer states,
   gradients, and parameters). The following config example partitions the model
   states to eight devices, and assumes the eight devices are located within a
   single node (`mics_hierarchical_params_gather` is `False`).

    .. code-block:: python
        :emphasize-lines: 3

        {
            "zero_optimization": {
                "stage": 3,
                "mics_shard_size": 8,
                "mics_hierarchical_params_gather": False,
            },
            ...
        }

Assumptions
===========

DeepSpeed automatically coordinates the collection (*i.e.,* all-gather),
partitioning (*i.e.,* scatter), and offloading of parameters at the
granularity of (sub)module ``forward()`` methods. The backward pass is
handled similarly. This strategy has two underlying assumptions:

#. The forward and backward passes of submodules must individually fit in device memory.
   If this not the case, :class:`deepspeed.zero.TiledLinear` implements
   **memory-centric tiling** and works with ZeRO-3 to break linear layers
   into a sequence of smaller submodules that can fit in memory.

#. A module's parameters are only accessed within its own ``__init__`` and ``forward()`` methods.
   Otherwise, DeepSpeed must be instructed to collect and re-partition the parameter.
   See :ref:`external-parameters` for manually coordinating parameters.


Constructing Massive Models
---------------------------

ZeRO-3 enables massive models whose parameters exceed the size of individual
nodes in a system. For the typical case of training without model parallelism,
you can simply allocate your model in our context:

.. code-block:: python

    with deepspeed.zero.Init():
        model = MyLargeModel()


.. autoclass:: deepspeed.zero.Init
    :members:


.. _external-parameters:

Manual Parameter Coordination
-----------------------------

Most models require no modification to be trained with ZeRO-3. However, in
some cases one may need to access model weights outside of the training loop,
or to share weights across submodules during training. DeepSpeed has
several mechanisms to coordinate partitioned weights for ZeRO-3.


Gathering Parameters
====================

DeepSpeed provides mechanisms for collecting (or *gathering*) a partitioned parameter.

Some models partitioned with :class:`deepspeed.zero.Init` may need to access
a module’s weights outside of the class constructor or its ``forward()``
method. We refer to these weights as **external parameters**, since these
parameters are accessed outside of the module that created them. To do so, use
:class:`deepspeed.zero.GatheredParameters` or :meth:`deepspeed.zero.register_external_parameter`.

.. autoclass:: deepspeed.zero.GatheredParameters
    :members:


Registering External Parameters
===============================

ZeRO-3 will automatically collect and partition the model parameters as they
are needed during the forward and backward passes. However, in some cases a
parameter may be used outside of its module's forward pass. We call these
*external* parameters. ZeRO-3 can coordinate these parameters if they are
registered either automatically or manually.


.. note::
    DeepSpeed version ``0.3.15`` includes automatic external parameter
    discovery and registration to support the most common cases. Parameters
    can still be manually registered if they cannot be automatically
    detected.


DeepSpeed can automatically detect the following external parameter scenarios:


#. Parameter access: consider the following pattern common in language models such as GPT:

   The tensor ``embeddings.weight`` is used in both ``embeddings.forward()`` and
   ``compute_logits()``. We call ``embeddings.weight`` an *external* parameter
   because it is used in the training loop outside of its owning module's
   forward pass.


   .. code-block:: python

       class LanguageModel(torch.nn.Module):
           ...
           def forward(self, inputs):
               embeds = self.embeddings(inputs)
               ...
               logits = compute_logits(output, self.embeddings.weight)
               ...


#. Returning a parameter:

   ``CustomLinear`` returns both an output and its own ``bias`` parameter. DeepSpeed
   will detect the external ``bias`` parameter and register it with submodules that
   use ``CustomLinear``.

   .. code-block:: python

       class CustomLinear(torch.nn.Linear):
           def forward(self, *input):
               output = super().forward(*input)
               return output, self.bias



.. autofunction:: deepspeed.zero.register_external_parameter

.. autofunction:: deepspeed.zero.unregister_external_parameter


.. `Module.apply <https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=module+apply#torch.nn.Module.apply>`_

Overriding Module.apply
===============================
A convenient mechanism for customizing model initialization is `Module.apply <https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=module+apply#torch.nn.Module.apply>`_.
With ZeRO stage 3, ``Module.apply`` implementations must account for parameter partitioning by ``zero.Init`` during model initialization. The default behavior of ZeRO stage 3 is to automatically
handle this issue by overriding ``Module.apply`` to ensure that parameters are gathered before access by ``Module.apply``. The benefit of this approach is development convenience, since
users are saved the burden of manual parameter coordination in ``Module.apply``. However, the downside is slow model initialization, since all the model parameters (e.g., billions) are gathered
even though the common usage of ``Module.apply`` is to customize a few parameters. Developers can disable this default behavior by setting the ``override_module_apply`` configuration knob to ``False``,
for faster model initialization at the cost of manually handling partitioned parameters in their ``Module.apply`` implementations.


Memory-Centric Tiling
---------------------

To reduce the working memory requirements of DL training for large models,
ZeRO-Infinity includes technique called *memory-centric tiling* that exploits
the data fetch and release pattern of ZeRO-3 to reduce the working memory
requirements by breaking down a large operator into smaller tiles that can be
executed sequentially. When combined with ZeRO-3, the parameter and gradients
of each tile can be fetched and released one at a time, reducing the working
memory proportional to the number of tiles. Therefore, ZeRO-Infinity can
support operators of arbitrary sizes, without refactoring for model
parallelism to fit them in limited GPU memory.


.. autoclass:: deepspeed.zero.TiledLinear
    :members:


Debugging
---------

Debugging ZeRO training is complicated by the partitioning of parameters, gradients, and optimizer states. None of these 3 groups of tensors (model states) can be normally accessed because of that. To overcome that DeepSpeed provides the following routines for accessing individual model states in both their partitioned (local) and unpartitioned (full) forms.

Important: Please note that, to access the unpartitioned (full) form, these utilities must be called by all processes participating in the training, even if you decide to do something with the result only in the main process. If all processes don't participate these utilities will hang waiting for all processes to send their contribution.

Additionally, you must be aware that these routines return correct data only in specific phases of the training. So for examples the gradients are valid after ``backward`` and before ``step``. The optimizer states are updated after ``step``. Same goes for fp32 master weights.

.. autofunction:: deepspeed.utils.safe_get_full_fp32_param

.. autofunction:: deepspeed.utils.safe_get_full_grad

.. autofunction:: deepspeed.utils.safe_get_full_optimizer_state

.. autofunction:: deepspeed.utils.safe_get_local_fp32_param

.. autofunction:: deepspeed.utils.safe_get_local_grad

.. autofunction:: deepspeed.utils.safe_get_local_optimizer_state


These routines can be used in a training loop as shown in the following snippet.

.. code-block:: python

    backward(loss)
    [...]
    from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad, safe_get_full_optimizer_state
    for n, lp in model.named_parameters():
        # 1. Access the full states
        #  1.1) gradient lookup
        # For zero1 and zero2, gradient lookup must be called after `backward` and before `step`
        # For zero3, gradient lookup must be called after `backward`
        hp_grad = safe_get_full_grad(lp)


        # 1.2) fp32 and optim states can probably be called anywhere in the training loop, but will be updated after `step`
        hp = safe_get_full_fp32_param(lp)
        exp_avg = safe_get_full_optimizer_state(lp, "exp_avg")
        exp_avg_sq = safe_get_full_optimizer_state(lp, "exp_avg_sq")

        # 2. Access the local states (zero3)
        # For zero3, all of the parameters, gradients, and optimizer states are partitioned,
        # and each process can access its corresponding local state.
        local_hp = safe_get_local_fp32_param(lp)
        local_hp_grad = safe_get_local_grad(lp)
        local_exp_avg = safe_get_local_optimizer_state(lp, "exp_avg")
        local_exp_avg_sq = safe_get_local_optimizer_state(lp, "exp_avg_sq")

    [...]
    optimizer.step()



Modifying Partitioned States
----------------------------

Sometimes, a user may want to modify parameters, gradients, or optimizer states outside of the regular training loop. This is currently difficult in ZeRO training because of partitioning. To overcome that, DeepSpeed provides the following routines for modifying the fp32 master parameters and the fp32 optimizer states.

.. autofunction:: deepspeed.utils.safe_set_full_fp32_param

.. autofunction:: deepspeed.utils.safe_set_full_optimizer_state

.. autofunction:: deepspeed.utils.safe_set_full_grad

.. autofunction:: deepspeed.utils.safe_set_local_fp32_param

.. autofunction:: deepspeed.utils.safe_set_local_grad

.. autofunction:: deepspeed.utils.safe_set_local_optimizer_state

.. autofunction:: deepspeed.utils.safe_update_full_grad_vectorized

The routines for modifying parameters and optimizer states can be used at any point after initialization of the DeepSpeed engine (i.e., ``deepspeed.initialize()``) as shown in the following snippet.

.. code-block:: python

    [...]
    from deepspeed.runtime.zero.utils import is_zero_param
    from deepspeed.utils import safe_set_full_fp32_param, safe_set_full_optimizer_state
    from deepspeed.utils import safe_set_local_fp32_param, safe_set_local_optimizer_state
    # Here is an example to zero all the fp32 parameters and optimizer states.
    for n, lp in model.named_parameters():
        # 1. For zero stage 1, 2, or 3 set the full fp32 and their full optim states
        zero_tensor = torch.zeros(lp.ds_shape) if is_zero_param(lp) else torch.zeros(lp.shape)

        safe_set_full_fp32_param(lp, zero_tensor)
        safe_get_full_optimizer_state(lp, zero_tensor, "exp_avg")
        safe_get_full_optimizer_state(lp, zero_tensor, "exp_avg_sq")

        # 2. For zero stage 3, each process sets its local fp32 parameters and their local optimizer states individually
        zero_tensor_local = torch.zeros(lp.ds_tensor.shape)

        safe_set_local_fp32_param(lp, zero_tensor_local)
        safe_set_local_optimizer_state(lp, zero_tensor_local, "exp_avg")
        safe_set_local_optimizer_state(lp, zero_tensor_local, "exp_avg_sq")

    [...]


The routines for modifying gradients can be used after ``backward`` but before ``step`` as shown in the following snippet.

.. code-block:: python

    backward(loss)
    [...]
    from deepspeed.runtime.zero.utils import is_zero_param
    from deepspeed.utils import safe_set_full_grad, safe_set_local_grad
    # Here is an example of how to zero all the gradients.
    for n, lp in model.named_parameters():
        # 1. For zero stage 1, 2, or 3 set the full gradient.
        zero_tensor = torch.zeros(lp.ds_shape) if is_zero_param(lp) else torch.zeros(lp.shape)

        safe_set_full_grad(lp, zero_tensor)

        # 2. For zero stage 3, each process sets its local gradient partition.
        zero_tensor_local = torch.zeros_like(lp.ds_tensor.shape)

        safe_set_local_grad(lp, zero_tensor_local)

    [...]
    optimizer.step()



GPU Memory Management
---------------------

By default at the end of training with ZeRO stage 3 some parameters could remain unpartitioned and use up some gpu memory.
This is done on purpose as an optimization should you resume training again. If you'd like to clear out the cached
parameters that use up gpu memory, you can call ``empty_partition_cache`` method of a DeepSpeed engine.

.. autofunction::deepspeed.DeepSpeedEngine.empty_partition_cache

The following code snippet illustrates this functionality.

.. code-block:: python

    with zero.Init():
        model = MyLargeModel()

    ds_engine, _, _, _ = deepspeed.initialize(model, ...)
    for batch in ...:
        loss = ds_engine(batch)
        ds_engine.backward(batch)
        ds_engine.step()

    # Free GPU memory consumed by model parameters
    ds_engine.empty_partition_cache()


Offload States
--------------

The DeepSpeed engine maintains a set of states in device memory (e.g., CUDA memory). The following API allows you to offload these states to a different device (currently, only CPU memory is supported), reducing the memory footprint on the device.

.. code-block:: python

    def offload_states(self,
                       include: Container[OffloadStateTypeEnum] = None,
                       device: OffloadDeviceEnum = OffloadDeviceEnum.cpu,
                       pin_memory: bool = True,
                       non_blocking: bool = False) -> None:
        """Offload the engine's states to the specified device.

        Arguments:
            include: Optional. The set of states to offload. If not provided, all states are offloaded.
            device: Optional. The device to move the ZeRO optimizer buffers to. Currently only `OffloadDeviceEnum.cpu` is supported.
            pin_memory: Optional. Whether to pin the memory of the offloaded states.
            non_blocking: Optional. Whether to offload the states asynchronously.
        """

You can selectively offload specific states by specifying the ``OffloadStateTypeEnum`` in the include argument. ``OffloadStateTypeEnum`` is an enum that defines the states that can be offloaded. The following states are supported:

* ``OffloadStateTypeEnum.optim_states``: Optimizer states. Currently, only states of DeepSpeed's FusedAdam optimizer are supported.
* ``OffloadStateTypeEnum.hp_params``: FP32 parameters.
* ``OffloadStateTypeEnum.lp_params``: BF16/FP16 parameters.
* ``OffloadStateTypeEnum.lp_grads``: BF16/FP16 gradients.
* ``OffloadStateTypeEnum.contiguous_grad_buffer``: The contiguous gradient buffer for reduce operations.

Note that offloading states comes with a trade-off between memory savings and computational overhead. This API allows states to be reloaded back into device memory when needed.

.. code-block:: python

    def reload_states(self, non_blocking: bool = False) -> None:
        """Reload the engine states to the original device.

        Arguments:
            non_blocking: Optional. Whether to offload the states asynchronously.
        """

Below is an example code snippet demonstrating how to offload FP32 parameters and optimizer states to CPU memory:

.. code-block:: python

    # Offload after forward, backward, and step
    ds_engine.offload_states(include=[OffloadStateTypeEnum.hp_params, OffloadStateTypeEnum.optim_states])

    # Do something requiring a lot of device memory
    ...
    # Load states back to device memory
    ds_engine.reload_states()

``deepspeed.runtime.zero.offload_states.get_state_devices`` returns devices of the specified state.

.. code-block:: python

    def get_state_devices(model, state: OffloadStateTypeEnum) -> Set[torch.device]:
        """Retrieve the devices of the specified state of the model.

        Args:
            model (DeepSpeedEngine): The model whose device allocations are to be checked.
            state (OffloadStateTypeEnum): The specific state for which the devices should be retrieved.

        Returns:
            Set[torch.device]: A set of devices of the specified state.

        """
