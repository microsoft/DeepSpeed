ZeRO Stage 3
############


Assumptions
-----------
#. Individual parameter weights and gradients must fit in worker memory.

#. A module's parameters are only accessed in the owning module's ``forward()``. For exceptions, see :class:`deepspeed.GatheredParameters` and :meth:`register_external_parameter()`.



Partitioned Allocation for Massive Models
-----------------------------------------

.. code-block:: python

    with deepspeed.zero.InitContext():
        model = MyModel(*args)

.. autoclass:: deepspeed.zero.InitContext
    :members:


Manual Parameter Collection
---------------------------

Some models partitioned with :class:`deepspeed.zero.InitContext` may need to access
a module's weights outside of the class constructor or ``forward()``. To do
so outside of the backwards computation graph, use the context
:class:`deepspeed.zero.GatheredParameters`.


.. autoclass:: deepspeed.zero.GatheredParameters
    :members:



Registering External Parameters
-------------------------------

.. autofunction:: deepspeed.zero.register_external_parameter
