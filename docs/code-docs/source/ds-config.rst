DeepSpeed Configurations
========================

.. _deepspeed-configs:

DeepSpeed Config Dictionaries
---------------------------
DeepSpeed APIs use nested dictionaries to specify most parameters.

.. _zero-config:

Zero Config
-----------

.. autopydantic_model:: deepspeed.runtime.zero.config.DeepSpeedZeroConfig

.. autoclass:: deepspeed.runtime.zero.config.ZeroStageEnum

.. autopydantic_model:: deepspeed.runtime.zero.config.DeepSpeedZeroOffloadParamConfig

.. autopydantic_model:: deepspeed.runtime.zero.config.DeepSpeedZeroOffloadOptimizerConfig

.. autoclass:: deepspeed.runtime.zero.offload_config.OffloadDeviceEnum
  :members:
  :undoc-members:
  :member-order: bysource
