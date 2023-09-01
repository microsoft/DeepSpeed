Communication Logging
=====================

DeepSpeed provides a flexible communication logging tool which can
automatically detect and record communication operations launched via
`deepspeed.comm`. NOTE: All logging communication calls are synchronized in
order to provide accurate timing information. This may hamper performance if
your model heavily uses asynchronous communication operations.

Once the logs are populated, they can be summarized with
`deepspeed.comm.log_summary()`. For more detail and example usage, see the
[tutorial](https://www.deepspeed.ai/tutorials/comms-logging/).

The behavior of communication logging can be controlled with values in the
`comms_logger` dictionary of the main DeepSpeed config:

.. _DeepSpeedCommsConfig:
.. autopydantic_model:: deepspeed.comm.config.DeepSpeedCommsConfig
