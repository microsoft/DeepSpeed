Autotuning
==============
One pain point in model training is to figure out a good performance-relevant configurations such as: micro batch size to fully utilize the hardware and achieve a high throughput number. This configuration exploring process is commonly done manually, but is important since model training is repeated many times and benefits of using a good configuration. Not only is the hand-tuning process time-consuming, but it's outcome is hardware-dependent. This means that a good configuration on one hardware might not be the best on another different hardware. The user thus has to hand tune the configuration again. With DeepSpeed (DS), there are more configuration parameters that could potentially affect the training speed, thus making it more tedious to manually tune the configuration. The had-tuning processes is not needed with Deepspeed's Autotuning framework.

The DeepSpeed Autotuner aims to mitigate this pain point and automatically discover the optimal DeepSpeed configuration that delivers good training speed.
The DeepSpeed Autotuner uses model information, system information, and heuristics to efficiently tune system knobs that affect compute and memory efficiencies, such as ZeRO optimization stages, micro-batch sizes, and many other ZeRO optimization configurations.
It not only reduces the time and resources user spend on tuning, but also can discover configurations better than hand-tuned methods. Moreover, DeepSpeed Autotuning is easy to use, requiring no code change from DeepSpeed users.

Please see the the `Autotuning tutorial <https://www.deepspeed.ai/tutorials/autotuning/>`_ for usage details.

Autotuner
---------------------------------------------------
.. autoclass:: deepspeed.autotuning.autotuner
   :members:
