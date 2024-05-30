# PaRO

## Implement of PaRO-DP

* `NIG`: [ParoNIGOptimizer](deepspeed/runtime/zero/stage_NIG.py)
* `NII`: [ParoNIIOptimizer](deepspeed/runtime/zero/stage_NII.py)
* `ING`: [ParoINGOptimizer](deepspeed/runtime/zero/stage_ING.py)
* `IIG`: [ParoIIGOptimizer](deepspeed/runtime/zero/stage_IIG.py)
* `IGG`: [ParoIGGOptimizer](deepspeed/runtime/zero/stage_IGG.py)

## Implement of PaRO-CC

`PaRO-CC`: [InterIntraCommInfo](deepspeed/comm/comm.py)

## Examples of PaRO

[demo](examples/paro/run.sh)

```json
{
  "zero_optimization": {
    "paro_strategy": "NIG"
  },
  "bf16": {
    "enabled": true
  },
  ...
}
```

`paro_strategy` can be set to one of the following values: `NIG`, `NII`, `ING`, `IIG`, `IGG`.