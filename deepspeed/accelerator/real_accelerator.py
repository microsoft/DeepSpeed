ds_accelerator = None


def init_accelerator():
    global ds_accelerator

    if ds_accelerator is not None:
        return

    # Ideally, adding a new accelerator should require just following snippet.
    # try:
    #     from infinity.stones import Infinity_Accelerator
    #     ds_accelerator = Infinity_Accelerator()
    #     return
    # except:
    #     pass

    try:
        # TODO: This import should reference an external module to DeepSpeed
        from deepspeed.accelerator.cuda_accelerator import CUDA_Accelerator
    except ImportError as e:
        pass
    else:
        ds_accelerator = CUDA_Accelerator()
        assert ds_accelerator.is_available(), \
            f'CUDA_Accelerator fails is_available() test (import was successful)'
        return

    try:
        # TODO: This import should reference an external module to DeepSpeed
        from deepspeed.accelerator.xpu_accelerator import XPU_Accelerator
    except ImportError as e:
        pass
    else:
        ds_accelerator = XPU_Accelerator()
        assert ds_accelerator.is_available(), \
            f'XPU_Accelerator fails is_available() test (import was successful)'
        return

    assert ds_accelerator is not None, f'failed to instantiate a DeepSpeed accelerator'


init_accelerator()
