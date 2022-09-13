ds_accelerator = None


def init_accelerator():
    global ds_accelerator

    if ds_accelerator is not None:
        return

    try:
        # TODO: This import should reference an external module to DeepSpeed
        from deepspeed.accelerator.cuda_accelerator import CUDA_Accelerator
        ds_accelerator = CUDA_Accelerator()
        return
    except:
        pass

    try:
        # TODO: This import should reference an external module to DeepSpeed
        from deepspeed.accelerator.xpu_accelerator import XPU_Accelerator
        ds_accelerator = XPU_Accelerator()
        return
    except:
        pass

    assert ds_accelerator is not None, f'failed to instantiate a DeepSpeed accelerator'


init_accelerator()
