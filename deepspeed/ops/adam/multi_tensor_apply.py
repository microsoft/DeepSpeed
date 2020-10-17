import torch


class MultiTensorApply(object):
    available = False
    warned = False

    def __init__(self, chunk_size):
        MultiTensorApply.available = True
        self.chunk_size = chunk_size

    #     #try:
    #     #    import amp_C
    #     #    MultiTensorApply.available = True
    #     #    self.chunk_size = chunk_size
    #     #except ImportError as err:
    #     #    MultiTensorApply.available = False
    #     #    MultiTensorApply.import_err = err

    # def check_avail(self):
    #     if MultiTensorApply.available == False:
    #         raise RuntimeError(
    #             "Attempted to call MultiTensorApply method, but MultiTensorApply "
    #             "is not available, possibly because Apex was installed without "
    #             "--cpp_ext --cuda_ext.  Original import error message:",
    #             MultiTensorApply.import_err)

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        # self.check_avail()
        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)
