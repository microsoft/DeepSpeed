import torch

class CompiledModuleWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self._wrapped = module
        self.is_compiled = False

    def __getattr__(self, name):
        if name == 'compile':
            return self.compile

        return super().__getattr__(name)


    def forward(self, *args, **kwargs):
        if not self.is_compiled:
            self._wrapped = torch.compile(self._wrapped, backend="inductor")
            self.is_compiled = True

        return self._wrapped(*args, **kwargs)

    def _custom_method(self, *args, **kwargs):
        pass