import torch

class CompiledModuleWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()

        modules = self.__dict__.get('_modules')
        modules['wrapped'] = module
        self.__dict__['wrapped'] = module
        self.is_compiled = False

    def __getattr__(self, name):
        return getattr(self.__dict__['wrapped'], name)


    def forward(self, *args, **kwargs):
        if not self.is_compiled:
            self.__dict__['wrapped'] = torch.compile(self.wrapped)
            self.is_compiled = True

        return self.__dict__['wrapped'](*args, **kwargs)
