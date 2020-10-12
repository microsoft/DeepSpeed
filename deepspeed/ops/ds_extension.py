import os
import time
import torch
from abc import ABC
from pathlib import Path

LOCK = 'build.lock'
DEFAULT_TORCH_EXTENSION_PATH = '/tmp/torch-extensions'


class DSExtension(ABC):
    def __init__(self):
        self.op = None

    def unsafe_load(self):
        # Extension specific implementation to build/load the op itself
        raise NotImplementedError()

    def load_op(self):
        if self.op is None:
            self.op = self.safer_load()
        return self.op

    @staticmethod
    def _wait_if_build_started(ext_path):
        while os.path.isfile(os.path.join(ext_path, LOCK)):
            time.sleep(1000)

    def safer_load(self):
        from torch.utils.cpp_extension import load
        torch_ext_path = os.environ.get('TORCH_EXTENSIONS_DIR',
                                        DEFAULT_TORCH_EXTENSION_PATH)
        ext_path = os.path.join(torch_ext_path, self.ext_name)
        os.makedirs(ext_path, exist_ok=True)

        # Attempt to mitigate build race conditions
        DSExtension._wait_if_build_started(ext_path)
        Path(os.path.join(ext_path, LOCK)).touch()

        op = self.unsafe_load()

        os.remove(os.path.join(ext_path, LOCK))

        return op

    @staticmethod
    def deepspeed_src_path():
        return Path(__file__).parent.absolute()
