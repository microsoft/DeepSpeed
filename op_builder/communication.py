"""
Customized wrap for collective communications
"""

from .builder import OpBuilder
import os

class CommunicationBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_COLL_COMM"
    NAME = "communication"
    
    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def include_paths(self):
        NCCL_HOME = os.getenv('NCCL_HOME')
        if not NCCL_HOME:
            NCCL_HOME='/usr/local/cuda'

        inc_path = os.path.join(NCCL_HOME, 'include')
        return [inc_path]

    def sources(self):
        return ['csrc/communication/collective_comm.cpp']