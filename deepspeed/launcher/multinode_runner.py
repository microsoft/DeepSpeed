import os
import sys
import shutil
from abc import ABC

from ..utils import logger
from .constants import PDSH_MAX_FAN_OUT


class MultiNodeRunner(ABC):
    def __init__(self, args, world_info_base64):
        self.args = args
        self.user_arguments = self.parse_user_args()
        self.user_script = args.user_script
        self.world_info_base64 = world_info_base64
        self.exports = {}

    def backend_exists(self):
        raise NotImplementedError()

    def get_cmd(self, environment, active_resources):
        raise NotImplementedError()

    def add_export(self, key, var):
        self.exports[key.strip()] = var.strip()

    def parse_user_args(self):
        return list(
            map(lambda x: x if x.startswith("-") else "'{}'".format(x),
                self.args.user_args))


class PDSHRunner(MultiNodeRunner):
    def __init__(self, args, world_info_base64):
        super(PDSHRunner, self).__init__(args, world_info_base64)

    def backend_exists(self):
        return shutil.which('pdsh')

    def get_cmd(self, environment, active_resources):
        environment['PDSH_RCMD_TYPE'] = 'ssh'

        active_workers = ",".join(active_resources.keys())
        logger.info("Running on the following workers: %s" % active_workers)

        # PDSH flags for max node fan out and specific hosts to launch on
        # See https://linux.die.net/man/1/pdsh for flag details
        pdsh_cmd_args = ['pdsh', '-f', str(PDSH_MAX_FAN_OUT), '-w', active_workers]

        exports = ""
        for key, val in self.exports.items():
            exports += "export {}={}; ".format(key, val)

        deepspeed_launch = [
            exports,
            "cd {};".format(os.path.abspath('.')),
            sys.executable,
            "-u",
            "-m",
            "deepspeed.launcher.launch",
            '--world_info={}'.format(self.world_info_base64),
            "--node_rank=%n",
            "--master_addr={}".format(self.args.master_addr),
            "--master_port={}".format(self.args.master_port)
        ]

        return pdsh_cmd_args + deepspeed_launch + [self.user_script
                                                   ] + self.user_arguments


class OpenMPIRunner(MultiNodeRunner):
    def __init__(self, args, world_info_base64, resource_pool):
        super(OpenMPIRunner, self).__init__(args, world_info_base64)
        self.resource_pool = resource_pool

    def backend_exists(self):
        #TODO: check for openmpi existance, not just mpirun
        #TODO: if IB is available we should suggestion mvapich
        return shutil.which('mpirun')

    def get_cmd(self, environment, active_resources):
        '''
        DELETEME: Example for 1-bit adam
        mpirun  -n 8 \
		-hostfile hosts \
		--mca btl ^openib \
		--mca btl_tcp_if_include eth0 \
		-x UCX_TLS=tcp \
		-x PYTHONPATH=$PYTHONPATH \
		-x NCCL_SOCKET_IFNAME=eth0 \
		-x NCCL_IB_DISABLE=1 \
		-x NCCL_IB_CUDA_SUPPORT=0 \
		-x NCCL_DEBUG=INFO \
		model.py
        '''
        #FIXME: Allow for include/exclude at node-level but not gpu-level
        assert self.args.include == "" and self.args.exclude == "", 'openmpi backend does not support worker include/exclusion'
        assert self.args.num_nodes == -1 and self.args.num_gpus == -1, 'openmpi backend does not support limiting num nodes/gpus'
        total_process_count = sum(self.resource_pool.values())

        mpirun_cmd = [
            'mpirun',
            '-n', f'{total_process_count}',
            '-hostfile' ,f'{self.args.hostfile}',
            '--mca', 'btl', '^openib',
            '--mca', 'btl_tcp_if_include', 'eth0',
            '-x', 'UCX_TLS=tcp'
        ]

        export_cmd = []
        for k, v in self.exports.items():
            export_cmd += ['-x', f'{k}={v}']

        python_exec = [sys.executable,
            "-u",
        ]

        return mpirun_cmd + export_cmd + python_exec + [self.user_script] + self.user_arguments


class MVAPICHRunner(MultiNodeRunner):
  pass
