# Copyright 2020 The Microsoft DeepSpeed Team
"""
DeepSpeed runner is the main front-end to launching multi-worker
training jobs with DeepSpeed. By default this uses pdsh to parallel
ssh into multiple worker nodes and launch all the neccisary processes
per rank for training.
"""

import os
import sys
import json
import shutil
import base64
import argparse
import subprocess
import collections
from copy import deepcopy

import torch.cuda

from .multinode_runner import PDSHRunner, OpenMPIRunner, MVAPICHRunner
from .constants import PDSH_LAUNCHER, OPENMPI_LAUNCHER, MVAPICH_LAUNCHER
from ..constants import TORCH_DISTRIBUTED_DEFAULT_PORT
from ..utils import logger
from ..elasticity.constants import DEEPSPEED_ELASTICITY_CONFIG, RUNNER_PID_FILE

DLTS_HOSTFILE = "/job/hostfile"
EXPORT_ENVS = ["NCCL", "PYTHON", "MV2", 'UCX']
DEEPSPEED_ENVIRONMENT_NAME = ".deepspeed_env"
DEEPSPEED_ENVIRONMENT_PATHS = [os.path.expanduser("~"), '.']
PDSH_MAX_FAN_OUT = 1024

ELASTIC_TRAINING = 'IS_ELASTIC_TRAINING_JOB'
ELASTIC_TRAINING_DEFAULT = 'false'


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="DeepSpeed runner to help launch distributed "
        "multi-node/multi-gpu training jobs.")

    parser.add_argument("-H",
                        "--hostfile",
                        type=str,
                        default=DLTS_HOSTFILE,
                        help="Hostfile path (in MPI style) that defines the "
                        "resource pool available to the job (e.g., "
                        "worker-0 slots=4)")

    parser.add_argument("-i",
                        "--include",
                        type=str,
                        default="",
                        help='''Specify hardware resources to use during execution.
                        String format is
                                NODE_SPEC[@NODE_SPEC ...],
                        where
                                NODE_SPEC=NAME[:SLOT[,SLOT ...]].
                        If :SLOT is omitted, include all slots on that host.
                        Example: -i "worker-0@worker-1:0,2" will use all slots
                        on worker-0 and slots [0, 2] on worker-1.
                        ''')

    parser.add_argument("-e",
                        "--exclude",
                        type=str,
                        default="",
                        help='''Specify hardware resources to NOT use during execution.
                        Mutually exclusive with --include. Resource formatting
                        is the same as --include.
                        Example: -e "worker-1:0" will use all available
                        resources except slot 0 on worker-1.
                        ''')

    parser.add_argument("--num_nodes",
                        type=int,
                        default=-1,
                        help="Total number of worker nodes to run on, this will use "
                        "the top N hosts from the given hostfile.")

    parser.add_argument("--num_gpus",
                        type=int,
                        default=-1,
                        help="Max number of GPUs to use on each node, will use "
                        "[0:N) GPU ids on each node.")

    parser.add_argument("--master_port",
                        default=TORCH_DISTRIBUTED_DEFAULT_PORT,
                        type=int,
                        help="(optional) Port used by PyTorch distributed for "
                        "communication during training.")

    parser.add_argument("--master_addr",
                        default="",
                        type=str,
                        help="(optional) IP address of node 0, will be "
                        "inferred via 'hostname -I' if not specified.")

    parser.add_argument("--launcher",
                        default=PDSH_LAUNCHER,
                        type=str,
                        help="(optional) choose launcher backend for multi-node "
                        "training. Options currently include PDSH, OpenMPI, MVAPICH.")

    parser.add_argument("--launcher_args",
                        default="",
                        type=str,
                        help="(optional) pass launcher specific arguments as a "
                        "single quoted argument.")

    parser.add_argument("--force_multi",
                        action="store_true",
                        help="Force multi-node launcher mode, helps in cases where user "
                        "wants to launch on single remote node.")

    parser.add_argument("user_script",
                        type=str,
                        help="User script to launch, followed by any required "
                        "arguments.")
    parser.add_argument('user_args', nargs=argparse.REMAINDER)
    return parser.parse_args(args=args)


def fetch_hostfile(hostfile_path):
    if not os.path.isfile(hostfile_path):
        logger.warning("Unable to find hostfile, will proceed with training "
                       "with local resources only.")
        return None

    # e.g., worker-0 slots=16
    with open(hostfile_path, 'r') as fd:
        resource_pool = collections.OrderedDict()
        for line in fd.readlines():
            line = line.strip()
            if line == '':
                # skip empty lines
                continue
            try:
                hostname, slots = line.split()
                _, slot_count = slots.split("=")
                slot_count = int(slot_count)
            except ValueError as err:
                logger.error("Hostfile is not formatted correctly, unable to "
                             "proceed with training.")
                raise err
            if hostname in resource_pool:
                logger.error("Hostfile contains duplicate hosts, unable to "
                             "proceed with training.")
                raise ValueError("host {} is already defined".format(hostname))
            resource_pool[hostname] = slot_count

    return resource_pool


def parse_resource_filter(host_info, include_str="", exclude_str=""):
    '''Parse an inclusion or exclusion string and filter a hostfile dictionary.

    String format is NODE_SPEC[@NODE_SPEC ...], where
        NODE_SPEC = NAME[:SLOT[,SLOT ...]].
    If :SLOT is omitted, include/exclude all slots on that host.

    Examples:
        include_str="worker-0@worker-1:0,2" will use all slots on worker-0 and
          slots [0, 2] on worker-1.
        exclude_str="worker-1:0" will use all available resources except
          slot 0 on worker-1.
    '''

    # Constants that define our syntax
    NODE_SEP = '@'
    SLOT_LIST_START = ':'
    SLOT_SEP = ','

    # Ensure include/exclude are mutually exclusive
    if (include_str != "") and (exclude_str != ""):
        raise ValueError('include_str and exclude_str are mutually exclusive.')

    # no-op
    if (include_str == "") and (exclude_str == ""):
        return host_info

    # Either build from scratch or remove items
    filtered_hosts = dict()
    if include_str:
        parse_str = include_str
    if exclude_str != "":
        filtered_hosts = deepcopy(host_info)
        parse_str = exclude_str

    # foreach node in the list
    for node_config in parse_str.split(NODE_SEP):
        # Node can either be alone or node:slot,slot,slot
        if SLOT_LIST_START in node_config:
            hostname, slots = node_config.split(SLOT_LIST_START)
            slots = [int(x) for x in slots.split(SLOT_SEP)]

            # sanity checks
            if hostname not in host_info:
                raise ValueError("Hostname '{}' not found in hostfile".format(hostname))
            for s in slots:
                if s not in host_info[hostname]:
                    raise ValueError("No slot '{}' specified on host '{}'".format(
                        s,
                        hostname))

            # If include string, build the list from here
            if include_str:
                filtered_hosts[hostname] = slots
            elif exclude_str:
                for s in slots:
                    logger.info('removing {} from {}'.format(s, hostname))
                    filtered_hosts[hostname].remove(s)

        # User just specified the whole node
        else:
            hostname = node_config
            # sanity check hostname
            if hostname not in host_info:
                raise ValueError("Hostname '{}' not found in hostfile".format(hostname))

            if include_str:
                filtered_hosts[hostname] = host_info[hostname]
            elif exclude_str:
                filtered_hosts[hostname] = []

    # Post-processing to remove duplicates and empty nodes
    del_keys = []
    for hostname in filtered_hosts:
        # Remove duplicates
        filtered_hosts[hostname] = list(set(filtered_hosts[hostname]))
        # Remove empty hosts
        if len(filtered_hosts[hostname]) == 0:
            del_keys.append(hostname)
    for name in del_keys:
        del filtered_hosts[name]

    # Lastly, go over filtered_hosts and convert to a OrderedDict() to ensure
    # we map ranks to nodes correctly by maintaining host_info ordering.
    ordered_hosts = collections.OrderedDict()
    for host in host_info:
        if host in filtered_hosts:
            ordered_hosts[host] = filtered_hosts[host]

    return ordered_hosts


def parse_inclusion_exclusion(resource_pool, inclusion, exclusion):
    active_resources = collections.OrderedDict()
    for hostname, slots in resource_pool.items():
        active_resources[hostname] = list(range(slots))

    return parse_resource_filter(active_resources,
                                 include_str=inclusion,
                                 exclude_str=exclusion)


def encode64(world_info, json_dump=True):
    if json_dump:
        world_info_json = json.dumps(world_info).encode('utf-8')
    else:
        world_info_json = world_info.encode('utf-8')
    world_info_base64 = base64.urlsafe_b64encode(world_info_json).decode('utf-8')
    return world_info_base64


def get_env(name, env_list, default=None):
    value = default
    for env in env_list:
        if name in env:
            value = env[name]
    assert value is not None, f"Unable to find {name} in env"
    return value


def rescale_resources(args):
    # after elastic scale up/down event re-read hostfile
    resource_pool = fetch_hostfile(args.hostfile)
    active_resources = parse_inclusion_exclusion(resource_pool,
                                                 args.include,
                                                 args.exclude)
    return active_resources, encode64(active_resources)


def main(args=None):
    args = parse_args(args)

    if args.num_nodes >= 0 or args.num_gpus >= 0:
        if args.include != "" or args.exclude != "":
            raise ValueError("Cannot specify num_nodes/gpus with include/exclude")

    multi_node_exec = True
    resource_pool = fetch_hostfile(args.hostfile)
    if not resource_pool:
        resource_pool = {}
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("Unable to proceed, no GPU resources available")
        resource_pool['localhost'] = device_count
        args.master_addr = "127.0.0.1"
        multi_node_exec = False

    if not multi_node_exec and args.num_nodes > 1:
        raise ValueError("Num nodes is >1 but no extra nodes available via hostfile")

    active_resources = parse_inclusion_exclusion(resource_pool,
                                                 args.include,
                                                 args.exclude)

    env = os.environ.copy()

    if not args.master_addr:
        first_host = list(active_resources.keys())[0]
        hostname_cmd = ["ssh {} hostname -I".format(first_host)]
        result = subprocess.check_output(hostname_cmd, shell=True)
        args.master_addr = result.decode('utf-8').split()[0]
        logger.info("Using IP address of {} for node {}".format(
            args.master_addr,
            first_host))

    if args.num_nodes > 0:
        updated_active_resources = collections.OrderedDict()
        for count, hostname in enumerate(active_resources.keys()):
            if args.num_nodes == count:
                break
            updated_active_resources[hostname] = active_resources[hostname]
        active_resources = updated_active_resources

    if args.num_gpus > 0:
        updated_active_resources = collections.OrderedDict()
        for hostname in active_resources.keys():
            updated_active_resources[hostname] = list(range(args.num_gpus))
        active_resources = updated_active_resources

    # encode world info as base64 to make it easier to pass via command line
    world_info_base64 = encode_world_info(active_resources)

    multi_node_exec = args.force_multi or len(active_resources) > 1

    env_file = {}
    for environ_path in DEEPSPEED_ENVIRONMENT_PATHS:
        environ_file = os.path.join(environ_path, DEEPSPEED_ENVIRONMENT_NAME)
        if os.path.isfile(environ_file):
            with open(environ_file, 'r') as fd:
                for var in fd.readlines():
                    key, val = var.split('=')
                    env_file[key] = val.strip()

    if DEEPSPEED_ELASTICITY_CONFIG in env:
        elastic_config = get_env(DEEPSPEED_ELASTICITY_CONFIG, [env, env_file])
        elastic_config_json = json.loads(elastic_config)

        assert DEEPSPEED_ELASTICITY_CONFIG not in env_file
        env_file[DEEPSPEED_ELASTICITY_CONFIG] = encode64(elastic_config, json_dump=False)

        from ..elasticity import compute_elastic_config
        from .. import __version__

        world_size = sum(map(lambda w: len(w), active_resources.values()))

        final_batch_size, valid_gpus, micro_batch_size, final_world_size = compute_elastic_config(
            ds_config=elastic_config_json,
            target_deepspeed_version=__version__,
            world_size=world_size)

        if world_size != final_world_size:
            logger.info(
                f"Modifying world size to be within the valid gpus range needed for elastic training: {world_size} -> {final_world_size}"
            )
            curr_world_size = sum(map(lambda w: len(w), active_resources.values()))
            while curr_world_size != final_world_size:
                last_worker = list(active_resources.keys())[-1]
                if len(active_resources[last_worker]) > 0:
                    active_resources[last_worker].pop()
                else:
                    active_resources.pop(last_worker)
                curr_world_size = sum(map(lambda w: len(w), active_resources.values()))
            logger.info(f"Updated active resources: {active_resources}")

    # This auto support will work for the deepspeed launcher only
    auto_elasticity_enabled = False
    if ELASTIC_TRAINING in env or ELASTIC_TRAINING in env_file:
        is_elastic_training = get_env(ELASTIC_TRAINING, [env, env_file])
        if is_elastic_training.lower() == 'true':
            auto_elasticity_enabled = True
            logger.info(
                "DeepSpeed Auto Elasticity Enabled. Ignoring all arguments to deepspeed launcher."
            )
        # add ELASTIC_TRAINING to environment file if it's not there already
        if ELASTIC_TRAINING in env and ELASTIC_TRAINING not in env_file:
            env_file[ELASTIC_TRAINING] = is_elastic_training.lower()

    if auto_elasticity_enabled:
        relaunch_cmd = ["deepspeed"] + ["--master_port={}".format(args.master_port)
                                        ] + [args.user_script] + args.user_args
        encoded_cmd = encode64(relaunch_cmd)
        assert args.hostfile == DLTS_HOSTFILE, "auto elasticity doesn't support custom hostfile paths"
        assert args.include == "" and args.exclude == "" and args.num_nodes == -1 and args.num_gpus == -1, "auto elasticity doesn't support launching on subset of job"

    # encode world info as base64 to make it easier to pass via command line
    world_info_base64 = encode64(active_resources)

    if not multi_node_exec:
        deepspeed_launch = [
            sys.executable,
            "-u",
            "-m",
            "deepspeed.launcher.launch",
            "--world_info={}".format(world_info_base64),
            "--master_addr={}".format(args.master_addr),
            "--master_port={}".format(args.master_port),
        ]
        if auto_elasticity_enabled:
            cmd = deepspeed_launch + ["--ds_command={}".format(encoded_cmd)
                                      ] + [args.user_script] + args.user_args
        else:
            cmd = deepspeed_launch + [args.user_script] + args.user_args

        # update local environment with contents of environment file
        env.update(env_file)
    else:
        args.launcher = args.launcher.lower()
        if args.launcher == PDSH_LAUNCHER:
            runner = PDSHRunner(args, world_info_base64)
        elif args.launcher == OPENMPI_LAUNCHER:
            runner = OpenMPIRunner(args, world_info_base64, resource_pool)
        elif args.launcher == MVAPICH_LAUNCHER:
            runner = MVAPICHRunner(args, world_info_base64, resource_pool)
        else:
            raise NotImplementedError(f"Unknown launcher {args.launcher}")

        if not runner.backend_exists():
            raise RuntimeError(f"launcher '{args.launcher}' not installed.")

        curr_path = os.path.abspath('.')
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = curr_path + ":" + env['PYTHONPATH']
        else:
            env['PYTHONPATH'] = curr_path

        for var in env.keys():
            if any([var.startswith(name) for name in EXPORT_ENVS]):
                runner.add_export(var, env[var])

        for key, val in env_file.items():
            runner.add_export(key, val)

        if auto_elasticity_enabled:
            cmd = runner.get_cmd(env,
                                 active_resources,
                                 auto_elasticity_enabled,
                                 encoded_cmd)
        else:
            cmd = runner.get_cmd(env, active_resources)

    if auto_elasticity_enabled:
        logger.info(
            "Auto elasticity enabled, cmd used for relaunching will be = {}".format(
                ' '.join(json.loads(base64.urlsafe_b64decode(encoded_cmd)))))
        # remove relaunch signal if exists
        if os.path.isfile('/tmp/ds-requires-relaunch'):
            os.remove('/tmp/ds-requires-relaunch')

    logger.info("Launch cmd = {}".format(' '.join(cmd)))

    result = subprocess.Popen(cmd, env=env)

    #if auto_elasticity_enabled:
    #    pid_dict = {'runner': os.getpid(), 'launcher': result.pid}
    #    with open(RUNNER_PID_FILE, 'w') as fd:
    #        json.dump(pid_dict, fd)
    #    print(f'pid dict: {pid_dict}')

    result.wait()

    if auto_elasticity_enabled:
        while os.path.isfile('/tmp/ds-requires-relaunch'):
            logger.info(f"relaunching with cmd: {cmd}")
            os.remove('/tmp/ds-requires-relaunch')
            new_resources, new_world_info = rescale_resources(args)
            runner.world_info_base64 = new_world_info
            cmd = runner.get_cmd(env,
                                 new_resources,
                                 auto_elasticity_enabled,
                                 encoded_cmd)
            result = subprocess.Popen(cmd, env=env)
            result.wait()

    # In case of failure must propagate the error-condition back to the caller (usually shell). The
    # actual error and traceback should have been printed in the subprocess, so in order to avoid
    # unnecessary noise we just quietly exit here with the same code as the subprocess
    if result.returncode > 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
