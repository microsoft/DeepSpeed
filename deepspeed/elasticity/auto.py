"""
Copyright 2021 The Microsoft DeepSpeed Team
"""

import logging
import threading
import time
import re
import os
import json
import time
import base64
import sys
import subprocess
import torch.distributed as dist
import signal
import os

from deepspeed.utils import logger
from .constants import DETECTION_MODE_POLL, DETECTION_MODE_INOTIFY, RUNNER_PID_FILE

POLLING_INTERVAL = 5


def auto_enabled(ds_config: dict):
    if 'IS_ELASTIC_TRAINING_JOB' in os.environ:
        if os.environ['IS_ELASTIC_TRAINING_JOB'].lower() == 'true':
            return True
    return False


def relaunch(state):
    relaunch_rank = state['relaunch_rank']

    if dist.get_rank() == relaunch_rank:
        cmd = os.environ['DS_CMD']
        cmd = base64.urlsafe_b64decode(cmd)
        cmd = json.loads(cmd)
        logger.info(
            f"{dist.get_rank()} deepspeed relaunching at rank:{relaunch_rank} with cmd = {cmd}"
        )
        #results = subprocess.Popen(cmd)
        with open('/tmp/ds-requires-relaunch', 'w') as fd:
            fd.write('')
        logger.info(f"deepspeed relaunching at rank:{relaunch_rank} with cmd = {cmd}")

    logger.info(f"at rank:{dist.get_rank()}, finishing the program..")
    logger.info(f"relaunch, current processes: {os.getpid()}, {os.getppid()}")
    if 'parent-pids' in state:
        for pid in state['parent-pids']:
            logger.info(f"killing parent process {pid}")
            os.kill(pid, signal.SIGTERM)

    #if os.path.isfile(RUNNER_PID_FILE):
    #    with open(RUNNER_PID_FILE, 'r') as fd:
    #        pid_dict = json.load(fd)
    #    for key,pid in pid_dict.items():
    #        try:
    #            os.kill(pid, signal.SIGTERM)
    #            logger.info(f'killed {key} pid: {pid}')
    #        except ProcessLookupError:
    #            pass

    logger.info(f"killing our process {os.getpid()}")
    os.kill(os.getppid(), signal.SIGTERM)
    os.kill(os.getpid(), signal.SIGTERM)


def listen_for_changes(state):
    original_hostfile = open('/job/hostfile').read()
    original_hosts = set(re.findall("(worker-[0-9]+)", original_hostfile))

    #print(f"Running on {len(original_hosts)} nodes")
    #print("Original hostfile =", original_hostfile)
    ssh_config_file = os.environ['HOME'] + '/.ssh/config'

    interval = 5

    while True:
        # wait for some time
        time.sleep(interval)

        # read the file and check changes
        new_hostfile = open('/job/hostfile').read()
        new_hosts = set(re.findall("(worker-[0-9]+)", new_hostfile))

        config = open(ssh_config_file).read()
        config_hosts = set(re.findall("Host (worker-[0-9]+)", config))

        logger.info(
            f"config_hosts={config_hosts} and new_hosts={new_hosts} and old_hosts={original_hosts}"
        )

        if config_hosts == new_hosts and new_hosts != original_hosts:
            if not len(new_hosts) == len(original_hosts):
                sorted_hosts = list(new_hosts)
                sorted_hosts.sort()
                state['relaunch_rank'] = int(sorted_hosts[0].split("-")[1])
                logger.info(f"Relaunch rank = {state['relaunch_rank']}")
                #time.sleep(1)
                if len(new_hosts) > len(original_hosts):
                    state['scale_up'] = True
                    # DeepSpeedEngine will read this and call relaunch
                    exit(0)
                elif len(new_hosts) < len(original_hosts):
                    state['scale_down'] = True
                    #print("\n_______________________________________________________\n")
                    #time.sleep(2)
                    relaunch(state)


def get_relaunch_rank_for_pod_level_retry(new_hostset, old_hostset, host_to_ip_map, new_host_to_ip_map):
    # get list of hosts valid on both old and new, this ensures we are re-launched
    # on a node that is still alive and not new

    filtered_host = []
    for key in host_to_ip_map.keys():
        if host_to_ip_map[key] == new_host_to_ip_map[key]:
            filtered_host.append(host_to_ip_map[key])

    host_list = list(new_hostset.intersection(filtered_host))
    host_list.sort()
    first_host = host_list[0]

    assert 'DS_RANK_MAPPING' in os.environ, "Missing DS_RANK_MAPPING variable, unable to proceed with relaunch"
    rank_mapping = json.loads(os.environ['DS_RANK_MAPPING'])
    logger.info(f"Global rank mapping={rank_mapping}")

    # relaunch rank is first rank on first host
    relaunch_rank = rank_mapping[first_host][0]
    return int(relaunch_rank)


def get_relaunch_rank(new_hostset, old_hostset):
    # get list of hosts valid on both old and new, this ensures we are re-launched
    # on a node that is still alive and not new
    host_list = list(new_hostset.intersection(old_hostset))
    host_list.sort()
    first_host = host_list[0]

    assert 'DS_RANK_MAPPING' in os.environ, "Missing DS_RANK_MAPPING variable, unable to proceed with relaunch"
    rank_mapping = json.loads(os.environ['DS_RANK_MAPPING'])
    logger.info(f"Global rank mapping={rank_mapping}")

    # relaunch rank is first rank on first host
    relaunch_rank = rank_mapping[first_host][0]
    return int(relaunch_rank)


def wait_on_available_nodes(new_hosts):
    available = False
    while not available:
        checks = []
        for host in new_hosts:
            checks.append(
                subprocess.Popen(['ssh',
                                  host,
                                  "'hostname'"],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE))
        available = all([check.wait() == 0 for check in checks])
        logger.info("waiting for scale up nodes to become available")
        time.sleep(1)


def handle_pod_level_retry_event(state, new_hosts, old_hosts, new_config_hosts, host_to_ip_map, new_host_to_ip_map):
    assert len(new_hosts) == len(old_hosts), "new hosts and new config hosts don't align, {new_hosts} != {new_config_hosts}"
    state['relaunch_rank'] = get_relaunch_rank_for_pod_level_retry(new_hosts, old_hosts, host_to_ip_map, new_host_to_ip_map)
    logger.info(f"Relaunch rank = {state['relaunch_rank']}")
    # DeepSpeedEngine will read this and call relaunch
    wait_on_available_nodes(new_hosts)
    relaunch(state)


def handle_scaling_event(state, new_hosts, old_hosts, new_config_hosts):
    assert len(new_hosts) == len(new_config_hosts), "new hosts and new config hosts don't align, {new_hosts} != {new_config_hosts}"

    state['relaunch_rank'] = get_relaunch_rank(new_hosts, old_hosts)

    logger.info(f"Relaunch rank = {state['relaunch_rank']}")
    #time.sleep(1)
    assert len(new_hosts) != len(old_hosts)
    if len(new_hosts) > len(old_hosts):
        state['scale_up'] = True
        # DeepSpeedEngine will read this and call relaunch
        wait_on_available_nodes(new_hosts)
        relaunch(state)
    elif len(new_hosts) < len(old_hosts):
        state['scale_down'] = True
        print("\n_______________________________________________________\n")
        #time.sleep(2)
        relaunch(state)


def get_host_set(hostfile_path):
    #TODO: support host parsing for non worker-[0-9]+ pattern
    hostfile = open(hostfile_path, 'r').read()
    hostset = set(re.findall("(worker-[0-9]+)", hostfile))
    assert len(hostset) > 0, f"Unable to find any hosts in hostfile={hostfile}"
    return hostset


def get_config_host_set(ssh_config_path):
    #TODO: support host parsing for non worker-[0-9]+ pattern
    config = open(ssh_config_path, 'r').read()
    config_hostset = set(re.findall("Host (worker-[0-9]+)", config))
    assert len(config_hostset) > 0, f"Unable to find any hosts in config={config}"
    return config_hostset


def get_config_ip_set(ssh_config_path):
    #TODO: support host parsing for non worker-[0-9]+ pattern
    config = open(ssh_config_path, 'r').read()
    config_ipset = set(re.findall("HostName \d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", config))
    assert len(config_ipset) > 0, f"Unable to find any ips in config={config}"
    return config_ipset

def get_worker_ip_mapping(ssh_config_path):
    config = open(ssh_config_path, 'r').read()
    regex = r"Host worker(.*?)\n(.*?)HostName (.*?)$"
    matches = re.finditer(regex, config, re.MULTILINE)
    mapping = {}
    for matchNum, match in enumerate(matches, start=1):
        match.group()
        host_name = re.search('Host (worker-[0-9]+)', match.group())
        ip = re.search('HostName(.*?)$', match.group())
        mapping[host_name] = ip
    return mapping


def wait_on_hostfile_changes(hostfile_path, original_hosts, new_config_hosts):
    # shouldn't take more than 5min for hostfile to change once ssh config has changed
    max_wait_time = 300
    sleep_time = 2
    max_loops = max_wait_time / sleep_time
    loops = 0
    while True:
        new_hosts = get_host_set(hostfile_path)
        if new_hosts != original_hosts:
            # hostfile has changed, does it align with ssh config changes?
            assert new_hosts == new_config_hosts, \
                f"unable to handle scaling event, {hostfile_path} ({new_hosts}) and .ssh/config ({new_config_hosts}) hosts do not match"
            logging.info(f"detected new host file, new config and new hostfile matches")
            return new_hosts

        # Still waiting for hostfile to change
        time.sleep(sleep_time)
        loops += 1
        if loops >= max_loops:
            return new_hosts
        assert loops < max_loops, "waited {max_wait_time/60} minutes for hostfile to change after .ssh/config changed, unable to handle scaling event"
        logger.info("waiting for hostfile to change...")


def listen_for_changes_polling(state):
    hostfile_path = state['hostfile_path']
    ssh_config_path = state['ssh_config_path']

    logger.info(
        f"Auto elasticity is waiting for scaling event, listening changes at {hostfile_path} and {ssh_config_path}"
    )

    original_hosts = get_host_set(hostfile_path)
    original_config_hosts = get_config_host_set(ssh_config_path)

    original_config_ips = get_config_ip_set(ssh_config_path)
    host_to_ip_map = get_worker_ip_mapping(ssh_config_path)

    while not state['scale_up'] and not state['scale_down'] and not state['pod_level_retried']:
        new_config_hosts = get_config_host_set(ssh_config_path)
        new_config_ips = get_config_ip_set(ssh_config_path)
        if new_config_ips != original_config_ips and new_config_hosts == original_config_hosts:
            new_host_to_ip_map = get_worker_ip_mapping(ssh_config_path)
            new_hosts = get_host_set(hostfile_path)
            state['pod_level_retried'] = True
            handle_pod_level_retry_event(state, new_hosts, original_hosts, new_config_hosts, host_to_ip_map,
                                         new_host_to_ip_map)
            logger.info(f"{dist.get_rank()} Pod level retry event already triggered, monitoring stopped")

        if new_config_hosts != original_config_hosts:
            logger.info(
                "{dist.get_rank()} detected ssh config changed due to scaling event")
            new_hosts = wait_on_hostfile_changes(hostfile_path,
                                                 original_hosts,
                                                 new_config_hosts)
            state['hostfile_changed'] = True

            handle_scaling_event(state, new_hosts, original_hosts, new_config_hosts)
            logger.info(f"{dist.get_rank()} Scaling event already triggered, monitoring stopped")
        time.sleep(POLLING_INTERVAL)


def listen_for_changes_with_inotify(state):
    import inotify
    import inotify.adapters
    hostfile_path = state['hostfile_path']
    ssh_config_path = state['ssh_config_path']

    logger.info(
        "Auto elasticity is waiting for scaling event, listening changes at {hostfile_path} and {ssh_config_path}"
    )

    original_hosts = get_host_set(hostfile_path)
    original_config_hosts = get_config_host_set(ssh_config_path)

    i = inotify.adapters.Inotify()

    # Watch for modifications to ~/.ssh/, config changes signal a scaling event
    ssh_path = os.path.dirname(ssh_config_path)
    i.add_watch(ssh_path)

    for event in i.event_gen(yield_nones=False):
        (_, type_names, path, filename) = event
        #print("PATH=[{}] FILENAME=[{}] EVENT_TYPES={}".format(path,
        #                                                     filename,
        #                                                     type_names))
        if filename == 'config' and type_names[0] == 'IN_MODIFY':
            print("PATH=[{}] FILENAME=[{}] EVENT_TYPES={}".format(
                path,
                filename,
                type_names))
            new_config_hosts = get_config_host_set(ssh_config_path)
            if original_config_hosts == new_config_hosts:
                logger.info(
                    "Detected ssh changes but the file is unchanged, will not trigger scale event"
                )
                continue
            state['config_changed'] = True
            logger.info("detected ssh config changed due to scaling event")

            new_hosts = wait_on_hostfile_changes(hostfile_path,
                                                 original_hosts,
                                                 new_config_hosts)
            state['hostfile_changed'] = True

            handle_scaling_event(state, new_hosts, original_hosts, new_config_hosts)


def start_watching(state, detection_method=DETECTION_MODE_POLL):
    if detection_method == DETECTION_MODE_INOTIFY:
        try:
            import inotify
        except ImportError as err:
            logging.error("Please pip install inotify to use automatic elasticity")
            raise err
        thread = threading.Thread(target=listen_for_changes_with_inotify,
                                  args=(state,
                                        ),
                                  daemon=True)
        logger.info('detection method via inotify')
    elif detection_method == DETECTION_MODE_POLL:
        thread = threading.Thread(target=listen_for_changes_polling,
                                  args=(state,
                                        ),
                                  daemon=True)
        logger.info('detection method via polling')
    elif detection_method == "subprocess":
        from multiprocessing import Process, Manager
        state['parent-pids'] = [os.getpid(), os.getppid()]
        logger.info(f"MAIN parent-pids: {state['parent-pids']}")
        state = Manager().dict(state)
        thread = Process(target=listen_for_changes_polling, args=(state, ))
        logger.info('detection method via subprocess')
    else:
        raise ValueError(f"Detection method of {detection_method} is unknown!")

    logger.info(f"watching for elastic scaling events with {detection_method}")
    thread.start()
    return state


# just for debugging -- deepspeed engine will do this
keep_training = True
step = 0


def train(state):
    global step
    global keep_training

    print("Training ... step:", step)
    step += 1

    # actual training work
    time.sleep(2)

    # check if a scale up or scale down event has come and act accordingly
    if state['scale_up']:
        print(f"scaling up nodes, checkpointing, and restarting")
        keep_training = False

    if state['scale_down']:
        print(f"scaling down nodes and restarting")
        keep_training = False


if __name__ == "__main__":
    auto_state = {
        'scale_up': False,
        'scale_down': False,
        'config_changed': False,
        'hostfile_changed': False,
        'pod_level_retried': False,
        "hostfile_path": "/job/hostfile",
        "ssh_config_path": os.path.join(os.environ["HOME"],
                                        '.ssh/config')
    }
    start_watching(auto_state)

    while keep_training:
        train(auto_state)

    logging.info("Main    : wait for the thread to finish")
    logging.info("Main    : all done")
