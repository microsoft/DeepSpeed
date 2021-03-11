from subprocess import check_output
import re

CONNECTION_TYPES = ["X", "SYS", "NODE", "PHB", "PXB", "PIX", "NV[\d]+"]


def get_topology_str():
    return check_output(["nvidia-smi", "topo", "-m"]).decode()


def contains_nvlinks(topology):
    return any([is_nvlink(item) for sublist in topology for item in sublist])


def is_nvlink(connection_type):
    return re.search(CONNECTION_TYPES[-1], connection_type)


def get_nvlink_pairs(topology):
    """
    takes a topology matrix and outputs a list of pairs bridged by nvlink
    """
    out = set()
    for device_idx1, item1 in enumerate(topology):
        for device_idx2, item2 in enumerate(item1):
            if is_nvlink(item2):
                if (device_idx2, device_idx1) not in out:
                    out.add((device_idx1, device_idx2))
    return out


def get_cuda_visible_device_mapping(nvlink_pairs, local_gpu_ids=None):
    nvlink_pairs = [item for sublist in sorted(nvlink_pairs) for item in sublist]
    if local_gpu_ids is not None:
        nvlink_pairs = [item for item in nvlink_pairs if item in local_gpu_ids]
    # deduplicate incase there's > pair per gpu
    deduped = []
    for item in nvlink_pairs:
        if item not in deduped:
            deduped.append(item)
    return_string = ",".join(map(str, deduped))
    return return_string


def topology_from_string(string):
    output_per_gpu = string.strip().split('Legend:')[0].strip().split('\n')
    headers = output_per_gpu.pop(0)
    headers = headers.strip().split()
    headers = [i for i in headers if re.search('GPU[\d]+', i)]
    num_gpus = len(headers)

    topology = []
    for output in output_per_gpu:
        output = output.strip().split()
        gpu_id = output.pop(0)
        output = output[:num_gpus]
        if 'GPU' in gpu_id:
            links = []
            for idx, i in enumerate(output):
                if idx >= num_gpus:
                    break
                links.append(i.strip())
            topology.append(links)

    # check for consistency
    assert all([len(i) == len(topology) for i in topology])
    return topology


def detect_nvlink_pairs_and_map_visible_devices(rank, local_gpu_ids):
    string = get_topology_str()
    topology = topology_from_string(string)
    if contains_nvlinks(topology):
        pairs = get_nvlink_pairs(topology)
        remapping = get_cuda_visible_device_mapping(pairs, local_gpu_ids)
        return remapping
    else:
        print(f'No NVLINK detected on rank {rank}')
        return None


if __name__ == "__main__":
    detect_nvlink_pairs_and_map_visible_devices()
