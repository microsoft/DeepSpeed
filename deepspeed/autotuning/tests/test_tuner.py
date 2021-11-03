import argparse
import copy
import json
import shutil

import hjson
import transformers
from deepspeed.autotuning.config import DeepSpeedAutotuningConfig
from deepspeed.autotuning.constants import *
from deepspeed.autotuning.scheduler import ResourceManager
from deepspeed.autotuning.tuner import (GridSearchTuner, ModelBasedTuner, RandomTuner)
from deepspeed.autotuning.utils import *
from deepspeed.runtime.zero.constants import *

ARGS_MAPPINGS = {"train_micro_batch_size_per_gpu": "--per_device_train_batch_size"}
METRIC = AUTOTUNING_METRIC_THROUGHPUT

TUNING_SPACE = {
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": [5e7,
                               5e8,
                               1e9],
        "allgather_bucket_size": [5e7,
                                  5e8,
                                  1e9],
    }
}

TUNING_SPACE = {"train_micro_batch_size_per_gpu": 1, "zero_optimization": {'stage': 1}}

DS_CONFIG_PATH = 'ds_config.json'
MODEL_NAME = 'microsoft/deberta-v2-xxlarge'
TASK_NAME = 'mnli'

USER_ARGS = [
    '--deepspeed',
    DS_CONFIG_PATH,
    '--model_name_or_path',
    MODEL_NAME,
    '--task_name',
    TASK_NAME,
    '--do_train',
    '--max_seq_length',
    '256',
    '--per_device_train_batch_size',
    '1',
    '--learning_rate',
    '13e-6',
    '--num_train_epochs',
    '1',
    '--output_dir',
    './mnli/output',
    '--save_steps',
    '0',
    '--overwrite_output_dir'
]

HF_PATH = path = os.path.dirname(transformers.__file__)
USER_SCRIPT = os.path.join(HF_PATH,
                           '..',
                           '..',
                           'examples/pytorch/text-classification/run_glue.py')

DLTS_HOSTFILE = "/job/hostfile"


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--user_script",
                        type=str,
                        default=USER_SCRIPT,
                        help="user_script")
    parser.add_argument("--user_args",
                        nargs='+',
                        type=str,
                        default=USER_ARGS,
                        help="user_args")
    parser.add_argument("--master_port", type=int, default=29500, help="master port")
    parser.add_argument("--results_dir",
                        type=str,
                        default="autotuning_results",
                        help="output directory for results")
    parser.add_argument("-H",
                        "--hostfile",
                        type=str,
                        default=DLTS_HOSTFILE,
                        help="Hostfile path (in MPI style) that defines the "
                        "resource pool available to the job (e.g., "
                        "worker-0 slots=4)")
    parser.add_argument("--exps_dir",
                        type=str,
                        default="autotuning_exps",
                        help="output directory for experiments")
    parser.add_argument("-w",
                        "--overwrite",
                        action='store_true',
                        help="overwite the output directories")
    parser.add_argument("--model_info",
                        type=str,
                        default="",
                        help="output directory for experiments")
    args = parser.parse_args()

    return args


def generate_experiments(tuning_space, args):
    """ Generates the autotuning experiments given a tuning_space.
    """
    # each zero stage uses a different template configuration file
    config_zero = tuning_space.get("zero_optimization", {})
    stage = config_zero.get("stage", None)
    template_config = {}
    if stage == 0:
        template_path = DEFAULT_TEMPLATE_PATH_ZERO_0
        template_config = hjson.load(open(template_path, 'r'))
        prefix = "z0_"

    elif stage == 1:
        template_path = DEFAULT_TEMPLATE_PATH_ZERO_1
        template_config = hjson.load(open(template_path, 'r'))
        prefix = "z1_"

    elif stage == 2:
        template_path = DEFAULT_TEMPLATE_PATH_ZERO_2
        template_config = hjson.load(open(template_path, 'r'))
        prefix = "z2_"

    elif stage == 3:
        template_path = DEFAULT_TEMPLATE_PATH_ZERO_3
        template_config = hjson.load(open(template_path, 'r'))
        model_info = args.model_info
        if model_info and "hidden_size" in model_info:
            hs = model_info["hidden_size"]
            template_config["zero_optimization"]["reduce_bucket_size"] = hs * hs
            template_config["zero_optimization"][
                "stage3_prefetch_bucket_size"] = 0.9 * hs * hs
            template_config["zero_optimization"][
                "stage3_param_persistence_threshold"] = 10 * hs
        prefix = "z3_"

    replace_dict(tuning_space,
                 args.user_config,
                 [ZERO_OPTIMIZATION,
                  "train_micro_batch_size_per_gpu"])

    all_configs = get_all_configs(tuning_space)

    tuning_keys = get_tuning_keys(tuning_space)

    pruned_list = prune_configs(all_configs)

    exps = []
    for config in pruned_list:
        exp_config = copy.deepcopy(template_config)
        # fill the template with the expr config
        replace_dict(exp_config, config)

        # if the config does not use offloading, remove the offloading section
        config_zero = config.get("zero_optimization", None)
        if config_zero:
            if "offload_optimizer" not in config_zero and "offload_optimizer" in exp_config[
                    "zero_optimization"]:
                del exp_config["zero_optimization"]["offload_optimizer"]
            if "offload_param" not in config_zero and "offload_param" in exp_config[
                    "zero_optimization"]:
                del exp_config["zero_optimization"]["offload_param"]

        exp = {}
        # generate the expr name
        exp_name = canonical_name(exp_config, tuning_keys, prefix)
        exp['name'] = exp_name
        exp["ds_config"] = exp_config
        exp['num_gpus'] = args.exp_num_gpus
        exp['num_nodes'] = args.exp_num_nodes
        exps.append(exp)

    return exps


if __name__ == "__main__":

    assert os.path.exists(DS_CONFIG_PATH)
    user_config = json.load(open(DS_CONFIG_PATH, "r"))
    autotuning_config = DeepSpeedAutotuningConfig(user_config)

    args = parse_args()
    args.user_config = user_config
    args.exp_num_gpus = autotuning_config.num_gpus
    args.exp_num_nodes = autotuning_config.num_nodes
    print(f"args={args}")

    results_dir = args.results_dir
    exps_dir = args.exps_dir
    if autotuning_config.results_dir and autotuning_config.results_dir != "":
        results_dir = autotuning_config.results_dir
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir, ignore_errors=True)
    os.makedirs(results_dir)

    if autotuning_config.exps_dir and autotuning_config.exps_dir != "":
        exps_dir = autotuning_config.exps_dir
    if os.path.exists(exps_dir):
        shutil.rmtree(exps_dir, ignore_errors=True)
    os.makedirs(exps_dir)

    resource_pool = fetch_hostfile(args.hostfile)
    if not resource_pool:
        resource_pool = {}
        import torch
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("Unable to proceed, no GPU resources available")
        resource_pool['localhost'] = device_count
    active_resources = resource_pool

    logger.info(f"active_resources = {active_resources}")

    hosts = []
    ngpus_per_node = 100
    for hostname, slots in active_resources.items():
        hosts.append(hostname)
        ngpus_per_node = min(slots, ngpus_per_node)

    assert ngpus_per_node > 0, "no gpu is available"

    rm = ResourceManager(args=args,
                         hosts=hosts,
                         num_gpus_per_node=ngpus_per_node,
                         results_dir=results_dir,
                         exps_dir=exps_dir,
                         arg_mappings=ARGS_MAPPINGS)

    tuning_space = TUNING_SPACE
    exps = generate_experiments(tuning_space, args)
    logger.info(f"generated {len(exps)} exps")

    logger.info(f'tuner_type= {autotuning_config.tuner_type}')
    if autotuning_config.tuner_type == AUTOTUNING_TUNER_MODELBASED:
        t = ModelBasedTuner(exps, rm, tuning_space)
    elif autotuning_config.tuner_type == AUTOTUNING_TUNER_RANDOM:
        t = RandomTuner(exps, rm)
    else:
        t = GridSearchTuner(exps, rm)

    best_exp, best_metric_val = t.tune(metric=METRIC)

    print(best_exp, best_metric_val)
