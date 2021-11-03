import copy

import hjson
import numpy as np
from deepspeed.autotuning.utils import write_experiments
from deepspeed.utils import logger

from deepspeed.autotuning.constants import (AUTOTUNING,
                                            AUTOTUNING_METRIC_DEFAULT,
                                            AUTOTUNING_METRIC_PATH)
from .base_tuner import BaseTuner
from .cost_model2 import XGBoostCostModel
from .utils import *
from ..utils import get_tuning_keys
from deepspeed.runtime.constants import *

INIT_SAMPLED_NUM_CONFIGS_PER_MBS = 2


class ModelBasedTuner(BaseTuner):
    """Exploring the search space with a cost model"""
    def __init__(self, exps: list, resource_manager, metric, tuning_space):
        super().__init__(exps, resource_manager, metric)
        self.tuning_space = tuning_space
        self.best_iter = 0
        self.early_stopping = None

        # sort all experiments by micro batch size
        self.all_exps = sorted(
            exps,
            key=lambda kv: kv['ds_config'][TRAIN_MICRO_BATCH_SIZE_PER_GPU])
        self.all_configs = [e['ds_config'] for e in exps]

        self.num_all_configs = len(self.all_configs)

        self.visited = set([])
        self.current_sampled_batch = []

        self.mbs_list = self.tuning_space[TRAIN_MICRO_BATCH_SIZE_PER_GPU]
        self.num_configs_per_mbs = self.num_all_configs // len(self.mbs_list)

        for i, mbs in enumerate(self.mbs_list):
            for _ in range(
                    min(INIT_SAMPLED_NUM_CONFIGS_PER_MBS,
                        self.num_configs_per_mbs)):
                # Do batch size based sampling
                index = np.random.randint(
                    self.num_configs_per_mbs) + self.num_configs_per_mbs * i

                while index in self.visited:
                    index = np.random.randint(
                        self.num_configs_per_mbs) + self.num_configs_per_mbs * i
                self.current_sampled_batch.append(self.all_exps[index])
                self.visited.add(index)

        self.cost_model = XGBoostCostModel("rank", keys=get_tuning_keys(tuning_space))

        self.evaluated_configs = []
        self.evaluated_perf = []

        self.train_ct = 0

        self.random_exploration_ratio = 0.2  # do random exploration ratio
        # For now, explore 3 steps along batch_dim and then switch to entire space
        self.batch_dim_exploration = 3

    def find_estimated_top_configs(self):
        """Use the cost model to predict the estimated performance of
        configurations for the next round of evaluation"""
        estimates = self.cost_model.predict(self.all_configs)

        return estimates

    def next_batch(self, sample_size):
        sampled_batch = []
        sample_size = min(sample_size, (self.num_all_configs - len(self.visited)))

        # We will explore the batch size for the first fewer iterations
        # this will be a small portion, we do not need to check "if next config exists"
        if self.train_ct <= self.batch_dim_exploration:
            # do important thing first
            plan_size = np.ceil(sample_size / len(self.mbs_list))
            logger.info(f"type(plan_size) = {type(plan_size)}")
            for i, mbs in enumerate(self.mbs_list):
                if len(self.visited) >= len(self.all_configs):
                    break
                result_per_batch_size = self.predict_results[self.num_configs_per_mbs *
                                                             i:self.num_configs_per_mbs *
                                                             (i + 1)]
                sorted_index = np.argsort(
                    result_per_batch_size
                )  # TODO: check if we need descending or asending order, throughput or latency
                for _ in range(plan_size):
                    # we need to do batch size based sampling
                    index = sorted_index.pop(0) + self.num_configs_per_mbs * i

                    while index in self.visited:
                        index = sorted_index.pop(0) + self.num_configs_per_mbs * i

                    sampled_batch.append(self.all_exps[index])  # TODO
                    self.visited.add(index)
                    if len(self.visited) / len(self.mbs_list) >= len(
                            self.all_configs) / len(self.mbs_list):
                        break
        else:
            counter = 0
            sorted_index = np.argsort(
                self.predict_results
            )  # TODO: check if we need descending or asending order, throughput or latency

            while counter < sample_size:
                if len(self.visited) >= len(self.all_configs):
                    break

                rd = np.random.rand()
                if rd > self.random_exploration_ratio:
                    # do normal selection
                    index = sorted_index.pop(0)
                    while index in self.visited:
                        index = sorted_index.pop(0)
                else:
                    # do random selection
                    index = np.random.randint(self.num_all_configs)
                    while index in self.visited:
                        index = np.random.randint(self.num_all_configs)
                sampled_batch.append(self.exps[index])  # TODO
                self.visited.add(index)
                counter += 1

        logger.info(
            f"next_batch return {len(sampled_batch)}, {[e['name'] for e in sampled_batch]}"
        )
        return sampled_batch

    def has_next(self):
        return len(self.visited) < self.num_all_configs

    def update(self):
        best_exp = None
        best_metric_val = None
        metric = self.metric
        logger.info(f"update len(exps) = {len(self.rm.finished_experiments)}")

        for exp_id, (exp, err) in self.rm.finished_experiments.items():
            if err:
                logger.info(
                    f"Skipping exp_id = {exp_id}, exp_name = {exp['name']}, the experiment did not run succesfully with error = {err}, thus a metrics.txt does not exist for it. Please check the stderr.log in {exp['result_dir']}"
                )
                # ZY: Not so sure if add the value as zero is good or not
                # For out of memory: 0.0 is a good choice
                # self.evaluated_configs.append(feature_val)
                # self.evaluated_perf.append(0.0)
                # For some deepspeed isssue (e.g., did not exist properly), it is better to
                # ignore the config or put it back to the pool

                continue

            p = exp["ds_config"][AUTOTUNING][AUTOTUNING_METRIC_PATH]
            with open(p, 'r') as f:
                results = hjson.load(f)
                metric_val = results[metric]
                # TODO: For now assume the metric is throughput.
                if best_exp == None or (metric_val and metric_val > best_metric_val):
                    logger.info(f"update best using exp_name = {exp['name']}")
                    best_exp = exp
                    best_metric_val = metric_val
                    # ZY: need try exception since the run can be failed?
            self.evaluated_perf.append(metric_val)
            self.evaluated_configs.append(exp['ds_config'])

        if self.has_next():
            logger.info(f"after has_next()")

            self.cost_model.fit(self.evaluated_configs, self.evaluated_perf)
            logger.info(f"before find_estimated_top_configs")

            self.predict_results = self.find_estimated_top_configs()
            logger.info(f"self.evaluated_configs = {len(self.evaluated_configs)}")

        self.train_ct += 1

        return best_exp, best_metric_val

    def tune(self, sample_size=1, n_trials=100, early_stopping=3):
        i = 0
        sample_size = len(self.mbs_list)
        # try:
        while i == 0 or (i < n_trials and self.has_next()):
            logger.info(
                f"self.has_next()  = {self.has_next()}, visited = {self.visited}, all = {self.num_all_configs} at iter = {i}"
            )
            # for model-based tuner need to do fit/prediction after initialization
            if i == 0:
                sampled_exps = self.current_sampled_batch
            else:
                logger.info("before next_batch")
                sampled_exps = self.next_batch(sample_size)
                logger.info("after next_batch")

            logger.info(
                f"sampled {len(sampled_exps)} exps {[e['name'] for e in sampled_exps]} at iter = {i}"
            )
            exp_paths = write_experiments(sampled_exps, self.rm.exps_dir)
            self.rm.schedule_experiments(exp_paths)
            self.rm.run()
            exp, metric_val = self.update()
            self.rm.clear()
            if self.best_exp == None or (metric_val
                                         and metric_val > self.best_metric_val):
                self.best_exp = exp
                self.best_metric_val = metric_val
                self.best_iter = i

            i += 1

            # Early stopping
            if early_stopping and i >= self.best_iter + early_stopping:
                logger.info(
                    f"Early stopped. Stopped at iteration {i}. Best iteration is {self.best_iter}. Early stopping threshold is {early_stopping}"
                )
                break
        # except:
        #     pass
