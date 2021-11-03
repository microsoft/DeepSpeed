import numpy as np
from xgd_model import XGBoostCostModel
import json
import os
import itertools

model_type_list = ['bert', 'distilbert', 'deberta']
path = '/data/chengli1/autotuning'
model_type = model_type_list[1]

if model_type == 'bert':
    tmbspg = [32, 64, 128, 256, 320]
elif model_type == 'distilbert':
    tmbspg = [256, 512, 640]
elif model_type == 'deberta':
    tmbspg = [2, 4, 6]

all_tune_param = {
    # "stage": 2,
    "overlap_comm": [True,
                     False],
    "reduce_scatter": [False,
                       True],
    "reduce_bucket_size": [5e7,
                           5e8,
                           1e9],
    "allgather_partitions": [True,
                             False],
    "allgather_bucket_size": [5e7,
                              5e8,
                              1e9],
    "contiguous_gradients": [False,
                             True],
    "train_micro_batch_size_per_gpu": tmbspg,
}


# used to compute theoretical random performance
def probabilities(num_sample, num_good_sample, total_config):
    # num_sample: num of random samples
    # num_good_sample: number of accetaple sample
    # toal_config: total num of configs
    result = 1
    num_bad_sample = total_config - num_good_sample
    for i in range(0, num_good_sample):
        result = result * (total_config - num_sample - i) / (total_config - i)
    return 1 - result


# print(probabilities(20, 14, 720))

all_dirs = []

full_path = f"{path}/{model_type}/autotuning_results"
for i, directories in enumerate(os.listdir(full_path)):
    dir_name = os.path.join(full_path, directories)
    all_dirs.append(dir_name)

all_dirs.sort()

bs_num = len(tmbspg)

corresponding_hyparameters = [[] for i in range(bs_num)]
throught_list = [[] for i in range(bs_num)]

for dir_name in all_dirs:
    current_data = []
    with open(f'{dir_name}/ds_config.json') as f:
        data = json.load(f)
    for k, v in data["zero_optimization"].items():
        if k in all_tune_param.keys():
            current_data.append(int(v))
    _bs = data["train_micro_batch_size_per_gpu"]
    current_data.append(_bs)

    try:
        with open(f'{dir_name}/metrics.json') as f:
            metric = json.load(f)
            ind = tmbspg.index(_bs)
            # print(ind)
            throught_list[ind].append(metric['throughput'])
            corresponding_hyparameters[ind].append(current_data)
    except:
        # If the config fails, we do not record the performance since it's may be caused by unnormal deepspeed exit
        _a = 1
partition_list_s = [len(throught_list[i]) for i in range(bs_num)]
partition_list = [0] + [sum(partition_list_s[:i + 1]) for i in range(bs_num)]
print(model_type)
for i, bs in enumerate(tmbspg):
    print(
        f"BS={bs}: max throughput {max(throught_list[i])}, average throughput {np.mean(throught_list[i])}"
    )

corresponding_hyparameters = list(itertools.chain(*corresponding_hyparameters))
throught_list = list(itertools.chain(*throught_list))

print('*' * 100)

total_num = len(corresponding_hyparameters)
for i in range(85, 100, 5):
    print(
        f"# Config can reach {i}% performance of best one ({i / 100 * np.max(throught_list)} out of {np.max(throught_list)}): {np.sum(np.array(throught_list) > i / 100 * np.max(throught_list))}"
    )
print(
    f"# Config can reach 99% performance of best one ({.99 * np.max(throught_list)} out of {np.max(throught_list)}): {np.sum(np.array(throught_list) > .99 * np.max(throught_list))}"
)
print('*' * 100)

#  let's build the model
data = corresponding_hyparameters
label = throught_list
model = XGBoostCostModel("rank")

init_sample_size_per_bs = 2 if bs_num <= 4 else 1
each_iter_runs = 3
each_iter_random_exploration = 0

so_far_evaluated_cases_index = []
so_far_evaluated_cases_data = []
so_far_evaluated_cases_target = []
# for first generate, we sample 1/2 cases from each batch size
for i in range(bs_num):
    num = np.random.choice(np.arange(partition_list[i],
                                     partition_list[i + 1]),
                           size=init_sample_size_per_bs,
                           replace=False)
    so_far_evaluated_cases_index.extend(list(num))
    # print(num)

for i in so_far_evaluated_cases_index:
    so_far_evaluated_cases_data.append(data[i])
    so_far_evaluated_cases_target.append(label[i])

# print(so_far_evaluated_cases_data)
# print(so_far_evaluated_cases_target)

final_result = []
for i in so_far_evaluated_cases_index:
    final_result.append(label[i])
print(
    f'Random Sample -- After evaluate {len(final_result)} examples, our current best is: {max(final_result)} as campared to '
    f'Best {max(label)}')
print(
    f"      Number of configs that work better than current best: {np.sum(np.array(throught_list)>max(final_result))}"
)

# explore batch size first, for each iteration, we choose the best config among the same batch size
for i in range(2):
    model.fit(so_far_evaluated_cases_data, so_far_evaluated_cases_target)
    output = model.predict(data)
    for bs_index in range(bs_num):
        sort_index = list(output[partition_list[bs_index]:partition_list[bs_index +
                                                                         1]].argsort())
        while True:
            index = sort_index.pop()
            if index not in so_far_evaluated_cases_index:
                c_index = index + partition_list[bs_index]
                so_far_evaluated_cases_index.append(c_index)
                so_far_evaluated_cases_data.append(data[c_index])
                so_far_evaluated_cases_target.append(label[c_index])
                break

    final_result = []
    for i in so_far_evaluated_cases_index:
        final_result.append(label[i])
    print(
        f'Model Based -- After evaluate {len(final_result)} examples, our current best is: {max(final_result)} as campared to '
        f'Best {max(label)}')
    print(
        f"      Number of configs that work better than current best: {np.sum(np.array(throught_list)>max(final_result))}"
    )
    _sum = [0] * len(tmbspg)
    for d in so_far_evaluated_cases_data:
        _sum[tmbspg.index(d[-1])] += 1
    print('Each batch size: ', _sum)

# explore the entire search space
for i in range(3):
    model.fit(so_far_evaluated_cases_data, so_far_evaluated_cases_target)
    output = model.predict(data)
    sort_index = list(output.argsort())
    insert = 1
    while insert <= each_iter_runs:
        index = sort_index.pop()
        if index not in so_far_evaluated_cases_index:
            insert += 1
            so_far_evaluated_cases_index.append(index)
            so_far_evaluated_cases_data.append(data[index])
            so_far_evaluated_cases_target.append(label[index])

    # add random exploration, for now we set it to 0
    insert = 1
    while insert <= each_iter_random_exploration:
        index = np.random.randint(0, total_num)
        if index not in so_far_evaluated_cases_index:
            insert += 1
            so_far_evaluated_cases_index.append(index)
            so_far_evaluated_cases_data.append(data[index])
            so_far_evaluated_cases_target.append(label[index])

    final_result = []
    for i in so_far_evaluated_cases_index:
        final_result.append(label[i])
    print(
        f'Model Based -- After evaluate {len(final_result)} examples, our current best is: {max(final_result)} as campared to '
        f'Best {max(label)}')
    print(
        f"      Number of configs that work better than current best: {np.sum(np.array(throught_list)>max(final_result))}"
    )
    _sum = [0] * len(tmbspg)
    for d in so_far_evaluated_cases_data:
        _sum[tmbspg.index(d[-1])] += 1
    print('Each batch size: ', _sum)

_sum = [0] * len(tmbspg)
for d in so_far_evaluated_cases_data:
    _sum[tmbspg.index(d[-1])] += 1
print(_sum)
print('Each batch size: ', _sum)

# print('*' * 100)
# sort_index = list(np.array(so_far_evaluated_cases_target).argsort())
# select_config = so_far_evaluated_cases_data[sort_index[-1]]
# for bs in tmbspg:
#     tmp_config = select_config[:-1] + [bs]
#     index = data.index(tmp_config)
#     if index not in so_far_evaluated_cases_index:
#         so_far_evaluated_cases_index.append(index)
#         so_far_evaluated_cases_data.append(data[index])
#         so_far_evaluated_cases_target.append(label[index])

# final_result = []
# for i in so_far_evaluated_cases_index:
#     final_result.append(label[i])
# print(f'Model Based -- After evaluate {len(final_result)} examples, our current best is: {max(final_result)} as campared to '
#         f'Best {max(label)}')
# print(f"      Number of configs that work better than current best: {np.sum(np.array(throught_list)>max(final_result))}")

# print('*' * 100)
# sort_index = list(np.array(so_far_evaluated_cases_target).argsort())
# for i in range(5):
#     print(f'Selected # {i+1} Config: {so_far_evaluated_cases_data[sort_index.pop()]}')
