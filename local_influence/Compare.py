import json
from statistics import mean
import random
import numpy as np
import os
import sys
sys.path.append('/home/wentaos/Optima')
from reward.reward import result_stats, result_stats_byid
from transformers import AutoTokenizer
import sys
sys.path.append("/home/wentaos/Optima/local_influence")

from my_utils import DataArguments, get_datasets

dataset_path = "/data/user_data/wentaos/Optima/my_datasets/mwh_qa_dpo_mix_warmup_8000/iteration_0"

data_args = DataArguments(chat_template=None, dataset_mixer={dataset_path: 1.0}, text_column='text', dataset_splits=['train', 'test'], dataset_configs=None, preprocessing_num_workers=8, truncation_side=None, auto_insert_empty_system_msg=True)

###############
# Load datasets
###############
raw_datasets = get_datasets(
    data_args,
    splits=data_args.dataset_splits,
    configs=data_args.dataset_configs,
    columns_to_keep=[
        "messages",
        "chosen",
        "rejected",
        "prompt",
        "completion",
        "label",
    ],
    shuffle=True,
)

print(raw_datasets["train"][0])


dataset_path = "/data/user_data/wentaos/Optima/my_datasets/mwh_qa_dpo_mix_warmup_1700/iteration_0"

data_args = DataArguments(chat_template=None, dataset_mixer={dataset_path: 1.0}, text_column='text', dataset_splits=['train', 'test'], dataset_configs=None, preprocessing_num_workers=8, truncation_side=None, auto_insert_empty_system_msg=True)

###############
# Load datasets
###############
temp_datasets = get_datasets(
    data_args,
    splits=data_args.dataset_splits,
    configs=data_args.dataset_configs,
    columns_to_keep=[
        "messages",
        "chosen",
        "rejected",
        "prompt",
        "completion",
        "label",
    ],
    shuffle=True,
)

for i in range(100):
    if raw_datasets["train"][i] == temp_datasets["train"][i]:
        print("True")
    else:
        print("False")
