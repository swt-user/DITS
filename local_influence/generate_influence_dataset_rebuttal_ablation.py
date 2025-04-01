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

def generate_influence_dataset(
    origin_dpo_path: str,
    dataset_path: str,
    data_influence_path: str,
    output_dir: str,
    dataset_type: str,
    stepnum: int = 5,
    select_num: int = 7000,
    max_count: int = 100,
    iteration: int = 0,
    other_type: bool = False,
):
    chosen_value = []
    distance = []
    chosen = []
    count = 0
    with open(origin_dpo_path, 'r' ) as f:
        for line in f:
            data = json.loads(line)
            results = data["dpo_results"]
            for result in results:
                chosen.append(result["chosen"])
                chosen_value.append(result["chosen_value"])
                distance.append(result["distance"])
                count += 1

    # 验证是否对齐
    Len =  int(0.9 * count)
    random.seed(42)
    # numbers = random.sample(range(Len), Len)
    generator = np.random.default_rng(42)
    numbers = generator.permutation(Len)

    chosen_value_shuffled = [chosen_value[i] for i in numbers]
    distance_shuffled = [distance[i] for i in numbers]
    chosen_shuffled = [chosen[i] for i in numbers]

    

    # dataset_type = "hotpot_qa"

    # file_path = "/home/wentaos/Optima/inference_results/hotpot_qa_dpo_server_stepnum10_15000"
    # file_path = "/data/user_data/wentaos/Optima/inference_results/hotpot_qa_dpo_server_stepnum10_15000_dynamic"
    

    score_type = "f1-score"   # "f1-score"  "exact-match"
    if dataset_type == "hotpot_qa":
        pass
    elif dataset_type == "mwh_qa":
        pass
    elif dataset_type == "cbt":
        pass
    elif dataset_type == "gsm8k":
        score_type = "exact-match"
        # no_use_prompt_pool =True
    elif dataset_type == "math":
        score_type = "math"
    elif dataset_type == "trival_qa":
        pass
    elif dataset_type == "arc":
        score_type = "exact-match"
    elif dataset_type == "mmlu":
        score_type = "exact-match"
        
    tokenizer_path = "/data/user_data/wentaos/Huggingface_models/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    import pandas as pd

    # 定义一个空的 DataFrame，包含两列：id 和 best_100
    df = pd.DataFrame(columns=['id', 'best_100'])

    for item in os.listdir(data_influence_path):
        data_dict = {}
        id_item = int(item.split('_')[-1])
        if os.path.isdir(os.path.join(data_influence_path, item)):
            item_path = os.path.join(os.path.join(data_influence_path, item), f"iteration_{iteration}_0.jsonl")
            if os.path.exists(item_path):
                answer, _ = result_stats_byid(item_path, tokenizer, score_type, max_count=max_count) 
                data_dict = {"id": id_item, "best_100": answer}
                df.loc[len(df)] = [id_item, answer]

    ## 验证选择的能力
    sorted_df =  df.sort_values(by="best_100", ascending=False).id.tolist()
    # print(sorted_df)
    selected_data = [int(i) for i in sorted_df]

    print(f"Collect data influence number: {len(selected_data)}")

    # selected = {list(i.keys())[0] for i in sorted_data[:50] if list(i.keys())[0]!=0}
    selected = selected_data[:(select_num//stepnum)]

    
    selected_chosen_value = []
    all_chosen_value = []
    index_MATES = []
    index_value = []
    selected_distance = []
    for i in range(Len):
        if i in selected:
        # print(f"{i}: {mean(chosen_value[i*1:(i+1)*1])}")
            t = mean(chosen_value_shuffled[i*stepnum:(i+1)*stepnum])
            selected_chosen_value.append(t)
            index_MATES += list(range(i*stepnum,(i+1)*stepnum))
        if i in selected_data:
            t = chosen_value_shuffled[i*stepnum:(i+1)*stepnum]
            all_chosen_value += t
            index_value += list(range(i*stepnum,(i+1)*stepnum))
            # selected_distance += mean(distance_shuffled[i*5:(i+1)*5])
            # print(f"{i}:{mean(chosen_value_shuffled[i*5:(i+1)*5])}")
    print(f"Mean selected chosen value: {mean(selected_chosen_value)}")
    print(f"Len of selected chosen value: {len(selected_chosen_value)}")
    index_value = [index_value[x] for x in (np.argsort(all_chosen_value)[::-1][:len(selected_chosen_value)*stepnum])]
    print(f"Len of index value: {len(index_value)}")
    print(mean([all_chosen_value[x] for x in np.argsort(all_chosen_value)[::-1][:len(selected_chosen_value)*stepnum]]))

    sorted_df =  df.sort_values(by="best_100", ascending=False).id.tolist()
    # print(sorted_df)
    selected_data = [int(i) for i in sorted_df]

    min_val = df["best_100"].min()
    max_val = df["best_100"].max()
    normalize_df = df
    normalize_df["best_100"] = (df["best_100"] - min_val)/(max_val - min_val)
    print(normalize_df.sort_values(by="best_100", ascending=False).head())

    mix_df = pd.DataFrame(columns=['id', 'mix_value', 'mix_value_high_I_low_Q', 'mix_value_low_I_high_Q', 'chosen_value'])
    index_mix = []
    selected_distance = []
    for i in range(Len):
        if i in selected_data:
        # print(f"{i}: {mean(chosen_value[i*1:(i+1)*1])}")
            for t in range(i*stepnum, (i+1)*stepnum):
                # high influence, high Q
                mix_value = chosen_value_shuffled[t] + df.loc[df['id'] == i, 'best_100'].values[0]
                mix_value_high_I_low_Q = - chosen_value_shuffled[t] + df.loc[df['id'] == i, 'best_100'].values[0]
                mix_value_low_I_high_Q = chosen_value_shuffled[t] - df.loc[df['id'] == i, 'best_100'].values[0]
                mix_df.loc[len(mix_df)] = [t, mix_value, mix_value_high_I_low_Q, mix_value_low_I_high_Q, chosen_value_shuffled[t]]
            
    
    sorted_df =  mix_df.sort_values(by="mix_value_high_I_low_Q", ascending=False)[:select_num]
    index_mix = sorted_df.id.tolist()
    print(f"final selected chosen value mean: {sorted_df['chosen_value'].mean()}")



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
    
    raw_datasets["train"] = raw_datasets["train"].select(index_mix)
    raw_datasets.save_to_disk(os.path.join(output_dir, f"{dataset_type}_dpo_mixHILQ_warmup_{select_num}_{max_count}/iteration_{iteration}"))

 
    sorted_df =  mix_df.sort_values(by="mix_value_low_I_high_Q", ascending=False)[:select_num]
    index_mix = sorted_df.id.tolist()
    print(f"final selected chosen value mean: {sorted_df['chosen_value'].mean()}")



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
    
    raw_datasets["train"] = raw_datasets["train"].select(index_mix)
    raw_datasets.save_to_disk(os.path.join(output_dir, f"{dataset_type}_dpo_mixLIHQ_warmup_{select_num}_{max_count}/iteration_{iteration}"))

    
if __name__ == "__main__":
    dataset = "gsm8k"
    iteration = 0
    # generate_influence_dataset(
    #     origin_dpo_path = f"/data/group_data/cx_group/MCTS-agent/results/{dataset}_sft_dpo_DI/dpo/iteration_{iteration}_dpo_format.jsonl",
    #     dataset_path = f"/data/group_data/cx_group/MCTS-agent/my_datasets/{dataset}_sft_dpo_DI/dpo/iteration_{iteration}",
    #     data_influence_path = f"/data/user_data/wentaos/Optima/inference_results/{dataset}_dpo_iteration_{iteration}_server_stepnum5",
    #     output_dir="/data/user_data/wentaos/Optima/my_datasets",
    #     dataset_type = dataset,
    #     stepnum = 5,
    #     select_num = 6213,
    #     iteration = iteration,
    #     other_type=True,
    # )
   
    # single iteration
    # generate_influence_dataset(
    #     origin_dpo_path = f"/data/user_data/wentaos/Optima/results/{dataset}_dpo/iteration_{iteration}_dpo_format.jsonl",
    #     dataset_path = f"/data/user_data/wentaos/Optima/my_datasets/{dataset}_dpo/iteration_{iteration}",
    #     data_influence_path = f"/data/user_data/wentaos/Optima/inference_results/{dataset}_dpo_server_stepnum5_15000",
    #     output_dir="/data/user_data/wentaos/Optima/my_datasets",
    #     dataset_type = dataset,
    #     stepnum = 5,
    #     select_num = 2505,
    #     iteration = iteration,
    #     other_type=True,
    # )



    # dataset_type = "mwh_qa"
    # name = '6_8'
    # data_influence_path = f"/data/user_data/wentaos/Optima/inference_results/{dataset_type}_dpo_iteration_{iteration}_{name}_server_stepnum5/"
    # mid_dpo_jsonl_root_path = f"/data/group_data/cx_group/MCTS-agent/results/{dataset_type}_sft_dpo_scale/dpo_{name}/"
    # mid_dpo_dataset_root_path = f"/data/group_data/cx_group/MCTS-agent/my_datasets/{dataset_type}_sft_dpo_scale/dpo_{name}/"
    
    # # generate mix dataset
    # generate_influence_dataset(
    #     origin_dpo_path=os.path.join(mid_dpo_jsonl_root_path, f"iteration_{iteration}_dpo_format.jsonl"),
    #     dataset_path=os.path.join(mid_dpo_dataset_root_path, f"iteration_{iteration}"),
    #     data_influence_path=data_influence_path,
    #     output_dir=mid_dpo_dataset_root_path,
    #     dataset_type=dataset_type,
    #     stepnum= 5,
    #     iteration=iteration,
    #     select_num=6831,
    #     other_type=True,
    # )


    dataset = "mwh_qa"
    iteration = 0
    generate_influence_dataset(
        origin_dpo_path = f"/data/group_data/cx_group/MCTS-agent/results/{dataset}_sft_dpo_DI/dpo/iteration_{iteration}_dpo_format.jsonl",
        dataset_path = f"/data/group_data/cx_group/MCTS-agent/my_datasets/{dataset}_sft_dpo_DI/dpo/iteration_{iteration}",
        data_influence_path = f"/data/user_data/wentaos/Optima/inference_results/{dataset}_dpo_sever_stepnum5",
        output_dir="/data/user_data/wentaos/Optima/temp_datasets",
        dataset_type = dataset,
        stepnum = 5,
        select_num = 8000,
        iteration = iteration,
        max_count = 5,
        other_type= True,
    )