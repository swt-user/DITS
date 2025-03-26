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

def Analysis_distribution(
    origin_dpo_path: str,
    dataset_path: str,
    data_influence_path: str,
    output_dir: str,
    dataset_type: str,
    stepnum: int = 5,
    select_num: int = 7000,
    iteration: int = 0,
    initial_value: int = 0, 
    scale_name: str = '',
    other_type: bool = False,
):
    chosen_value = []
    distance = []
    chosen = []
    rejected = []
    count = 0
    with open(origin_dpo_path, 'r' ) as f:
        for line in f:
            data = json.loads(line)
            results = data["dpo_results"]
            for result in results:
                chosen.append(result["chosen"])
                rejected.append(result["rejected"])
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
    chosen_shuffled = [''.join(chosen[i]) for i in numbers]
    rejected_shuffled = [''.join(rejected[i]) for i in numbers]

    

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
        
    tokenizer_path = "/home/wentaos/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    import pandas as pd

    # 定义一个空的 DataFrame，包含两列：id 和 best_100
    df = pd.DataFrame(columns=['chosen', 'rejected', 'best_100', "chosen_value", "distance"])

    for item in os.listdir(data_influence_path):
        data_dict = {}
        try:
            id_item = int(item.split('_')[-1])
        except:
            continue
        if os.path.isdir(os.path.join(data_influence_path, item)):
            item_path = os.path.join(os.path.join(data_influence_path, item), f"iteration_{iteration}_0.jsonl")
            if os.path.exists(item_path):
                answer, _ = result_stats_byid(item_path, tokenizer, score_type, max_count=100) 
                data_dict = {"id": id_item, "best_100": answer}
                # print(answer)
                # print('\n'.join(chosen_shuffled[id_item*stepnum:(id_item+1)*stepnum]))
                # df.loc[len(df)] = [id_item, answer, mean(chosen_value_shuffled[id_item*stepnum:(id_item+1)*stepnum]), mean(distance_shuffled[id_item*stepnum:(id_item+1)*stepnum])]
                for t in range(id_item*stepnum, (id_item+1)*stepnum):
                    df.loc[len(df)] = [chosen_shuffled[t], rejected_shuffled[t], answer,  chosen_value_shuffled[t], distance_shuffled[t]]

    # df.to_csv(f'/home/wentaos/Optima/local_influence/{dataset_type}{scale_name}.txt', sep='\t', index=False)
    df.to_csv(f'/home/wentaos/Optima/local_influence/{dataset_type}_record_new.txt', sep='\t', index=False)
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 读取数据（以 CSV 文件为例）
    # df = pd.read_csv('your_file.csv')

    # 或者使用手动输入数据
    data = df["best_100"]

    # 计算大于零的占比
    greater_than_zero = np.sum(data > 0) / len(data)

    # 计算标准差
    std_dev = np.std(data)
    # 绘制密度分布图
    sns.kdeplot(data, color="orange", shade=True)

    # 设置图表标题和标签
    plt.title(f'Standard Deviation={std_dev}\nPos ({int(greater_than_zero*100)}%), Neg ({100-int(greater_than_zero*100)}%)')
    plt.xlabel('Oracle Data Influence')
    plt.ylabel('Probability Density')

    # plt.savefig(f"/home/wentaos/Optima/local_influence/{dataset_type}{scale_name}_figure.png", bbox_inches='tight')

if __name__ == "__main__":
    dataset_type = "mwh_qa"
    i = 0
    Analysis_distribution(
        origin_dpo_path = "/data/group_data/cx_group/MCTS-agent/results/mwh_qa_sft_dpo/dpo/iteration_0_dpo_format.jsonl",
        # origin_dpo_path = f"/data/group_data/cx_group/MCTS-agent/results/{dataset_type}_sft_dpo_DI/dpo/iteration_{i}_dpo_format.jsonl",
        # origin_dpo_path = "/data/user_data/wentaos/Optima/results/cbt_dpo/iteration_0_dpo_format.jsonl",
        # origin_dpo_path = "/data/group_data/cx_group/MCTS-agent/results/mwh_qa_sft_dpo_scale/dpo_3_4/iteration_0_dpo_format.jsonl",
        dataset_path = "str",
        # data_influence_path = f"/data/user_data/wentaos/Optima/inference_results/{dataset_type}_dpo_iteration_{i}_server_stepnum5",
        data_influence_path = "/data/user_data/wentaos/Optima/inference_results/mwh_qa_dpo_sever_stepnum5",
        # data_influence_path = "/data/user_data/wentaos/Optima/inference_results/mwh_qa_dpo_iteration_0_3_4_server_stepnum5",
        output_dir="/data/user_data/wentaos/Optima/my_datasets",
        dataset_type = dataset_type,
        stepnum = 5,
        select_num = 12000,
        iteration = i,
        initial_value = 0,
        scale_name = f'',
        other_type=True,
    )

    