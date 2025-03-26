import sys
sys.path.append("/home/wentaos/Optima/local_influence")
sys.path.append("/home/wentaos/Optima/")
from transformers import AutoTokenizer, AutoModelForCausalLM
from train.datagenerate import vllm_data_generate
from train.monte_carlo_deploy import monte_carlo_data_generate_deploy, monte_carlo_data_generate_deploy_lora
from dataloader.dataloader import (
    DataloaderForHotpotQA,
    DataloaderForMWHQA,
    DataloaderForCBT,
    DataloaderForGSM8K,
    DataloaderForMATH,
    DataloaderForTrivalQA,
    DataloaderForARC,
    DataloaderForMMLU,
)
from datasets import load_from_disk
import os
import yaml
import subprocess
import requests
import torch
import json
import numpy as np
import time
import os
import subprocess
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing
from queue import Queue, Empty

from generate_influence_dataset import generate_influence_dataset

def collect_data_influence_once(gpu_id, t, i, interval, dataset_type, name):
    # Calculate start_point and end_point
    print(f"Running task {t}")
    start_point = t 
    end_point = interval + t
    main_processs_port = 29000 + t
    print("start")
    process = subprocess.Popen(
        f"""source ~/.bashrc && \
        conda activate optima-train &&\
        cd /home/wentaos/Optima/alignment-handbook &&\
        CUDA_VISIBLE_DEVICES={gpu_id} ACCELERATE_LOG_LEVEL=info NCCL_P2P_DISABLE=1 accelerate launch --config_file /home/wentaos/Optima/alignment-handbook/recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port {main_processs_port} /home/wentaos/Optima/alignment-handbook/scripts/run_dpo_collect_noserver.py /home/wentaos/Optima/alignment-handbook/recipes/Llama3-8B-multi-scale/{dataset_type}_sft_dpo/dpo_iteration_{i}_lora_collect_warmup_scale_{name}.yaml --load_in_4bit=false --start_point={start_point} --end_point={end_point} --step_num=5
        """, 
        shell=True)
    process.wait()
    time.sleep(10)

    try: 
        process_port = 12000 + t
        process = subprocess.Popen(
            f"""source ~/.bashrc && \
            conda activate optima-vllm && \
            cd /home/wentaos/Optima && \
            export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True && \
            OUTLINES_CACHE_DIR=./outlines/{dataset_type}/{gpu_id}_{name} python inference_script_server_train_dynamic.py \
            --model_root_path /data/group_data/cx_group/MCTS-agent/checkpoints/{dataset_type}_sft_dpo_scale/sft_iteration_{i} \
            --model_name iteration_{i} \
            --model_lora_path /data/group_data/cx_group/MCTS-agent/checkpoints/{dataset_type}_sft_dpo_scale/dpo_iteration_{i}_{name}_lora/dpo_iteration_{i} \
            --tokenizer_path /home/wentaos/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a \
            --device {gpu_id} \
            --port {process_port} \
            --dataset_type {dataset_type} \
            --num_thread 16 \
            --num_gpu 1 \
            --start_id {start_point} \
            --end_id {end_point} \
            --output_root_path "/data/user_data/wentaos/Optima/inference_results/{dataset_type}_dpo_iteration_{i}_{name}_server_stepnum5/id" \
            --vllm_env optima-vllm
            """, 
            shell=True)
        process.wait()
        time.sleep(10)
    except Exception as e:
        # 打印错误信息和堆栈跟踪
        print(f"发生错误: {e}")
        traceback.print_exc()
        return e

    for id in range(start_point, end_point):
        checkpoint_path = f"/data/group_data/cx_group/MCTS-agent/checkpoints/{dataset_type}_sft_dpo_scale/dpo_iteration_{i}_{name}_lora/dpo_iteration_{i}_id_{id}"
        if os.path.exists(checkpoint_path):
            process = subprocess.Popen(f"""rm -r {checkpoint_path}""", shell=True)
            process.wait()

    return True


def run_task_on_gpu(task_function, gpu_id, *args, **kwargs):
    """在指定 GPU 上运行任务的包装函数"""
    with torch.cuda.device(gpu_id):
        print(f"Running task on GPU {gpu_id}")
        return task_function(*args, **kwargs)

# def collect_data_influence_task(t, i, interval, dataset_type):
#     """示例任务函数"""
#     print(f"Running task {t}")
#     # 模拟任务的运行
#     result = collect_data_influence_once(t, i, interval, dataset_type)
#     return result

# 调度任务
def schedule_tasks(args_list, available_gpus):
    with ThreadPoolExecutor(max_workers=len(available_gpus)) as executor:
        # 跟踪正在运行的任务和其对应的 GPU
        gpu_status = {gpu: True for gpu in available_gpus}  # True 表示空闲
        futures_to_gpu = {}

        while args_list or futures_to_gpu:
            # 分配任务到空闲 GPU
            for gpu_id, is_free in gpu_status.items():
                if is_free and args_list:
                    task_args = args_list.pop(0)
                    future = executor.submit(collect_data_influence_once, gpu_id, *task_args)
                    futures_to_gpu[future] = gpu_id
                    gpu_status[gpu_id] = False

            # 处理已完成任务
            completed_futures = []
            for future in as_completed(futures_to_gpu):
                gpu_id = futures_to_gpu.pop(future)
                gpu_status[gpu_id] = True
                try:
                    result = future.result()
                    print(f"Task completed on GPU {gpu_id}, result: {result}")
                except Exception as e:
                    print(f"Task on GPU {gpu_id} failed: {e}")
                completed_futures.append(future)

            # 移除已完成的任务
            for completed_future in completed_futures:
                futures_to_gpu.pop(completed_future, None)

if __name__ == "__main__":

    import argparse

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="一个简单的命令行参数示例")


    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--name', type=str)
    parser.add_argument('--dataset_type', type=str, default="mwh_qa", help='是否显示详细信息')
    parser.add_argument('--lora_path', default=None)
    # 解析参数
    args = parser.parse_args()

    data_influence_path = f"/data/user_data/wentaos/Optima/inference_results/{args.dataset_type}_dpo_iteration_{args.iteration}_{args.name}_server_stepnum5/"
            
    if os.path.exists(data_influence_path):
        print(f"data influence already exist {data_influence_path}")
        
        # 获取所有子文件夹的名称
        subfolders = [f for f in os.listdir(data_influence_path) if os.path.isdir(os.path.join(data_influence_path, f))]
        
        # 提取所有子文件夹名称中的 t 值
        id_values = [int(f.split('_')[-1]) for f in subfolders if f.startswith('id_')]
        
        # 找到最大的 t 值
        if id_values:
            base_num = max(id_values)
            print(f"max_id: {base_num}")
    else:
        base_num = 0


    origin_dpo_yaml_path_lora = f"/home/wentaos/Optima/alignment-handbook/recipes/Llama3-8B-multi-scale/{args.dataset_type}_sft_dpo/dpo/base_lora.yaml"
    mid_yaml_root_path = f"/home/wentaos/Optima/alignment-handbook/recipes/Llama3-8B-multi-scale/{args.dataset_type}_sft_dpo"
    dataset_path = f"/data/group_data/cx_group/MCTS-agent/my_datasets/{args.dataset_type}_sft_dpo_scale/dpo_{args.name}/iteration_{args.iteration}"
    check_point_root_path = f"/data/group_data/cx_group/MCTS-agent/checkpoints/{args.dataset_type}_sft_dpo_scale/dpo_iteration_{args.iteration}_{args.name}_lora/dpo_iteration_{args.iteration}"
    # dpo warmup  train     
    with open(origin_dpo_yaml_path_lora, "r") as f:
        config = yaml.safe_load(f)
    


    if args.lora_path == None:
        config["model_name_or_path"] = os.path.join(f"/data/group_data/cx_group/MCTS-agent/checkpoints/{args.dataset_type}_sft_dpo_scale/dpo_iteration_{args.iteration}_scale" + "_lora", "checkpoint-50")
    else:
        config["model_name_or_path"] = args.lora_path
    config["dataset_mixer"] = {dataset_path: 1.0}
    config["output_dir"] = check_point_root_path
    config["save_strategy"] = "no"
    config["warmup_steps"] = 0
    config["step_num"] = 5
    config["start_point"] = 0
    config["end_point"] = 300
    config["eval_num"] = 150
    # config.pop("warmup_ratio")
    with open(
        os.path.join(mid_yaml_root_path, f"dpo_iteration_{args.iteration}_lora_collect_warmup_scale_{args.name}.yaml"), "w"
    ) as fout:
        fout.write(yaml.safe_dump(config))
    
    
    # data influence collect
    train_dataset = load_from_disk(os.path.join(dataset_path, "train"), keep_in_memory=True)

    num_device = 8 # List of available GPU IDs
    interval = 5

    args_list = []
    for t in range(base_num, len(train_dataset)//5, interval):
        args_list.append((t, args.iteration, interval, args.dataset_type, args.name))
    
    available_gpus = list(range(num_device))
    if args_list:
        schedule_tasks(args_list, available_gpus)
        
        

        
    

    data_influence_path = f"/data/user_data/wentaos/Optima/inference_results/{args.dataset_type}_dpo_iteration_{args.iteration}_{args.name}_server_stepnum5/"
    mid_dpo_jsonl_root_path = f"/data/group_data/cx_group/MCTS-agent/results/{args.dataset_type}_sft_dpo_scale/dpo_{args.name}/"
    mid_dpo_dataset_root_path = f"/data/group_data/cx_group/MCTS-agent/my_datasets/{args.dataset_type}_sft_dpo_scale/dpo_{args.name}/"
    
    # generate mix dataset
    generate_influence_dataset(
        origin_dpo_path=os.path.join(mid_dpo_jsonl_root_path, f"iteration_{args.iteration}_dpo_format.jsonl"),
        dataset_path=os.path.join(mid_dpo_dataset_root_path, f"iteration_{args.iteration}"),
        data_influence_path=data_influence_path,
        output_dir=mid_dpo_dataset_root_path,
        dataset_type=args.dataset_type,
        stepnum= 5,
        iteration=args.iteration,
        select_num=len(train_dataset)//2,
        other_type=True,
    )