# from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from train.datagenerate import vllm_data_generate
from train.monte_carlo_deploy import monte_carlo_data_generate_deploy
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
from local_influence.generate_influence_dataset import generate_influence_dataset
import os
import yaml
import subprocess
import requests
import torch
import json
import numpy as np
import time
import sys


def dpo_train(
    origin_dpo_yaml_path: str,
    initial_model_path: str,
    initial_dataset_path: str,
    dataset_type: str,
    mid_yaml_root_path: str,
    mid_dpo_jsonl_root_path: str,
    mid_dpo_dataset_root_path: str,
    check_point_root_path: str,
    initial_episilon: float,
    initial_dpo_min_value: float,
    initial_dpo_episilon: float,
    iteration_times: int,
    port: int,
    devices: str,
    tokenizer_first_path: str,
    tokenizer_second_path: str,
    sample_count: int,
    monte_sample_count: int,
    explore_count: int,
    thread_count: int,
    prompt_pool_path: str,
    skipping: int,
):
    pass


def get_train_dataloader(dataset_type):
    score_type = "f1-score"
    if dataset_type == "hotpot_qa":
        loader = DataloaderForHotpotQA(split="train")
    elif dataset_type == "mwh_qa":
        loader = DataloaderForMWHQA(split="train")
    elif dataset_type == "cbt":
        print("cbt")
        loader = DataloaderForCBT(split="train")
    elif dataset_type == "gsm8k":
        print("gsm8k")
        score_type = "exact-match"
        loader = DataloaderForGSM8K(split="train")
    elif dataset_type == "math":
        print("math")
        loader = DataloaderForMATH(split="train")
        score_type = "math"
    elif dataset_type == "trival_qa":
        print("trival_qa")
        loader = DataloaderForTrivalQA(split="train")
    elif dataset_type == "arc":
        print("arc")
        loader = DataloaderForARC(split="train")
        score_type = "exact-match"
    elif dataset_type == "mmlu":
        print("mmlu")
        loader = DataloaderForMMLU(split="auxiliary_train")
        score_type = "exact-match"
    return score_type, loader
    
def dpo_train_v2_data_influence(
    origin_dpo_yaml_path: str,
    initial_model_path: str,
    initial_dataset_path: str,
    dataset_type: str,
    mid_yaml_root_path: str,
    mid_dpo_jsonl_root_path: str,
    mid_dpo_dataset_root_path: str,
    check_point_root_path: str,
    initial_episilon: float,
    initial_dpo_min_value: float,
    initial_dpo_episilon: float,
    iteration_times: int,
    port: int,
    devices: str,
    tokenizer_first_path: str,
    tokenizer_second_path: str,
    sample_count: int,
    monte_sample_count: int,
    explore_count: int,
    thread_count: int,
    prompt_pool_path: str,
    skipping: int,
    cal_ppl: int = 1,
    from_initial: bool = False,
    lambda1: float = -0.6,
    lambda2: float = 1,
    vllm_env:str = "",
    alignment_env:str = "",
    origin_dpo_yaml_path_lora:str=""
):
    model_path = initial_model_path
    episilon = initial_episilon
    dpo_episilon = initial_dpo_episilon
    dpo_min_value = initial_dpo_min_value
    loader = None
    num_device = int(devices)

    score_type, loader = get_train_dataloader(dataset_type)
    for _ in range(skipping):
        loader.sample_once()

    for i in range(iteration_times):

        dataset_path = os.path.join(mid_dpo_dataset_root_path, f"iteration_{i}")
        if os.path.exists(dataset_path):
            pass 
        else: 
            # deploy model for dpo
            ports = []
            num_device = int(devices)
            for j in range(num_device):
                process_occupy = subprocess.Popen(
                    f"""
                source ~/.bashrc && \
                conda activate {vllm_env} && \
                CUDA_VISIBLE_DEVICES={j+1} vllm serve {model_path} --host 0.0.0.0 --port {port+1+j} --served-model-name "Llama-3" --enable-prefix-caching
                """,
                    shell=True,
                )
                ports.append(port + 1 + j)

            process = subprocess.Popen(
                f"""source ~/.bashrc && \
                    conda activate {alignment_env} && \
                    python ppl_deploy.py --num_replicas 1
                    """,
                shell=True,
            )
            # check deploy
            while True:
                try:
                    ret = requests.post(
                        "http://localhost:8000/ppl", json={"texts": ["hi hi" * 5000]}
                    )
                    if ret.status_code == 200:
                        break
                except:
                    continue

            while True:
                try:
                    message_input = [{"role": "assistant", "content": "hello!"}]
                    headers = {"Content-Type": "application/json"}
                    data_json = {
                        "model": "Llama-3",
                        "messages": message_input,
                    }
                    response = requests.post(
                        f"http://0.0.0.0:{port+1}/v1/chat/completions",
                        headers=headers,
                        json=data_json,
                    )
                    print(response)
                    content = (response.json()["choices"][0]["message"]["content"],)
                    print(f"ready to generate data: {content}")
                    break
                except:
                    continue
            # sample 100
            try:
                process = subprocess.Popen(
                    f"""
                    source ~/.bashrc && \
                    conda activate {alignment_env} && \
                    OUTLINES_CACHE_DIR='./outlines' HF_DATASETS_CACHE="../huggingface_cache/huggingface_dataset_cache" python inference_main.py\
                    --output_path {os.path.join(mid_dpo_jsonl_root_path,f"tmp_iteration_{i}.jsonl")}\
                    --sample_count 100\
                    --num_thread 24\
                    --url_first "http://0.0.0.0:{port+1}/v1/chat/completions"\
                    --url_second "http://0.0.0.0:{port+1}/v1/chat/completions"\
                    --dataset_type {dataset_type}\
                    --skipping {loader.current_task_id+1}\
                    --temperature 0.7 \
                    --ports [{",".join([str(p) for p in ports])}]\
                    --add_name 0
                    """,
                    shell=True,
                )
                process.wait()
            except:
                print('----------------------A---------------------')
                for j in range(num_device):
                    os.system(
                        f"""pkill -f "vllm serve {model_path} --host 0.0.0.0 --port {port+j+1} --served-model-name Llama-3 --enable-prefix-caching" """
                    )
                os.system("""pkill -f "python ppl_deploy.py --num_replicas 1" """)
                torch.cuda.empty_cache()
                exit(0)
            # calculate quantile
            # token_count_list = []
            # with open(
            #     os.path.join(mid_dpo_jsonl_root_path, f"tmp_iteration_{i}.jsonl"), "r"
            # ) as f:
            #     for line in f:
            #         data = json.loads(line)
            #         token_count_list.append(data["results"][0]["token_count"])
            # percentiles = [85]
            # quantile = int(np.percentile(token_count_list, percentiles)[0]) + 1


            quantile = 2048
            # monte_carlo data generate
            time.sleep(10)
            try:
                monte_carlo_data_generate_deploy(
                    max_token=quantile,
                    model="http://localhost:8000/ppl",
                    tokenizer_path=initial_model_path,
                    sft_data_path="",
                    output_path=os.path.join(
                        mid_dpo_jsonl_root_path, f"iteration_{i}.jsonl"
                    ),
                    sample_count=monte_sample_count,
                    num_thread=thread_count,
                    dataloader=loader,
                    prompt_pool_path=prompt_pool_path,
                    model_url=f"http://0.0.0.0:{port}/v1/chat/completions",
                    min_value=dpo_episilon,
                    incremental_threshold=dpo_min_value,
                    ports=ports,
                    score_type=score_type,
                    cal_ppl=(cal_ppl == 1),
                    lambda1=lambda1,
                    lambda2=lambda2,
                )
            except:
                for j in range(num_device):
                    os.system(
                        f"""pkill -f "vllm serve {model_path} --host 0.0.0.0 --port {port+j+1} --served-model-name Llama-3 --enable-prefix-caching" """
                    )
                os.system("""pkill -f "python ppl_deploy.py --num_replicas 1" """)
                torch.cuda.empty_cache()
                exit(0)
            for j in range(num_device):
                os.system(
                    f"""pkill -f "vllm serve {model_path} --host 0.0.0.0 --port {port+j+1} --served-model-name Llama-3 --enable-prefix-caching" """
                )
            os.system("""pkill -f "python ppl_deploy.py --num_replicas 1" """)
            torch.cuda.empty_cache()
            time.sleep(10)
            # generate dpo dataset
            process = subprocess.Popen(
                f"""
                source ~/.bashrc && \
                conda activate {alignment_env} && \
                HF_DATASETS_CACHE="../huggingface_cache/huggingface_dataset_cache" python reward_main.py --cleaned_output_path {os.path.join(mid_dpo_jsonl_root_path,f"iteration_{i}_dpo_format.jsonl")}\
                --dpo_dataset_output_path {os.path.join(mid_dpo_dataset_root_path,f"iteration_{i}")}\
                --model_path {initial_model_path}
                """,
                shell=True,
            )
            process.wait()
            dataset_path = os.path.join(mid_dpo_dataset_root_path, f"iteration_{i}")


        # dpo lora train     
        with open(origin_dpo_yaml_path_lora, "r") as f:
            config = yaml.safe_load(f)
        if not from_initial:
            config["model_name_or_path"] = model_path
        else:
            config["model_name_or_path"] = initial_model_path
        config["dataset_mixer"] = {dataset_path: 1.0}
        config["output_dir"] = os.path.join(check_point_root_path + "_lora", f"dpo_iteration_{i}")
        config["save_strategy"] = "steps"
        config["save_steps"] = 50

        try:
            with open(
                os.path.join(mid_yaml_root_path, f"dpo_iteration_{i}_lora.yaml"), "r"
            ) as f:
                pass
        except:
            with open(
                os.path.join(mid_yaml_root_path, f"dpo_iteration_{i}_lora.yaml"), "w"
            ) as fout:
                fout.write(yaml.safe_dump(config))
            process = subprocess.Popen(
                f"""source ~/.bashrc &&\
                conda activate {alignment_env} && \
                ACCELERATE_LOG_LEVEL=info NCCL_P2P_DISABLE=1 accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_dpo.py  {os.path.join(mid_yaml_root_path[21:],f"dpo_iteration_{i}_lora.yaml")} --load_in_4bit=false""",
                shell=True,
                cwd="./alignment-handbook",
            )
            process.wait()


        

        data_influence_path = f"/data/user_data/wentaos/Optima/inference_results/${dataset_type}_dpo_server_stepnum5/"
        if os.path.exists(data_influence_path):
            pass
        else:
            # dpo warmup  train     
            with open(origin_dpo_yaml_path_lora, "r") as f:
                config = yaml.safe_load(f)
            
            config["model_name_or_path"] = os.path.join(os.path.join(check_point_root_path + "_lora", f"dpo_iteration_{i}"), "checkpoint-50")
            config["dataset_mixer"] = {dataset_path: 1.0}
            config["output_dir"] = os.path.join(check_point_root_path + "_lora", f"dpo_iteration_{i}")
            config["save_strategy"] = "no"
            config["warmup_steps"] = 0
            config["step_num"] = 5
            config["start_point"] = 0
            config["end_point"] = 300
            config["eval_num"] = 150
            with open(
                os.path.join(mid_yaml_root_path, f"dpo_iteration_{i}_lora_collect_warmup.yaml"), "w"
            ) as fout:
                fout.write(yaml.safe_dump(config))
            # data influence collect
            train_dataset = load_from_disk(os.path.join(dataset_path, "train"), keep_in_memory=True)
            step = len(train_dataset) // num_device

            for t in range(0, step, 5):
                
                for j in range(num_device + 1):
                    basenum = j * step
                    start_point=t + basenum
                    end_point=t + 5 + basenum

                    num_port = 12000 + basenum
                    process = subprocess.Popen(
                        f"""source ~/.bashrc &&\
                        conda activate {alignment_env} && \
                        CUDA_VISIBLE_DEVICES={j} ACCELERATE_LOG_LEVEL=info NCCL_P2P_DISABLE=1 accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port {num_port} scripts/run_dpo_collect_noserver.py /home/wentaos/Optima/alignment-handbook/recipes/Llama3-8b/{dataset_type}_dpo/dpo_iteration_{i}_lora_collect_warmup.yaml --load_in_4bit=false --start_point=${start_point} --end_point=${end_point} --step_num=5
                        """,
                        shell=True,
                        cwd="./alignment-handbook",
                    )
                    process.wait()

                    process = subprocess.Popen(
                        f"""
                        source ~/.bashrc &&\
                        conda activate {vllm_env} && \
                        export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True &&\
                        CUDA_VISIBLE_DEVICES={j} OUTLINES_CACHE_DIR=./outlines/${dataset_type}/${j} python inference_script_server_train_dynamic.py \
                            --model_root_path /data/user_data/wentaos/optima-checkpoints/${dataset_type}_sft/ \
                            --model_lora_path /data/user_data/wentaos/optima-checkpoints/${dataset_type}_dpo_lora/dpo_iteration_0 \
                            --tokenizer_path /home/wentaos/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a \
                            --device 0 \
                            --port {num_port} \
                            --dataset_type ${dataset_type} \
                            --num_thread 16 \
                            --num_gpu 1 \
                            --start_id ${start_point} \
                            --end_id ${end_point} \
                            --output_root_path "/data/user_data/wentaos/Optima/inference_results/${dataset_type}_dpo_server_stepnum5/id" \
                            --vllm_env optima-vllm &&\
                        for id in $(seq $start_point $((end_point-1))) \
                        do \
                            rm -r /data/user_data/wentaos/optima-checkpoints/${dataset_type}_dpo_lora/dpo_iteration_0_id_${id}\
                        done\
                        """,
                        shell=True,
                    )
                    process.wait()

        data_influence_path = f"/data/user_data/wentaos/Optima/inference_results/${dataset_type}_dpo_server_stepnum5/"
        # generate mix dataset
        generate_influence_dataset(
            origin_dpo_path=os.path.join(mid_dpo_jsonl_root_path, f"iteration_{i}_dpo_format.jsonl"),
            dataset_path=os.path.join(mid_dpo_dataset_root_path, f"iteration_{i}"),
            data_influence_path=data_influence_path,
            dataset_type=dataset_type,
            stepnum= 5,
            select_num=len(train_dataset)//2,
        )
         
        

        # dpo train
        with open(origin_dpo_yaml_path, "r") as f:
            config = yaml.safe_load(f)
        if not from_initial:
            config["model_name_or_path"] = model_path
        else:
            config["model_name_or_path"] = initial_model_path
        config["dataset_mixer"] = {os.path.join(mid_dpo_dataset_root_path + f"_mix_warmup_{len(train_dataset)//2}", f"iteration_{i}"): 1.0}
        config["output_dir"] = os.path.join(check_point_root_path + f"_mix_warmup_{len(train_dataset)//2}", f"dpo_iteration_{i}")

        try:
            with open(
                os.path.join(mid_yaml_root_path,, f"dpo_iteration_{i}.yaml" + f"_mix_warmup_{len(train_dataset)//2}"), "r"
            ) as f:
                pass
        except:
            with open(
                os.path.join(mid_yaml_root_path,, f"dpo_iteration_{i}.yaml" + f"_mix_warmup_{len(train_dataset)//2}"), "w"
            ) as fout:
                fout.write(yaml.safe_dump(config))
            process = subprocess.Popen(
                f"""source ~/.bashrc &&\
                conda activate {alignment_env} && \
                ACCELERATE_LOG_LEVEL=info NCCL_P2P_DISABLE=1 accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py  {os.path.join(mid_yaml_root_path[21:], f"dpo_iteration_{i}.yaml" + f"_mix_warmup_{len(train_dataset)//2}")} """,
                shell=True,
                cwd="./alignment-handbook",
            )
            process.wait()
        # update
        model_path = os.path.join(check_point_root_path, f"dpo_iteration_{i}")


