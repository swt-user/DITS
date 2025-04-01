# from datasets import load_from_disk
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
    DataloaderForMBPP,
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
from multiprocessing import Pool,  Process
from queue import Queue, Empty
import sys
sys.path.append("/home/wentaos/Optima/local_influence")
from generate_influence_dataset import generate_influence_dataset

def collect_data_influence(gpu_id, i, task_queue, lock, interval):
    # Calculate start_point and end_point

    while True:
        try:
            # Fetch the next task from the queue
            t, basenum, dataset_type = task_queue.get(timeout=5)  # Wait up to 5 seconds for a task
        except Empty:
            break 

        
        start_point = t + basenum
        end_point = t + interval + basenum

        # Set environment variables
        env = os.environ.copy()
        env["ACCELERATE_LOG_LEVEL"] = "info"
        env["NCCL_P2P_DISABLE"] = "1"
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        
        # Conda environments and working directories
        conda_env_1 = "optima-train"
        conda_env_2 = "optima-vllm"
        main_processs_port = 29000 + t
        # Task 1: Run accelerate command
        # accelerate_command = [
        #     "bash", "-c",
        #     f"""
        #     source ~/miniconda3/etc/profile.d/conda.sh &&
        #     conda activate {conda_env_1} &&
        #     cd /home/wentaos/Optima/alignment-handbook &&
        #     accelerate launch --config_file /home/wentaos/Optima/alignment-handbook/recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port {main_processs_port} \
        #     /home/wentaos/Optima/alignment-handbook/scripts/run_dpo_collect_noserver.py \
        #     /home/wentaos/Optima/alignment-handbook/recipes/Llama3-8B-multi-DI/{dataset_type}_sft_dpo/dpo_iteration_{i}_lora_collect_warmup.yaml \
        #     --load_in_4bit=false \
        #     --start_point={start_point} \
        #     --end_point={end_point} \
        #     --step_num=5
        #     """
        # ]
        # print("Accelerate Command:", " ".join(accelerate_command))
        print(f"{gpu_id} start")
        process = subprocess.Popen(
            f"""source ~/.bashrc && \
            conda activate {conda_env_1} &&\
            cd /home/wentaos/Optima/alignment-handbook &&\
            accelerate launch --config_file /home/wentaos/Optima/alignment-handbook/recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port {main_processs_port} \
            /home/wentaos/Optima/alignment-handbook/scripts/run_dpo_collect_noserver.py \
            /home/wentaos/Optima/alignment-handbook/recipes/Llama3-8B-multi-DI/{dataset_type}_sft_dpo/dpo_iteration_{i}_lora_collect_warmup.yaml \
            --load_in_4bit=false \
            --start_point={start_point} \
            --end_point={end_point} \
            --step_num=5
            """, 
            env=env, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        process.wait()
        # 获取输出和错误
        stdout, stderr = process.communicate()

        # 检查进程的返回状态
        if process.returncode == 0:
            print("Command succeeded with output:")
            print(stdout)
        else:
            print("Command failed with error:")
            print(stderr)

        print("Task 1 finish")
        # Task 2: Run inference_script_server_train_dynamic.py
        inference_command = [
            "bash", "-c",
            f"""
            conda init &&
            conda activate {conda_env_2} &&
            cd /home/wentaos/Optima &&
            OUTLINES_CACHE_DIR=./outlines/{dataset_type}/{gpu_id} \
            python inference_script_server_train_dynamic.py \
            --model_root_path /data/group_data/cx_group/MCTS-agent/checkpoints/{dataset_type}_sft_dpo_DI/sft_iteration_{i} \
            --model_name iteration_0 \
            --model_lora_path /data/group_data/cx_group/MCTS-agent/checkpoints/{dataset_type}_sft_dpo_DI_lora/dpo_iteration_{i} \
            --tokenizer_path /home/wentaos/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a \
            --device {gpu_id} \
            --port {main_processs_port} \
            --dataset_type {dataset_type} \
            --num_thread 16 \
            --num_gpu 1 \
            --start_id {start_point} \
            --end_id {end_point} \
            --output_root_path "/data/user_data/wentaos/Optima/inference_results/{dataset_type}_dpo_iteration_{i}_server_stepnum5/id" \
            --vllm_env optima-vllm
            """
        ]
        try:
            subprocess.run(inference_command, env=env, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with return code {e.returncode}")
            print(f"Error output: {e.stderr}")

        # Cleanup: Remove checkpoints for the current range
        for id in range(start_point, end_point):
            checkpoint_path = f"/data/group_data/cx_group/MCTS-agent/checkpoints/{dataset_type}_sft_dpo_DI_lora/dpo_iteration_{i}_id_{id}"
            if os.path.exists(checkpoint_path):
                subprocess.run(["rm", "-r", checkpoint_path])


def sft_dpo_train_v2_DI(
    origin_sft_yaml_path: str,
    origin_dpo_yaml_path: str,
    origin_dpo_yaml_path_lora: str,
    initial_model_path: str,
    initial_dataset_path: str,
    dataset_type: str,
    mid_yaml_root_path: str,
    mid_sft_jsonl_root_path: str,
    mid_dpo_jsonl_root_path: str,
    mid_sft_dataset_root_path: str,
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
    lambda1: float = -0.6,
    lambda2: float = 1,
    from_initial: bool = False,
    vllm_env:str = "",
    alignment_env:str = ""
):
    model_path = initial_model_path
    episilon = initial_episilon
    dpo_episilon = initial_dpo_episilon
    dpo_min_value = initial_dpo_min_value
    loader = None
    score_type = "f1-score"
    no_use_prompt_pool = False
    is_math = False
    if dataset_type == "hotpot_qa":
        loader = DataloaderForHotpotQA(split="train")
    elif dataset_type == "mwh_qa":
        loader = DataloaderForMWHQA(split="train")
    elif dataset_type == "cbt":
        print("cbt")
        loader = DataloaderForCBT(split="train")
    elif dataset_type == "gsm8k":
        print("gsm8k")
        is_math = True
        score_type = "exact-match"
        loader = DataloaderForGSM8K(split="train")
    elif dataset_type == "math":
        is_math = True
        print("math")
        loader = DataloaderForMATH(split="train")
        score_type = "math"
        # no_use_prompt_pool = True
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
    elif dataset_type == "mbpp":
        print("mbpp")
        loader = DataloaderForMBPP(split="train")
        score_type = "code"

    for _ in range(skipping):
        loader.sample_once()
    for i in range(iteration_times):
        
        
        dataset_path = os.path.join(mid_sft_dataset_root_path, f"iteration_{i}")
        if os.path.exists(dataset_path):
            print(f"dataset already exists at {dataset_path}")
        else:
            ports = []
            process = subprocess.Popen(
                f"""
            source ~/.bashrc && \
            conda activate {vllm_env} && \
            CUDA_VISIBLE_DEVICES={devices} vllm serve {model_path} --host 0.0.0.0 --port {port} --served-model-name "Llama-3" --enable-prefix-caching
            """,
                shell=True,
            )
            ports.append(port)
            for j in range(7):
                process_occupy = subprocess.Popen(
                    f"""
                source ~/.bashrc && \
                conda activate {vllm_env} && \
                CUDA_VISIBLE_DEVICES={j} vllm serve {model_path} --host 0.0.0.0 --port {port+j+1} --served-model-name "Llama-3" --enable-prefix-caching
                """,
                    shell=True,
                )
                ports.append(port + j + 1)
            while True:
                try:
                    message_input = [{"role": "assistant", "content": "hello!"}]
                    headers = {"Content-Type": "application/json"}
                    data_json = {
                        "model": "Llama-3",
                        "messages": message_input,
                    }
                    response = requests.post(
                        f"http://0.0.0.0:{port}/v1/chat/completions",
                        headers=headers,
                        json=data_json,
                    )
                    content = (response.json()["choices"][0]["message"]["content"],)
                    print(f"ready to generate data: {content}")
                    break
                except:
                    continue
            pid = process.pid
            vllm_data_generate(
                "Llama-3",
                "Llama-3",
                url_first=f"http://0.0.0.0:{port}/v1/chat/completions",
                url_second=f"http://0.0.0.0:{port}/v1/chat/completions",
                tokenizer_path_first=tokenizer_first_path,
                tokenizer_path_second=tokenizer_second_path,
                sample_count=sample_count,
                explore_count=explore_count,
                output_path=os.path.join(mid_sft_jsonl_root_path, f"iteration_{i}.jsonl"),
                thread_count=thread_count,
                prompt_pool_path=prompt_pool_path,
                dataloader=loader,
                no_use_prompt_pool=((i != 0) or no_use_prompt_pool),
                temperature=0.7 if ((i != 0) or no_use_prompt_pool or is_math) else 0.3,
                ports=ports,
                iteration=i,
            )
            time.sleep(10)
            for j in range(8):
                os.system(
                    f"""pkill -f "vllm serve {model_path} --host 0.0.0.0 --port {port+j} --served-model-name Llama-3 --enable-prefix-caching" """
                )
            time.sleep(30)
            process = subprocess.Popen(
                f"""source ~/.bashrc && \
                    conda activate {alignment_env} && \
                    python ppl_deploy.py 
                    """,
                shell=True,
            )

            while True:
                try:
                    ret = requests.post(
                        "http://localhost:8000/ppl", json={"texts": ["hi hi" * 5000]}
                    )
                    if ret.status_code == 200:
                        break
                except:
                    continue

            # reward
            process = subprocess.Popen(
                f"""source ~/.bashrc && \
                    conda activate {alignment_env} && \
                    HF_DATASETS_CACHE="../huggingface_cache/huggingface_dataset_cache" python reward_main.py --raw_data_path {os.path.join(mid_sft_jsonl_root_path, f"iteration_{i}.jsonl")}\
                    --rewarded_output_path {os.path.join(mid_sft_jsonl_root_path, f"rewarded_iteration_{i}.jsonl")}\
                    --cleaned_output_path {os.path.join(mid_sft_jsonl_root_path, f"cleaned_iteration_{i}.jsonl")}\
                    --model_path {initial_model_path}\
                    --sft_dataset_output_path {os.path.join(mid_sft_dataset_root_path,f"iteration_{i}")}\
                    --score 1 --clean 1\
                    --episilon {episilon}\
                    --deploy 1\
                    --num_thread {thread_count}\
                    --score_type {score_type}\
                    --cal_ppl {cal_ppl}\
                    --lambda1 {lambda1}\
                    --lambda2 {lambda2}\
                    --prompt_type {loader.data_type}
                    """,
                shell=True,
            )
            process.wait()
            os.system("""pkill -f "python ppl_deploy.py" """)
            dataset_path = os.path.join(mid_sft_dataset_root_path, f"iteration_{i}")
            time.sleep(20)
        
        
        # train
        with open(origin_sft_yaml_path, "r") as f:
            config = yaml.safe_load(f)
        # if not from_initial:
        #     config["model_name_or_path"] = "."+model_path
        # else:
        #     config["model_name_or_path"] = initial_model_path
        # config["dataset_mixer"] = {"."+dataset_path: 1.0}
        # config["output_dir"] = os.path.join("."+check_point_root_path, f"sft_iteration_{i}")

        config["model_name_or_path"] = initial_model_path
        config["dataset_mixer"] = {dataset_path: 1.0}
        config["output_dir"] = os.path.join(check_point_root_path, f"sft_iteration_{i}")

        try:
            with open(
                os.path.join(mid_yaml_root_path, f"sft_iteration_{i}.yaml"), "r"
            ) as f:
                pass
        except:
            with open(
                os.path.join(mid_yaml_root_path, f"sft_iteration_{i}.yaml"), "w"
            ) as fout:
                fout.write(yaml.safe_dump(config))
            process = subprocess.Popen(
                f"""source ~/.bashrc &&\
                conda activate {alignment_env} && \
                ACCELERATE_LOG_LEVEL=info NCCL_P2P_DISABLE=1 accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py  {os.path.join(mid_yaml_root_path[21:],f"sft_iteration_{i}.yaml")} """,
                shell=True,
                cwd="./alignment-handbook",
            )
            process.wait()
        model_path = os.path.join(check_point_root_path, f"sft_iteration_{i}")
        time.sleep(15)


        dataset_path = os.path.join(mid_dpo_dataset_root_path, f"iteration_{i}")
        if os.path.exists(dataset_path):
            pass
        else:
            # deploy model for dpo
            ports = []
            for j in range(7):
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
            process = subprocess.Popen(
                f"""
                source ~/.bashrc && \
                conda activate {alignment_env} && \
                HF_DATASETS_CACHE="../huggingface_cache/huggingface_dataset_cache" python inference_main.py\
                --output_path {os.path.join(mid_dpo_jsonl_root_path,f"tmp_iteration_{i}.jsonl")}\
                --sample_count 100\
                --num_thread 24\
                --url_first "http://0.0.0.0:{port+1}/v1/chat/completions"\
                --url_second "http://0.0.0.0:{port+1}/v1/chat/completions"\
                --dataset_type {dataset_type}\
                --skipping {loader.current_task_id+1}\
                --temperature 0.7\
                --ports [{",".join([str(p) for p in ports])}]\
                --add_name 1
                """,
                shell=True,
            )
            process.wait()
            # calculate quantile
            token_count_list = []
            with open(
                os.path.join(mid_dpo_jsonl_root_path, f"tmp_iteration_{i}.jsonl"), "r"
            ) as f:
                for line in f:
                    data = json.loads(line)
                    token_count_list.append(data["results"][0]["token_count"])
            percentiles = [80]
            quantile = int(np.percentile(token_count_list, percentiles)[0]) + 1
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
            except Exception as e:
                # 打印错误信息和堆栈跟踪
                print(f"发生错误: {e}")
                traceback.print_exc()
                for j in range(7):
                    os.system(
                        f"""pkill -f "vllm serve {model_path} --host 0.0.0.0 --port {port+j+1} --served-model-name Llama-3 --enable-prefix-caching" """
                    )
                os.system("""pkill -f "python ppl_deploy.py --num_replicas 1" """)
                torch.cuda.empty_cache()
                raise
                exit(0)
            for j in range(7):
                os.system(
                    f"""pkill -f "vllm serve {model_path} --host 0.0.0.0 --port {port+j+1} --served-model-name Llama-3 --enable-prefix-caching" """
                )
            os.system("""pkill -f "python ppl_deploy.py --num_replicas 1" """)
            torch.cuda.empty_cache()
            time.sleep(30)
            # dpo dataset
            process = subprocess.Popen(
                f"""
                source ~/.bashrc && \
                conda activate {alignment_env} && \
                HF_DATASETS_CACHE="../huggingface_cache/huggingface_dataset_cache" python /home/wentaos/Optima/reward_main.py --cleaned_output_path {os.path.join(mid_dpo_jsonl_root_path,f"iteration_{i}_dpo_format.jsonl")}\
                --dpo_dataset_output_path {os.path.join(mid_dpo_dataset_root_path,f"iteration_{i}")} --model_path {initial_model_path}
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


        

        data_influence_path = f"/data/user_data/wentaos/Optima/inference_results/{dataset_type}_dpo_iteration_{i}_server_stepnum5/"
        
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
        # config.pop("warmup_ratio")
        with open(
            os.path.join(mid_yaml_root_path, f"dpo_iteration_{i}_lora_collect_warmup.yaml"), "w"
        ) as fout:
            fout.write(yaml.safe_dump(config))
        
        
        # data influence collect
        train_dataset = load_from_disk(os.path.join(dataset_path, "train"), keep_in_memory=True)

        num_device = int(devices)+1 # List of available GPU IDs
        interval = 50

        main_processs_port = 29000

        for t in range(base_num, len(train_dataset)//5, interval):
            start_point = t 
            end_point = interval + t
            main_processs_port = 29000 + t
            print("start")
            process = subprocess.Popen(
                f"""source ~/.bashrc && \
                conda activate optima-train &&\
                cd /home/wentaos/Optima/alignment-handbook &&\
                ACCELERATE_LOG_LEVEL=info NCCL_P2P_DISABLE=1 accelerate launch --config_file /home/wentaos/Optima/alignment-handbook/recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port {main_processs_port} /home/wentaos/Optima/alignment-handbook/scripts/run_dpo_collect_noserver.py /home/wentaos/Optima/alignment-handbook/recipes/Llama3-8B-multi-DI/{dataset_type}_sft_dpo/dpo_iteration_{i}_lora_collect_warmup.yaml --load_in_4bit=false --start_point={start_point} --end_point={end_point} --step_num=5
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
                    OUTLINES_CACHE_DIR=./outlines/{dataset_type} python inference_script_server_train_dynamic.py \
                    --model_root_path /data/group_data/cx_group/MCTS-agent/checkpoints/{dataset_type}_sft_dpo_DI/sft_iteration_{i} \
                    --model_name iteration_{i} \
                    --model_lora_path /data/group_data/cx_group/MCTS-agent/checkpoints/{dataset_type}_sft_dpo_DI_lora/dpo_iteration_{i} \
                    --tokenizer_path /home/wentaos/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a \
                    --device 0 \
                    --port {process_port} \
                    --dataset_type {dataset_type} \
                    --num_thread {num_device*16} \
                    --num_gpu {num_device} \
                    --start_id {start_point} \
                    --end_id {end_point} \
                    --output_root_path "/data/user_data/wentaos/Optima/inference_results/{dataset_type}_dpo_iteration_{i}_server_stepnum5/id" \
                    --vllm_env optima-vllm
                    """, 
                    shell=True)
                process.wait()
                time.sleep(10)
            except Exception as e:
                # 打印错误信息和堆栈跟踪
                print(f"发生错误: {e}")
                traceback.print_exc()

            for id in range(start_point, end_point):
                checkpoint_path = f"/data/group_data/cx_group/MCTS-agent/checkpoints/{dataset_type}_sft_dpo_DI_lora/dpo_iteration_{i}_id_{id}"
                if os.path.exists(checkpoint_path):
                    process = subprocess.Popen(f"""rm -r {checkpoint_path}""", shell=True)
                    process.wait()
            

            
            print("Collect Data Influence completed.")
            # train_dataset = load_from_disk(os.path.join(dataset_path, "train"), keep_in_memory=True)
            # step = len(train_dataset) // num_device

            # for t in range(0, step, 5):
                
            #     for j in range(num_device + 1):
            #         basenum = j * step
            #         start_point=t + basenum
            #         end_point=t + 5 + basenum

            #         num_port = 12000 + basenum
            #         process = subprocess.Popen(
            #             f"""source ~/.bashrc &&\
            #             conda activate {alignment_env} && \
            #             CUDA_VISIBLE_DEVICES={j} ACCELERATE_LOG_LEVEL=info NCCL_P2P_DISABLE=1 accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port {num_port} scripts/run_dpo_collect_noserver.py /home/wentaos/Optima/alignment-handbook/recipes/Llama3-8b/{dataset_type}_dpo/dpo_iteration_{i}_lora_collect_warmup.yaml --load_in_4bit=false --start_point=${start_point} --end_point=${end_point} --step_num=5
            #             """,
            #             shell=True,
            #             cwd="./alignment-handbook",
            #         )
            #         process.wait()

            #         process = subprocess.Popen(
            #             f"""
            #             source ~/.bashrc &&\
            #             conda activate {vllm_env} && \
            #             export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True &&\
            #             CUDA_VISIBLE_DEVICES={j} OUTLINES_CACHE_DIR=./outlines/${dataset_type}/${j} python inference_script_server_train_dynamic.py \
            #                 --model_root_path /data/user_data/wentaos/optima-checkpoints/${dataset_type}_sft/ \
            #                 --model_lora_path /data/user_data/wentaos/optima-checkpoints/${dataset_type}_dpo_lora/dpo_iteration_0 \
            #                 --tokenizer_path /home/wentaos/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a \
            #                 --device 0 \
            #                 --port {num_port} \
            #                 --dataset_type ${dataset_type} \
            #                 --num_thread 16 \
            #                 --num_gpu 1 \
            #                 --start_id ${start_point} \
            #                 --end_id ${end_point} \
            #                 --output_root_path "/data/user_data/wentaos/Optima/inference_results/${dataset_type}_dpo_iteration_{i}_server_stepnum5/id" \
            #                 --vllm_env optima-vllm &&\
            #             for id in $(seq $start_point $((end_point-1))) \
            #             do \
            #                 rm -r /data/user_data/wentaos/optima-checkpoints/${dataset_type}_dpo_lora/dpo_iteration_0_id_${id}\
            #             done\
            #             """,
            #             shell=True,
            #         )
            #         process.wait()

        data_influence_path = f"/data/user_data/wentaos/Optima/inference_results/${dataset_type}_dpo_iteration_{i}_server_stepnum5/"
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
                os.path.join(mid_yaml_root_path, f"dpo_iteration_{i}.yaml" + f"_mix_warmup_{len(train_dataset)//2}"), "r"
            ) as f:
                pass
        except:
            with open(
                os.path.join(mid_yaml_root_path, f"dpo_iteration_{i}.yaml" + f"_mix_warmup_{len(train_dataset)//2}"), "w"
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
        model_path = os.path.join(check_point_root_path, f"dpo_iteration_{i}")


