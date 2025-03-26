from reward.reward import result_stats
from argparse import ArgumentParser
from utils.config import llama3_path_a100, llama3_path_a800
from transformers import AutoTokenizer
from train.inference import inference
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
import time
import os
import subprocess
import requests
import random
import numpy as np 
import torch

argumentParser = ArgumentParser()
argumentParser.add_argument("--model_root_path", type=str, required=True)
argumentParser.add_argument("--model_name", type=str, required=True)
argumentParser.add_argument("--model_lora_path", type=str, required=True)
argumentParser.add_argument("--tokenizer_path", type=str, default=llama3_path_a800)
argumentParser.add_argument("--device", type=int, default=7)
argumentParser.add_argument("--port", type=int, default=8000)
argumentParser.add_argument("--dataset_type", type=str, default="hotpot_qa")
argumentParser.add_argument("--num_thread", type=int, default=128)
argumentParser.add_argument("--num_gpu", type=int, default=1)
argumentParser.add_argument("--start_id", type=int, default=1)
argumentParser.add_argument("--end_id", type=int, default=10)
argumentParser.add_argument("--output_root_path", type=str, required=True)
argumentParser.add_argument(
    "--vllm_env",
    type=str,
    required=True
)
args = argumentParser.parse_args()

def set_seed_everything(seed=42):
    random.seed(seed)                    # 设置Python标准库random的种子
    np.random.seed(seed)                 # 设置NumPy的随机种子
    torch.manual_seed(seed)              # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)         # 设置PyTorch的单个GPU随机种子
    torch.cuda.manual_seed_all(seed)     # 设置PyTorch的所有GPU随机种子
    torch.backends.cudnn.deterministic = True   # 使CUDNN后端行为确定性
    torch.backends.cudnn.benchmark = False

def get_dataloader(dataset_type):
    if args.dataset_type == "hotpot_qa":
        loader = DataloaderForHotpotQA(split="train", current_id=15000)
    elif args.dataset_type == "mwh_qa":
        loader = DataloaderForMWHQA(split="train", current_id=20000)
    elif args.dataset_type == "cbt":
        print("cbt")
        loader = DataloaderForCBT(split="validation", current_id=0)
    elif args.dataset_type == "gsm8k":
        print("gsm8k")
        is_math = True
        loader = DataloaderForGSM8K(split="train", current_id=7000)
        score_type = "exact-match"
        # no_use_prompt_pool =True
    elif args.dataset_type == "math":
        score_type = "math"
        is_math = True
        # no_use_prompt_pool = True
        print("math")
        loader = DataloaderForMATH(split="train", current_id=7000)
    elif args.dataset_type == "trival_qa":
        print("trival_qa")
        loader = DataloaderForTrivalQA(split="train", current_id=15000)
    elif args.dataset_type == "arc":
        print("arc")
        score_type = "exact-match"
        loader = DataloaderForARC(split="validation", current_id=0)
    elif args.dataset_type == "mmlu":
        print("mmlu")
        score_type = "exact-match"
        loader = DataloaderForMMLU(split="auxiliary_train", current_id=15000)
    
    return loader
    



if __name__ == "__main__":
    # models = os.listdir(args.model_root_path)
    # models = [(model, os.path.join(args.model_root_path, model)) for model in models]
    # model_paths = [("base",llama3_path_a800)]
    model_paths = [(args.model_name, args.model_root_path)]
    # model_paths.extend(models)
    loader = None

    set_seed_everything()

    loader = get_dataloader(args.dataset_type)


    for model, model_path in model_paths:
        ports = []
        for i in range(args.num_gpu):
            print(model_path)
            process = subprocess.Popen(
                f"""
            source ~/.bashrc && \
            conda activate {args.vllm_env} && \
            CUDA_VISIBLE_DEVICES={args.device+i} vllm serve {model_path} --enable-lora --host 0.0.0.0 --port {args.port+i} --served-model-name "Llama-3" --enable-prefix-caching --max-lora-rank 64
            """,
                shell=True,
            )
            ports.append(args.port + i)
        while True:
            try:
                message_input = [{"role": "assistant", "content": "hello!"}]
                headers = {"Content-Type": "application/json"}
                data_json = {
                    "model": "Llama-3",
                    "messages": message_input,
                }
                response = requests.post(
                    f"http://0.0.0.0:{args.port}/v1/chat/completions",
                    headers=headers,
                    json=data_json,
                )
                content = (response.json()["choices"][0]["message"]["content"],)
                print(f"ready to generate data: {content}")
                break
            except:
                continue
        time.sleep(15)
        loader.current_task_id = 0

        

        for id in range(args.start_id, args.end_id):

            if not os.path.exists(args.output_root_path + f"_{id}"):
                os.makedirs(args.output_root_path + f"_{id}")

            # dynamic serve lora adapter
            for i in range(args.num_gpu):
                load_url = f"http://0.0.0.0:{args.port+i}/v1/load_lora_adapter"
                headers = {
                    "Content-Type": "application/json"
                }
                data = {
                    "lora_name": "sql_lora",
                    "lora_path": args.model_lora_path + f"_id_{id}"
                }
                response = requests.post(load_url, headers=headers, json=data)

            for step in range(1):
                set_seed_everything()
                loader = get_dataloader(args.dataset_type)
                
                inference(
                    "sql_lora",
                    "sql_lora",
                    f"http://0.0.0.0:{args.port}/v1/chat/completions",
                    f"http://0.0.0.0:{args.port}/v1/chat/completions",
                    args.tokenizer_path,
                    args.tokenizer_path,
                    sample_count=20,
                    explore_count=1,
                    output_path=os.path.join(args.output_root_path+f"_{id}", f"{model}_{step}.jsonl"),
                    thread_count=args.num_thread,
                    prompt_pool_path="",
                    train_data_path="",
                    dataloader=loader,
                    temperature=0.0,
                    no_use_prompt_pool=True,
                    ports=ports,
                )

            # dynamic unload lora
            for i in range(args.num_gpu):
                unload_url = f"http://0.0.0.0:{args.port+i}/v1/unload_lora_adapter"
                data = {
                    "lora_name": "sql_lora",
                }
                response = requests.post(unload_url, headers=headers, json=data)

        for i in range(args.num_gpu):
            os.system(
                f"""pkill -f "vllm serve {model_path} --enable-lora --host 0.0.0.0 --port {args.port+i} --served-model-name Llama-3 --enable-prefix-caching --max-lora-rank 64" """
            )
        time.sleep(5)
