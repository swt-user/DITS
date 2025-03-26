from reward.reward import result_stats
from argparse import ArgumentParser
from utils.config import llama3_path_a100, llama3_path_a800
from transformers import AutoTokenizer
from train.inference_noserver import inference_noserver
from transformers import AutoModelForCausalLM, set_seed
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
argumentParser.add_argument("--model_lora_path", type=str, required=True)
argumentParser.add_argument("--tokenizer_path", type=str, default=llama3_path_a800)
argumentParser.add_argument("--device", type=int, default=7)
argumentParser.add_argument("--port", type=int, default=8000)
argumentParser.add_argument("--dataset_type", type=str, default="hotpot_qa")
argumentParser.add_argument("--num_thread", type=int, default=128)
argumentParser.add_argument("--num_gpu", type=int, default=1)
argumentParser.add_argument("--num_test", type=int, default=3)
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

if __name__ == "__main__":
    models = os.listdir(args.model_root_path)
    models = [(model, os.path.join(args.model_root_path, model)) for model in models]
    # model_paths = [("base",llama3_path_a800)]
    model_paths = []
    model_paths.extend(models)
    loader = None

    set_seed_everything()

    if args.dataset_type == "hotpot_qa":
        print("hotpot")
        loader = DataloaderForHotpotQA(split="validation")
        print("load dataset success")
    elif args.dataset_type == "mwh_qa":
        print("mwh")
        loader = DataloaderForMWHQA(split="dev")
    elif args.dataset_type == "cbt":
        print("cbt")
        loader = DataloaderForCBT(split="test")
    elif args.dataset_type == "gsm8k":
        print("gsm8k")
        loader = DataloaderForGSM8K(split="test")
    elif args.dataset_type == "math":
        print("math")
        loader = DataloaderForMATH(split="test")
    elif args.dataset_type == "trival_qa":
        print("trival_qa")
        loader = DataloaderForTrivalQA(split="validation")
    elif args.dataset_type == "arc":
        print("arc")
        loader = DataloaderForARC(split="test")
    elif args.dataset_type == "mmlu":
        print("mmlu")
        loader = DataloaderForMMLU(split="test")
    for model, model_path in model_paths:
        
        if not os.path.exists(args.output_root_path):
            os.makedirs(args.output_root_path)
        
        inference_noserver(
            model_path,
            model_path,
            f"cuda:{args.device}",
            f"cuda:{args.device}",
            args.tokenizer_path,
            args.tokenizer_path,
            sample_count=100,
            explore_count=1,
            output_path=os.path.join(args.output_root_path, f"{model}.jsonl"),
            thread_count=args.num_thread,
            prompt_pool_path="",
            train_data_path="",
            dataloader=loader,
            temperature=0,
            no_use_prompt_pool=True,
            lora_first_path=args.model_lora_path,
            lora_second_path=args.model_lora_path,
        )
        
        time.sleep(5)
