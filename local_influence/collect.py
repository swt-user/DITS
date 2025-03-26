from pathlib import Path
import argparse
import random
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from datasets import Dataset, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import transformers
# from utils import PartialMaskTokenizer, reformat_to_chat

from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, PeftConfig
from my_utils import (get_tokenizer, process_dpo_format_to_dataset, 
        apply_chat_template, decontaminate_humaneval, dpo_generate_and_tokenize_prompt, 
        SFT_generate_and_tokenize_prompt, PreferenceCollator, DataArguments, get_datasets, compute_dpo_loss)
from functools import partial

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='0'

os.environ["HF_HOME"]="~/.cache/huggingface"


def get_dpo_dataloader(
    accelerator: Accelerator,
    args: Dict[str, any],
    split: str,
    tokenizer,
    batch_size=8, 
) -> DataLoader:
    

    # partialMaskTokenizer = PartialMaskTokenizer(tokenizer, max_length=1024)
    # partialMaskTokenizer.format_tokenizer()
    # tokenizer = partialMaskTokenizer.tokenizer

    # get train_dataloader
    raw_datasets = process_dpo_format_to_dataset(args["data_path"])
    # train_dataset = Dataset.from_dict(
    #     {
    #         "instruction": [
    #             ins
    #             for ins in raw_train_dataset["instruction"]
    #             for _ in range(train_args["num_gpus"])
    #         ],
    #         "response": [
    #             res
    #             for res in raw_train_dataset["response"]
    #             for _ in range(train_args["num_gpus"])
    #         ],
    #     }
    # )

    # tokenizer = get_tokenizer(model_args, data_args)
    column_names = list(raw_datasets["train"].features)
    #####################
    # Apply chat template
    #####################
    with accelerator.main_process_first():
        raw_datasets = raw_datasets.map(
            apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "task": "dpo",
                "auto_insert_empty_system_msg": True,
            },
            num_proc=1,
            # num_proc=1,
            remove_columns=column_names,
            desc="Formatting comparisons with prompt template",
        )
        ##########################
        # Decontaminate benchmarks
        ##########################
        num_raw_train_samples = len(raw_datasets["train"])
        raw_datasets = raw_datasets.filter(
            decontaminate_humaneval,
            fn_kwargs={"text_column": "text_chosen"},
            batched=True,
            batch_size=10_000,
            num_proc=1,
            desc="Decontaminating HumanEval samples",
        )
        num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
        

        # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
        for split in ["train", "test"]:
            raw_datasets[split] = raw_datasets[split].rename_columns(
                {
                    "text_prompt": "prompt",
                    "text_chosen": "chosen",
                    "text_rejected": "rejected",
                }
            )

        processed_datasets = raw_datasets.map(
            partial(dpo_generate_and_tokenize_prompt, tokenizer=tokenizer),
            remove_columns=raw_datasets["train"].column_names,
        )

    print(f"original train dataset:{raw_datasets}\n")

    data_collator= PreferenceCollator()
  
    train_dataloader = DataLoader(
        processed_datasets[split], shuffle=False, collate_fn=data_collator, batch_size=batch_size, pin_memory=True
    )
    # for i in range(len(train_dataset)):
    #     accelerator.print(train_dataset[i])
    # train_dataset = train_dataset.shuffle(seed=42)
    # accelerator.print("train dataset after shuffle:\n")
    # for i in range(len(train_dataset)):
    #     accelerator.print(train_dataset[i])
    # train_dataset = Dataset.from_dict(
    #     {
    #         "messages": reformat_to_chat(
    #             input=train_dataset["instruction"], output=train_dataset["response"]
    #         )
    #     }
    # )
    # with accelerator.main_process_first():
    #     tokenized_train_dataset = train_dataset.map(
    #         partialMaskTokenizer.preprocess,
    #         fn_kwargs={"mask_inputs": True, "add_generation_prompt": False},
    #         batched=True,
    #     )

    # tokenized_train_dataset.set_format("torch")

    # train_dataloader = DataLoader(tokenized_train_dataset, batch_size=1, shuffle=False)

    # # get eval dataloader
    # eval_dataset = load_from_disk(eval_args["eval_dataset_path"])
    # eval_dataset = eval_dataset.shuffle(seed=42).select(range(eval_args["eval_nums"]))
    # # accelerator.print("size of eval dataset: ", len(eval_dataset))
    # # accelerator.print("batch size of eval: ", eval_args["batch_size"])
    # eval_dataset = Dataset.from_dict(
    #     {
    #         "messages": reformat_to_chat(
    #             input=eval_dataset["instruction"], output=eval_dataset["response"]
    #         )
    #     }
    # )
    # with accelerator.main_process_first():
    #     tokenized_eval_dataset = eval_dataset.map(
    #         partialMaskTokenizer.preprocess,
    #         fn_kwargs={"mask_inputs": True, "add_generation_prompt": False},
    #         batched=True,
    #     )

    # tokenized_eval_dataset.set_format("torch")

    # eval_dataloader = DataLoader(
    #     tokenized_eval_dataset, batch_size=eval_args["batch_size"], shuffle=False
    # )

    return train_dataloader

def get_sft_dataloader(accelerator: Accelerator,
    args: Dict[str, any],
    split: str,
    tokenizer,
    batch_size=8,
) -> DataLoader:
    
    data_args = DataArguments(chat_template=None, dataset_mixer={args["eval_dataset_path"]: 1.0}, text_column='text', dataset_splits=['train', 'test'], dataset_configs=None, preprocessing_num_workers=8, truncation_side=None, auto_insert_empty_system_msg=True)

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
        shuffle=False,
    )

    column_names = list(raw_datasets["train"].features)
    with accelerator.main_process_first():
        raw_datasets = raw_datasets.map(
                apply_chat_template,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "task": "sft",
                    "auto_insert_empty_system_msg": True,
                },
                num_proc=8,
                remove_columns=column_names,
                desc="Applying chat template",
            )
        
        processed_datasets = raw_datasets.map(
            partial(SFT_generate_and_tokenize_prompt, tokenizer=tokenizer),
            remove_columns=raw_datasets["train"].column_names,
        )
    print(f"original train dataset:{raw_datasets}\n")

    test_dataset = processed_datasets[split]
    data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    
    
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, collate_fn=data_collator, batch_size=batch_size, pin_memory=True
    )

    return test_dataloader

def one_step_train(
    model: AutoModelForCausalLM,
    optimizer: AdamW,
    accelerator: Accelerator,
    data_batch: Dict[str, torch.Tensor],
    eval_dataloader: DataLoader,
    step: int,
    avg_loss_before: float,
) -> float:
    device = "cuda:0"
    model.train()
    optimizer.zero_grad()
    data_batch = {k: v.to(device) for k, v in batch.items()}
    loss = compute_dpo_loss(model, data_batch['prompt_input_ids'], data_batch['prompt_attention_mask'], 
                                data_batch['chosen_input_ids'], data_batch['chosen_attention_mask'],
                                data_batch['rejected_input_ids'], data_batch['rejected_attention_mask'])
    
    loss = loss / accelerator.num_processes
    # if accelerator.is_local_main_process:
    #     print(f"{accelerator.device} loss: {loss:.10f}")
    # accelerator.print("train loss: ", loss.item())
    # print(f"train loss {loss.item()} on device {accelerator.device}")
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

    # eval
    model.eval()
    total_loss = 0
    for index, batch in tqdm(
        enumerate(eval_dataloader),
        total=len(eval_dataloader),
        desc="Evaluating",
        disable=not accelerator.is_local_main_process,
    ):
        batch = {k: v.to(device) for k, v in batch.items()}
    
        with torch.no_grad():
            loss = compute_dpo_loss(model, batch['prompt_input_ids'], batch['prompt_attention_mask'], 
                                                        batch['chosen_input_ids'], batch['chosen_attention_mask'],
                                                        batch['rejected_input_ids'], batch['rejected_attention_mask'])
            
        loss_withoutnan = torch.nanmean(accelerator.gather(loss))
        total_loss += loss_withoutnan.item()
    avg_loss_after = total_loss / len(eval_dataloader)
    accelerator.print(f"eval loss after one step train: {avg_loss_after}")
    accelerator.print(f"eval loss diff: {avg_loss_after - avg_loss_before}")
    return avg_loss_after


def collect_local_data_influence(config: Dict[str, any]) -> None:
    set_seed(42)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True

    accelerator = Accelerator(mixed_precision="bf16")
    accelerator.print(
        f"datetime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True
    )


    

    peft_config = PeftConfig.from_pretrained(config["lora_path"])
    # peft_config.base_model_name_or_path = config["model_name_or_path"]


    
    raw_model = LlamaForCausalLM.from_pretrained(config["model_name_or_path"], 
                                                torch_dtype=torch.bfloat16,
                                                device_map="cuda",
                                                cache_dir=os.environ["HF_HOME"],
                                                )
    raw_model = PeftModel.from_pretrained(raw_model, config["lora_path"], torch_dtype=torch.bfloat16, adapter_name="first adapter")
    for n, p in raw_model.named_parameters():
        if 'lora' in n:
            print(n)
            p.requires_grad = True
    # raw_model = AutoModelForCausalLM.from_pretrained(
    #     config["model_name_or_path"],
    #     torch_dtype=torch.bfloat16,
    #     device_map="cpu",
    #     cache_dir=os.environ["HF_HOME"],
    # )
    model = accelerator.prepare(raw_model)

    tokenizer = get_tokenizer(config["tokenizer_path"], truncation_side="left") 

    train_args, eval_args, tokenizer_path = (
        config["train_args"],
        config["eval_args"],
        config["tokenizer_path"],
    )

    
    # train_dataloader = get_dpo_dataloader(accelerator, train_args, "train", tokenizer)
    eval_dataloader = get_sft_dataloader(accelerator, eval_args, "test", tokenizer)

    optimizer = AdamW(params=model.parameters())

    optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        optimizer, train_dataloader, eval_dataloader
    )

    # accelerator.load_state(config["warm_up_path"])

    # obtain baseline reference loss
    model.eval()
    total_loss = 0
    for index, batch in tqdm(
        enumerate(eval_dataloader),
        total=len(eval_dataloader),
        desc="Evaluating",
        disable=not accelerator.is_local_main_process,
    ):
        batch = {
            k: v.unsqueeze(0) if v.ndim == 1 else v
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask", "labels"]
        }
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        loss = accelerator.gather(loss).mean()
        total_loss += loss.item()
    avg_loss_before = total_loss / len(eval_dataloader)
    accelerator.print(f"baseline reference loss: {avg_loss_before}")
    # Write the baseline reference loss to a file
    baseline_ref_loss_path = os.path.join(
        Path(config["train_args"]["data_path"]).parent, "baseline_ref_loss.txt"
    )
    print(f"baseline reference loss path: {baseline_ref_loss_path}")
    with open(baseline_ref_loss_path, "w") as f:
        f.write(f"{avg_loss_before}")

    score_list = []
    for step, batch in tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc="Training",
        disable=not accelerator.is_local_main_process,
    ):
        accelerator.print(f"..........train batch: {step}...........")
        t1 = datetime.now()
        loss = one_step_train(
            model, optimizer, accelerator, batch, eval_dataloader, step, avg_loss_before
        )
        # accelerator.print(
        #     f"Epoch {epoch}, Step {step}, Loss on {accelerator.device}: {loss}"
        # )
        score_list.append(loss)
        accelerator.load_state(config["warm_up_path"])
        t2 = datetime.now()
        accelerator.print(f"time taken: {(t2 - t1).total_seconds()}")
    # print(score_list)
    raw_train_dataset = raw_train_dataset.add_column("scores", score_list)
    raw_train_dataset.save_to_disk(config["scores_dataset_path"])
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()
    with open(args.config_file, "r") as config_f:
        config = yaml.safe_load(config_f)
        collect_local_data_influence(config)