#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
import sys
import os
os.environ["WANDB_API_KEY"] = 'ee39227b6f2f9d6623026b6f9b5d5d293b949651'
os.environ["WANDB_DISABLED"] = 'true'
import torch
import transformers
import copy
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from peft import PeftConfig, PeftModel
from trl import DPOTrainer
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class NewArguments:
    start_point: int = field(default=0, metadata={"help": "Number of processes to use for distributed training"})
    eval_num: int = field(default=1559, metadata={"help": "Local rank for distributed training"})
    name_str: str = field(default='', metadata={"help": "Local rank for distributed training"})


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig, NewArguments))
    model_args, data_args, training_args, new_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

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
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = (
        "left"  # Truncate from left to ensure we don't lose labels in final turn
    )
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "dpo",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
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
    logger.info(
        f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples/num_raw_train_samples * 100:.2f}%) samples from the training set."
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {
                "text_prompt": "prompt",
                "text_chosen": "chosen",
                "text_rejected": "rejected",
            }
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(
            f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}"
        )
        logger.info(
            f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}"
        )
        logger.info(
            f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}"
        )

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        quantization_config=quantization_config.to_dict()

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision) is True:
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(
            model_args.model_name_or_path, revision=model_args.model_revision
        )
        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            use_flash_attention_2=model_args.use_flash_attention_2,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=(
                get_kbit_device_map() if quantization_config is not None else None
            ),
            quantization_config=quantization_config,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )
        model_kwargs = None
    ### Important !!!!
    ref_model = None
    ref_model_kwargs = None

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None
    
    print(model.print_trainable_parameters())
    for n, p in model.named_parameters():
        if 'lora' in n:
            # print(n)
            p.requires_grad = True
    print(model.print_trainable_parameters())

    for name, param in model.named_parameters():
        if 'lora' in name:
            print(param.grad)
            params_before = param.clone().detach()
            break
    
    initial_state = copy.deepcopy(model.state_dict())
    
    if not os.path.exists(f"/home/wentaos/Optima/local_influence/case0_1example_{new_args.name_str}_{new_args.eval_num}"):
        os.mkdir(f"/home/wentaos/Optima/local_influence/case0_1example_{new_args.name_str}_{new_args.eval_num}")

    try:
        with open(f"/home/wentaos/Optima/local_influence/case0_1example_{new_args.name_str}_{new_args.eval_num}/{new_args.start_point}.txt", "r") as file:
            seen_ids = {int(line.split(":", 1)[0]) for line in file if ":" in line}
    except:
        seen_ids = set()
        
    ####################
    ###Evaluate First###
    ####################
    trainer = DPOTrainer(
            model,
            ref_model,
            model_init_kwargs=model_kwargs,
            ref_model_init_kwargs=ref_model_kwargs,
            args=training_args,
            beta=training_args.beta,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets["test"].select(list(range(new_args.eval_num))),
            tokenizer=tokenizer,
            # max_length=training_args.max_length,
            # max_prompt_length=training_args.max_prompt_length,
            # peft_config=get_peft_config(model_args),
            # loss_type=training_args.loss_type,
    )
    if trainer.accelerator.is_main_process:
        # trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        # trainer.model.config.save_pretrained(training_args.output_dir)

        ##########
        # Evaluate
        ##########
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            metrics["eval_samples"] = len(raw_datasets["test"].select(list(range(new_args.eval_num))))
            trainer.log_metrics("eval", metrics)
            # trainer.save_metrics("eval", metrics)

        with open(f"/home/wentaos/Optima/local_influence/case0_1example_{new_args.name_str}_{new_args.eval_num}/{new_args.start_point}.txt", 'a', encoding='utf-8') as file:
            # 确保内容以换行符结束
            file.write(f"0: margins:{ metrics['eval_rewards/margins']:.4f}; accuracies: { metrics['eval_rewards/accuracies']:.4f}; loss: { metrics['eval_loss']:.4f} \n")
    
    # training_args.rpo_alpha=1
    # breakpoint()
    # print(f"sdasdsasdsadsa:{training_args}")
    # training_args.saodsajdksdj=121
    training_args.rpo_alpha =1.0
    # print(f"llklkklklklkl:{training_args}")

    random.seed(42) # 设置种子值为42
    numbers = random.sample(range(len(raw_datasets["train"])), len(raw_datasets["train"]))

    

    for i in range(new_args.start_point, len(raw_datasets["train"])):

        if i in seen_ids:
            continue
        temp = [k for k in range(i, (i+1))]
        temp_dataset = raw_datasets["train"].select(temp)
        #########################
        # Instantiate DPO trainer
        #########################
        model.load_state_dict(initial_state)

        trainer = DPOTrainer(
            model,
            ref_model,
            model_init_kwargs=model_kwargs,
            ref_model_init_kwargs=ref_model_kwargs,
            args=training_args,
            beta=training_args.beta,
            train_dataset=temp_dataset,
            eval_dataset=raw_datasets["test"].select(list(range(new_args.eval_num))),
            tokenizer=tokenizer,
            # max_length=training_args.max_length,
            # max_prompt_length=training_args.max_prompt_length,
            # peft_config=get_peft_config(model_args),
            # loss_type=training_args.loss_type,
        )

        
        ###############
        # Training loop
        ###############
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(raw_datasets["train"])
        trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)
        # trainer.save_state()

        logger.info("*** Training complete ***")
        for name, param in model.named_parameters():
            if 'lora' in name:
                print(param.grad)
                params_after = param.clone().detach().cpu()
                break
        print("------------------diff=------------------------------")
        print(torch.sum(torch.abs(params_after - params_before)).item())
    # ##################################
    # # Save model and create model card
    # ##################################
    # logger.info("*** Save model ***")
    # trainer.save_model(training_args.output_dir)
    # logger.info(f"Model saved to {training_args.output_dir}")

        # Save everything else on main process
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "dataset": list(data_args.dataset_mixer.keys()),
            "dataset_tags": list(data_args.dataset_mixer.keys()),
            "tags": ["alignment-handbook"],
        }
        if trainer.accelerator.is_main_process:
            trainer.create_model_card(**kwargs)
            # Restore k,v cache for fast inference
            trainer.model.config.use_cache = True
            # trainer.model.config.save_pretrained(training_args.output_dir)

            ##########
            # Evaluate
            ##########
            if training_args.do_eval:
                logger.info("*** Evaluate ***")
                metrics = trainer.evaluate()
                metrics["eval_samples"] = len(raw_datasets["test"].select(list(range(new_args.eval_num))))
                trainer.log_metrics("eval", metrics)
                # trainer.save_metrics("eval", metrics)

            with open(f"/home/wentaos/Optima/local_influence/case0_1example_{new_args.eval_num}_{new_args.start_point}.txt", 'a', encoding='utf-8') as file:
                # 确保内容以换行符结束
                file.write(f"{i}: margins:{ metrics['eval_rewards/margins']:.4f}; accuracies: { metrics['eval_rewards/accuracies']:.4f}; loss: { metrics['eval_loss']:.4f} \n")
                

            logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
