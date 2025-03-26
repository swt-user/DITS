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
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
import copy

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


logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()

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
        shuffle=False,    # no  shuffle 
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

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None
    
    # training_args.rpo_alpha=1
    # breakpoint()
    # print(f"sdasdsasdsadsa:{training_args}")
    # training_args.saodsajdksdj=121
    training_args.rpo_alpha =1.0
    # print(f"llklkklklklkl:{training_args}")
    #########################
    # Instantiate DPO trainer
    #########################
    trainer = DPOTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        # peft_config=get_peft_config(model_args),
        loss_type=training_args.loss_type,
    )

    initial_state = copy.deepcopy(trainer.model.state_dict())

    train_dataloader = trainer.get_train_dataloader()
    eval_dataloader = trainer.get_eval_dataloader()
    # 评估当前模型
    # trainer.model.eval()
    # with torch.no_grad():
    #     eval_loss = 0
    #     for eval_batch in eval_dataloader:
    #         loss, metrics = trainer.compute_loss(trainer.model, eval_batch, return_outputs=True)
    #         eval_loss += metrics["rewards/margins"].item()
    #     eval_loss /= len(eval_dataloader)
    #     # eval_losses.append(eval_loss)
    
    # before_train_margins = eval_loss
    # print(f"Batch {batch_idx + 1}/{len(train_dataloader)}:")
    # print(f"  Training loss: {train_losses[-1]:.4f}")
    # print(f"  Initial Evaluation reward_margins: {before_train_margins:.4f}")
    ###############
    # Training loop
    ###############
    # 记录训练和验证loss
    train_losses = []
    eval_losses = []
    
    # 对每个训练样本进行一步梯度下降
    train_dataloader = trainer.get_train_dataloader()
    eval_dataloader = trainer.get_eval_dataloader()
    print(trainer.model.print_trainable_parameters())
    for n, p in model.named_parameters():
        if 'lora' in n:
            # print(n)
            print(p.grad)
            break
    # ("Set up Lora Finished")
    # print(trainer.model.print_trainable_parameters())
            
    for batch_idx, batch in enumerate(train_dataloader):
        # 重置模型到初始状态
        trainer.model.load_state_dict(initial_state)
        trainer.model.train()
        
        # trainer.create_optimizer_and_scheduler(num_training_steps=1)
        # # 单步训练
        # print(trainer.optimizer)
        # loss = trainer.compute_loss(trainer.model, batch)
        # loss.backward()
        # train_losses.append(loss.detach().item())
        # trainer.optimizer.step()
        trainer.train(max_steps=1)
        for n, p in model.named_parameters():
            if 'lora' in n:
                # print(n)
                print(p.grad)
                break

        trainer.optimizer.zero_grad()
        
        torch.cuda.empty_cache()

        if batch_idx == 0:
            print(batch)
        # 评估当前模型
        trainer.model.eval()
        with torch.no_grad():
            eval_loss = 0
            for eval_batch in eval_dataloader:
                loss, metrics = trainer.compute_loss(trainer.model, eval_batch, return_outputs=True)
                # print(metrics["rewards/margins"])
                eval_loss += metrics["rewards/margins"].item()
            eval_loss /= len(eval_dataloader)
            eval_losses.append(eval_loss)
        
        print(f"Batch {batch_idx + 1}/{len(train_dataloader)}:")
        print(f"  Training loss: {train_losses[-1]:.4f}")
        print(f"  Evaluation reward_margins: {eval_losses[-1]:.4f} and reward margins difference: {(eval_losses[-1] - before_train_margins):.4f}")
        with open("/home/wentaos/Optima/local_influence/case0.txt", 'a', encoding='utf-8') as file:
            # 确保内容以换行符结束
            file.write(f"{batch_idx + 1}:{eval_losses[-1]:.4f}\n")
    logger.info("*** Training complete! ***")

    return {
        'train_losses': train_losses,
        'eval_losses': eval_losses,
    }

    

    


if __name__ == "__main__":
    main()
