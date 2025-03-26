from typing import List, Dict, Union, Any
import torch
from pydantic import BaseModel
from abc import abstractmethod
from message.message import llmMessage
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
# from vllm import LLM, SamplingParams
# from vllm.lora.request import LoRARequest
import threading
import time
import queue

class BaseLLM(BaseModel):
    device: str = "cuda:1"
    model: Any = None
    tokenizer: Any = None
    max_new_tokens: int = 1000
    do_sample: bool = True
    temperature: float = 0

    @abstractmethod
    def generate_response(self, messages: List[Dict]) -> dict:
        pass

    @abstractmethod
    def generate_vllm_response(self, messages: List[Dict],  name: str = "", lora_request=None) -> dict:
        pass


class Llama3(BaseLLM):
    def generate_response(self, messages: List[Dict], name: str = "") -> dict:
        messages_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        #self.tokenizer.eos_token_id = [128001, 128009]
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.temperature=None
        self.model.generation_config.top_p=None

        if name != "":
            messages_input += name + ":"
        messages_input = self.tokenizer(messages_input, return_tensors="pt")[
            "input_ids"
        ].to(self.device)
        
        
        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        with torch.inference_mode():
            response = self.model.generate(
                messages_input,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                eos_token_id=terminators,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
        response = self.tokenizer.batch_decode(response)
        response = (
            response[0].split("<|end_header_id|>")[-1].strip().strip("<|eot_id|>")
        )
        return llmMessage(
            role="assistant",
            content=response,
        )
    
    def generate_vllm_response(self, messages: List[Dict], name: str = "", lora_request=None) -> dict:
        # print("DDDDDDDDDDDD")
        messages_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # print(messages_input)
        # if name != "":
        #     messages_input += name + ":"
        # messages_input = self.tokenizer(messages_input, return_tensors="pt")[
        #     "input_ids"
        # ].to(self.device)

        terminators = [
            "<|end_header_id|>",
            "<|eot_id|>",
        ]

        sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                stop=terminators
            )
        
        with torch.inference_mode():
            # response = self.model.generate(
            #     messages_input,
            #     max_new_tokens=self.max_new_tokens,
            #     do_sample=self.do_sample,
            #     temperature=self.temperature,
            #     eos_token_id=terminators,
            # )
            # print(messages_input)
            response = self.model.generate(
                messages_input,
                sampling_params,
                lora_request=lora_request, 
            )
            # print(response)
            
        # response = self.tokenizer.batch_decode(response)
        # response = (
        #     response[0].split("<|end_header_id|>")[-1].strip().strip("<|eot_id|>")
        # )
        response = response[0].outputs[0].text
        print(f"------------------{response}-------------------")
        return llmMessage(
            role="assistant",
            content=response,
        )

    def generate_vllm_response_multiprocess(self, messages: List[Dict], name: str = "", lora_request=None, lock=None, result_queue=[], final_results=[], thread_count=[]) -> dict:
        # print("DDDDDDDDDDDD")
        messages_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # print(messages_input)
        # if name != "":
        #     messages_input += name + ":"
        # messages_input = self.tokenizer(messages_input, return_tensors="pt")[
        #     "input_ids"
        # ].to(self.device)

        terminators = [
            "<|end_header_id|>",
            "<|eot_id|>",
        ]

        sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                stop=terminators
            )
        
        with torch.inference_mode():
            # response = self.model.generate(
            #     messages_input,
            #     max_new_tokens=self.max_new_tokens,
            #     do_sample=self.do_sample,
            #     temperature=self.temperature,
            #     eos_token_id=terminators,
            # )
            # print(messages_input)
            # response = self.model.generate(
            #     messages_input,
            #     sampling_params,
            #     lora_request=lora_request, 
            # )
            result_queue.put((int(threading.current_thread().name), messages_input))  # 假设中间结果是数据的两倍

            # 增加线程计数
            with lock:
                thread_count[0] += 1

            # 等待主线程提交结果
            while True:
                with lock:
                    if int(threading.current_thread().name) in final_results:
                        # 获取分配的结果
                        response = final_results[int(threading.current_thread().name)]
                        final_results.pop(int(threading.current_thread().name))
                        break
                time.sleep(0.1) 
            # print(response)
            
        # response = self.tokenizer.batch_decode(response)
        # response = (
        #     response[0].split("<|end_header_id|>")[-1].strip().strip("<|eot_id|>")
        # )
        # print(f"------------------{response}-------------------")
        response = response.outputs[0].text
        # print(f"------------------{response}-------------------")
        return llmMessage(
            role="assistant",
            content=response,
        )