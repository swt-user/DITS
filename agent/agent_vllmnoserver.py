from typing import List, Dict
from pydantic import BaseModel
from message.message import llmMessage
from model.llm import BaseLLM
from string import Template
import requests
from vllm.lora.request import LoRARequest
import os

class BaseAgent:

    def no_memory_step(self):
        pass

    def step(self):
        pass

    def init_system_prompt(self, template: str, args: dict):
        pass

    def add_memory(self):
        pass

    def reset(self):
        pass


class Agent(BaseAgent):
    llm: BaseLLM = None
    prompt_template: str = ""
    system_prompt: llmMessage = llmMessage(role="system", content="")
    memory: List[llmMessage] = []
    name: str = ""

    def __init__(self, 
                llm: BaseLLM = None,
                prompt_template: str = "",
                system_prompt: llmMessage = llmMessage(role="system", content=""),
                memory: List[llmMessage] = [],
                name: str = "",
                temperature: float = 0.0,
                lora_name: str = None,
                lora_path: str = None):
        self.llm = llm
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.memory = memory
        self.name = name 

        self.llm.do_sample = False

    def step(self) -> llmMessage:
        message_input = [
            {"role": message.role, "content": message.content}
            for message in self.memory
        ]
        response = self.llm.generate_response(message_input, self.name)

        self.add_memory(response)

        return response

    def init_system_prompt(self, template: str, args: dict):
        self.system_prompt.content = Template(template).safe_substitute(args)
        self.memory.append(self.system_prompt)

    def add_memory(self, new_memory: llmMessage):
        self.memory.append(new_memory)

    def reset(self):
        self.memory = []
        self.system_prompt = llmMessage(role="system", content=self.prompt_template)


class VLLMAgentNoServer(BaseAgent):
    
    
    def __init__(self, 
                llm: BaseLLM = None,
                prompt_template: str = "",
                system_prompt: llmMessage = llmMessage(role="system", content=""),
                memory: List[llmMessage] = [],
                name: str = "",
                temperature: int = 0,
                lora_name: str = None,
                lora_path: str = None):
        self.llm = llm
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.memory = memory
        self.my_model_name = name
        self.name = name
        self.temperature = temperature
        
        if lora_path == None:
            self.lora_request = None 
        else:
            self.lora_request = LoRARequest(lora_name, 1, lora_path)
            self.lora_name = lora_name
            self.lora_path = lora_path

    def step(self) -> llmMessage:
        # print("BBBBBBB")
        message_input = [
            {"role": message.role, "content": message.content}
            for message in self.memory
        ]
        # print(message_input)
        # print(self.llm)
        response = self.llm.generate_vllm_response(message_input, self.name, self.lora_request)
        # print(f"ccccc {response}")
        self.add_memory(response)

        return response

    def init_system_prompt(self, template: str, args: dict):
        self.system_prompt.content = Template(template).safe_substitute(args)
        self.memory.append(self.system_prompt)

    def add_memory(self, new_memory: llmMessage):
        self.memory.append(new_memory)

    def reset(self):
        self.memory = []
        self.system_prompt = llmMessage(role="system", content=self.prompt_template)


class VLLMAgentNoServerMultiProcess(BaseAgent):
    
    
    def __init__(self, 
                llm: BaseLLM = None,
                prompt_template: str = "",
                system_prompt: llmMessage = llmMessage(role="system", content=""),
                memory: List[llmMessage] = [],
                name: str = "",
                temperature: int = 0,
                lora_name: str = None,
                lora_path: str = None):
        self.llm = llm
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.memory = memory
        self.my_model_name = name
        self.name = name
        self.temperature = temperature
        
        if lora_path == None:
            self.lora_request = None 
        else:
            self.lora_request = LoRARequest(lora_name, 1, lora_path)
            self.lora_name = lora_name
            self.lora_path = lora_path

    def step(self, lock, result_queue, final_results, thread_count) -> llmMessage:
        # print("BBBBBBB")
        message_input = [
            {"role": message.role, "content": message.content}
            for message in self.memory
        ]
        # print(message_input)
        # print(self.llm)
        response = self.llm.generate_vllm_response_multiprocess(message_input, self.name, self.lora_request,  lock, result_queue, final_results, thread_count)
        # print(f"ccccc {response}")
        self.add_memory(response)

        return response

    def init_system_prompt(self, template: str, args: dict):
        self.system_prompt.content = Template(template).safe_substitute(args)
        self.memory.append(self.system_prompt)

    def add_memory(self, new_memory: llmMessage):
        self.memory.append(new_memory)

    def reset(self):
        self.memory = []
        self.system_prompt = llmMessage(role="system", content=self.prompt_template)




class VllmAgent(BaseAgent):
    """
    The agent class is based on VLLM.
    It handles communication , manages the conversation context (memory),
    and formats the input/output in the required structure.
    """

    def __init__(self, url: str, my_model_name: str, name: str, temperature: float):
        self.url = url
        self.prompt_template = ""
        self.system_prompt: llmMessage = llmMessage(role="system", content="")
        self.memory: List[llmMessage] = []
        self.my_model_name = my_model_name
        self.name = name
        self.temperature = temperature

    def init_system_prompt(self, template: str, args: dict):
        self.system_prompt.content = Template(template).safe_substitute(args)
        self.memory.append(self.system_prompt)

    def add_memory(self, new_memory: llmMessage):
        self.memory.append(new_memory)

    def reset(self):
        self.memory = []
        self.system_prompt = llmMessage(role="system", content=self.prompt_template)

    # step and update memory
    def step(self):
        message_input = [
            {"role": message.role, "content": message.content}
            for message in self.memory
        ]

        headers = {"Content-Type": "application/json"}
        is_iteration_0 = False
        if (
            "You should start your utterance with" in self.system_prompt.content
            or "You must begin your response with" in self.system_prompt.content
        ):
            is_iteration_0 = True

        if "Llama" in os.environ["INITIAL_MODEL_PATH"]: 
            chat_template = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}{%- if is_alice %}\n    {{- \'Alice:\' }}\n{%- endif %}\n{%- if is_bob %}\n    {{- \'Bob:\' }}\n{%- endif %}"""
        elif "Qwen" in os.environ["INITIAL_MODEL_PATH"]: 
            chat_template = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|im_start|>' + message['role'] + '<|im_end|>\n\n'+ message['content'] | trim + '<|im_end|>' %}{% if loop.index0 == 0 %}{% set content = content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_end|>\n\n' }}{% endif %}{%- if is_alice %}\n    {{- 'Alice:' }}\n{%- endif %}\n{%- if is_bob %}\n    {{- 'Bob:' }}\n{%- endif %}"""
        
        data_json = {
            "model": self.my_model_name,
            "messages": message_input,
            "temperature": self.temperature,
            "chat_template": chat_template,
            "chat_template_kwargs": {
                "is_alice": (self.name == "Alice") and (not is_iteration_0),
                "is_bob": (self.name == "Bob") and (not is_iteration_0),
            },
            "max_tokens": 2000 if os.environ["DATASET"]!="math" else 5000,
        }
        response = requests.post(self.url, headers=headers, json=data_json)
        if response.status_code == 400:
            return llmMessage(role="assistant", content="error")
        content: str = response.json()["choices"][0]["message"]["content"]
        if not content.startswith(self.name) and not is_iteration_0:
            content = f"{self.name}:{content}"
        response = llmMessage(
            role="assistant",
            content=content,
        )
        self.add_memory(response)

        return response

    # step but don't update memory
    def no_memory_step(self):
        message_input = [
            {"role": message.role, "content": message.content}
            for message in self.memory
        ]

        headers = {"Content-Type": "application/json"}
        data_json = {
            "model": self.my_model_name,
            "messages": message_input,
            "temperature": self.temperature,
            "max_tokens": 2000 if os.environ["DATASET"]!="math" else 5000,
        }

        response = requests.post(self.url, headers=headers, json=data_json)
        if response.status_code == 400:
            return llmMessage(role="assistant", content="error")
        response = llmMessage(
            role="assistant",
            content=response.json()["choices"][0]["message"]["content"],
        )

        return response
