
import numpy as np
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, Dataset
import json
import sys
sys.path.append('/home/wentaos/Optima/alignment-handbook/src')
import os
import transformers
from transformers import default_data_collator
import torch.nn.functional as F
from torch.func import functional_call, vmap
from torch.func import grad
from transformers.data.data_collator import DataCollatorMixin
from tqdm import tqdm
from string import Template

DEFAULT_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set offset = 1 %}{% else %}{% set offset = 0 %}{% endif %}{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n\n' }}{% endif %}"


max_completion_length = 1000
if "llama-3":
    padding_value = 128002

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )
        
def dpo_tokenize(data_point, tokenizer, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    batch = {}
    for word in ["prompt", "chosen", "rejected"]:
        temp_batch = tokenizer(
            data_point[word],
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=False,
            return_tensors=None,
        )
        for name in ["input_ids", "attention_mask"]:
            batch[word+"_"+name] = temp_batch[name]
    
    
    # batch["input_ids"] = batch["prompt_input_ids"]
    
    return batch

def dpo_generate_and_tokenize_prompt(data_point, tokenizer=None):
    
    tokenized_full_prompt = dpo_tokenize(data_point, tokenizer)
    
    return tokenized_full_prompt

def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})


def is_openai_format(messages: Any) -> bool:
    """
    Check if the input messages are in OpenAI format.
    Args:
        messages (`Any`):
            Messages to check.
    Returns:
        `bool`: Whether the messages are in OpenAI format.
    """
    if isinstance(messages, list) and all(
        isinstance(message, dict) for message in messages
    ):
        return all("role" in message and "content" in message for message in messages)
    return False


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"],
    auto_insert_empty_system_msg: bool = True,
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(chosen_messages, tokenizer)
                maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False
            )
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False
            )
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task in ["dpo", "orpo"]:
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(
                example["rejected"]
            ):
                raise ValueError(
                    f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
                )

            # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                # Now we extract the final turn to define chosen/rejected responses
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)

            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False
            )
            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False
            )# .strip("<|begin_of_text|>")
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False
            )# .strip("<|begin_of_text|>")
            tokenized_chosen = tokenizer.tokenize(example["text_chosen"])
            tokenized_rejected = tokenizer.tokenize(example["text_rejected"])
            if tokenized_chosen[0] == tokenizer.bos_token:
                example["text_chosen"] = example["text_chosen"][len(tokenizer.bos_token):]
            if tokenized_rejected[0] == tokenizer.bos_token:
                example["text_rejected"] = example["text_rejected"][len(tokenizer.bos_token):]
        else:
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require either the "
                f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example





def pad(tensors: List[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`List[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
        >>> import torch
        >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
        tensor([[[1, 2],
                [3, 4]],

                [[5, 6],
                [0, 0]]])
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output
    
class PreferenceCollator(DataCollatorMixin):
    """
    Data collator used for preference data. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl import PreferenceCollator
    >>> collator = PreferenceCollator(pad_token_id=0)
    >>> examples = [
    ...     {"prompt_input_ids": [1, 2, 3], "chosen_input_ids": [4, 5], "rejected_input_ids": [6]},
    ...     {"prompt_input_ids": [7, 8], "chosen_input_ids": [9, 10], "rejected_input_ids": [11, 12, 13]}
    ... ]
    >>> collator(examples)
    {'prompt_input_ids': tensor([[1, 2, 3],
                                 [0, 7, 8]]),
     'prompt_attention_mask': tensor([[1, 1, 1],
                                      [0, 1, 1]]),
     'chosen_input_ids': tensor([[ 4,  5],
                                 [ 9, 10]]),
     'chosen_attention_mask': tensor([[1, 1],
                                      [1, 1]]),
     'rejected_input_ids': tensor([[ 6,  0,  0],
                                   [11, 12, 13]]),
     'rejected_attention_mask': tensor([[1, 0, 0],
                                        [1, 1, 1]])
    }
    ```
    """

    pad_token_id: int = 128002
    return_tensors: str = "pt"

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]
        if "pixel_values" in examples[0]:
            pixel_values = [torch.tensor(example["pixel_values"]) for example in examples]
        if "pixel_attention_mask" in examples[0]:
            pixel_attention_mask = [torch.tensor(example["pixel_attention_mask"]) for example in examples]

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        output["chosen_input_ids"] = pad(chosen_input_ids, padding_value=self.pad_token_id)
        output["chosen_attention_mask"] = pad(chosen_attention_mask, padding_value=0)
        output["rejected_input_ids"] = pad(rejected_input_ids, padding_value=self.pad_token_id)
        output["rejected_attention_mask"] = pad(rejected_attention_mask, padding_value=0)
        if "pixel_values" in examples[0]:
            output["pixel_values"] = pad(pixel_values, padding_value=0.0)
        if "pixel_attention_mask" in examples[0]:
            output["pixel_attention_mask"] = pad(pixel_attention_mask, padding_value=0)

        return output
    




def compute_dpo_loss(model, params, buffers, prompt_input_ids, prompt_attention_mask, chosen_input_ids, chosen_attention_mask, rejected_input_ids, rejected_attention_mask):

    rpo_alpha = 1.0
    beta = 0.1
    max_length = 800

    
    
    num_examples = prompt_input_ids.shape[0]//2

    
    # For the prompt, the input_ids are the same for both the chosen and rejected responses
    prompt_input_ids = torch.cat([prompt_input_ids, prompt_input_ids], dim=0)
    prompt_attention_mask = torch.cat(
        [prompt_attention_mask, prompt_attention_mask], dim=0
    )

    

    # Concatenate the chosen and rejected completions
    max_completion_length = max(chosen_input_ids.shape[1], rejected_input_ids.shape[1])
    completion_input_ids = torch.cat(
        (
            pad_to_length(chosen_input_ids, max_completion_length, pad_value=padding_value),
            pad_to_length(rejected_input_ids, max_completion_length, pad_value=padding_value),
        ),
    )
    completion_attention_mask = torch.cat(
        (
            pad_to_length(chosen_attention_mask, max_completion_length, pad_value=0),
            pad_to_length(rejected_attention_mask, max_completion_length, pad_value=0),
        ),
    )

    
    input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
    attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
    # Mask the prompt but not the completion for the loss
    loss_mask = torch.cat(
        (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
        dim=1,
    )

    if max_length is not None:
        input_ids = input_ids[:, : max_length]
        attention_mask = attention_mask[:, : max_length]
        loss_mask = loss_mask[:, : max_length]
    # # Flush left to reduce the memory usage
    # # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
    # #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
    # for i in range(attention_mask.size(0)):
    #     first_one_idx = torch.nonzero(attention_mask[i])[0].item()
    #     input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
    #     attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
    #     loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

    # # Get the first column idx that is all zeros and remove every column after that
    # empty_cols = torch.sum(attention_mask, dim=0) == 0
    # first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else attention_mask.size(1)
    # input_ids = input_ids[:, :first_empty_col]
    # attention_mask = attention_mask[:, :first_empty_col]
    # loss_mask = loss_mask[:, :first_empty_col]

    labels = input_ids[:, 1:].clone()
    loss_mask = loss_mask[:, 1:].bool()
    # labels[~loss_mask] = 0
    labels = torch.where(loss_mask, labels, torch.tensor(0, dtype=labels.dtype, device=labels.device))
    
    outputs = functional_call(model, (params, buffers), args=input_ids, 
                                  kwargs={'attention_mask': attention_mask, 
                                          # 'labels': labels
                                         })
    
    
    logits = outputs.logits[:, :-1, :]
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    # per_token_logps[~loss_mask] = 0
    per_token_logps = torch.where(loss_mask, per_token_logps, torch.tensor(0, dtype=per_token_logps.dtype, device=per_token_logps.device))
    all_logps = per_token_logps.sum(-1)
    chosen_logps = all_logps[:num_examples]
    rejected_logps = all_logps[num_examples:]

    
        
    if rpo_alpha is not None:
        chosen_logits = logits[:num_examples]
        chosen_labels = labels[:num_examples]

        # Compute the log probabilities of the labels
        nll_loss = F.cross_entropy(
            torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
        )

    with model.disable_adapter():
        
        ref_outputs = functional_call(model, (params, buffers), args=input_ids, 
                                      kwargs={'attention_mask': attention_mask, 
                                              # 'labels': labels
                                })
        ref_logits = ref_outputs.logits[:, :-1, :]
        ref_per_token_logps = torch.gather(ref_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        # ref_per_token_logps[~loss_mask] = 0
        ref_per_token_logps = torch.where(loss_mask, ref_per_token_logps, torch.tensor(0, dtype=ref_per_token_logps.dtype, device=ref_per_token_logps.device))
        ref_all_logps = ref_per_token_logps.sum(-1)
        ref_chosen_logps = ref_all_logps[:num_examples]
        ref_rejected_logps = ref_all_logps[num_examples:]


    
    pi_logratios = chosen_logps - rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    differences = pi_logratios - ref_logratios
    losses = -F.logsigmoid(beta * differences)
    
    loss = losses + rpo_alpha * nll_loss
    
    # print(loss)
    return loss.squeeze(0) # must be a scaler

def process_dpo_format_to_dataset(dpo_format_data_path: str):
    """
    Processes a dataset in DPO format into a structured Hugging Face dataset.

    Args:
        dpo_format_data_path (str): Path to the input dataset in DPO format.
        output_path (str): Path where the processed dataset will be saved.

    Returns:
        DatasetDict: The processed dataset with training and testing splits saved to disk.
    """
    dataset_dict = {"chosen": [], "rejected": []}
    with open(dpo_format_data_path, "r") as fin:
        for line in fin:
            data = json.loads(line)
            results = data["dpo_results"]
            for result in results:
                chosen = result["chosen"]
                rejected = result["rejected"]
                chosen_conversation = [
                    {"role": "assistant", "content": chosen_speech}
                    for chosen_speech in chosen
                ]
                rejected_conversation = [
                    {"role": "assistant", "content": rejected_speech}
                    for rejected_speech in rejected
                ]
                chosen_conversation[0]["role"] = "system"
                rejected_conversation[0]["role"] = "system"
                dataset_dict["chosen"].append(chosen_conversation)
                dataset_dict["rejected"].append(rejected_conversation)
    train_dict = {
        "chosen": dataset_dict["chosen"][: int(0.9 * len(dataset_dict["chosen"]))],
        "rejected": dataset_dict["rejected"][
            : int(0.9 * len(dataset_dict["rejected"]))
        ],
    }
    test_dict = {
        "chosen": dataset_dict["chosen"][int(0.9 * len(dataset_dict["chosen"])) :],
        "rejected": dataset_dict["rejected"][
            int(0.9 * len(dataset_dict["rejected"])) :
        ],
    }
    train = Dataset.from_dict(train_dict)
    test = Dataset.from_dict(test_dict)
    datasetDict = DatasetDict({"train": train, "test": test})
    # datasetDict.save_to_disk(output_path)
    return datasetDict


def get_tokenizer(tokenizer_path, truncation_side):
    
    auto_set_chat_template = True

    tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            revision='main',
            trust_remote_code=True,
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if truncation_side is not None:
        tokenizer.truncation_side = truncation_side

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 1024

    elif auto_set_chat_template and tokenizer.chat_template is None and tokenizer.default_chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
        
    if "llama-3":
        tokenizer.pad_token_id = 128002

    return tokenizer


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def decontaminate_humaneval(
    samples: List[Dict[str, Any]], text_column: str = "text", filter_out: Dict[str, List[str]] = {}
) -> List[Dict[str, Any]]:
    """
    filter_out: Dict[str, List[str]] mapping from benchmark name to list of strings that need to be
    filtered-out.
    Return a list where each element is True if the corresponding file should be included in the dataset.
    Otherwise, the element is False.
    """
    output = []

    for content in samples[text_column]:
        content = normalize_whitespace(content.lower())
        matched = False
        for _, substrings in filter_out.items():
            for substring in substrings:
                if normalize_whitespace(substring.lower()) in content:
                    matched = True
                    break
            if matched:
                break
        # we keep files that are not matched
        output.append(not matched)

    return output


import numpy as np
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

max_completion_length = 500
if "llama-3":
    padding_value = 128002

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )
        



import numpy as np
# mask token



def SFT_tokenize(prompt, tokenizer=None, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    you_are_Alice_token = [128006, 9125, 128007, 271, 2675, 527, 30505]
    you_are_Bob_token = [128006, 9125, 128007, 271, 2675, 527, 14596]
    Alice_token = [128006, 78191, 128007, 271, 62786, 25]
    Bob_token = [128006, 78191, 128007, 271, 33488, 25]

    batch = tokenizer(
        prompt,
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding=False,
        return_tensors=None,
    )
    ####
    # batch["input_ids"][-1] = tokenizer.eos_token_id
    # batch["attention_mask"][-1] = 1
    ####
    response_token_ids_idxs = []
    human_token_ids_idxs = []

    batch["labels"] = np.array(batch["input_ids"].copy())
    # import pdb
    # pdb.set_trace()
    # print(batch["labels"])
    # print(np.where(batch["labels"] == you_are_Alice_token[0]))
    for assistant_idx in np.where(batch["labels"] == you_are_Alice_token[0])[
        0
    ]:
        if (
            you_are_Alice_token
            == batch["labels"][
                assistant_idx : assistant_idx + len(you_are_Alice_token)
            ].tolist()
        ):
            response_token_ids = Alice_token
            instruction_token_ids = Bob_token
    for assistant_idx in np.where(batch["labels"] == you_are_Bob_token[0])[
        0
    ]:
        if (
            you_are_Bob_token
            == batch["labels"][
                assistant_idx : assistant_idx + len(you_are_Bob_token)
            ].tolist()
        ):
            response_token_ids = Bob_token
            instruction_token_ids = Alice_token

    for assistant_idx in np.where(
        batch["labels"] == response_token_ids[0]
    )[0]:
        # find the indexes of the start of a response.
        if (
            response_token_ids
            == batch["labels"][
                assistant_idx : assistant_idx + len(response_token_ids)
            ].tolist()
        ):
            response_token_ids_idxs.append(
                assistant_idx + len(response_token_ids)
            )

    human_token_ids = instruction_token_ids
    for human_idx in np.where(batch["labels"] == human_token_ids[0])[0]:
        # find the indexes of the start of a human answer.
        if (
            human_token_ids
            == batch["labels"][
                human_idx : human_idx + len(human_token_ids)
            ].tolist()
        ):
            human_token_ids_idxs.append(human_idx)

    if (
        len(human_token_ids_idxs) > 0
        and len(response_token_ids_idxs) > 0
        and human_token_ids_idxs[0] > response_token_ids_idxs[0]
    ):
        human_token_ids_idxs = [0] + human_token_ids_idxs

    for idx, (start, end) in enumerate(
        zip(human_token_ids_idxs, response_token_ids_idxs)
    ):
        # Make pytorch loss function ignore all non response tokens
        if idx != 0:
            batch["labels"][start:end] = -100
        else:
            batch["labels"][:end] = -100

    if len(response_token_ids_idxs) < len(human_token_ids_idxs):
        batch["labels"][human_token_ids_idxs[-1] :] = -100

    return batch

def SFT_generate_and_tokenize_prompt(data_point, tokenizer=None):
    
    tokenized_full_prompt = SFT_tokenize(data_point['text'], tokenizer)
    
    return tokenized_full_prompt


from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple
from datasets import load_from_disk, concatenate_datasets

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    dataset_mixer: Optional[Dict[str, float]] = field(
        default=None,
        metadata={"help": ("Datasets and their proportions to be used for training ift/rl.")},
    )
    text_column: Optional[str] = field(
        default="text",
        metadata={"help": "The column name to use for the text in the dataset (only used for continued pretraining)."},
    )
    dataset_splits: Optional[List[str]] = field(
        default_factory=lambda: ["train", "test"],
        metadata={"help": ("List of train test splits to use in the dataset")},
    )
    dataset_configs: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of dataset config names. If given must be the same length as 'dataset_mixer' keys."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    truncation_side: Optional[str] = field(
        default=None, metadata={"help": "Truncation side to use for the tokenizer."}
    )
    auto_insert_empty_system_msg: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to automatically insert an empty system message as the first message if `system` is mentioned in the chat template."
            )
        },
    )
    
def get_datasets(
    data_config: DataArguments | dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle: bool = True,
) -> DatasetDict:
    
    if type(data_config) is DataArguments:
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        dataset_mixer = data_config.dataset_mixer
    elif isinstance(data_config, dict):
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(
        dataset_mixer,
        splits=splits,
        configs=configs,
        columns_to_keep=columns_to_keep,
        shuffle=shuffle,
    )
    return raw_datasets

def mix_datasets(
    dataset_mixer: dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle=True,
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'dataset_mixer' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    splits = ["train", "test"] if splits is None else splits
    configs = [None] * len(dataset_mixer) if not configs else configs
    columns_to_keep = [] if columns_to_keep is None else columns_to_keep

    if configs is not None and len(configs) != len(dataset_mixer):
        raise ValueError(
            "The number of given dataset config names must be the same as the given number of datasets."
        )

    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for (ds, frac), ds_config in zip(dataset_mixer.items(), configs):
        fracs.append(frac)
        for split in splits:
            try:
                # check local dataset
                dataset = load_from_disk(os.path.join(ds, split), keep_in_memory=True)
                print('--------------------------')
                print(dataset)
                print('--------------------------')
            except FileNotFoundError:
                # if dataset on a Hub repo
                dataset = load_dataset(ds, ds_config, split=split)

            # Remove redundant columns to avoid schema conflicts on load
            dataset = dataset.remove_columns(
                [col for col in dataset.column_names if col not in columns_to_keep]
            )
            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(
                    f"Split type {split} not recognized as one of test or train."
                )

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(
                seed=42
            )
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with splits {splits}. Check the dataset has been correctly formatted."
        )

    print(f'------------------------------{len(raw_datasets)}-----------------')

    return raw_datasets