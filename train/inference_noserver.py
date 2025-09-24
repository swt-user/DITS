from transformers import AutoTokenizer, AutoModelForCausalLM
from train.datagenerate import conversation, vllm_data_generate, data_generate
from train.datagenerate_noserver import vllm_data_generate_noserver
from train.datagenerate_noserver_multiprocess import vllm_data_generate_noserver_multiprocess


def inference_noserver(
    model_first: str,
    model_second: str,
    device_first: str,
    device_second: str,
    tokenizer_path_first: str,
    tokenizer_path_second: str,
    sample_count: int,
    explore_count: int,
    output_path: str,
    thread_count: int,
    prompt_pool_path: str,
    train_data_path: str = None,
    dataloader=None,
    temperature: float = 0,
    no_use_prompt_pool: bool = True,
    ports: list = [],
    add_name: int = 0,
    lora_first_path = None,
    lora_second_path = None,
):
    if train_data_path is None:
        raise FileNotFoundError("train_data_path is None")

    vllm_data_generate_noserver(
        model_first,
        model_second,
        device_first,
        device_second,
        tokenizer_path_first,
        tokenizer_path_second,
        sample_count,
        explore_count,
        output_path,
        thread_count,
        prompt_pool_path,
        train_data_path,
        dataloader=dataloader,
        no_use_prompt_pool=no_use_prompt_pool,
        temperature=temperature,
        ports=ports,
        iteration=1 if add_name == 0 else 0,
        lora_first_path = lora_first_path,
        lora_second_path = lora_second_path,
    )