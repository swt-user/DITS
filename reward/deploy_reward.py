import logging
import multiprocessing.pool
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from answerParser.parser import is_equiv
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
from pydantic import BaseModel
import json
from reward.reward import cal_f1_score
from typing import List
from utils import lambda1, lambda2
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing
import re
import os

app = FastAPI()
logger = logging.getLogger("ray.serve")

reward_file_lock = threading.Lock()
multi_lock = multiprocessing.Lock()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Params(BaseModel):
    texts: list[str]


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, model_handle: DeploymentHandle) -> None:
        self.handle = model_handle

    @app.post("/ppl")
    async def ppl(self, params: Params):
        return await self.handle.ppl.remote(params.texts)


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    num_replicas=1,
)
class RewardModel:
    def __init__(self):
        model_path = os.environ["INITIAL_MODEL_PATH"]
        self.reward_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # torch_dtype="auto",
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )
        self.tokenizer.pad_token_id = 128002
        self.reward_model.eval()

    def calc_example_losses(self, logits, labels):
        
        # logits = logits.float()
        logits = logits.to(dtype=torch.bfloat16)
        bsz, seq_len = labels.shape
        
        # Shift so that tokens < n predict n
        shift_logits = logits[..., 4:-1, :].contiguous()
        shift_labels = labels[..., 5:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(
            reduction="none", ignore_index=self.tokenizer.pad_token_id
        )
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.view(bsz, seq_len - 5)
        return loss.sum(dim=-1) / (loss != 0).sum(dim=-1)


    def calc_example_losses_subbatch(self, logits, labels, sub_batch_size=1, max_seq_len=2000):
        # 确保logits为bfloat16精度
        logits = logits.to(dtype=torch.bfloat16)
        bsz, seq_len = labels.shape

        # Shift so that tokens < n predict n
        shift_logits = logits[..., 4:-1, :].contiguous()  # (bsz, seq_len-5, vocab_size)
        shift_labels = labels[..., 5:].contiguous()  # (bsz, seq_len-5)
        shift_labels = shift_labels.to(shift_logits.device)

        # 初始化损失和有效Token数的累积变量
        loss_total = torch.zeros(bsz, device=logits.device)  # 存储每个样本的总损失
        valid_counts = torch.zeros(bsz, device=logits.device)  # 存储每个样本的有效Token数

        # 定义损失函数
        loss_fct = CrossEntropyLoss(
            reduction="none", ignore_index=self.tokenizer.pad_token_id
        )

        # 分批处理
        for i in range(0, bsz, sub_batch_size):
            # 获取当前子批次
            sub_logits = shift_logits[i:i + sub_batch_size]  # (sub_batch_size, seq_len-5, vocab_size)
            sub_labels = shift_labels[i:i + sub_batch_size]  # (sub_batch_size, seq_len-5)
            
            # 分批处理序列长度
            for j in range(0, seq_len, max_seq_len):
                # 获取当前子序列的logits和labels
                seq_logits = sub_logits[:, j:j + max_seq_len, :]  # (sub_batch_size, min(max_seq_len, seq_len-5-j), vocab_size)
                seq_labels = sub_labels[:, j:j + max_seq_len]  # (sub_batch_size, min(max_seq_len, seq_len-5-j))

                # 展平当前子序列的logits和labels
                seq_logits = seq_logits.view(-1, seq_logits.size(-1))  # (sub_batch_size * seq_len_j, vocab_size)
                seq_labels = seq_labels.view(-1)  # (sub_batch_size * seq_len_j)

                # 计算当前子序列的损失
                seq_loss = loss_fct(seq_logits, seq_labels)  # (sub_batch_size * seq_len_j)
                seq_loss = seq_loss.view(sub_batch_size, -1)  # (sub_batch_size, seq_len_j)

                seq_valid = (seq_labels != self.tokenizer.pad_token_id).view(sub_batch_size, -1)  # (sub_batch_size, seq_len_j)

                # 累积损失和有效Token数
                loss_total[i:i + sub_batch_size] += seq_loss.sum(dim=1)  # 累积每个样本的总损失
                valid_counts[i:i + sub_batch_size] += seq_valid.sum(dim=1) 
                # with open("/home/wentaos/Optima/bash_math_multi/depoly_write.txt", 'a') as f:
                #      f.write(f"logits.shape: {shift_logits.shape}   {sub_logits.shape} \n")
        # 计算每个样本的平均损失
        return loss_total / valid_counts

    def ppl(self, texts: list[str]):
        # with torch.inference_mode():
        #     inputs = self.tokenizer(
        #         texts,
        #         truncation=True,
        #         padding=True,
        #         return_tensors="pt",
        #         max_length=self.reward_model.config.max_position_embeddings,
        #         add_special_tokens=False,
        #     ).to("cuda")
        #     outputs = self.reward_model(**inputs)
            
        #     loss = self.calc_example_losses_subbatch(outputs.logits, inputs["input_ids"])
        #     return loss.cpu().tolist()

        with torch.inference_mode():
            result = []
            for t in texts:
                inputs = self.tokenizer(
                    [t],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=self.reward_model.config.max_position_embeddings,
                    add_special_tokens=False,
                ).to("cuda")
                outputs = self.reward_model(**inputs)
            
                loss = self.calc_example_losses(outputs.logits, inputs["input_ids"])
                result.append(loss.cpu().tolist()[0])
        
        return result


# # Create the deployment
# entrypoint = APIIngress.bind(BPCModel.bind())


def serve_reward_model(model_path, num_replicas=1, gpu_ids: list = [0, 1, 2, 3, 4, 5, 6, 7]):
    # Initialize Ray
    ray.init(
        address="local",
        runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": f"{gpu_ids}"[1:-1]}},
    )

    # Create the deployment
    entrypoint = APIIngress.bind(RewardModel.options(num_replicas=num_replicas).bind())

    # serve.start(http_options={"max_concurrent_requests": 1})
    # Run the deployment
    serve.run(entrypoint, blocking=True)


def reward_batch_based_on_deploy(
    task_results: dict,
    model_url,
    tokenizer,
    _lambda1: float,
    _lambda2: float,
    score_type: str = "f1-score",
    cal_ppl: bool = True,
):
    tokenizer.pad_token_id = 128002
    results_list: List[dict] = task_results["results"]
    rewarded_results_list: List[dict] = []
    max_token_count = max(
        [
            result["token_count"]
            for result in results_list
            if "large" not in result["conversation"]
        ]
    )
    is_mix = False
    if score_type == "mix":
        is_mix = True
    for result in results_list:
        if is_mix:
            score_type = result["score_type"]
        try:
            if isinstance(result["answer"], list):
                if score_type == "f1-score":
                    all_score = [
                        cal_f1_score(answer, result["final_answer"], tokenizer)
                        for answer in result["answer"]
                    ]
                    correct_score = max(all_score)
                elif score_type == "exact-match":
                    answers = [answer.strip().lower for answer in result["answer"]]
                    correct_score = (
                        1 if result["answer"].strip().lower() in answers else 0
                    )
            else:
                if score_type == "f1-score":
                    correct_score = cal_f1_score(
                        result["answer"], result["final_answer"], tokenizer
                    )
                elif score_type == "exact-match":
                    correct_score = (
                        1
                        if result["answer"].strip().lower()
                        == result["final_answer"].strip().strip("\\").lower()
                        else 0
                    )
                elif score_type == "math":
                    correct_score = (
                        1
                        if is_equiv(
                            result["answer"].strip().lower(),
                            result["final_answer"].strip().lower(),
                        )
                        else 0
                    )
        except:
            correct_score = 0
        token_score = 0
        try:
            token_score = _lambda1 * result["token_count"] / max_token_count
        except:
            token_score = 0
        if "large" in result["conversation"]:
            token_score = -1
        if result["token_count"] <= 40:
            token_score = -1
        ppl_score = 0

        if cal_ppl:
            max_token = 2000 if os.environ["DATASET"]!="math" else 5000
            if result["token_count"] > max_token or len(result["conversation"]) == 0:
                ppl_score = -1
            else:
                tokenized_sentences = []
                record = []
                repeat = False
                for sentence in result["conversation"]:
                    try:
                        if (
                            sentence.strip().strip("Alice:").strip("Bob:").lower()
                            in record
                        ):
                            repeat = True
                            break
                        record.append(
                            sentence.strip().strip("Alice:").strip("Bob:").lower()
                        )
                    except:
                        pass
                    sentence = [{"role": "assistant", "content": sentence}]
                    tokenized_sentence = tokenizer.apply_chat_template(
                        sentence, tokenize=False
                    )
                    tokenized_sentences.append(tokenized_sentence)
                if not repeat:
                    ret = requests.post(model_url, json={"texts": tokenized_sentences})
                    ret = [float(ppl) for ppl in ret.text[1:-1].split(",")]
                    max_ppl = max(ret)
                    ppl_score = _lambda2 / max_ppl
                else:
                    ppl_score = 0

        reward = correct_score + token_score + ppl_score
        result["reward"] = reward
        result["correct_score"] = correct_score
        result["token_score"] = token_score
        result["ppl_score"] = ppl_score
        rewarded_results_list.append(result)
        print(
            f"correct: {correct_score},token: {token_score},ppl:{ppl_score},reward:{correct_score + token_score + ppl_score}"
        )
    return rewarded_results_list


def process_raw_conversation_data_based_on_deploy_once(
    data, model_url, tokenizer, _lambda1, _lambda2, score_type, cal_ppl, output_path
):
    print(f"-----------{data['task_id']}-------------")
    data["results"] = reward_batch_based_on_deploy(
        data,
        model_url,
        tokenizer,
        _lambda1,
        _lambda2,
        score_type=score_type,
        cal_ppl=cal_ppl,
    )
    with reward_file_lock:
        with open(output_path, "a") as fout:
            try:
                fout.write(json.dumps(data) + "\n")
            except:
                pass


def process_raw_conversation_data_based_on_deploy_once_lock(
    data,
    model_url,
    tokenizer,
    _lambda1,
    _lambda2,
    score_type,
    cal_ppl,
    output_path,
):
    print(f"-----------{data['task_id']}-------------")
    data["results"] = reward_batch_based_on_deploy(
        data,
        model_url,
        tokenizer,
        _lambda1,
        _lambda2,
        score_type=score_type,
        cal_ppl=cal_ppl,
    )
    with multi_lock:
        with open(output_path, "a") as fout:
            try:
                fout.write(json.dumps(data) + "\n")
            except:
                pass


def process_raw_conversation_data_based_on_deploy(
    model_url,
    tokenizer,
    input_path,
    output_path,
    _lambda1: float = lambda1,
    _lambda2: float = lambda2,
    cal_ppl: bool = True,
    score_type: str = "f1-score",
    num_thread: int = 24,
):
    """
    Process raw conversation data and compute scores based on the specified model.

    Parameters:
        model_url (str): The URL of the model used for processing.
        tokenizer: The tokenizer used to tokenize input data.
        input_path (str): Path to the input data file.
        output_path (str): Path to the output data file.
        _lambda1 (float): Weight for token scoring.
        _lambda2 (float): Weight for perplexity scoring.
        cal_ppl (bool): Whether to calculate perplexity.
        score_type (str): Type of score to calculate ("f1-score", "exact-match", "math").
        num_thread (int): Number of threads to use for processing.

    Returns:
        None
    """
    record_set = set()
    skipping = 0
    try:
        with open(output_path, "r") as fin:
            for line in fin:
                data = json.loads(line)
                skipping += 1
                record_set.add(data["task_id"])
    except:
        pass
    print(f"skipping: {skipping}")
    print(f"{record_set}")

    fout = open(output_path, "a")
    args_list = []
    with open(input_path, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if data["task_id"] in record_set:
                continue
            args_list.append(data)
    if score_type != "math":
        with ThreadPoolExecutor(max_workers=num_thread) as executor:
            futures = [
                executor.submit(
                    process_raw_conversation_data_based_on_deploy_once,
                    data,
                    model_url,
                    tokenizer,
                    _lambda1,
                    _lambda2,
                    score_type,
                    cal_ppl,
                    output_path,
                )
                for data in args_list
            ]
            for future in as_completed(futures):
                print(f"----------------")
    else:
        with ProcessPoolExecutor(max_workers=num_thread) as executor:
            futures = [
                executor.submit(
                    process_raw_conversation_data_based_on_deploy_once_lock,
                    data,
                    model_url,
                    tokenizer,
                    _lambda1,
                    _lambda2,
                    score_type,
                    cal_ppl,
                    output_path,
                )
                for data in args_list
            ]
            for future in as_completed(futures):
                print(f"----------------")
    fout.close()


def get_score_deploy(
    result: dict,
    max_token_count: int,
    tokenizer,
    model_url: str,
    cal_ppl: bool = True,
    _lambda1: float = lambda1,
    _lambda2: float = lambda2,
    score_type: str = "f1-score",
):
    """
    Calculate the reward score based on the result, including correctness, token score, and perplexity.

    Parameters:
        result (dict): A dictionary containing the answer, final answer, conversation, and token count.
        max_token_count (int): The maximum token count for normalization.
        tokenizer: Tokenizer used to process text.
        model_url (str): The URL for the model used to calculate perplexity.
        cal_ppl (bool): A flag to indicate whether to calculate perplexity.
        _lambda1 (float): Weight for the token score.
        _lambda2 (float): Weight for the perplexity score.
        score_type (str): The type of score to calculate ("f1-score", "exact-match", or "math").

    Returns:
        tuple: A tuple containing the total reward, correct score, and perplexity score.
    """
    correct_score = 0
    try:
        if isinstance(result["answer"], list):
            if score_type == "f1-score":
                all_score = [
                    cal_f1_score(answer, result["final_answer"], tokenizer)
                    for answer in result["answer"]
                ]
                correct_score = max(all_score)
            elif score_type == "exact-match":
                answers = [answer.strip().lower for answer in result["answer"]]
                correct_score = 1 if result["answer"].strip().lower() in answers else 0
        else:
            if score_type == "f1-score":
                correct_score = cal_f1_score(
                    result["answer"], result["final_answer"], tokenizer
                )
            elif score_type == "exact-match":
                correct_score = (
                    1
                    if result["answer"].strip().lower()
                    == result["final_answer"].strip().strip("\\").lower()
                    else 0
                )
            elif score_type == "math":
                correct_score = (
                    1
                    if is_equiv(
                        result["answer"].strip().lower(),
                        result["final_answer"].strip().lower(),
                    )
                    else 0
                )
    except:
        correct_score = 0
    token_score = _lambda1 * result["token_count"] / max_token_count
    if "large" in result["conversation"]:
        token_score = -1
    if result["token_count"] <= 40:
        token_score = -1

    ppl_score = 0
    if cal_ppl:
        max_token = 2000 if os.environ["DATASET"]!="math" else 5000
        if result["token_count"] > max_token or len(result["conversation"]) == 0:
            ppl_score = -1
        else:
            tokenized_sentences = []
            record = []
            repeat = False
            for sentence in result["conversation"]:
                if re.search(r"Alice:", sentence) and re.search(r"Bob:", sentence):
                    correct_score = 0
                    break
                elif (
                    len(re.findall(r"Alice:", sentence)) >= 2
                    or len(re.findall(r"Bob:", sentence)) >= 2
                ):
                    correct_score = 0
                    break
                try:
                    if sentence.strip().strip("Alice:").strip("Bob:").lower() in record:
                        repeat = True
                        break
                    record.append(
                        sentence.strip().strip("Alice:").strip("Bob:").lower()
                    )
                except:
                    pass
                sentence = [{"role": "assistant", "content": sentence}]
                tokenized_sentence = tokenizer.apply_chat_template(
                    sentence, tokenize=False
                )
                tokenized_sentences.append(tokenized_sentence)
            if not repeat:
                if len(tokenized_sentences)>0:
                    ret = requests.post(model_url, json={"texts": tokenized_sentences})
                    ret = [float(ppl) for ppl in ret.text[1:-1].split(",")]
                    max_ppl = max(ret)
                    ppl_score = _lambda2 / max_ppl
                else:
                    ppl_score = 0
            else:
                ppl_score = 0
    reward = correct_score + token_score + ppl_score
    print(
        f"correct: {correct_score},token: {token_score},ppl:{ppl_score},reward:{correct_score + token_score + ppl_score}"
    )
    return reward, correct_score, ppl_score
