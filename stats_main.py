from reward.reward import result_stats
from argparse import ArgumentParser
from utils.config import llama3_path_a100, llama3_path_a800
from transformers import AutoTokenizer

argumentParser = ArgumentParser()
argumentParser.add_argument("--input_path", type=str, required=True)
argumentParser.add_argument("--tokenizer_path", type=str, default="/home/wentaos/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a")
argumentParser.add_argument("--score_type",type=str,default="f1-score")
argumentParser.add_argument("--is_consistence",type=int,default=0)
args = argumentParser.parse_args()

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    result_stats(args.input_path, tokenizer,args.score_type,(args.is_consistence==1))
