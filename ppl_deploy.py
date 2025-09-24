from argparse import ArgumentParser
from reward.deploy_reward import serve_reward_model


argumentParser = ArgumentParser()
argumentParser.add_argument("--num_replicas", type=int, default=8)
argumentParser.add_argument("--model_path", type=str, default=None)
args = argumentParser.parse_args()

if __name__ == "__main__":
    serve_reward_model(args.model_path, args.num_replicas, gpu_ids=[i for i in range(args.num_replicas)])
