import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.mpe import simple_tag_v2
from pettingzoo.utils import to_parallel
from magent_wrappers import MAgengtPettingZooEnv, MAgentParallelPettingZooEnv
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy, PPOTorchPolicy
from ray.rllib.agents.dqn import DQNTrainer, DQNTorchPolicy
from ray.rllib.offline.json_reader import JsonReader
import torch
import os
import argparse
import numpy as np
from pathlib import Path
import pandas as pd

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "1"
# num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK"))
# print("num cpus are ", num_cpus)

parser = argparse.ArgumentParser()
parser.add_argument("--path", metavar="P", type=str, help="path")
parser.add_argument(
    "--chkpt-path",
    metavar="CP",
    type=str,
    default="C:/Users/Andrew Williams/Documents/github/marl/json_files",
    help="checkpoint path",
)
args = parser.parse_args()
print(args)


if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

    ray.init(include_dashboard=False, num_cpus=1, num_gpus=0)
    assert ray.is_initialized() == True

    def env_creator(args):
        env = simple_tag_v2.env(
            num_good=1,
            num_adversaries=3,
            num_obstacles=2,
            max_cycles=100,
            continuous_actions=False,
        )
        env = to_parallel(env)
        return ParallelPettingZooEnv(env)

    env = env_creator({})
    register_env("simple_tag", env_creator)
    print(set(env.agents), len(env.agents))

    obs_space = env.observation_space
    act_space = env.action_space

    # base_path = args.path
    # p = Path(base_path)

    # checkpoint_paths = [f for f in p.iterdir() if f.is_dir()]
    # print("path is ", base_path)

    # def gen_policies(agents_list):
    #     policies = {}
    #     policies["policy_adversary"] = (PPOTorchPolicy, obs_space, act_space, {})
    #     policies["policy_agent"] = (PPOTorchPolicy, obs_space, act_space, {})
    #     return policies

    # policies = gen_policies(env.agents)

    # def policy_map(agent_id, episode, **kwargs):
    #     # if args.team:
    #     assert isinstance(agent_id, str)
    #     if "adversary" in agent_id:
    #         return "policy_adversary"
    #     if "agent" in agent_id:
    #         return "policy_agent"

    # policies = gen_policies(env.agents)

    # def policy_map(agent_id, episode, **kwargs):
    #     assert isinstance(agent_id, str)
    #     if "adversary" in agent_id:
    #         return "policy_adversary"
    #     if "agent" in agent_id:
    #         return "policy_agent"

    evaluation_num_episodes = 100000

    def get_config(output_path):
        return {
            "env": "simple_tag",
            # "record_env": True,
            "multiagent": {
                "policies": set(env.agents),
                "policy_mapping_fn": (
                    lambda agent_id, episode, **kwargs: agent_id
                ),
            },
            "num_gpus": 0,
            # "log_level": "DEBUG",
            # "num_workers": num_cpus-1,
            "evaluation_num_episodes": evaluation_num_episodes,
            "evaluation_num_workers": 1,
            "framework": "torch",
            "output": output_path,
            # "log_level": "DEBUG",
        }

    # for path in checkpoint_paths:
    if args.chkpt_path:
        path = Path(args.chkpt_path)
        json_exists = False
        path_str = str(path)

        try:
            json_file = [f for f in path.iterdir() if f.suffix == ".json"]
            assert len(json_file) == 0
        except AssertionError:
            json_exists = True
            if len(json_file) == 1:
                print(f"{len(json_file)} json file(s) already exists")

        # ppo_config = get_config(path_str)
        # trainer = PPOTrainer(config=ppo_config)
        # chekpoint_file = [
        #     f for f in path.iterdir() if f.suffix == ".tune_metadata"
        # ]
        # assert len(chekpoint_file) == 1
        # chekpoint_file = os.path.join(path, chekpoint_file[0].stem)
        # trainer.restore(chekpoint_file)

        # models = {}
        # for agent in set(env.agents):
        #     models[agent] = trainer.get_policy(agent).model

        # print("models are ", models)

        # if not json_exists:
        #     trainer.evaluate()

        for file in os.listdir(path):
            if file.endswith(".json"):
                json_path = os.path.join(path, file)
                print(json_path)
                json_f = JsonReader(json_path)
                # json_f2 = json.load(open(json_path))

                with open(json_path, "r") as f:
                    # parse the JSON data
                    text = f.read()
                    lines = text.split("\n")

                    # count the number of lines
                    num_lines = len(lines)

                # print the number of lines/objects
                evaluation_num_episodes = num_lines

                # print(len(json_f2))
                device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu"
                )

                ep_num = 0
                data = pd.DataFrame()

                # TODO: optimize this loop
                rows_list = []

                while ep_num < evaluation_num_episodes:
                    ma_batch = json_f.next()
                    p_batches = ma_batch.policy_batches
                    for policy_name in set(env.agents):
                        for row in p_batches[policy_name].rows():
                            row["policy_name"] = policy_name
                            row["eval_episode"] = ep_num
                            rows_list.append(row)

                    ep_num += 1

                    # if ep_num % 20 == 0:
                    print(f"ep {ep_num} of {evaluation_num_episodes}")

                data = pd.DataFrame(rows_list)

                assert data is not None
                data.to_pickle(path_str + "_data.pkl")
