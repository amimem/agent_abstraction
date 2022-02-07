from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.magent import battle_v3
from pettingzoo.utils import to_parallel
from magent_wrappers import MAgengtPettingZooEnv, MAgentParallelPettingZooEnv
from ray.rllib.agents.dqn import DQNTrainer, DQNTFPolicy, DQNTorchPolicy
from argparse import ArgumentParser
import os
import ray

os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "1"
num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK'))
print("num cpus are ", num_cpus)

parser = ArgumentParser()

parser.add_argument("-t", "--team", action="store_true",
                    help="use separate policies for each team")
parser.add_argument("-n", "--number", type=int, default=1,
                    help="number of policies per team")

args = parser.parse_args()

# Based on code from github.com/parametersharingmadrl/parametersharingmadrl
if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

    ray.init(include_dashboard=False, num_cpus=num_cpus, num_gpus=1)
    assert ray.is_initialized() == True

    def env_creator(args):
        env = battle_v3.env(map_size=15)
        # env = to_parallel(env)
        return MAgengtPettingZooEnv(env, flatten_dict_observations=True)

    env = env_creator({})
    register_env("battle_v3", env_creator)
    print(set(env.agents), len(env.agents))

    obs_space = env.observation_space
    act_space = env.action_space

    def gen_policies(num_policies):
        policies = {}
        for i in range(num_policies):
            policies[f"policy_{i}"] = (DQNTorchPolicy, obs_space, act_space, {})
        return policies

    policies = set(env.agents) if not args.team else gen_policies(args.number)

    def policy_map(agent_id, episode, **kwargs):
        if args.team:
            assert isinstance(agent_id, str)
            if "red" in agent_id:
                return "policy_0"
            elif "blue" in agent_id:
                return "policy_1"
        else:
            return agent_id

    save_dir = os.getenv('SLURM_TMPDIR')

    tune.run(
        "DQN",
        stop={"episodes_total": 5000},
        local_dir=save_dir,
        checkpoint_freq=500,
        checkpoint_at_end=True,
        config={
            # Enviroment specific
            "env": "battle_v3",
            # "simple_optimizer":True,
            # General
            "framework": "torch",
            "num_gpus": 1,
            "num_workers": num_cpus-1,
            # "num_envs_per_worker": 0.25,
            # "model": {"dim": 42, "conv_filters": [[16, [4, 4], 2], [32, [4, 4], 2], [512, [11, 11], 1]]},
            # "num_atoms": 51
            "noisy": True,
            "n_step": 3,
            # "v_min": -10,
            # "v_max": 10,
            # "model": {
            #     # "fcnet_hiddens": [64],
            #     "dim": 15,
            #     "conv_filters": [
            #         [32, [15, 15], 1],
            #         ],
            # },
            # Method specific
             "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_map,
                # Keep this many policies in the "policy_map" (before writing
                # least-recently used ones to disk/S3).
                # "policy_map_capacity": len(env.agents),
                # Where to store overflowing (least-recently used) policies?
                # Could be a directory (str) or an S3 location. None for using
                # the default output dir.
                # "policy_map_cache": save_dir,
                # Function mapping agent ids to policy ids.
            },
            # "record_env": True,
            # "train_batch_size": 600,
            # "log_level": "DEBUG",
            # "wandb": {
            #     "project": "marl",
            #     "api_key_file": "wandb_api_key.txt",
            #     "log_config": True
            # }
        },
        # callbacks=[WandbLoggerCallback(project="marl", api_key_file="wandb_api_key.txt",log_config=True)],
        )
