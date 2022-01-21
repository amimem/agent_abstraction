from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.magent import battle_v3
from pettingzoo.utils import to_parallel
from magent_wrappers import MAgengtPettingZooEnv, MAgentParallelPettingZooEnv
import os

os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "1"

# Based on code from github.com/parametersharingmadrl/parametersharingmadrl
if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

    def env_creator(args):
        env = battle_v3.env(map_size=15)
        # env = to_parallel(env)
        return MAgengtPettingZooEnv(env, flatten_dict_observations=False)

    env = env_creator({})
    register_env("battle_v3", env_creator)
    print(set(env.agents), len(env.agents))

    save_dir = os.getenv('SLURM_TMPDIR')

    tune.run(
        "DQN",
        stop={"episodes_total": 60000},
        checkpoint_freq=500,
        local_dir=save_dir,
        checkpoint_at_end=True,
        config={
            # Enviroment specific
            # "env": "battle_v3",
            "env": "battle_v3",
            # "simple_optimizer":True,
            # General
            "framework": "torch",
            "num_gpus": 1,
            "num_workers": 10,
            # "model": {"dim": 42, "conv_filters": [[16, [4, 4], 2], [32, [4, 4], 2], [512, [11, 11], 1]]},
            "model": {
                # "fcnet_hiddens": [64],
                "dim": 15,
                "conv_filters": [
                    [32, [15, 15], 1],
                    ],
            },
            # Method specific
             "multiagent": {
                "policies": set(env.agents),
                "policy_mapping_fn": (
                    lambda agent_id, episode, **kwargs: agent_id),
                # Keep this many policies in the "policy_map" (before writing
                # least-recently used ones to disk/S3).
                # "policy_map_capacity": len(env.agents),
                # Where to store overflowing (least-recently used) policies?
                # Could be a directory (str) or an S3 location. None for using
                # the default output dir.
                # "policy_map_cache": save_dir,
                # Function mapping agent ids to policy ids.
            },
            "record_env": True,
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
