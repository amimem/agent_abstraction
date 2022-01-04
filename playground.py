from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.mpe import simple_push_v2
from pettingzoo.mpe import simple_adversary_v2
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_tag_v2
import ray.rllib.contrib.maddpg as maddpg
import ray.rllib.agents.dqn as dqn  # DQNTrainer
import ray.rllib.agents.a3c.a2c as a2c  # A2CTrainer
import ray.rllib.agents.ddpg.td3 as td3  # TD3Trainer
import supersuit

# Based on code from github.com/parametersharingmadrl/parametersharingmadrl
if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

    def env_creator(args):
        env = simple_adversary_v2.env()
        env = supersuit.pad_observations_v0(env)
        return ParallelPettingZooEnv(env)

    env = env_creator({})
    register_env("simple_adversary_v2", env_creator)

    # env = env_creator({"e":simple_push_v2})
    # register_env("simple_push_v2", env_creator)

    # env = env_creator({"e":simple_spread_v2})
    # register_env("simple_spread_v2", env_creator)

    # env = env_creator({"e":simple_tag_v2})
    # register_env("simple_tag_v2", env_creator)

    tune.run(
        "DQN",
        stop={"episodes_total": 60000},
        checkpoint_freq=1000,
        config={
            # Enviroment specific
            "env": "simple_adversary_v2",
            # General
            # "framework": "torch",
            "num_gpus": 0,
            # "num_workers": 2,
            # Method specific
            "multiagent": {
                "policies": set(env.agents),
                "policy_mapping_fn": (
                    lambda agent_id, episode, **kwargs: agent_id),
            }
        },
    )
