from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
# from pettingzoo.mpe import simple_push_v2
# from pettingzoo.mpe import simple_adversary_v2
# from pettingzoo.mpe import simple_spread_v2
# from pettingzoo.mpe import simple_tag_v2
from pettingzoo.magent import battle_v3
# from pettingzoo.magent import adversarial_pursuit_v3
# from pettingzoo.magent import gather_v3
# from pettingzoo.magent import battlefield_v3
# from pettingzoo.magent import combined_arms_v5
# from pettingzoo.magent import tiger_deer_v3
# import ray.rllib.contrib.maddpg as maddpg
# import ray.rllib.agents.dqn as dqn  # DQNTrainer
# import ray.rllib.agents.a3c.a2c as a2c  # A2CTrainer
# import ray.rllib.agents.ddpg.td3 as td3  # TD3Trainer
# from ray.tune.integration.wandb import WandbLogger
# from ray.tune.integration.wandb import WandbLoggerCallback
# from ray.tune.logger import DEFAULT_LOGGERS
from pettingzoo.utils import to_parallel
# import supersuit
from supersuit import flatten_v0

# Based on code from github.com/parametersharingmadrl/parametersharingmadrl
if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

    def env_creator(args):
        env = battle_v3.env(map_size=15, minimap_mode=True)
        env = flatten_v0(env)
        return PettingZooEnv(env)
        # env = to_parallel(env)
        # env = supersuit.pad_observations_v0(env)
        # env = supersuit.pad_action_space_v0(env)

    env = env_creator({})
    register_env("battle_v3", env_creator)
    print(set(env.agents))

    tune.run(
        "DQN",
        stop={"episodes_total": 60000},
        checkpoint_freq=1000,
        config={
            # Enviroment specific
            # "env": "battle_v3",
            "env": "battle_v3",
            # "simple_optimizer":True,
            # General
            "framework": "torch",
            "num_gpus": 2,
            "num_workers": 10,
            # Method specific
            "multiagent": {
                "policies": set(env.agents),
                "policy_mapping_fn": (
                    lambda agent_id, episode, **kwargs: agent_id),
            },
            # "log_level": "DEBUG",
            # "wandb": {
            #     "project": "marl",
            #     "api_key_file": "wandb_api_key.txt",
            #     "log_config": True
            # }
        },
        # callbacks=[WandbLoggerCallback(project="marl", api_key_file="wandb_api_key.txt",log_config=True)],
        )
