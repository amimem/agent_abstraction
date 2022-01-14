from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.magent import battle_v3
from pettingzoo.utils import to_parallel
from magent_wrappers import MAgengtPettingZooEnv, MAgentParallelPettingZooEnv
# import supersuit
from supersuit import flatten_v0

# Based on code from github.com/parametersharingmadrl/parametersharingmadrl
if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

    def env_creator(args):
        env = battle_v3.env(map_size=15)
        env = flatten_v0(env)
        return MAgengtPettingZooEnv(env)

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
