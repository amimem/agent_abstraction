import ray
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.utils import to_parallel
from pettingzoo.magent import battle_v3
import ray.rllib.contrib.maddpg as maddpg
from ray.rllib.agents.dqn import DQNTrainer, DQNTFPolicy, DQNTorchPolicy
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy, PPOTorchPolicy
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import os
from supersuit import flatten_v0

if __name__ == "__main__":

    ray.init(include_dashboard=False)

    def env_creator(args):
        env = battle_v3.env(map_size=15, minimap_mode=True)
        env = flatten_v0(env)
        return PettingZooEnv(env)

    env = env_creator({})
    register_env("battle_v3", env_creator)
    print(set(env.agents))

    obs_space = env.observation_space
    act_space = env.action_space

    # print(obs_space)
    # print(act_space)

    policies = {
        "ppo_policy": (PPOTorchPolicy, obs_space, act_space, {}),
    }

    print(policies)

    ppo_trainer = PPOTrainer(
        env="battle_v3",
        config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn":(
                    lambda agent_id, episode, **kwargs: "ppo_policy"),
            },
            # "model": {
                # "fcnet_hiddens": [64]
                # "conv_filters": [[13, 13, 5]],
                # "vf_share_layers": True,
            # },
            # "num_sgd_iter": 6,
            # "rollout_fragment_length": 20,
            # Number of timesteps collected for each SGD round. This defines the size
            # of each SGD epoch.
            # "train_batch_size": 60,
            # Total SGD batch size across all devices for SGD. This defines the
            # minibatch size within each epoch.
            # "sgd_minibatch_size": 20,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": 2,
            "num_workers": 10,
            "framework": "torch",
            # "log_level": "DEBUG",
        })

    print(ppo_trainer)

    for i in range(1000):
        print("== Iteration", i, "==")

        # improve the PPO policy
        print("-- PPO --")
        result_ppo = ppo_trainer.train()
        print(pretty_print(result_ppo))
