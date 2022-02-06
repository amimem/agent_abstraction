import ray
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.utils import to_parallel
from pettingzoo.magent import battle_v3
from ray.rllib.agents.dqn import DQNTrainer, DQNTFPolicy, DQNTorchPolicy
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from magent_wrappers import MAgengtPettingZooEnv, MAgentParallelPettingZooEnv
import os
from supersuit import flatten_v0

os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "1"

if __name__ == "__main__":

    # set local_mode=True if OOM
    ray.init(include_dashboard=False, num_cpus=10)

    def env_creator(args):
        env = battle_v3.env(map_size=15, minimap_mode=False)
        return MAgengtPettingZooEnv(env)

    env = env_creator({})
    register_env("battle_v3", env_creator)
    print(set(env.agents))

    obs_space = env.observation_space
    act_space = env.action_space

    # print(obs_space)
    # print(act_space)

    policies = {
        "dqn_policy": (DQNTorchPolicy, obs_space, act_space, {}),
    }

    print(policies)

    dqn_trainer = DQNTrainer(
        env="battle_v3",
        config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn":(
                    lambda agent_id, episode, **kwargs: "dqn_policy"),
            },
            "model": {
                # "fcnet_hiddens": [64],
                "conv_filters": [[32, [15, 15], 1]],
                # "vf_share_layers": True,
            },
            # "num_sgd_iter": 6,
            # "rollout_fragment_length": 20,
            # Number of timesteps collected for each SGD round. This defines the size
            # of each SGD epoch.
            # "train_batch_size": 60,
            # Total SGD batch size across all devices for SGD. This defines the
            # minibatch size within each epoch.
            # "sgd_minibatch_size": 20,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": 1,
            "num_workers": 9,
            "framework": "torch",
            # "log_level": "DEBUG",
        })

    print(dqn_trainer)

    for i in range(1000):
        print("== Iteration", i, "==")

        # improve the DQN policy
        print("-- DQN --")
        result_dqn = dqn_trainer.train()
        print(pretty_print(result_dqn))

        if i % 50 == 0:
            checkpoint = dqn_trainer.save()
            print("checkpoint saved at", checkpoint)
