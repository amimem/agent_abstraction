from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.magent import battle_v3
from pettingzoo.utils import to_parallel
from magent_wrappers import MAgengtPettingZooEnv, MAgentParallelPettingZooEnv
from ray.rllib.agents.dqn import DQNTrainer, DQNTorchPolicy
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

    policies = {
        "dqn_policy": (DQNTorchPolicy, obs_space, act_space, {}),
    }

    save_dir = os.getenv('SLURM_TMPDIR')

    dqn_trainer = DQNTrainer(
        env="battle_v3",
        config={
            "record_env": True,
            "multiagent": {
                "policies": set(env.agents),
                "policy_mapping_fn":(
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
            "model": {
                # "fcnet_hiddens": [64],
                "conv_filters": [32, [15, 15], 1],
                # "vf_share_layers": True,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": 1,
            "num_workers": 0,
            "framework": "torch",
            # "log_level": "DEBUG",
        })

    tune.run(
        dqn_trainer,
        stop={"episodes_total": 1000},
        checkpoint_freq=100,
        local_dir=save_dir,
        checkpoint_freq=50,
        checkpoint_at_end=True,
        )
