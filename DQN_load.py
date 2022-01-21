import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.magent import battle_v3
from pettingzoo.utils import to_parallel
from magent_wrappers import MAgengtPettingZooEnv, MAgentParallelPettingZooEnv
from ray.rllib.agents.dqn import DQNTrainer, DQNTorchPolicy
from ray.rllib.offline.json_reader import JsonReader
import torch
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', metavar='P', type=str, help='path')
parser.add_argument('--checkpoint', metavar='C', type=str, help='checkpoint')
parser.add_argument('--json', metavar='J', type=str, help='json')
parser.add_argument('--eval', metavar='e', type=bool, help='eval', default=False)
args = parser.parse_args()

def on_train_result(self, *, trainer: "Trainer", result: dict,
                        **kwargs) -> None:
        """Called at the end of Trainable.train().

        Args:
            trainer: Current trainer instance.
            result: Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """

        if self.legacy_callbacks.get("on_train_result"):
            self.legacy_callbacks["on_train_result"]({
                "trainer": trainer,
                "result": result,
            })


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

    obs_space = env.observation_space
    act_space = env.action_space

    policies = {
        "dqn_policy": (DQNTorchPolicy, obs_space, act_space, {}),
    }

    base_path = args.path
    checkpoint_path = os.path.join(base_path, args.checkpoint)
    print("path is ", base_path)

    dqn_config = {
        "env":"battle_v3",
        # "record_env": True,
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
            "conv_filters": [
                [32, [15, 15], 1],
                ],
            # "vf_share_layers": True,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 1,
        "num_workers": 1,
        "evaluation_num_workers":1 if args.eval else 0,
        "evaluation_num_episodes": 1,
        "framework": "torch",
        "output": base_path,
        # "log_level": "DEBUG",
    }

    
    trainer=DQNTrainer(config=dqn_config)
    trainer.restore(checkpoint_path)

    models = {}
    for agent in set(env.agents):
        models[agent] = trainer.get_policy(agent).model
    
    # print(models['red_0'])

    if args.eval:
        print(trainer.evaluate())

    if not args.eval:
        prefix = args.checkpoint
        json_path = os.path.join(base_path, args.json)
        json = JsonReader(json_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ma_batch = json.next()
        p_batches = ma_batch.policy_batches
        s_batch = p_batches['red_0']

        with open(os.path.join(base_path, prefix + '_states.npy'), 'wb') as f:
            np.save(f, s_batch['obs'])

        s_batch = s_batch.to_device(device, framework="torch")

        for agent in set(env.agents):
            model_out = models[agent](s_batch)
            v = models[agent].get_state_value(model_out[0])
            with open(os.path.join(base_path, prefix + f'_values_{agent}.npy'), 'wb') as f:
                np.save(f, v.cpu().detach().numpy())
