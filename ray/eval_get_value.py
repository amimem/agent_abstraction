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
from pathlib import Path


os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "1"
num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK'))
print("num cpus are ", num_cpus)

parser = argparse.ArgumentParser()
parser.add_argument('--path', metavar='P', type=str, help='path')
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

    policies = {
        "dqn_policy": (DQNTorchPolicy, obs_space, act_space, {}),
    }

    base_path = args.path
    p = Path(base_path)

    checkpoint_paths = [f for f in p.iterdir() if f.is_dir()]
    print("path is ", base_path)

    def get_config(output_path):
        return {
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
            "n_step": 3,
            "noisy": True,
            # "model": {
            #     # "fcnet_hiddens": [64],
            #     "conv_filters": [
            #         [32, [15, 15], 1],
            #         ],
            #     # "vf_share_layers": True,
            # },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": 1,
            # "num_workers": num_cpus-1,
            "evaluation_num_workers":num_cpus,
            "evaluation_num_episodes": 1,
            "framework": "torch",
            "output": output_path,
            # "log_level": "DEBUG",
        }


    for path in checkpoint_paths:

        json_exists = False
        path_str = str(path)

        try:
            json_file = [f for f in path.iterdir() if f.suffix == '.json']
            assert len(json_file) == 0
        except AssertionError:
            json_exists = True
            if len(json_file) == 1:
                print(f"{len(json_file)} json file(s) already exists")
            
        dqn_config = get_config(path_str)
        trainer=DQNTrainer(config=dqn_config)
        chekpoint_file = [f for f in path.iterdir() if f.suffix == '.tune_metadata']
        assert len(chekpoint_file) == 1
        chekpoint_file = os.path.join(path, chekpoint_file[0].stem)
        trainer.restore(chekpoint_file)

        models = {}
        for agent in set(env.agents):
            models[agent] = trainer.get_policy(agent).model

        if not json_exists:
            print(trainer.evaluate())

        for file in os.listdir(path):
            if file.endswith(".json"):
                json_path = os.path.join(path, file)
                print(json_path)
                json = JsonReader(json_path)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                ma_batch = json.next()
                p_batches = ma_batch.policy_batches
                s_batch = p_batches['red_0']

                states_path = Path(path_str + '_states.npy')
                if not states_path.exists():
                    with open(states_path, 'wb') as f:
                        np.save(f, s_batch['obs'])

                s_batch = s_batch.to_device(device, framework="torch")

                for agent in set(env.agents):
                    value_path = Path(path_str + f'_values_{agent}.npy')
                    if not value_path.exists():
                        model_out = models[agent](s_batch)
                        v = models[agent].get_state_value(model_out[0])
                        with open(value_path, 'wb') as f:
                            np.save(f, v.cpu().detach().numpy())