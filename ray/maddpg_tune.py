from statistics import mode
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.mpe import simple_tag_v2
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy, PPOTorchPolicy
from argparse import ArgumentParser
import os
import ray

os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "1"
num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK'))
# print("num cpus are ", num_cpus)

# Based on code from github.com/parametersharingmadrl/parametersharingmadrl
if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

    ray.init(include_dashboard=False, num_cpus=num_cpus, num_gpus=1)
    assert ray.is_initialized() == True

    def env_creator(args):
        env = simple_tag_v2.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=False)
        # env = to_parallel(env)
        return ParallelPettingZooEnv(env)

    env = env_creator({})
    register_env("simple_tag", env_creator)
    print(set(env.agents), len(env.agents))

    obs_space = env.observation_space
    act_space = env.action_space

    def gen_policies(agents_list):
        policies = {}
        policies["policy_adversary"] = (PPOTorchPolicy, obs_space, act_space, {})
        policies["policy_agent"] = (PPOTorchPolicy, obs_space, act_space, {})
        return policies

    policies = gen_policies(env.agents)

    def policy_map(agent_id, episode, **kwargs):
        # if args.team:
        assert isinstance(agent_id, str)
        if "adversary" in agent_id:
            return "policy_adversary"
        if "agent" in agent_id:
            return "policy_agent"

    save_dir = os.getenv('SLURM_TMPDIR')

    policies = gen_policies(env.agents)

    print(policies)

    def policy_map(agent_id, episode, **kwargs):
        assert isinstance(agent_id, str)
        if "adversary" in agent_id:
            return "policy_adversary"
        if "agent" in agent_id:
            return "policy_agent"

    save_dir = os.getenv('SLURM_TMPDIR')

    dqn_trainer = PPOTrainer(
        env="simple_tag",
        config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_map,
            },
            "vf_clip_param": 30,
            # "render_env": True,
            # Evaluate once per training iteration.
            # "evaluation_interval": 1,
            # Run evaluation on (at least) two episodes
            # "evaluation_duration": 1,
            # ... using one evaluation worker (setting this to 0 will cause
            # evaluation to run on the local evaluation worker, blocking
            # training until evaluation is done).
            # "evaluation_num_workers": 0,
            # Special evaluation config. Keys specified here will override
            # the same keys in the main config, but only for evaluation.
            # "evaluation_config": {
                # Render the env while evaluating.
                # Note that this will always only render the 1st RolloutWorker's
                # env and only the 1st sub-env in a vectorized env.
                # "render_env": True,
            # },
            # "log_level": "DEBUG",
        }
        )

    print(dqn_trainer)

    for i in range(100000):
        print("== Iteration", i, "==")

        # improve the DQN policy
        print("-- DQN --")
        result_dqn = dqn_trainer.train()
        print(pretty_print(result_dqn))

        if i % 100 == 0:
            checkpoint = dqn_trainer.save()
            print("checkpoint saved at", checkpoint)