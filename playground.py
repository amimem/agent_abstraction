from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.mpe import simple_push_v2
from pettingzoo.mpe import simple_adversary_v2
import ray.rllib.contrib.maddpg as maddpg
# import supersuit
# Based on code from github.com/parametersharingmadrl/parametersharingmadrl

if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN
    def env_creator(args):
        env = supersuit.aec_wrappers.pad_observations(simple_adversary_v2.env()) 
        # env = simple_push_v2.env()
        return PettingZooEnv(env)

    env = env_creator({})
    register_env("simple", env_creator)

    # num_agents = env.num_agents
    # agent_ids = list(range(num_agents))

    # observation_space_dict = _make_dict(env.observation_space)
    # action_space_dict = _make_dict(env.action_space)

    # def gen_policy(i):
    #     return (
    #         None,
    #         env.observation_space_dict[i],
    #         env.action_space_dict[i],
    #         {
    #             "agent_id": i,
    #             "obs_space_dict": env.observation_space_dict,
    #             "act_space_dict": env.action_space_dict,
    #         }
    #     )

    # policies = {"policy_%d" %i: gen_policy(i) for i in range(len(env.observation_space_dict))}
    # policy_ids = list(policies.keys())

    tune.run(
        "DQN",
        # stop={"episodes_total": 60000},
        checkpoint_freq=10,
        config={
            # Enviroment specific
            "env": "simple",
            # General
            # "num_gpus": 1,
            # "num_workers": 2,
            # Method specific
            "multiagent": {
                "policies": set(env.agents),
                "policy_mapping_fn": (
                    lambda agent_id, episode, **kwargs: agent_id),
            }
            # "multiagent": {
            #         "policies": policies,
            #         "policy_mapping_fn": ray.tune.function(
            #             lambda i: policy_ids[i]
            #         )
            #     },
            # Method specific.
            # "multiagent": {
            #     # We only have one policy (calling it "shared").
            #     # Class, obs/act-spaces, and config will be derived
            #     # automatically.
            #     "policies": {"shared_policy"},
            #     # Always use "shared" policy.
            #     "policy_mapping_fn": (
            #         lambda agent_id, episode, **kwargs: "shared_policy"),
            # },
        },
    )
