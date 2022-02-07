from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from gym.spaces import Box

class MAgengtPettingZooEnv(PettingZooEnv):
    def __init__(self, env, flatten_dict_observations=False):

        # the overrridden reset function in called inside super, so flatten_dict_observations must be defined before that
        self.flatten_dict_observations = flatten_dict_observations

        super().__init__(env)

        # Get first action space, assuming all agents have equal space
        self.observation_space = self.env.state_space

        if self.flatten_dict_observations:
            new_shape = 1
            for e in self.observation_space.shape:
                new_shape *= e
                
            self.observation_space = Box(low=0., high=2., shape=(new_shape,))

    def step(self, action):
        self.env.step(action[self.env.agent_selection])
        obs_d = {}
        state_d = {}
        rew_d = {}
        done_d = {}
        info_d = {}
        while self.env.agents:
            obs, rew, done, info = self.env.last()
            state = self.env.state()
            a = self.env.agent_selection
            obs_d[a] = obs
            rew_d[a] = rew
            done_d[a] = done
            info_d[a] = info

            if self.flatten_dict_observations:
                state_d[a] = state.reshape([-1])
            else:
                state_d[a] = state

            if self.env.dones[self.env.agent_selection]:
                self.env.step(None)
            else:
                break

        all_done = not self.env.agents
        done_d["__all__"] = all_done

        return state_d, rew_d, done_d, info_d

    def reset(self):
        self.env.reset()
        return {
            self.env.agent_selection: self.env.state() if not self.flatten_dict_observations else self.env.state().reshape([-1])
        }


class MAgentParallelPettingZooEnv(ParallelPettingZooEnv):
    def __init__(self, env, flatten_dict_observations=False):
        super().__init__(env)
        
        # Get dictionaries of obs_spaces and act_spaces
        self.observation_space = self.par_env.state_space

    def step(self, action_dict):
        obss, rews, dones, infos = self.par_env.step(action_dict)
        state = self.par_env.state()
        for a in obss:
            obss[a] = state
        dones["__all__"] = all(dones.values())
        return obss, rews, dones, infos