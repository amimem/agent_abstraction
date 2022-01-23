import gym
from stable_baselines3 import DQN
from stable_baselines3.dqn import CnnPolicy
from pettingzoo.magent import battle_v3
import supersuit as ss

env = battle_v3(map_size=13)
model = DQN(CnnPolicy, env)
model.learn(total_timesteps=2000000)
model.save("dqn_policy")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()