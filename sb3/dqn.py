import gym

from stable_baselines3 import DQN
from stable_baselines3.dqn import CnnPolicy
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from pettingzoo.magent import battle_v3
from pettingzoo.utils import to_parallel

from wandb.integration.sb3 import WandbCallback

import supersuit as ss
import wandb


config = {
    "policy_type": type(MlpPolicy),
    "total_timesteps": 5e6,
    "env_name": "battle_v3",
}
run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

env = battle_v3.env(map_size=20)
env = to_parallel(env)
env = ss.black_death_v2(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, num_cpus=0, base_class='stable_baselines3')
env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 50000 == 0, video_length=1000)
model = DQN(MlpPolicy, env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(total_timesteps=config["total_timesteps"],
            callback=WandbCallback(
                model_save_path=f"models/{run.id}",
                verbose=2,
                )
            )
model.save(f"dqn_policy_{run.id}")
run.finish()

# Test trained agent
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     if done.all():
#       obs = env.reset()