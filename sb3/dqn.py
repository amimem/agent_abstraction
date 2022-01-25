import gym
from stable_baselines3 import DQN
from stable_baselines3.dqn import CnnPolicy
from stable_baselines3.dqn import MlpPolicy
from pettingzoo.magent import battle_v3
from pettingzoo.utils import to_parallel
from wandb.integration.sb3 import WandbCallback
import supersuit as ss
import wandb


config = {
    "policy_type": type(MlpPolicy),
    "total_timesteps": 1e4,
    "env_name": "battle_v3",
}
run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

env = battle_v3.env(map_size=13)
env = to_parallel(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, num_cpus=0, base_class='stable_baselines3')
model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=config["total_timesteps"],
            callback=WandbCallback(
                model_save_path=f"models/{run.id}",
                verbose=2,
                )
            )
model.save("dqn_policy")
run.finish()

# Test trained agent
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     if done.all():
#       obs = env.reset()