
"""
Taken from rlpyt examples

Runs one instance of the MultiGrid environment and optimizes using DQN algorithm.
Can use a GPU for the agent (applies to both sample and train). No parallelism
employed, so everything happens in one python process; can be easier to debug.

The kwarg snapshot_mode="last" to logger context will save the latest model at
every log point (see inside the logger for other options).

In viskit, whatever (nested) key-value pairs appear in config will become plottable
keys for showing several experiments.  If you need to add more after an experiment, 
use rlpyt.utils.logging.context.add_exp_param().

"""

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.envs.gym import make as gym_make
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context

from gym.envs.registration import register
from gym_multigrid.envs.collect_game import CollectGameEnv
from gym_multigrid.envs.soccer_game import SoccerGameEnv

register(
    id='multigrid-soccer-v0',
    entry_point='gym_multigrid.envs:SoccerGameEnv',
    kwargs={'size': None, 'zero_sum': True, 'partial_obs': False}
)

register(
    id='multigrid-collect-v0',
    entry_point='gym_multigrid.envs:CollectGameEnv',
    kwargs={'size': 10, 'partial_obs': False},

)

def build_and_train(game="multigrid-collect-v0", run_ID=0, cuda_idx=None):
    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=game),
        eval_env_kwargs=dict(id=game),
        batch_T=4,  # Four time-steps per sampler iteration.
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )
    algo = DQN(min_steps_learn=1e3)  # Run with defaults.
    agent = DqnAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = dict(game=game)
    name = "dqn_" + game
    log_dir = "example_1"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='multigrid-collect-v0')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    args = parser.parse_args()
    build_and_train(
        game=args.game,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
