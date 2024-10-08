# MARL - Agent Abstraction

This repository contains the source code used for our experiments on agent abstraction in multi-agent reinforcement learning. It includes, but is not limited to, the code necessary for reproducing the plots from our ICLR 2022 workshop paper, [Summarizing Societies: Agent Abstraction in Multi-Agent Reinforcement Learning](https://openreview.net/forum?id=Sc9ESMyTZ9).

## Mila Cluster First Time Setup:

If you are in the interactive mode:
```zsh
source /etc/profile
```
Then:
```zsh
module load python/3.7 pytorch/1.8.1
cd ~; git clone https://github.com/amimem/marl.git
cd marl; python -m venv ./env/marl_env
source ./env/marl_env/bin/activate
pip install --no-index -r requirements.txt
```

Once the setup is done, for future use just:
```zsh
module load python/3.7 pytorch/1.8.1
source ~/marl/.env/marl_env/bin/activate
```

And run the script that you want:
```zsh
python ray/playground.py
```


## Compute Canada Setup:
The code does not work on CC due to dependency and package build issues.

## Script Help:
For now `ray` includes all the scripts; main experiments are in `playgorund.py`, once the reliablity and usability of data is established, we can move the setting to separate files (`DQN.py`, `PPO.py` for example).

`magent_wrappers.py` includes the custom wrappers we wrote for RLLib + PettingZoo. Only the AEC (non-parallel) wrapper is tested.
