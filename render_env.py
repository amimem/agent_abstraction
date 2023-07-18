import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
import datetime
import pickle5 as pickle
import os
import argparse


def load_checkpoint_data(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        df_whole = pickle.load(f)
    return df_whole


def get_episode_ids(df_whole, n=10, selection="first"):
    # separate df_whole into one dataframe per agent
    df_agent_0 = df_whole[df_whole["agent_index"] == 0]  # 0,1,2 are predators

    if selection == "max_reward":
        # sum the rewards of each agent across episodes and sort them in descending order
        rewards_agent_0 = (
            df_agent_0.groupby("eps_id")["rewards"]
            .sum()
            .sort_values(ascending=False)
        )

    elif selection == "min_reward":
        rewards_agent_0 = (
            df_agent_0.groupby("eps_id")["rewards"]
            .sum()
            .sort_values(ascending=True)
        )

    elif selection == "first":
        rewards_agent_0 = df_agent_0.groupby("eps_id")["rewards"].sum()

    elif selection == "random":
        rewards_agent_0 = (
            df_agent_0.groupby("eps_id")["rewards"].sum().sample(frac=1)
        )
    else:
        raise ValueError(
            "selection must be 'max_reward', 'min_reward', 'first' or 'random'"
        )

    return rewards_agent_0.index.unique()[:n]


def select_episodes(df_whole, selection=None, n=10):
    if selection is None:
        print("No episodes provided, returning all episodes...")
        return df_whole.sort_values(
            by=["eps_id", "t", "agent_index"], inplace=True
        )

    elif selection == "max_reward":
        return (
            df_whole[
                df_whole["eps_id"].isin(
                    get_episode_ids(df_whole, n, selection="max_reward")
                )
            ]
            .sort_values(by=["eps_id", "t", "agent_index"], inplace=False)
            .drop_duplicates(["eps_id", "agent_index", "t"])
        )

    elif selection == "min_reward":
        return (
            df_whole[
                df_whole["eps_id"].isin(
                    get_episode_ids(df_whole, n, selection="min_reward")
                )
            ]
            .sort_values(by=["eps_id", "t", "agent_index"], inplace=False)
            .drop_duplicates(["eps_id", "agent_index", "t"])
        )

    elif selection == "first":
        return (
            df_whole[
                df_whole["eps_id"].isin(
                    get_episode_ids(df_whole, n, selection="first")
                )
            ]
            .sort_values(by=["eps_id", "t", "agent_index"], inplace=False)
            .drop_duplicates(["eps_id", "agent_index", "t"])
        )

    elif selection == "random":
        return (
            df_whole[
                df_whole["eps_id"].isin(
                    get_episode_ids(df_whole, n, selection="random")
                )
            ]
            .sort_values(by=["eps_id", "t", "agent_index"], inplace=False)
            .drop_duplicates(["eps_id", "agent_index", "t"])
        )

    # elif eps_ids is iterable, such as a list or nparray:
    elif isinstance(selection, (list, np.ndarray)):
        return df_whole[df_whole["eps_id"].isin(selection)].sort_values(
            by=["eps_id", "t", "agent_index"], inplace=True
        )

    else:
        raise ValueError(
            "selection must be None, 'max_reward', 'min_reward', 'first', 'random' or an iterable such as a list or nparray"
        )


def plot_action_probs(df_whole, filename, train_test_split=0.9):
    # get a list of all the episode ids
    episode_ids = df_whole["eps_id"].unique()

    # take the first half of the episode ids
    episode_ids_train = episode_ids[: int(train_test_split * len(episode_ids))]
    # take the last 10% of the episode ids
    episode_ids_test = episode_ids[int(train_test_split * len(episode_ids)) :]
    # select the first 90% of the episodes as the training set
    df_train = df_whole[df_whole["eps_id"].isin(episode_ids_train)]
    df_test = df_whole[df_whole["eps_id"].isin(episode_ids_test)]

    # plot a histogram of action probs for each agent across all episodes in the training set
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for agent_id in range(4):
        ax[agent_id].hist(
            df_train[df_train["agent_index"] == agent_id]["action_prob"],
            bins=100,
        )
        ax[agent_id].set_title(f"Agent {agent_id} action_prob, train set")
    plt.savefig(filename + "_train_action_probs.png")

    # plot a histogram of action probs for each agent across all episodes in the test set
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for agent_id in range(4):
        ax[agent_id].hist(
            df_test[df_test["agent_index"] == agent_id]["action_prob"], bins=100
        )
        ax[agent_id].set_title(f"Agent {agent_id} action_prob, test set")
    # save plots to file
    plt.savefig(filename + "_test_action_probs.png")
    print(f"Saved action probability plots to {filename}_test_action_probs.png")


def plot_reward_distribution(df_whole, filename, train_test_split=0.9):
    # get a list of all the episode ids
    episode_ids = df_whole["eps_id"].unique()

    # take the first half of the episode ids
    episode_ids_train = episode_ids[: int(train_test_split * len(episode_ids))]
    # take the last 10% of the episode ids
    episode_ids_test = episode_ids[int(train_test_split * len(episode_ids)) :]
    # select the first 90% of the episodes as the training set
    df_train = df_whole[df_whole["eps_id"].isin(episode_ids_train)]
    # select the last 10% of the episodes as the test set
    df_test = df_whole[df_whole["eps_id"].isin(episode_ids_test)]

    # plot a histogram of action probs for each agent across all episodes in the training set
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for agent_id in range(4):
        ax[agent_id].hist(
            df_train[df_train["agent_index"] == agent_id]
            .groupby("eps_id")["rewards"]
            .sum(),
            bins=20,
        )
        ax[agent_id].set_title(f"Agent {agent_id} rewards, train set")
        ax[agent_id].set_yscale("log")
        ax[agent_id].set_xlim(
            min(
                df_train[df_train["agent_index"] == agent_id]
                .groupby("eps_id")["rewards"]
                .sum()
            ),
            max(
                df_train[df_train["agent_index"] == agent_id]
                .groupby("eps_id")["rewards"]
                .sum()
            ),
        )
    plt.savefig(filename + "_train_reward_distribution.png")
    print(
        f"Saved reward distribution plots to {filename}_train_reward_distribution.png"
    )

    # plot a histogram of action probs for each agent across all episodes in the test set
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for agent_id in range(4):
        ax[agent_id].hist(
            df_test[df_test["agent_index"] == agent_id]
            .groupby("eps_id")["rewards"]
            .sum(),
            bins=20,
        )
        ax[agent_id].set_title(f"Agent {agent_id} rewards, test set")
        ax[agent_id].set_yscale("log")
        ax[agent_id].set_xlim(
            min(
                df_train[df_train["agent_index"] == agent_id]
                .groupby("eps_id")["rewards"]
                .sum()
            ),
            max(
                df_train[df_train["agent_index"] == agent_id]
                .groupby("eps_id")["rewards"]
                .sum()
            ),
        )
    plt.savefig(filename + "_test_reward_distribution.png")


def render_episode_selection(
    df_whole, selection, filename, n=10, num_interpolation_points=10
):
    print(f"Rendering {n} {selection} episodes...")
    plotting_eps = select_episodes(df_whole, selection=selection, n=n)

    num_episodes = len(plotting_eps["eps_id"].unique())
    num_agents = len(plotting_eps["agent_index"].unique())
    num_timesteps = len(plotting_eps["t"].unique())

    # reshape observations to be ndarray (num_episodes, num_timesteps, num_agents, _observation_size)
    observations = np.stack(plotting_eps["obs"].to_numpy())
    observations = np.reshape(
        observations,
        (
            num_episodes,
            num_timesteps,
            num_agents,
            -1,
        ),
    )

    # only keep one agent's rewards per episode since it's a zero-sum game
    rewards = plotting_eps["rewards"].values.reshape(
        num_episodes, -1, num_agents
    )
    rewards = rewards[:, :, -1]

    # get the agent velocities and positions, and the entities positions from the observations
    agent_velocities = observations[:, :, :, 0:2]
    agent_positions = observations[:, :, :, 2:4]
    entities_positions = observations[:, :, 0, 4:8].reshape(
        num_episodes, num_timesteps, -1, 2
    ) + agent_positions[:, :, 0, :].reshape(num_episodes, num_timesteps, 1, 2)

    # join the positions and entities_positions to be (num_episodes, num_timesteps, num_agents + num_entities, 2)
    positions = np.concatenate([agent_positions, entities_positions], axis=2)

    num_episodes, num_timesteps, num_agents, _num_coordinates = positions.shape
    x = np.arange(num_timesteps)

    # interpolate the positions and rewards in each batch
    positions_interp = np.zeros(
        (
            num_episodes,
            (num_timesteps) * num_interpolation_points + 1,
            num_agents,
            2,
        )
    )

    rewards_interp = np.zeros(
        (
            num_episodes,
            (num_timesteps) * num_interpolation_points + 1,
        )
    )

    timesteps_interp = np.linspace(
        0, num_timesteps, num_timesteps * num_interpolation_points + 1
    )

    # TODO: vectorize this
    for episode in range(num_episodes):
        for agent_id in range(num_agents):
            for axis in range(2):
                interp_func = interp1d(
                    x,
                    positions[episode, :, agent_id, axis],
                    kind="linear",
                    fill_value="extrapolate",
                )
                positions_interp[episode, :, agent_id, axis] = interp_func(
                    timesteps_interp
                )

        # interpolate the rewards by nearest value filling
        rewards_interp_func = interp1d(
            x, rewards[episode, :], kind="nearest", fill_value="extrapolate"
        )
        rewards_interp[episode] = rewards_interp_func(timesteps_interp)

    # flatten the positions and rewards across episodes, repeat the timesteps
    positions_interp = positions_interp.reshape(-1, num_agents, 2)
    timesteps_interp = np.tile(timesteps_interp, num_episodes)
    rewards_interp = rewards_interp.reshape(-1)

    # render the animation
    fig, ax = plt.subplots()

    ax.set_xlim(
        positions_interp[:, :, 0].min(), positions_interp[:, :, 0].max()
    )
    ax.set_ylim(
        positions_interp[:, :, 1].min(), positions_interp[:, :, 1].max()
    )

    # predators are red dots, prey is a green dot, entities are blue squares
    scatters = [
        ax.scatter(
            positions_interp[0, agent_id, 0],
            positions_interp[0, agent_id, 1],
            s=np.pi * 18**2,
            color="red",
            label=f"Predator {agent_id}",
        )
        for agent_id in [0, 1, 2]
    ]

    scatters.append(
        ax.scatter(
            positions_interp[0, 3, 0],
            positions_interp[0, 3, 1],
            label="Prey",
            s=np.pi * 12**2,
            color="green",
        )
    )

    scatters.append(
        ax.scatter(
            positions_interp[0, 4, 0],
            positions_interp[0, 4, 1],
            label="Entity 1",
            marker="s",
            s=400,
            color="blue",
        )
    )

    scatters.append(
        ax.scatter(
            positions_interp[0, 5, 0],
            positions_interp[0, 5, 1],
            label="Entity 2",
            marker="s",
            s=400,
            color="blue",
        )
    )

    ax.set_title(f"Predators {selection} - {n} episodes", fontsize=16)

    # selection is title
    timestep_text = ax.text(
        0.05,
        0.95,
        f"Timestep: {timesteps_interp[0]}",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
    )

    # add the timesteps to the plot

    # add the rewards to the plot
    reward_text = ax.text(
        0.05,
        0.90,
        f"Reward: {rewards_interp[0]}",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
    )

    # Each interpolated timestep is a frame in the animation
    frames = positions_interp.shape[0]

    ani = animation.FuncAnimation(
        fig,
        update_plot,
        frames=frames,
        fargs=(
            positions_interp,
            scatters,
            timesteps_interp,
            timestep_text,
            rewards_interp,
            reward_text,
        ),
        interval=1000,
    )

    ani.save(
        f"{filename}_render_{selection}_{n}_episodes.mp4",
        writer="ffmpeg",
        fps=50,
    )
    print(f"{filename}_render_{selection}_{n}_episodes.mp4")
    plt.close()


def update_plot(
    frame,
    coordinates,
    scatters,
    timesteps,
    timestep_text,
    rewards_interp,
    reward_text,
):
    # for agent_id, scatter in zip(agent_ids, scatters[0:4]):
    #     scatter.set_offsets(positions[frame, agent_id])
    for agent_id, scatter in enumerate(scatters):
        scatter.set_offsets(coordinates[frame, agent_id])
    # update timestep_text to show the current timestep with one decimal place
    timestep_text.set_text(f"Timestep: {timesteps[frame]:.1f}")
    reward_text.set_text(f"Reward: {rewards_interp[frame]:.1f}")

    return scatters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="output-2023-06-15_17-17-14_worker-4_0_data.pkl",
        help="The name of the file to load",
    )
    parser.add_argument(
        "--n",
        type=int,
        help="The number of episodes to render",
    )
    args = parser.parse_args()

    # get the file name and number of episodes to render from the command line
    if args.file is not None:
        file = args.file
        print(f"Loading data from {file}.")
    else:
        raise ValueError("Please specify a file to load with --file.")
    if args.n is not None:
        n = args.n
    else:
        print(
            "No number of rendering episodes specified with --n, defaulting to 1..."
        )
        n = 1

    # load the data from the checkpoint file
    df_whole = load_checkpoint_data(file)
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # create folder in root directory of repo to save the plots
    if not os.path.exists(f"plots/{file.split('.')[0]}"):
        print(f"Creating folder plots/{file.split('.')[0]}")
        os.makedirs(f"plots/{file.split('.')[0]}")

    # change the working directory to the plots folder
    os.chdir(f"plots/{file.split('.')[0]}")

    # #plot the action probs
    plot_action_probs(
        df_whole,
        filename=file.split(".")[0] + datetime_str,
    )

    # #plot the reward distribution
    plot_reward_distribution(
        df_whole,
        filename=file.split(".")[0] + datetime_str,
    )

    # render the first n episodes, the n episodes with the highest reward, and the n episodes with the lowest reward
    render_episode_selection(
        df_whole,
        selection="random",
        filename=file.split(".")[0] + datetime_str,
        n=n,
    )

    render_episode_selection(
        df_whole,
        selection="max_reward",
        filename=file.split(".")[0] + datetime_str,
        n=n,
    )

    render_episode_selection(
        df_whole,
        selection="min_reward",
        filename=file.split(".")[0] + datetime_str,
        n=n,
    )
