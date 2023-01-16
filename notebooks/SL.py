import torch
import torch.nn as nn
from torch.utils.data import Sampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import glob
import argparse
import random

# from torchmetrics import Accuracy

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

n_input = 20
n_hidden = 256
n_out = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device", flush=True)

# get arguments using argparse, such as path, batch size, learning rate, split start and a binary flag for whether to use the separated data or not, the seed to use

parser = argparse.ArgumentParser(description='Train a neural network to predict actions from observations')
parser.add_argument('--path', type=str, default='/home/mila/m/memariaa/scratch', help='path to the data')
parser.add_argument('--batch_size', type=int, default=25, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--split_start', type=float, default=0.0, help='start of the validation split')
parser.add_argument('--num_models', type=int, default=0, help='whether to use the separated data or not')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--sampler', type=str, default='random', help='sampler to use')
args = parser.parse_args()

seed: int = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

epochs = 1000
learning_rate = args.learning_rate
batch_size = args.batch_size
split_start = args.split_start
sampler = args.sampler
num_models: int = args.num_models

# %%
# find all pickle files in the current directory using glob

path = args.path
# all_files = glob.glob(path + "/*.pkl")

# # create a list of dataframes
# li = []

# for filename in all_files:
#     df = pd.read_pickle(filename)
#     li.append(df)

# concatenate the list of dataframes into one dataframe
# data = pd.concat(li, axis=0, ignore_index=True)
data = pd.read_hdf(f"{path}/new/data.h5", key="df")
# try:
#     # Attempt to load the pickle file using the latest version of pandas.
#     data = pd.read_pickle('/home/mila/m/memariaa/scratch/new/data.pkl')
# except AttributeError:
#     # If an AttributeError is raised, it is likely because the pickle file was
#     # created using an older version of pandas. In this case, we can try
#     # loading the file using an older version of the pickle protocol.
#     with open('/home/mila/m/memariaa/scratch/new/data.pkl', 'rb') as f:
#         data = pickle.load(f, fix_imports=True, encoding="bytes")
#     data = pd.DataFrame(data)

# %%
# save observation column of arrays to a numpy array
obs = np.array(data['obs'].to_list())

# %%
act = np.array(data['actions'].to_list())

# %%
ids = np.array(data['agent_index'].to_list())

# %%
class SequentialBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        
    def __iter__(self):
        n = len(self.data_source)
        #  starting index should have remainder of 0 
        idx = [i for i in range(0, n, self.batch_size) if i % self.batch_size == 0]
        random.shuffle(idx)
        # yield sequential batches 
        for i in range(0, len(idx)):
            yield list(range(idx[i], idx[i] + self.batch_size))
            
    def __len__(self):
        return len(self.data_source) // self.batch_size


def get_dataloaders(data, num_models = num_models, split_start = split_start, batch_size = batch_size, sampler = sampler):

    # Shuffling by episode
    # groups = [data for _, data in data.groupby('eps_id')]
    # np.random.shuffle(groups)
    # data = pd.concat(groups).reset_index(drop=False)

    ratios=[split_start, 0.8, 0.9]

    data = data.sort_values(by=['eps_id','agent_index', 't'],ignore_index=True)
    data = data.rename(columns={"action_dist_inputs": "logits"})
    data['probs'] = data['logits'].transform(scipy.special.softmax)

    # do train test validation split on the data
    # the data is split into ratios[0:1] train, ratios[1:2] test, ratios[2:]validation
    # the data is shuffled before splitting
    # splitting dataset by episodes
    num_episodes = len(data['eps_id'].unique())
    length_of_epi = max(data['t'].unique()) + 1
    num_agents = len(data['agent_index'].unique())
    _, train, test, val = np.split(data, [int(ratios[0]*length_of_epi*num_agents*num_episodes),int(ratios[1]*length_of_epi*num_agents*num_episodes), int(ratios[2]*length_of_epi*num_agents*num_episodes)])


    print("train shape: ", train.shape, flush=True)
    print("test shape: ", test.shape, flush=True)
    print("val shape: ", val.shape, flush=True)


    # create test, train and validation dataloaders
    # the dataloaders will be used to train the neural network
    # the dataloaders will return a batch of observations and actions
    # the batch size is set to 100
    # the shuffle parameter is set to False so that the data is shuffled before each epoch

    all_train_dataloaders = []
    all_test_dataloaders = []
    all_val_dataloaders = []

    # 'separated' decides size of train, test and validation sets. 
    # If False, obs and actions are concatenated for separate agents. 
    # If False, obs and actions are separate for separate agents.

    train_partitions = []
    test_partitions = []
    val_partitions = []

    if num_models == 2:
        teams_idx = [[0,1,2], [3]]
        for idx in teams_idx:
            # if agent idx is in idx, then it is in the team
            train_partitions.append(train[train['agent_index'].isin(idx)])
            test_partitions.append(test[test['agent_index'].isin(idx)])
            val_partitions.append(val[val['agent_index'].isin(idx)])
    elif num_models == 1:
        train_partitions.append(train)
        test_partitions.append(test)
        val_partitions.append(val)
    elif num_models == num_agents:
        for agent_ind in np.arange(num_agents):
            train_partitions.append(train[train['agent_index'] == agent_ind])
            test_partitions.append(test[test['agent_index'] == agent_ind])
            val_partitions.append(val[val['agent_index'] == agent_ind])
    else:
        raise ValueError("num_models must be 1, 2 or num_agents")


    for i in np.arange(num_models):

        obs_train, obs_test, obs_val = np.array(train_partitions[i]['obs'].to_list()), np.array(test_partitions[i]['obs'].to_list()), np.array(val_partitions[i]['obs'].to_list())
        act_train, act_test, act_val = np.array(train_partitions[i]['actions'].to_list()), np.array(test_partitions[i]['actions'].to_list()), np.array(val_partitions[i]['actions'].to_list())
        act_prob_train, act_prob_test, act_prob_val = np.array(train_partitions[i]['action_prob'].to_list()), np.array(test_partitions[i]['action_prob'].to_list()), np.array(val_partitions[i]['action_prob'].to_list())
        ids_train, ids_test, ids_val = np.array(train_partitions[i]['agent_index'].to_list()), np.array(test_partitions[i]['agent_index'].to_list()), np.array(val_partitions[i]['agent_index'].to_list())
        logits_train, logits_test, logits_val = np.array(train_partitions[i]['logits'].to_list()), np.array(test_partitions[i]['logits'].to_list()), np.array(val_partitions[i]['logits'].to_list())
        probs_train, probs_test, probs_val = np.array(train_partitions[i]['probs'].to_list()), np.array(test_partitions[i]['probs'].to_list()), np.array(val_partitions[i]['probs'].to_list())


        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(obs_train), torch.from_numpy(act_train),torch.from_numpy(act_prob_train), torch.from_numpy(ids_train),torch.from_numpy(logits_train),torch.from_numpy(probs_train))
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(obs_test), torch.from_numpy(act_test), torch.from_numpy(act_prob_test), torch.from_numpy(ids_test),torch.from_numpy(logits_test),torch.from_numpy(probs_test))
        val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(obs_val), torch.from_numpy(act_val), torch.from_numpy(act_prob_val), torch.from_numpy(ids_val),torch.from_numpy(logits_val),torch.from_numpy(probs_val))

        if sampler == 'custom':
            train_sampler = SequentialBatchSampler(train_dataset, batch_size=batch_size)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler)

            test_sampler = SequentialBatchSampler(test_dataset, batch_size=batch_size)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_sampler)

            val_sampler = SequentialBatchSampler(val_dataset, batch_size=batch_size)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler)

        else:
            if sampler == 'random':
                shuffle_flag = True
            else:
                shuffle_flag = False
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_flag)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_flag)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_flag)

        all_train_dataloaders.append(train_dataloader)
        all_test_dataloaders.append(test_dataloader)
        all_val_dataloaders.append(val_dataloader)

    return all_train_dataloaders, all_test_dataloaders, all_val_dataloaders

# %%
class PolicyNetwork(nn.Module):
    def __init__(self, index):
        super(PolicyNetwork, self).__init__()
        self.index = index
        self.model = nn.Sequential(nn.Linear(n_input, n_hidden, bias=True),
                      nn.Tanh(),
                      nn.Linear(n_hidden, n_hidden, bias=True),
                      nn.Tanh(),
                      nn.Linear(n_hidden, n_hidden, bias=True),
                      nn.Tanh(),
                      nn.Linear(n_hidden, n_out, bias=True))

    def forward(self, x):
        logits = self.model(x)
        return logits


def policy_model(num_networks = 1):
    
    model = [PolicyNetwork(i).to(device) for i in range(num_networks)]
    # print(model)

    loss_function = nn.CrossEntropyLoss()  
    optimizer = [torch.optim.SGD(m.parameters(), lr=learning_rate) for m in model]

    return model, loss_function, optimizer

def train_loop(dataloaders, models, loss_fn, optimizers):

    epoch_losses = []
    epoch_accuracies = []

    for model, optimizer, dataloader in zip(models, optimizers, dataloaders):

        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        running_train_loss = 0.0 
        running_accuracy = 0.0 

        for batch, (obs, act, prob, idx, logits, probs) in enumerate(dataloader):

            # Compute prediction and loss
            obs = obs.to(device)
            pred = model(obs)
            act = act.to(device)
            loss = loss_fn(pred, act)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            correct = (predicted == act).sum().item()
            running_accuracy += correct

            if batch % 1000 == 0:
                loss, current = loss.item(), batch * len(obs)
                print(f"model: {model.index} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]",flush=True)
        
        epoch_loss = running_train_loss / num_batches
        epoch_losses.append(epoch_loss)
        epoch_accuracy = running_accuracy / size
        epoch_accuracies.append(epoch_accuracy)

        print(f"model: {model.index} Train loss: {epoch_loss:>8f} Train accuracy: {epoch_accuracy:>8f}",flush=True)

    return epoch_losses, epoch_accuracies

def test_loop(dataloaders, models, loss_fn):

    epoch_losses = []
    epoch_accuracies = []

    with torch.no_grad():
        for model, dataloader in zip(models, dataloaders):

            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            
            test_loss, correct = 0, 0
            for batch, (obs, act, prob, idx, logits, probs) in enumerate(dataloader):

                obs = obs.to(device)
                pred = model(obs)
                act = act.to(device)
                test_loss += loss_fn(pred, act).item()
                correct += (pred.argmax(1) == act).type(torch.float).sum().item()

            test_loss /= num_batches
            correct /= size

            epoch_losses.append(test_loss)
            epoch_accuracies.append(correct)

            print(f"Mode: {model.index} \n Test Error: \n Accuracy:{(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n",flush=True)
            
    return epoch_losses, epoch_accuracies

if __name__ == "__main__":

    policies, loss_fn, optimizers = policy_model(num_networks= num_models)

    train_loader, test_loader, val_loader = get_dataloaders(data=data)

    train_losses = []
    train_accuracies = []

    test_losses = []
    test_accuracies = []

    for t in range(epochs):

        print(f"Epoch {t+1}\n-------------------------------", flush=True)
        train_loss, train_acc =  train_loop(train_loader, policies, loss_fn, optimizers)
        test_loss, test_acc = test_loop(test_loader, policies, loss_fn)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # save model
        if (t+1) % 200 == 0:
            for i in range(len(policies)): torch.save(policies[i].state_dict(), f"{path}/model_p{i}_{t+1}_{split_start}_{num_models}_{seed}_{learning_rate}_{sampler}.pth") 
            print("Saved PyTorch Model State to model.pth", flush=True)

            # save train and test losses and accuracies as numpy arrays
            np.save(f'{path}/train_losses_{t+1}_{split_start}_{num_models}_{seed}_{learning_rate}_{sampler}.npy', train_losses)
            np.save(f'{path}/train_accuracies_{t+1}_{split_start}_{num_models}_{seed}_{learning_rate}_{sampler}.npy', train_accuracies)
            np.save(f'{path}/test_losses_{t+1}_{split_start}_{num_models}_{seed}_{learning_rate}_{sampler}.npy', test_losses)
            np.save(f'{path}/test_accuracies_{t+1}_{split_start}_{num_models}_{seed}_{learning_rate}_{sampler}.npy', test_accuracies)

    print("Done!")