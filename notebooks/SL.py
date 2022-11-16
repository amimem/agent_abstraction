# %%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from torchmetrics import Accuracy

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# %%
import glob

# %% [markdown]
# ### Parameters

# %%
n_input = 20
n_hidden = 256
n_out = 5
batch_size = 100
learning_rate = 0.01

# %% [markdown]
# ### Data

# %%
# find all pickle files in the current directory using glob

path = 
all_files = glob.glob(path + "/*.pkl")

# create a list of dataframes
li = []

for filename in all_files:
    df = pd.read_pickle(filename)
    li.append(df)

# concatenate the list of dataframes into one dataframe
data = pd.concat(li, axis=0, ignore_index=True)

# %%
# save observation column of arrays to a numpy array
obs = np.array(data['obs'].to_list())
obs.shape

# %%
act = np.array(data['actions'].to_list())
act.shape

# %%
ids = np.array(data['agent_index'].to_list())
ids.shape

# %%
# create a pytorch dataloader from the obervation and action arrays
# the dataloader will be used to train the neural network
# the dataloader will return a batch of observations and actions
# the batch size is set to 100
# the shuffle parameter is set to True so that the data is shuffled before each epoch

dataset = torch.utils.data.TensorDataset(torch.from_numpy(obs).float(), torch.from_numpy(act).float())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %% [markdown]
# ## Functions

# %%
def data_prep(data):

    # do train test validation split on the data
    # the data is split into 80% train, 10% test, 10% validation
    # the data is shuffled before splitting

    train, test, val = np.split(data.sample(frac=1), [int(.8*len(data)), int(.9*len(data))])

    print("train shape: ", train.shape)
    print("test shape: ", test.shape)
    print("val shape: ", val.shape)

    # 

    obs_train, obs_test, obs_val = np.array(train['obs'].to_list()), np.array(test['obs'].to_list()), np.array(val['obs'].to_list())
    act_train, act_test, act_val = np.array(train['actions'].to_list()), np.array(test['actions'].to_list()), np.array(val['actions'].to_list())
    ids_train, ids_test, ids_val = np.array(train['agent_index'].to_list()), np.array(test['agent_index'].to_list()), np.array(val['agent_index'].to_list())

    # create test, train and validation dataloaders
    # the dataloaders will be used to train the neural network
    # the dataloaders will return a batch of observations and actions
    # the batch size is set to 100
    # the shuffle parameter is set to True so that the data is shuffled before each epoch

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(obs_train), torch.from_numpy(act_train), torch.from_numpy(ids_train))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(obs_test), torch.from_numpy(act_test), torch.from_numpy(ids_test))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(obs_val), torch.from_numpy(act_val), torch.from_numpy(ids_val))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader, val_dataloader

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

# %%
class Model():
    
    def __init__(self, num_agents, learning_rate):
        self.num_agents = num_agents
        self.policy_networks = [PolicyNetwork() for _ in range(num_agents)]
        self.optimizers = [torch.optim.Adam(policy_network.parameters(), lr=learning_rate) for policy_network in self.policy_networks]
        self.loss_fn = nn.CrossEntropyLoss()
    
    def train(self, dataloader):
        for policy_network, optimizer in zip(self.policy_networks, self.optimizers):
            for batch_idx, (obs, act) in enumerate(dataloader):
                optimizer.zero_grad()
                logits = policy_network(obs)
                loss = self.loss_fn(logits, act.long())
                loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0:
                    print('Loss: {:.6f}'.format(
                        100. * batch_idx / len(dataloader), loss.item()))

    def test(self, dataloader):
        with torch.no_grad():
            for policy_network, optimizer in zip(self.policy_networks, self.optimizers):
                for batch_idx, (obs, act) in enumerate(dataloader):             
                    logits = policy_network(obs)
                    loss = self.loss_fn(logits, act.long())

                    # get accuracy of the model
                    _, predicted = torch.max(logits.data, 1)
                    total = act.size(0)
                    correct = (predicted == act).sum().item()
                    accuracy = 100 * correct / total

                    if batch_idx % 100 == 0:
                        print('Loss: {:.6f}, Accuracy: {}%'.format(
                            100. * batch_idx / len(dataloader), loss.item(), accuracy))

# %%
def policy_model(num_networks = 1):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    model = [PolicyNetwork(i).to(device) for i in range(num_networks)]
    # print(model)

    loss_function = nn.CrossEntropyLoss()  
    optimizer = [torch.optim.SGD(m.parameters(), lr=learning_rate) for m in model]

    return model, loss_function, optimizer

# %%
def train_loop(dataloader, models, loss_fn, optimizers):
    size = len(dataloader.dataset)
    for model, optimizer in zip(models, optimizers):
        for batch, (obs, act, ids) in enumerate(dataloader):
            # slice batch based on agent index
            obs = obs[ids == model.index]
            act = act[ids == model.index]

            # print (obs.shape, act.shape)
            
            # Compute prediction and loss
            pred = model(obs)
            loss = loss_fn(pred, act)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 500 == 0:
                loss, current = loss.item(), batch * len(obs)
                print(f"model: {model.index} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, models, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for model in models:
            for batch, (obs, act, ids) in enumerate(dataloader):
                # slice batch based on agent index
                obs = obs[ids == model.index]
                act = act[ids == model.index]

                pred = model(obs)
                test_loss += loss_fn(pred, act).item()
                correct += (pred.argmax(1) == act).type(torch.float).sum().item()

            test_loss /= num_batches
            correct /= size
            print(f"Mode: {model.index} \n Test Error: \n Accuracy:{(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
# %%
    policies, loss_fn, optimizers = policy_model(num_networks=4)

    # %%
    # get dataloaders for train, test and validation data
    train_dataloader, test_dataloader, val_dataloader = data_prep(data)

    epochs = 10000
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, policies, loss_fn, optimizers)
        test_loop(test_dataloader, policies, loss_fn)
    print("Done!")




