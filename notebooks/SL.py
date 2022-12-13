import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import glob
import argparse
# from torchmetrics import Accuracy

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

n_input = 20
n_hidden = 256
n_out = 5
batch_size = 100
learning_rate = 0.01

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device", flush=True)

# get arguments using argparse, such as path, batch size, learning rate, split start and a binary flag for whether to use the separated data or not

parser = argparse.ArgumentParser(description='Train a neural network to predict actions from observations')
parser.add_argument('--path', type=str, default='/home/mila/m/memariaa/scratch/new/data.h5', help='path to the data')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--split_start', type=float, default=0.0, help='start of the validation split')
parser.add_argument('--separated', type=bool, default=False, help='whether to use the separated data or not')
args = parser.parse_args()

epochs = 400
spilt_start = args.split_start
is_separated: bool = args.separated

# %%
# find all pickle files in the current directory using glob

path = "/home/mila/m/memariaa/scratch"
# all_files = glob.glob(path + "/*.pkl")

# # create a list of dataframes
# li = []

# for filename in all_files:
#     df = pd.read_pickle(filename)
#     li.append(df)

# concatenate the list of dataframes into one dataframe
# data = pd.concat(li, axis=0, ignore_index=True)
data = pd.read_hdf("/home/mila/m/memariaa/scratch/new/data.h5", key="df")
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
def get_dataloaders(data, separated = is_separated, ratios=[spilt_start, 0.8, 0.9]):

    # Shuffling by episode
    groups = [data for _, data in data.groupby('eps_id')]
    np.random.seed(0)
    np.random.shuffle(groups)
    data = pd.concat(groups).reset_index(drop=False)
    data = data.sort_values(by=['eps_id','t','agent_index'],ignore_index=False)
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
    if separated is True:
        for agent_ind in np.arange(num_agents):

            train_agent = train[train['agent_index'] == agent_ind]
            test_agent = test[test['agent_index'] == agent_ind]
            val_agent = val[val['agent_index'] == agent_ind]

            obs_train, obs_test, obs_val = np.array(train_agent['obs'].to_list()), np.array(test_agent['obs'].to_list()), np.array(val_agent['obs'].to_list())
            act_train, act_test, act_val = np.array(train_agent['actions'].to_list()), np.array(test_agent['actions'].to_list()), np.array(val_agent['actions'].to_list())
            act_prob_train, act_prob_test, act_prob_val = np.array(train_agent['action_prob'].to_list()), np.array(test_agent['action_prob'].to_list()), np.array(val_agent['action_prob'].to_list())
            ids_train, ids_test, ids_val = np.array(train_agent['agent_index'].to_list()), np.array(test_agent['agent_index'].to_list()), np.array(val_agent['agent_index'].to_list())
            logits_train, logits_test, logits_val = np.array(train_agent['logits'].to_list()), np.array(test_agent['logits'].to_list()), np.array(val_agent['logits'].to_list())
            probs_train, probs_test, probs_val = np.array(train_agent['probs'].to_list()), np.array(test_agent['probs'].to_list()), np.array(val_agent['probs'].to_list())


            train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(obs_train), torch.from_numpy(act_train),torch.from_numpy(act_prob_train), torch.from_numpy(ids_train),torch.from_numpy(logits_train),torch.from_numpy(probs_train))
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            all_train_dataloaders.append(train_dataloader)

            test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(obs_test), torch.from_numpy(act_test), torch.from_numpy(act_prob_test), torch.from_numpy(ids_test),torch.from_numpy(logits_test),torch.from_numpy(probs_test))
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            all_test_dataloaders.append(test_dataloader)

            val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(obs_val), torch.from_numpy(act_val), torch.from_numpy(act_prob_val), torch.from_numpy(ids_val),torch.from_numpy(logits_val),torch.from_numpy(probs_val))
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            all_val_dataloaders.append(val_dataloader)
    else:
        obs_train, obs_test, obs_val = np.array(train['obs'].to_list()), np.array(test['obs'].to_list()), np.array(val['obs'].to_list())
        act_train, act_test, act_val = np.array(train['actions'].to_list()), np.array(test['actions'].to_list()), np.array(val['actions'].to_list())
        act_prob_train, act_prob_test, act_prob_val = np.array(train['action_prob'].to_list()), np.array(test['action_prob'].to_list()), np.array(val['action_prob'].to_list())
        ids_train, ids_test, ids_val = np.array(train['agent_index'].to_list()), np.array(test['agent_index'].to_list()), np.array(val['agent_index'].to_list())
        logits_train, logits_test, logits_val = np.array(train['logits'].to_list()), np.array(test['logits'].to_list()), np.array(val['logits'].to_list())
        probs_train, probs_test, probs_val = np.array(train['probs'].to_list()), np.array(test['probs'].to_list()), np.array(val['probs'].to_list())


        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(obs_train), torch.from_numpy(act_train), torch.from_numpy(act_prob_train) ,torch.from_numpy(ids_train),torch.from_numpy(logits_train),torch.from_numpy(probs_train))
        all_train_dataloaders = [torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)]

        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(obs_test), torch.from_numpy(act_test), torch.from_numpy(act_prob_test) ,torch.from_numpy(ids_test),torch.from_numpy(logits_test),torch.from_numpy(probs_test))
        all_test_dataloaders = [torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)]

        val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(obs_val), torch.from_numpy(act_val), torch.from_numpy(act_prob_val) ,torch.from_numpy(ids_val),torch.from_numpy(logits_val),torch.from_numpy(probs_val))
        all_val_dataloaders = [torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)]


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

    size = len(dataloaders[0].dataset)
    num_batches = len(dataloaders[0])

    for model, optimizer, dataloader in zip(models, optimizers, dataloaders):

        running_train_loss = 0.0 
        running_accuracy = 0.0 

        epoch_losses = []
        epoch_accuracies = []

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
    size = len(dataloaders[0].dataset)
    num_batches = len(dataloaders[0])

    with torch.no_grad():
        for model, dataloader in zip(models, dataloaders):
            test_loss, correct = 0, 0
            epoch_losses = []
            epoch_accuracies = []
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

    policies, loss_fn, optimizers = policy_model(num_networks= 4 if args.separated else 1)

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
            torch.save(policies[0].state_dict(), f"{path}/model_{t+1}_{spilt_start}_{is_separated}.pth")
            print("Saved PyTorch Model State to model.pth", flush=True)

            # save train and test losses and accuracies as numpy arrays
            np.save(f'{path}/train_losses_{t+1}_{spilt_start}_{is_separated}.npy', train_losses)
            np.save(f'{path}/train_accuracies_{t+1}_{spilt_start}_{is_separated}.npy', train_accuracies)
            np.save(f'{path}/test_losses_{t+1}_{spilt_start}_{is_separated}.npy', test_losses)
            np.save(f'{path}/test_accuracies_{t+1}_{spilt_start}_{is_separated}.npy', test_accuracies)

    print("Done!")