import torch
import torch.nn as nn
import torch.nn.functional as F
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device", flush=True)

# get arguments using argparse, such as path, batch size, learning rate, split start and a binary flag for whether to use the separated data or not, the seed to use

parser = argparse.ArgumentParser(description='Train a neural network to predict actions from observations')
parser.add_argument('--path', type=str, default='/Users/ens/repos/marl/notebooks', help='path to the data')
parser.add_argument('--batch_size', type=int, default=25, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--split_start', type=float, default=0.0, help='start of the validation split')
parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--num_hidden', type=int, default=256, help='number of neurons in the hidden layer')
parser.add_argument("--mode", type=str, default="team", help="team or individual")
parser.add_argument('--seed', type=int, default=0, help='seed')
args = parser.parse_args()

n_input = 20
n_hidden = args.num_hidden
n_out = 5
mode = args.mode

seed: int = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

epochs = 1000
learning_rate = args.learning_rate
batch_size = args.batch_size
split_start = args.split_start
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
data = pd.read_hdf(f"{path}/data.h5", key="df")
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

def get_datasets(data, split_start = 0.7):

    # Shuffling by episode
    # groups = [data for _, data in data.groupby('eps_id')]
    # np.random.shuffle(groups)
    # data = pd.concat(groups).reset_index(drop=False)

    ratios=[split_start, 0.8, 0.9]

    data = data.sort_values(by=['eps_id','agent_index', 't'], ignore_index=True)
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


    train_partitions = []
    test_partitions = []
    val_partitions = []

    for agent_ind in np.arange(num_agents):
        train_partitions.append(train[train['agent_index'] == agent_ind])
        test_partitions.append(test[test['agent_index'] == agent_ind])
        val_partitions.append(val[val['agent_index'] == agent_ind])

    obs_train = []
    obs_test = []
    obs_val = []
    act_train = []
    act_test = []
    act_val = []
    act_prob_train = []
    act_prob_test = []
    act_prob_val = []
    ids_train = []
    ids_test = []
    ids_val = []
    logits_train = []
    logits_test = []
    logits_val = []
    probs_train = []
    probs_test = []
    probs_val = []


    for i in np.arange(num_agents):

        obs_train.append(np.array(train_partitions[i]['obs'].to_list()))
        obs_test.append(np.array(test_partitions[i]['obs'].to_list()))
        obs_val.append(np.array(val_partitions[i]['obs'].to_list()))
        act_train.append(np.array(train_partitions[i]['actions'].to_list()))
        act_test.append(np.array(test_partitions[i]['actions'].to_list()))
        act_val.append(np.array(val_partitions[i]['actions'].to_list()))
        act_prob_train.append(np.array(train_partitions[i]['action_prob'].to_list()))
        act_prob_test.append(np.array(test_partitions[i]['action_prob'].to_list()))
        act_prob_val.append(np.array(val_partitions[i]['action_prob'].to_list()))
        ids_train.append(np.array(train_partitions[i]['agent_index'].to_list()))
        ids_test.append(np.array(test_partitions[i]['agent_index'].to_list()))
        ids_val.append(np.array(val_partitions[i]['agent_index'].to_list()))
        logits_train.append(np.array(train_partitions[i]['logits'].to_list()))
        logits_test.append(np.array(test_partitions[i]['logits'].to_list()))
        logits_val.append(np.array(val_partitions[i]['logits'].to_list()))
        probs_train.append(np.array(train_partitions[i]['probs'].to_list()))
        probs_test.append(np.array(test_partitions[i]['probs'].to_list()))
        probs_val.append(np.array(val_partitions[i]['probs'].to_list()))

    obs_train = np.array(obs_train)
    obs_test = np.array(obs_test)
    obs_val = np.array(obs_val)
    act_train = np.array(act_train)
    act_test = np.array(act_test)
    act_val = np.array(act_val)
    act_prob_train = np.array(act_prob_train)
    act_prob_test = np.array(act_prob_test)
    act_prob_val = np.array(act_prob_val)
    ids_train = np.array(ids_train)
    ids_test = np.array(ids_test)
    ids_val = np.array(ids_val)
    logits_train = np.array(logits_train)
    logits_test = np.array(logits_test)
    logits_val = np.array(logits_val)
    probs_train = np.array(probs_train)
    probs_test = np.array(probs_test)
    probs_val = np.array(probs_val)

    print("obs_train shape: ", obs_train.shape)

    train_dataset = (obs_train, act_train)
    test_dataset = (obs_test, act_test)
    val_dataset = (obs_val, act_val)
    
    return train_dataset, test_dataset, val_dataset

def data_generator(data, batch_size):
    """
    A generator that yields batches of data.
    """
    X, y = data

    # shuffle the data
    randomize = np.arange(len(y))
    np.random.shuffle(randomize)
    X = X[randomize]
    y = y[randomize]
    
    num_samples = y.shape[-1]
    num_batches = num_samples // batch_size
    if num_samples % batch_size != 0:
        num_batches += 1
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i+1) * batch_size
        batch_X, batch_y = X[:, start_idx:end_idx], y[:, start_idx:end_idx]
        yield batch_X, batch_y

class MultiInputMultiOutputNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, mode = 'team'):
        super(MultiInputMultiOutputNet, self).__init__()

        self.mode = mode

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc3 = nn.Linear(input_size, hidden_size)
        self.fc4 = nn.Linear(input_size, hidden_size)
        self.hidden11 = nn.Linear(hidden_size, hidden_size)
        self.hidden12 = nn.Linear(hidden_size, hidden_size)
        self.hidden21 = nn.Linear(hidden_size, hidden_size)
        self.hidden22 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.fc6 = nn.Linear(hidden_size, output_size)
        self.fc7 = nn.Linear(hidden_size, output_size)
        self.fc8 = nn.Linear(hidden_size, output_size)

    def forward(self, input1, input2, input3, input4):
        x1 = F.tanh(self.fc1(input1))
        x2 = F.tanh(self.fc2(input2))
        x3 = F.tanh(self.fc3(input3))
        x4 = F.tanh(self.fc4(input4))

        h11_in = x1+x2+x3 if self.mode == 'team' else x1+x2+x3+x4

        h11_out = self.hidden11(h11_in)
        h11_act = F.tanh(h11_out)
        h21 = self.hidden21(h11_act)
        h21_act = F.tanh(h21)

        if self.mode == 'team':
            h12_in = x4
            h12_out = self.hidden12(h12_in)
            h12_act = F.tanh(h12_out)
            h22 = self.hidden22(h21_act)
            h22_act = F.tanh(h22)

        output1 = self.fc5(h12_act)
        output2 = self.fc6(h12_act)
        output3 = self.fc7(h12_act)
        output4 = self.fc8(h22_act) if self.mode == 'team' else self.fc8(h21_act)
        return output1, output2, output3, output4

def train_model(net, train_data):
    # Move the model to the GPU if available
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    my_generator = data_generator(train_data, batch_size = 25)

    running_train_loss = 0.0

    # Iterate over the generator to get batches of data
    counter = 0
    for batch_X, batch_y in my_generator:
        counter += 1

        input1, input2, input3, input4 = batch_X[0], batch_X[1], batch_X[2], batch_X[3]
        target1, target2, target3, target4 = batch_y[0], batch_y[1], batch_y[2], batch_y[3]

        # Tensorize the data form numpy arrays
        input1 = torch.from_numpy(input1).float()
        input2 = torch.from_numpy(input2).float()
        input3 = torch.from_numpy(input3).float()
        input4 = torch.from_numpy(input4).float()
        target1 = torch.from_numpy(target1).long()
        target2 = torch.from_numpy(target2).long()
        target3 = torch.from_numpy(target3).long()
        target4 = torch.from_numpy(target4).long()
        
        # Forward pass
        output1, output2, output3, output4 = net(input1, input2, input3, input4)
        loss1 = criterion(output1, target1)
        acc1 = (output1.argmax(1) == target1).float().mean()
        loss2 = criterion(output2, target2)
        acc2 = (output2.argmax(1) == target2).float().mean()
        loss3 = criterion(output3, target3)
        acc3 = (output3.argmax(1) == target3).float().mean()
        loss4 = criterion(output4, target4)
        acc4 = (output4.argmax(1) == target4).float().mean()
        loss = loss1 + loss2 + loss3 + loss4
        acc = (acc1 + acc2 + acc3 + acc4) / 4
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

        if counter % 100 == 0:
            print(f'Batch {counter}, Loss: {loss.item():.4f}')

    epoch_loss = running_train_loss / counter

    print(f"Train loss: {epoch_loss:>8f}" ,flush=True)

    return epoch_loss

def test_model(net, test_data):
    # Move the model to the GPU if available
    net.to(device)
    criterion = nn.CrossEntropyLoss()

    my_generator = data_generator(test_data, batch_size = 25)

    running_test_loss = 0.0

    # Iterate over the generator to get batches of data
    counter = 0
    with torch.no_grad():
        for batch_X, batch_y in my_generator:
            counter += 1

            input1, input2, input3, input4 = batch_X[0], batch_X[1], batch_X[2], batch_X[3]
            target1, target2, target3, target4 = batch_y[0], batch_y[1], batch_y[2], batch_y[3]

            # Tensorize the data form numpy arrays
            input1 = torch.from_numpy(input1).float()
            input2 = torch.from_numpy(input2).float()
            input3 = torch.from_numpy(input3).float()
            input4 = torch.from_numpy(input4).float()
            target1 = torch.from_numpy(target1).long()
            target2 = torch.from_numpy(target2).long()
            target3 = torch.from_numpy(target3).long()
            target4 = torch.from_numpy(target4).long()
            
            # Forward pass
            output1, output2, output3, output4 = net(input1, input2, input3, input4)
            loss1 = criterion(output1, target1)
            loss2 = criterion(output2, target2)
            loss3 = criterion(output3, target3)
            loss4 = criterion(output4, target4)
            loss = loss1 + loss2 + loss3 + loss4
            
            running_test_loss += loss.item()

            if counter % 100 == 0:
                print(f'Batch {counter}, Loss: {loss.item():.4f}')

    epoch_loss = running_test_loss / counter

    print(f"Test loss: {epoch_loss:>8f}" ,flush=True)

    return epoch_loss

if __name__ == "__main__":

    train_ds, test_ds, val_ds = get_datasets(data=data)
    model = MultiInputMultiOutputNet(input_size = n_input, hidden_size = n_hidden, output_size = n_out, mode=mode)

    train_losses = []
    train_accuracies = []

    test_losses = []
    test_accuracies = []

    for t in range(epochs):

        print(f"Epoch {t+1}\n-------------------------------", flush=True)

        train_loss = train_model(model, train_ds)
        test_loss = test_model(model, test_ds)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # save model
        if (t+1) % 10 == 0:
            torch.save(model.state_dict(), f'{path}/model_{t+1}_{split_start}_{seed}_{learning_rate}_{n_hidden}_{mode}.pth')
            print("Saved PyTorch Model State to model.pth", flush=True)

            # save train and test losses and accuracies as numpy arrays
            np.save(f'{path}/train_losses_{t+1}_{split_start}_{seed}_{learning_rate}_{n_hidden}_{mode}.npy', train_losses)
            np.save(f'{path}/train_accuracies_{t+1}_{split_start}_{seed}_{learning_rate}_{n_hidden}_{mode}.npy', train_accuracies)
            np.save(f'{path}/test_losses_{t+1}_{split_start}_{seed}_{learning_rate}_{n_hidden}_{mode}.npy', test_losses)
            np.save(f'{path}/test_accuracies_{t+1}_{split_start}_{seed}_{learning_rate}_{n_hidden}_{mode}.npy', test_accuracies)

    print("Done!")