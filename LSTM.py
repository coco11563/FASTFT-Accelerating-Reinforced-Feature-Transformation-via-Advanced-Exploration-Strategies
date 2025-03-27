import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import DataLoader
import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
import statistics
import os
import time


class CONFIG():
    def __init__(self):
        self.num_epochs = 20
        self.patience = 3
        self.batch_size = 128
        self.hidden_dim = 16
        self.hidden_dim_1 = 4
        self.layers_num = 2
        self.mlplayers = 1
        self.aug = 3
        self.dropout_1 = 0
        self.dropout_2 = 0
        self.maxval = 0
        self.minval = 0
        self.selective_aug = 197  
        self.aug_mode = 'all_aug'
        self.image_path = 'image/'
        self.lr = 0.0002
        # self.lr = 0.001
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.thresh = 3


config = CONFIG()
config_dict = vars(config)



O1 = ['sqrt', 'square', 'sin', 'cos', 'tanh', 'stand_scaler',
      'minmax_scaler', 'quan_trans', 'sigmoid', 'log', 'reciprocal']
O2 = ['+', '-', '*', '/']
O3 = ['stand_scaler', 'minmax_scaler', 'quan_trans']

O1_len = len(O1)
O2_len = len(O2)
O3_len = len(O3)

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

class stateListDataset(Dataset):
    def __init__(self, data_path, label_path, name):
        if name == 'fetal_health' or name == 'breast_cancer' or name == 'cardio_train' or name == 'alzheimers':
            Dg = pd.read_csv('./data/processed/' + name + '.csv')
        else:
            Dg = pd.read_hdf('./data/processed/' + name + '.hdf')   
        original_features = list(Dg.columns)
        original_len = len(original_features)
        self.eos_token = original_len+O1_len+O2_len+O3_len+2
        self.pad_token = original_len+O1_len+O2_len+O3_len+3

        data = load_data(data_path)
        labels = load_data(label_path)
        self.labels = []
        self.data = []
        self.min_val = min(labels)
        self.max_val = max(labels)
        config.minval = self.min_val
        config.maxval = self.max_val
        print('min:', self.min_val)
        print('max:', self.max_val)
        labels = (labels - self.min_val) / (self.max_val - self.min_val)
        self.q1 = statistics.quantiles(labels, n=4)[0]
        self.q3 = statistics.quantiles(labels, n=4)[2]
        print(self.q1)
        print(self.q3)
        max_len = 0
        for i, feature_list in enumerate(data):
            j = 0
            if labels[i] <= self.q3 and labels[i] >= self.q1:
                aug = config.aug
            else:
                aug = config.aug * 3
            while j<aug:
                label = labels[i]
                self.labels.append(label)
                feature_list_lstm = ''
                for feature in feature_list:
                    feature = str(feature)
                    if feature == str(original_features[-1]):
                        continue
                    for index, x in enumerate(original_features):
                        if str(x) in feature:
                            feature = feature.replace(str(x), '='+str(index)+'=')
                    for index, x in enumerate(O1):
                        if x in feature:
                            feature = feature.replace(x, '='+str((index+original_len))+'=')
                    for index, x in enumerate(O2):
                        if x in feature:
                            feature = feature.replace(x, '='+str((index+original_len+O1_len))+'=')
                    for index, x in enumerate(O3):
                        if x in feature:
                            feature = feature.replace(x, '='+str((index+original_len+O1_len+O2_len))+'=')
                    feature_list_lstm = feature_list_lstm + feature
                feature_list_lstm = feature_list_lstm.replace('_','')
                feature_list_lstm = feature_list_lstm.split('=')
                feature_list_lstm = [item for item in feature_list_lstm if item != '']
                feature_list_lstm = [int(item) for item in feature_list_lstm]
                self.data.append(feature_list_lstm.copy())
                if len(self.data[-1]) > max_len:
                    max_len = len(self.data[-1])
                random.shuffle(feature_list)
                j = j + 1
        for i in range(len(self.data)):
            if len(self.data[i])<max_len:
                for j in range(max_len - len(self.data[i])):
                    self.data[i].append(self.pad_token)
        print('maxlen:', max_len)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        label = self.labels[index]
        state_list = torch.Tensor(self.data[index]).long()
        return state_list, label


class Encoder(nn.Module):
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size):
        super().__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

    def infer(self, x, predict_lambda, direction='-'):

        encoder_outputs, encoder_hidden, seq_emb, predict_value = self(x)
        grads_on_outputs = torch.autograd.grad(predict_value, encoder_outputs, torch.ones_like(predict_value))[0]
        if direction == '+':
            new_encoder_outputs = encoder_outputs + predict_lambda * grads_on_outputs
        elif direction == '-':
            new_encoder_outputs = encoder_outputs - predict_lambda * grads_on_outputs
        else:
            raise ValueError('Direction must be + or -, got {} instead'.format(direction))
        new_encoder_outputs = F.normalize(new_encoder_outputs, 2, dim=-1)
        new_seq_emb = torch.mean(new_encoder_outputs, dim=1)
        new_seq_emb = F.normalize(new_seq_emb, 2, dim=-1)
        return encoder_outputs, encoder_hidden, seq_emb, predict_value, new_encoder_outputs, new_seq_emb

    def forward(self, x):
        pass



class RNNEncoder(Encoder):
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 mlp_layers,
                 mlp_hidden_size,
                 mlp_dropout,
                 init = True
                 ):
        super(RNNEncoder, self).__init__(layers, vocab_size, hidden_size)

        self.mlp_layers = mlp_layers
        self.mlp_hidden_size = mlp_hidden_size

        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True)
        self.mlp = nn.Sequential()
        for i in range(self.mlp_layers):
            if i == 0:
                linear_layer = nn.Linear(self.hidden_size, self.mlp_hidden_size)
                if init:
                    nn.init.normal_(linear_layer.weight, mean=0, std=1)
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    linear_layer,
                    ))
            else:
                linear_layer = nn.Linear(mlp_hidden_size, mlp_hidden_size)
                if init:
                    nn.init.normal_(linear_layer.weight, mean=0, std=1)
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
                    ))
        self.regressor = nn.Linear(self.hidden_size if self.mlp_layers == 0 else self.mlp_hidden_size, 1)
        nn.init.normal_(self.regressor.weight, mean=0, std=1)
        
    def forward(self, x):
        embedded = self.embedding(x)  # batch x length x hidden_size
        out, hidden = self.rnn(embedded)
        out = F.normalize(out, 2, dim=-1)
        encoder_outputs = out  # final output
        encoder_hidden = hidden  # layer-wise hidden

        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=-1)
        seq_emb = out

        out = self.mlp(out)
        out = self.regressor(out)
        predict_value = out
        predict_value = torch.sigmoid(out)
        return predict_value


def predict(model, feature_list, name):
    if name == 'fetal_health' or name == 'breast_cancer' or name == 'cardio_train' or name == 'alzheimers':
            Dg = pd.read_csv('./data/processed/' + name + '.csv')
    else:
            Dg = pd.read_hdf('./data/processed/' + name + '.hdf')  
    original_features = list(Dg.columns)
    original_len = len(original_features)

    feature_list_lstm = ''
    for feature in feature_list:
        feature = str(feature)
        if feature == str(original_features[-1]):
            continue          
        for index, x in enumerate(original_features):
            if str(x) in feature:
                feature = feature.replace(str(x), '='+str(index)+'=')
        for index, x in enumerate(O1):
            if x in feature:
                feature = feature.replace(x, '='+str((index+original_len))+'=')
        for index, x in enumerate(O2):
            if x in feature:
                feature = feature.replace(x, '='+str((index+original_len+O1_len))+'=')
        for index, x in enumerate(O3):
            if x in feature:
                feature = feature.replace(x, '='+str((index+original_len+O1_len+O2_len))+'=')
        feature_list_lstm = '' + feature
    feature_list_lstm = feature_list_lstm.replace('_','')
    feature_list_lstm = feature_list_lstm.split('=')
    feature_list_lstm = [item for item in feature_list_lstm if item != '']
    feature_list_lstm = [int(item) for item in feature_list_lstm]
    data = torch.Tensor(feature_list_lstm).long()
    data = data.reshape(1, -1)
    data = data.to(config.device)
    return model(data).reshape(-1).float()

def train(data_path, performance_path, name):
    print('Using LSTM threshold: ' + str((config.thresh+1)/10))
    alldata = stateListDataset(data_path, performance_path, name)
    train_size = int(len(alldata) * 0.8)
    test_size = len(alldata) - train_size
    train_data, test_data = torch.utils.data.random_split(alldata, [train_size, test_size])
    train_loader=DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader=DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    
    train_losses = []
    test_losses = []
    model = RNNEncoder(config.layers_num, alldata.eos_token + 2, config.hidden_dim, config.dropout_1, config.mlplayers, config.hidden_dim_1, config.dropout_2)
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_min = 100000000
    start_time = time.time()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i, epoch in enumerate(range(config.num_epochs)):
        model.train()
        train_loss = 0
        targets = []
        outputs = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config.device), target.to(config.device)
            optimizer.zero_grad()
            output = model(data)
            loss_fn = nn.MSELoss()
            loss = loss_fn(output.float(), target.float())
            loss.backward()
            optimizer.step()
            target = target.float().cpu()
            for t in target:
                targets.append(t.item())
            for o in output:
                outputs.append(o.reshape(-1).cpu().detach().item())
            if(batch_idx + 1) % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                            100. * (batch_idx + 1) / len(train_loader), loss.item()))
            train_loss += loss.item()

        print("train loss:", train_loss * config.batch_size / train_size)


        model.eval()

        test_loss = 0
        for test_batch, target in test_loader:
            test_batch, target = test_batch.to(config.device), target.to(config.device)
            output = model(test_batch).float()
            test_loss += loss_fn(output, target.float()).float()

        print('Test Loss: {:.6f}'.format(test_loss*config.batch_size/test_size))
        if test_loss.item()/test_size < loss_min:
            loss_min = test_loss.item()/test_size
            count = 0
            best_model = model
            best_o = outputs
            best_t = targets
        else:
            count = count + 1
            if count>config.patience:
                break
    print('training end')
    data = [[x, y] for (x, y) in zip(best_t, best_o)]
    test_loss = 0
    model.eval()
    best_model = model
    best_model.eval()
    i = 0
    losses = []
    for test_batch, target in test_loader:
        test_batch, target = test_batch.to(config.device), target.to(config.device)
        loss = loss_fn(best_model(test_batch).float(), target.float()).float()
        losses.append(loss)
        test_loss += loss
        target = target.float().cpu()
        out = best_model(test_batch).float().cpu()
        if i % 5 == 0:
            print(f"gold  Target: {target}  Out:{out}")
        i = i + 1
    end_event.record()
    torch.cuda.synchronize()
    gpu_time_ms = start_event.elapsed_time(end_event)
    time_cost = time.time() - start_time
    print("GPU Time:", gpu_time_ms, "milliseconds")
    print("time:", time_cost, "seconds")
    print(outputs)
    print('Test Loss: {:.6f}'.format(test_loss*config.batch_size/test_size))
    with open('./lstm_data.pkl', 'wb') as f:
        pickle.dump(losses, f)
    if config.thresh == 10:
        threshold = statistics.quantiles(best_o, n=20)[18]
    elif config.thresh == -1:
        threshold = min(best_o)
    elif config.thresh == 3:
        threshold = max(best_o)
    else:
        threshold = statistics.quantiles(best_o, n=4)[config.thresh]
    print('threshold_2:', threshold)
    return best_model, alldata.max_val, alldata.min_val, threshold


def retrain(model, name, feature_lists, performances):
    print('Using LSTM threshold: ' + str((config.thresh+1)/10))
    model.to(config.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_min = 100000000

    if name == 'fetal_health' or name == 'breast_cancer' or name == 'cardio_train' or name == 'alzheimers':
            Dg = pd.read_csv('./data/processed/' + name + '.csv')
    else:
            Dg = pd.read_hdf('./data/processed/' + name + '.hdf')  
    original_features = list(Dg.columns)
    original_len = len(original_features)
    retrain_data = []
    retrain_label = []
    max_len = 0
    pad_token = original_len+O1_len+O2_len+O3_len+3
    for i, feature_list in enumerate(feature_lists):
        j = 0
        if feature_list == None:
            continue
        while j < config.aug:
            label = [performances[i]]
            label = torch.Tensor(label)
            label = label.reshape(1, -1)
            label = label.to(config.device)
            retrain_label.append(label)
            feature_list_lstm = ''
            for feature in feature_list:
                feature = str(feature)
                if feature == str(original_features[-1]):
                    continue
                for index, x in enumerate(original_features):
                    if str(x) in feature:
                        feature = feature.replace(str(x), '='+str(index)+'=')
                for index, x in enumerate(O1):
                    if x in feature:
                        feature = feature.replace(x, '='+str((index+original_len))+'=')
                for index, x in enumerate(O2):
                    if x in feature:
                        feature = feature.replace(x, '='+str((index+original_len+O1_len))+'=')
                for index, x in enumerate(O3):
                    if x in feature:
                        feature = feature.replace(x, '='+str((index+original_len+O1_len+O2_len))+'=')
                feature_list_lstm = feature_list_lstm + feature
            feature_list_lstm = feature_list_lstm.replace('_','')
            feature_list_lstm = feature_list_lstm.split('=')
            feature_list_lstm = [item for item in feature_list_lstm if item != '']
            feature_list_lstm = [int(item) for item in feature_list_lstm]
            data = torch.Tensor(feature_list_lstm.copy()).long()
            data = data.reshape(1, -1)
            data = data.to(config.device)
            retrain_data.append(data)
            if len(retrain_data[-1]) > max_len:
                max_len = len(retrain_data[-1])
            random.shuffle(feature_list)
            j = j + 1
        
        for i in range(len(retrain_data)):
            if len(retrain_data)<max_len:
                for j in range(max_len - len(retrain_data[i])):
                    retrain_data[i].append(pad_token)
    
    for epoch in range(5):
        train_loss = 0.0
        target = torch.Tensor(retrain_label).long().to(config.device)
        data = torch.Tensor(data).to(config.device)
        optimizer.zero_grad()
        output = model(data)
        loss_fn = nn.MSELoss()
        loss = loss_fn(output.float(), target.float())
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        print("train loss:", train_loss / len(retrain_data))
        
    return model