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
import numpy as np
import pickle


class CONFIG():
    num_epochs = 10
    patience = 10
    batch_size = 1
    hidden_dim = 16
    hidden_dim_1 = 8
    layers_num = 2
    aug = 1 
    aug_mode = 'all_aug'
    lr = 0.0008
    thresh = 3
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


config = CONFIG()


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
class featureDataset(Dataset):
    def __init__(self, data_path, label_path, name):
        if name == 'fetal_health' or name == 'breast_cancer' or name == 'cardio_train' or name == 'alzheimers':
            Dg = pd.read_csv('./data/processed/' + name + '.csv')
        else:
            Dg = pd.read_hdf('./data/processed/' + name + '.hdf')  
        original_features = list(Dg.columns)
        original_len = len(original_features)
        data = load_data(data_path)
        self.eos_token = original_len+O1_len+O2_len+O3_len+2
        features = []
        self.data = []
        for featurelist in data:
            for feature in featurelist:
                feature = str(feature)
                if feature in features:
                    continue
                else:
                    features.append(feature)
                    if feature == original_features[-1]:
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
                    feature = feature.replace('_','')
                feature = feature.split('=')
                feature = [item for item in feature if item != '']
                feature = [int(item) for item in feature]
                self.data.append(feature.copy())
                print(feature)
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        state_list = torch.Tensor(self.data[index]).long()
        return state_list, 0


class stateListDataset(Dataset):
    def __init__(self, data_path, label_path, name):
        if name == 'fetal_health' or name == 'breast_cancer' or name == 'cardio_train' or name == 'alzheimers':
            Dg = pd.read_csv('./data/processed/' + name + '.csv')
        else:
            Dg = pd.read_hdf('./data/processed/' + name + '.hdf')  
        original_features = list(Dg.columns)
        original_len = len(original_features)
        self.eos_token = original_len+O1_len+O2_len+O3_len+2

        data = load_data(data_path)
        features = []
        self.data = []
        for featurelist in enumerate(data):
            for feature in featurelist[1]:
                feature = str(feature)
                if feature in original_features:
                    continue
                elif feature in features:
                    continue
                else:
                    features.append(feature)
                    if feature == original_features[-1]:
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
                    feature = feature.replace('_','')
                feature = feature.split('=')
                feature = [item for item in feature if item != '']
                feature = [int(item) for item in feature]


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        state_list = torch.Tensor(self.data[index]).long()
        return state_list, 0

class LSTMmodel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, layer_num):
        super(LSTMmodel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=layer_num, batch_first=True, bidirectional=True).to(config.device)
        self.reg = nn.Linear(hidden_dim * 2, 1).to(config.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out, _ = self.lstm(input)
        out = self.reg(out[:, -1, :])
        out = self.sigmoid(out)
        return out

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

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential()
        for i in range(self.mlp_layers):
            if i == 0:
                linear_layer = nn.Linear(self.hidden_size, self.mlp_hidden_size)
                if init:
                    nn.init.normal_(linear_layer.weight, mean=0, std=1)
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    linear_layer,
                    nn.Dropout(p=mlp_dropout)))
            else:
                linear_layer = nn.Linear(mlp_hidden_size, mlp_hidden_size)
                if init:
                    nn.init.normal_(linear_layer.weight, mean=0, std=1)
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
                    nn.Dropout(p=mlp_dropout)))
        self.regressor = nn.Linear(self.hidden_size if self.mlp_layers == 0 else self.mlp_hidden_size, 1)
        nn.init.normal_(self.regressor.weight, mean=0, std=1)

    def forward(self, x):
        embedded = self.embedding(x)  # batch x length x hidden_size
        embedded = self.dropout(embedded)
        out, hidden = self.rnn(embedded)
        out = F.normalize(out, 2, dim=-1)
        encoder_outputs = out  # final output
        encoder_hidden = hidden  # layer-wise hidden

        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=-1)
        seq_emb = out

        out = self.mlp(out)
        out = self.regressor(out)
        predict_value = torch.sigmoid(out)
        return predict_value


def predict(model, feature, name):
    if name == 'fetal_health' or name == 'breast_cancer' or name == 'cardio_train' or name == 'alzheimers':
            Dg = pd.read_csv('./data/processed/' + name + '.csv')
    else:
            Dg = pd.read_hdf('./data/processed/' + name + '.hdf')  
    original_features = list(Dg.columns)
    original_len = len(original_features)
    feature = str(feature)
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
    feature = feature.replace('_','')
    feature = feature.split('=')
    feature = [item for item in feature if item != '']
    feature = [int(item) for item in feature]
    data = torch.Tensor(feature).long()
    data = data.reshape(1, -1)
    data = data.to(config.device)
    out, y1, y2 = model(data)
    return out.reshape(-1).cpu().detach().item()


def predict_all(model, feature_list, name):
    intrinsic_list = []
    for feature in feature_list[:-1]:
        intrinsic = predict(model, feature, name)
        intrinsic_list.append(intrinsic * intrinsic)
    return np.mean(intrinsic_list)

class RND(nn.Module):
    def __init__(self,
                 token,
                 ):
        super().__init__()
        self.rnd = RNNEncoder(config.layers_num, token, config.hidden_dim, 0, 0, config.hidden_dim_1, 0, False)
        self.imitate = RNNEncoder(config.layers_num, token, config.hidden_dim, 0, 2, config.hidden_dim_1, 0, False)
        for param in self.rnd.parameters():
            param.requires_grad = False
        lstm1_weights = [param for name, param in self.rnd.rnn.named_parameters() if 'weight' in name]
        lstm2_weights = [param for name, param in self.imitate.rnn.named_parameters() if 'weight' in name]
        concatenated_weights = torch.cat(lstm1_weights + lstm2_weights, dim=0)
        nn.init.orthogonal_(concatenated_weights, 16.0)
        concatenated_weights = concatenated_weights * 3.0
        start_idx = 0
        for name, param in self.rnd.rnn.named_parameters():
            if 'weight' in name:
                end_idx = start_idx + param.data.size(0)
                param.data = concatenated_weights[start_idx:end_idx].clone()
                start_idx = end_idx
        for name, param in self.imitate.rnn.named_parameters():
            if 'weight' in name:
                end_idx = start_idx + param.data.size(0)
                param.data = concatenated_weights[start_idx:end_idx].clone()
                start_idx = end_idx

    def forward(self, x):
        y_1 = self.rnd(x)
        y_2 = self.imitate(x)
        return y_1 - y_2, y_1, y_2

def train(data_path, performance_path, name):
    print('Using RND threshold: ' + str((config.thresh+1)/4))
    alldata = featureDataset(data_path, performance_path, name)
    train_size = int(len(alldata) * 0.8)
    valid_size = int(len(alldata) * 0.2)
    train_data = Subset(alldata, range(0, train_size))
    valid_data = Subset(alldata, range(train_size, train_size + valid_size))
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False)

    train_losses = []
    valid_losses = []
    
    model = RND(alldata.eos_token + 1)
    for param in model.rnd.parameters():
        param.requires_grad = False
    
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_min = 100000000
    for epoch in range(config.num_epochs):
        model.train(mode=True)
        train_loss = 0
        targets = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config.device), target.to(config.device)
            optimizer.zero_grad()
            output, y_1, y_2 = model(data)
            if (batch_idx + 1) % 25 == 0:
                print("train batch:", y_1, y_2)
            loss_fn = nn.MSELoss()
            loss = loss_fn(output, target.float())
            loss.backward()
            optimizer.step()
            target = target.float().cpu()
            targets.append(target.item())
            if(batch_idx + 1) % 200 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                            100. * (batch_idx + 1) / len(train_loader), loss.item()))
            train_loss += loss.item()
        print('Train Loss: {:.6f}'.format(train_loss / train_size))
        train_losses.append(train_loss / train_size)
        for name, param in model.rnd.named_parameters():
            if 'weight' in name:
                break
        model.eval()
        valid_loss = 0.0
        outputs = []
        for valid_batch, target in valid_loader:
            valid_batch, target = valid_batch.to(config.device), target.to(config.device)
            output, y_1, y_2 = model(valid_batch)
            valid_loss += loss_fn(output.reshape(-1).float(), target.float()).float()
            outputs.append(output.reshape(-1).float().cpu().detach())
        print('Valid Loss: {:.6f}'.format(valid_loss/valid_size))
        valid_losses.append(valid_loss.item()/valid_size)
        if config.thresh == 10:
            new_threshold = statistics.quantiles([abs(x) for x in outputs], n=20)[18]
        elif config.thresh == -1:
            new_threshold = min([abs(x) for x in outputs])
        elif config.thresh == 3:
            new_threshold = max([abs(x) for x in outputs])
        else:
            new_threshold = statistics.quantiles([abs(x) for x in outputs], n=4)[config.thresh]
        count = 0
        if train_loss < loss_min:
            loss_min = train_loss
            count = 0
            best_model = model
            threshold = new_threshold
        else:
            count = count + 1
            if count>config.patience:
                break
    with open('./lstmrnd_data.pkl', 'wb') as f:
        pickle.dump(outputs, f)
    print('training end')
    print('threshold_1:', threshold)
    return best_model, threshold


def retrain(model, name, feature_lists):
    print('Using RND threshold: ' + str((config.thresh+1)/4))
    model.to(config.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_min = 100000000
    retrain_data = []

    if name == 'fetal_health' or name == 'breast_cancer' or name == 'cardio_train' or name == 'alzheimers':
        Dg = pd.read_csv('./data/processed/' + name + '.csv')
    else:
        Dg = pd.read_hdf('./data/processed/' + name + '.hdf')  
    original_features = list(Dg.columns)
    original_len = len(original_features)
    features = []
    data = []
    for featurelist in feature_lists:
        if featurelist == None:
            continue
        for feature in featurelist:
            feature = str(feature)
            if feature in features:
                continue
            else:
                features.append(feature)
                if feature == original_features[-1]:
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
                feature = feature.replace('_','')
            feature = feature.split('=')
            feature = [item for item in feature if item != '']
            feature = [int(item) for item in feature]
            data = torch.Tensor(feature.copy()).long()
            data = data.reshape(1, -1)
            data = data.to(config.device)
            retrain_data.append(data)

    for epoch in range(5):
        train_loss = 0.0
        for batch_idx, data in enumerate(retrain_data):
            target = [0.0]
            target = torch.Tensor(target)
            target = target.reshape(1, -1)
            data, target = data.to(config.device), target.to(config.device)
            optimizer.zero_grad()
            output, _, _ = model(data)
            loss_fn = nn.MSELoss()
            loss = loss_fn(output.float(), target.float())
            loss.backward()
            optimizer.step()
            if batch_idx%50==0:
                target = target.float().cpu()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, (batch_idx + 1) * len(data), len(retrain_data),
                            100. * (batch_idx + 1) / len(retrain_data), loss.item()))
            train_loss += loss.item()
        print("train loss:", train_loss / len(retrain_data))
    return model
