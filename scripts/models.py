####################################################
# Import
####################################################

import numpy as np
import pandas as pd
import wandb
from matplotlib import pyplot as plt
import pickle
import sys
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
# from torch.nn.utils import weight_norm

import time


# my_net: LSTM -> FC -> RELU 
class LSTM_(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_outputs):
        super(LSTM_, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        # or:
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        ###### fc layers
        self.fc1 = nn.Linear(hidden_size, 112)
        self.fc2 = nn.Linear(112, 112)
        # self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(112, hidden_size)

        self.fc = nn.Linear(hidden_size, num_outputs)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(globals.device) 
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(globals.device) #just for lstm
        x = x.float() # x: (b_size, seq_s, input_s), h0: (n_layers, b_size, h_size)
        # Forward propagate RNN
        # out, _ = self.gru(x, h0) #/gru  
        # or:
        out, _ = self.lstm(x,)  # (h0,c0)
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
                
        # Decode the hidden state of the last time step
        out = out[:, -1, :] # out: (b_size, 128)

        ###### fc layers
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        # out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        ######
                
        out = self.fc(out) # out: (n, 10)
        
        out = self.tanh(out)
        return out

# ConFNet4 : eran net - cnn - lstm - fc - bn    (non sequencial data)
class ConFNet4(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConFNet4, self).__init__()
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=1, stride=1, padding=1) 
        self.cnn2 = nn.Conv1d(8, 64, kernel_size=1, stride=1, padding=1) 
        self.rnn = nn.LSTM(input_size=32, hidden_size=5, num_layers=1, batch_first=True)
        self.linear_1 = nn.Linear(320, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64,32)#(64, 256)
        # self.linear_4 = nn.Linear(256, 16)
        # self.linear_5 = nn.Linear(16, 256)
        # self.linear_6 = nn.Linear(256, 64)
        # self.linear_7 = nn.Linear(64, 32)
        self.linear_8 = nn.Linear(32, output_dim)
        self.activation = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.bn3 = nn.BatchNorm1d(num_features=32) #256->32
        # self.bn4 = nn.BatchNorm1d(num_features=16)
        # self.bn5 = nn.BatchNorm1d(num_features=256)
        self.bnc = nn.BatchNorm1d(num_features=8) 
        self.bnc2 = nn.BatchNorm1d(num_features=64)
        self.Dropout = nn.Dropout(0.1)
        self.LKR = nn.LeakyReLU(negative_slope=0.01)
        self.Sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=False)  

    def forward(self, x):
        x_c = self.LKR(self.bnc(self.cnn1(x)))
        x_c2 = self.LKR(self.bnc2(self.cnn2(x_c)))
        outRNN, h_h = self.rnn(x_c2)
        x_flatten = outRNN.reshape(-1, 64*5)
        x1 = self.activation(self.bn1(self.linear_1(x_flatten)))
        x1_d = self.Dropout(x1)
        x2 = self.activation(self.bn2(self.linear_2(x1_d)))
        x2_d = self.Dropout(x2)
        ## x2_d = x2_d + x1_d
        x3 = self.activation(self.bn3(self.linear_3(x2_d)))
        x3_d = self.Dropout(x3)
        ## x3_d = x3_d + x2_d
        # x4 = self.activation(self.bn4(self.linear_4(x3_d)))
        # x4_4 = self.Dropout(x4)
        ## x4_4 = x4_4 + x3_d
        # x5 = self.activation(self.bn5(self.linear_5(x4_4)))
        # x5 = x3_d + x5
        # x6 = self.activation(self.linear_6(x5))
        # x6 = x6 + x2_d
        # x7 = self.activation(self.linear_7(x6))
        ## x7 = x7 + x6
        x8 =self.relu(self.linear_8(x3_d)) #x3_d->x7 #self.relu -> self.Sigmoid
        return x8

# Regressor : simple mlp (daniel) (regressor)   (non sequencial data)
class Regressor(torch.nn.Module):
    def __init__(self,input_shape,depth) -> None:
        super().__init__()
        self.depth = depth

        self.l1 = torch.nn.Linear(input_shape,56)
        self.l2 = torch.nn.Linear(56,224) ### 112 -> 224
        
        self.ln = torch.nn.ModuleList([torch.nn.Linear(224,224) for i in range(depth)]) #torch.nn.ModuleList
        self.drop = torch.nn.ModuleList([torch.nn.Dropout(np.random.rand()/3) for i in range(depth)]) #torch.nn.ModuleList
        
        self.l3 = torch.nn.Linear(224,56)
        self.l4 = torch.nn.Linear(56,10)

        self.relu = torch.nn.ReLU(inplace=False)     

    def forward(self,x):
        y = self.l1(x)
        y = self.relu(y)

        y = self.l2(y)
        y = self.relu(y)

        for i in range(self.depth):
            y = self.ln[i](y)
            y = self.drop[i](y)
            y = self.relu(y)

        y = self.l3(y)
        y = self.relu(y)

        y = self.l4(y)
        y = self.relu(y)

        return y
# CNN : (not working good) 
class CNN_(nn.Module):
    def __init__(self, input_size, hidden_size, num_outputs, k=3, hidden_2 = 48):
        super(CNN_, self).__init__()
        
        ###### cnn layers 
        ## input cnn (batch=64, input(7X4) hidden=54)
        self.conv1 = nn.Conv2d(1, hidden_size, k, stride=1, padding=1) #?= Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(hidden_size, hidden_2, k, stride=1, padding=1)
        self.fc1 = nn.Linear(2*k*(hidden_2), 120)
        self.fc2 = nn.Linear(120, hidden_size)
        self.fc = nn.Linear(hidden_size, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, x):

        ###### cnn layers
        out = x.unsqueeze(1)
        ## out = out.permute(1,0,2,3)
        out = self.conv1(out)
        out = self.pool(out)
        out = self.conv2(out)
        c = out.shape
        c2 = c[1]*c[2]*c[3]
        out = out.view(-1, c2) #
        out = self.fc1(out)
        out = self.fc2(out)                
        out = self.fc(out) # out: (n, 10)
        
        out = self.relu(out)
        return out
    
    
class FullyC_oupinp(torch.nn.Module):
    def __init__(self,input_shape,depth, hidden_size) -> None:
        super().__init__()
        self.depth = depth
        self.l1 = torch.nn.Linear(input_shape,76)
        self.l2 = torch.nn.Linear(76,hidden_size)
        self.ln = torch.nn.ModuleList([torch.nn.Linear(hidden_size,hidden_size) for i in range(depth)]) 
        self.drop = torch.nn.ModuleList([torch.nn.Dropout(np.random.rand()/3) for i in range(depth)]) 
        self.l3 = torch.nn.Linear(hidden_size,50)
        self.l4 = torch.nn.Linear(50,10)
        self.relu = torch.nn.ReLU(inplace=False)     

    def forward(self,x):
        x = x.float()
        y = self.l1(x)
        y = self.relu(y)
        y = self.l2(y)
        y = self.relu(y)
        for i in range(self.depth):
            y = self.ln[i](y)
            y = self.drop[i](y)
            y = self.relu(y)

        y = self.l3(y)
        y = self.relu(y)
        y = self.l4(y)
        y = self.relu(y)

        return y
    
    

if __name__=='__main__':
    pass