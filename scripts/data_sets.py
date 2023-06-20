####################################################
# Import
####################################################

import globals 
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

import time


class Allsight_Dataset(Dataset):
  def __init__(self,csvpath ,seq ,mode = 'train',transforms=None):
        self.mode = mode       
        self.transform = transforms
        df = pd.read_csv(csvpath)
        np_train_f = np.array(df.iloc[:,1:-10], dtype=np.float32)
        np_train_l = np.array(df.iloc[:,-10:], dtype=np.float32)
        a,b = np_train_f.shape
        np_f = np.zeros((a, seq, b))
        for i in range(len(np_train_f)):
          for k in range(seq):
            if(i-seq+1+k < 0): l=0
            else: l = i - (seq -1 -k)
            # print(i,k,l)
            np_f[i,k,:] = np_train_f[l,:]

        self.inp = torch.from_numpy(np_f)
        self.oup = torch.from_numpy(np_train_l)

  def __len__(self):
    return len(self.inp)


  def __getitem__(self, idx):

    inp = self.inp[idx]
    oup = self.oup[idx]

    if self.transform:
            for t in self.transform:
                inp = t(inp) # Chose class transform based on y

    return inp, oup

class WhiteNoise(torch.nn.Module):
    def __init__(self,noiseFactor = 0.1, prob=0.5) -> None:
        super().__init__()
        self.prob = prob
        self.noiseFactor = noiseFactor

    def forward(self,x):
        if random.random() < self.prob:
            x_T = x + self.noiseFactor*torch.normal(0,0.1,size=(x.shape))
            return x_T
        return x

class Normalize(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x):
        mean = torch.mean(x,axis=0)
        std = torch.std(x,axis=0)
        std[std==0] = 1
        y = (x - mean)/std 
        return y

class vec2mat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x):
        x = x.detach().numpy()
        xx = np.array([[x[11],x[12],x[7],x[13],x[9],x[10],x[8]],
              [x[23],x[27],x[25],x[26],x[24],x[22],x[21]],
              [x[0],x[2],x[4],x[5],x[6],x[1],x[3]],
              [x[14],x[18],x[20],x[19],x[16],x[17],x[15]]]) 
        xx = torch.from_numpy(xx)
        return xx.T

if __name__=='__main__':
    pass