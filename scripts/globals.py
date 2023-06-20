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
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn

def init(): 

    global wandb_on 
    wandb_on = 0 # (1 = True, 0 = False)

    global run_name
    run_name = 'TCN_finetuning_oron'
    
    #### Device ####
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

if __name__=='__main__':
    pass