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
from torch.nn.utils import weight_norm

import globals 
from models import LSTM_, ConFNet4, Regressor, CNN_
from train import train_gan
from data_sets import HandDataset_seq, WhiteNoise, Normalize, vec2mat
from utils import angles_error, estimate_model, check_model, load_data


####################################################
# Main function
####################################################

def main():

    globals.init()

    ####################################################
    # Define initial config
    ####################################################
    
    config = { 
        "input_dim" : 28,
        "output_dim" : 10,
        "batch_size" : 128,
        "n_hidden" : 54,
        "num_layers" : 2,
        "seq" : 60,
        "lr" : 0.00001,
        "max_epochs" : 50,
        "depth" : 8,
        "run_group": "non",
        "kernel_num": 10,
        "prob_": 0.3,
        "noise_fac": 0.05,
        "hidden_2": 28,
        "size": 100,
        "n_lay": 2,
        "dropout": 0.1,
    }    
    
    ####################################################
    # starts a wandb session
    ####################################################
    
    if(globals.wandb_on == 1):
        run = wandb.init(entity="roblab_tau_alon",config = config)
        config = wandb.config
    print(config)

    ####################################################
    # Define all Models (using your config)
    ####################################################
    
    gan_models = {
            "fc_" : Regressor(config["input_dim"],
                                config["depth"]),
            "lstm_" : LSTM_(input_size=config["input_dim"],
                                hidden_size=config["n_hidden"], 
                                num_layers=config["num_layers"], 
                                num_outputs=config["output_dim"]),
            "conFnet4" : ConFNet4(input_dim=config["input_dim"], 
                                    output_dim =config["output_dim"]),
            "cnn_" : CNN_(input_size=config["input_dim"],
                                hidden_size=config["n_hidden"], 
                                num_outputs=config["output_dim"],
                                k=config["kernel_num"], 
                                hidden_2 =config["hidden_2"]),
    }
    
    ####################################################
    # Load and Arange the Data 
    ####################################################
    seq = config["seq"]
    b_size = config["batch_size"]
    train_data_path = '/home/roblab20/Documents/repose/new_system/Tele_manipulation_of_robotic_hand/Regression/Data/Data_after_preproccesing/train_001.csv'
    test_data_path = '/home/roblab20/Documents/repose/new_system/Tele_manipulation_of_robotic_hand/Regression/Data/Data_after_preproccesing/test_001.csv'


    # Augmantations
    transforms = nn.Sequential(
                        # vec2mat(),
                        # Normalize(),
                        # WhiteNoise(config["noise_fac"],config["prob_"])
    )

    test_transforms = nn.Sequential(
                        # vec2mat(),
                        # Normalize()
    )
    
    train_dataloader, test_dataloader = load_data(train_data_path, test_data_path, None, None, b_size, seq,transforms, test_transforms) 


    ####################################################
    # Choose a model 
    ####################################################

    model = gan_models["tcn_"]
    model = model.to(globals.device)

    ####################################################
    # Training 
    ####################################################
    
    epochs = config["max_epochs"]
    lr = config["lr"]
    model  = train_gan(model, train_dataloader, test_dataloader, epochs, lr)

    ###################################################
    # Test
    ###################################################
    check_model(model, train_dataloader, test_dataloader)

    if(globals.wandb_on == 1):
        wandb.finish()

####################################################

####################################################

if __name__ == "__main__":
  
    main()