####################################################
# Import
####################################################

import globals 
from utils import angles_error, estimate_model, check_model, load_data

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


###############################################
# Train
###############################################
    
def train_gan(model, train_dataloader, test_loader, epochs = 100, lr=0.02):
    test_ang_err =100
    prevTestLoss = 100
    print('Starts fitting')

    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss #more options: L1Loss, MSELoss(l2), 

    # train loop 
    n_total_steps = len(train_dataloader)

    for epoch in range(epochs):
        losses = 0
        b = 0
        model.train()
        for i, (features, labels) in enumerate(train_dataloader): 
            features=features.to(globals.device)
            labels=labels.to(globals.device)
            # Forward pass
            prediction = model(features)
            loss = loss_func(prediction, labels)
            losses = losses + loss
            b = b + 1
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            if (i+1) % 1400 == 0:
                print (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.5f}')

        test_ang_err = estimate_model(model, test_loader, loss_func, losses, b, epoch, epochs)
        if(test_ang_err < prevTestLoss):
            checkpoint_path = f"/home/roblab20/Documents/repose/new_system/Tele_manipulation_of_robotic_hand/Regression/scripts/torch_models_1/saved_models/checkpoint{globals.run_name}_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_ang_err,
                }, checkpoint_path)
            if(globals.wandb_on == 1):
                wandb.save(checkpoint_path)    
            prevTestLoss = test_ang_err

    model.eval()

    return model


if __name__=='__main__':

    pass