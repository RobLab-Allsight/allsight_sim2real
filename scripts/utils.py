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

import globals 
# from models import LSTM_, ConFNet4
# from train import train_regressor
from data_sets import HandDataset_seq, WhiteNoise, Normalize, HandDataset , HandDataset_seq_2, HandDataset_oupinp

####################################################
# Utils function
####################################################

def angles_error(outputs, labels):
    aver_err_ang = labels.detach().numpy() - outputs.detach().numpy()
    aver_err_ang = abs(aver_err_ang)
    if(globals.seq_model == 2): # not used
        aver_err_ang = (aver_err_ang)/(labels.detach().numpy())
        angles_error = np.mean(aver_err_ang[:,:],axis=0)*100
    else:
        angles_error = np.mean(aver_err_ang[:,:],axis=0)*90
    return angles_error

def estimate_model(model, test_loader, lossfunc, losses, b, epoch, epochs): # in the trainning
    model.eval()
    k = 0 
    t_losses = 0
    te_ang_s = 0
    
    br = 0
    
    with torch.no_grad():
        for i, (fea, lab) in enumerate(test_loader):     
            fea=fea.to(globals.device)
            lab=lab.to(globals.device)
            # Forward pass
            outs = model(fea)
            loss = lossfunc(outs, lab)
            k+=1
            t_losses = t_losses + loss  
            te_ang = angles_error(outs.cpu(), lab.cpu())
            te_ang_s = te_ang_s + te_ang
            
    te_loss = t_losses/k
    test_ang_err_mean = np.mean(te_ang_s/k)
    tr_loss = losses/b
    if(globals.wandb_on == 1):
        wandb.log({"train_loss_trainning": tr_loss,
                "test_loss_trainning": te_loss,
                "test_angles_trainning": test_ang_err_mean,
                }) 
    
    # if(epoch>4 and (te_loss-tr_loss)>0.025 and (te_loss)> 0.051): br = 1 
    
    print (f'Epoch [{epoch+1}/{epochs}], Train loss: {tr_loss:.5f}, Test loss: {te_loss:.5f}, Test angles error: {test_ang_err_mean:.5f}')
    return test_ang_err_mean#, br

def check_model(model, train_loader, test_loader): # end of trainning

    model.eval()
    lossfunc = torch.nn.MSELoss()
    k = 0
    b = 0 
    train_losses = 0
    tr_ang_s = 0
    test_losses = 0
    te_ang_s = 0
    with torch.no_grad():
        for i, (f_test, l_test) in enumerate(test_loader):     
            f_test=f_test.to(globals.device)
            l_test=l_test.to(globals.device)
            # Forward pass
            outs_test = model(f_test)
            loss_test = lossfunc(outs_test, l_test)
            k+=1
            te_ang = angles_error(outs_test.cpu(), l_test.cpu())
            te_ang_s = te_ang_s + te_ang
            test_losses = test_losses + loss_test   
            
        for j, (f_train, l_train) in enumerate(train_loader):
            f_train=f_train.to(globals.device)
            l_train=l_train.to(globals.device)     
            # Forward pass
            outs_train = model(f_train)
            loss_train = lossfunc(outs_train, l_train)
            b+=1
            tr_ang = angles_error(outs_train.cpu(), l_train.cpu())
            tr_ang_s = tr_ang_s + tr_ang
            train_losses = train_losses + loss_train 

    test_loss_final = test_losses/k
    train_loss_final = train_losses/b 
    test_ang_err = te_ang_s/k
    train_ang_err = tr_ang_s/b    
    print(f'Train angles error: {np.round(train_ang_err,2)}')  
    print(f'Train angles std: {np.std(train_ang_err):.3f}')
    print(f'Test angles error: {np.round(test_ang_err,2)}')
    print(f'Test angles std: {np.std(test_ang_err):.3f}')
    a_e_train = np.mean(train_ang_err)
    a_e_test = np.mean(test_ang_err)
    if(globals.wandb_on == 1):
        wandb.log({"train_loss_final": train_loss_final,
                "test_loss_final": test_loss_final,
                "train_angles_err": a_e_train,
                "test_angles_err": a_e_test
                }) 
    print (f'Results: Train loss: {train_loss_final:.5f}, Test loss: {test_loss_final:.5f}, train_angles_err: {a_e_train:.5f}, test_angles_err: {a_e_test:.5f}')
    return

def load_data(path_1, path_2, path_3, path_4, b_size, seq, transforms=None, test_transforms=None):
   
    
    if(globals.seq_model == 1):
        train_data = HandDataset_seq(path_1, seq,transforms=transforms)
        test_data = HandDataset_seq(path_2, seq,transforms=test_transforms)
        
    elif(globals.seq_model == 2):
        train_data = HandDataset_seq_2(path_1,path_2, transforms=transforms)
        test_data = HandDataset_seq_2(path_3,path_4, transforms=test_transforms)
    else:
        train_data = HandDataset(path_1, transforms=transforms)
        test_data = HandDataset(path_2, transforms=test_transforms)
        
        
    train_dataloader = DataLoader(dataset = train_data, batch_size=b_size,num_workers=16,persistent_workers=False, shuffle=True) 
    test_data_loader = DataLoader(dataset = test_data, batch_size=b_size,shuffle=True)

    return train_dataloader, test_data_loader


def load_data2(path_1, path_2, path_3, path_4, seq,transforms=None,test_transforms=None):
       
    
    if(globals.seq_model == 1): ############### tamporary
        train_data = HandDataset_oupinp(path_1, seq,transforms=transforms)
        test_data = HandDataset_oupinp(path_2, seq,transforms=test_transforms)
        
    elif(globals.seq_model == 2):
        train_data = HandDataset_seq_2(path_1,path_2, transforms=transforms)
        test_data = HandDataset_seq_2(path_3,path_4, transforms=test_transforms)
    else:
        train_data = HandDataset(path_1, transforms=transforms)
        test_data = HandDataset(path_2, transforms=test_transforms)
        
        
    train_dataloader = DataLoader(dataset = train_data, batch_size=256,num_workers=8,persistent_workers=False, shuffle=True) 
    test_data_loader = DataLoader(dataset = test_data, batch_size=256,shuffle=True)

    return train_dataloader, test_data_loader


if __name__=='__main__':
    pass