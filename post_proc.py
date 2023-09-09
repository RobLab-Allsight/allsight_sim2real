# import
import os
import cv2
import re
import argparse
import sys
import time
import os
import json
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import sys

import copy
from train_allsight_regressor.misc import normalize, unnormalize, normalize_max_min, unnormalize_max_min, save_df_as_json
from train_allsight_regressor.vis_utils import Arrow3D
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR  # Learning rate schedulers
from train_allsight_regressor.models import PreTrainedModel, PreTrainedModelWithRef
from train_allsight_regressor.vis_utils import data_for_cylinder_along_z, data_for_sphere_along_z, set_axes_equal
from train_allsight_regressor.datasets import TactileSimDataset, output_map, get_buffer_paths_sim, CircleMaskTransform
from train_allsight_regressor.surface import create_finger_geometry
from train_allsight_regressor.geometry import convert_quat_wxyz_to_xyzw, convert_quat_xyzw_to_wxyz
from transformations import quaternion_matrix
from scipy import spatial
from tqdm import tqdm
import random
from models import networks, pre_process
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
np.set_printoptions(suppress=True, linewidth=np.inf)  # to widen the printed array

pc_name = os.getlogin()

random.seed(42)
torch.manual_seed(42)


def main(args):
    random.seed(42)
    
    real_regressor = networks.define_regressor('./checkpoints/regression_models/white/real_8_ref.pth', [0])
    json_gan_p = f'./datasets/data_Allsight/json_data/{args.gan_type}_test_{args.gan_num}_{args.data_num}_{args.gan_epoch}_transformed_ref.json'
    json_gan_p_new = f'./datasets/data_Allsight/json_data/{args.gan_type}_test_{args.gan_num}_{args.data_num}_{args.gan_epoch}_transformed_opt.json'
    
    loss_lim = 0.05
    
    df_data = pd.read_json(json_gan_p).transpose()
    
    idx_list = []
    loss_list = []
    
    Y = np.array([df_data.iloc[id].pose_transformed[0][:3] for id in range(df_data.shape[0])])


    y_mean = Y.mean(axis=0)
    y_std = Y.std(axis=0)
    y_max = Y.max(axis=0)
    y_min = Y.min(axis=0)

    data_statistics = {'mean': y_mean, 'std': y_std, 'max': y_max, 'min': y_min}  
    
    for idx in range(len(df_data)):
        with torch.no_grad():
            ref_frame = cv2.resize(cv2.cvtColor(cv2.imread(df_data['ref_frame'][idx]), cv2.COLOR_BGR2RGB),(224,224))
            ref_frame = torch.from_numpy(ref_frame.transpose((2, 0, 1))).float().to(device).unsqueeze(0)
            frame = cv2.resize(cv2.cvtColor(cv2.imread(df_data['frame'][idx]), cv2.COLOR_BGR2RGB),(224,224))
            frame = torch.from_numpy(frame.transpose((2, 0, 1))).float().to(device).unsqueeze(0)
            
            pred_px = real_regressor(frame, ref_frame).to(device)
            
            y = torch.Tensor(Y[idx])
            y = normalize(y, data_statistics['mean'], data_statistics['std']).float()
            y = y.view(1, -1).to(device)            
            loss = nn.functional.mse_loss(pred_px,y)
            loss_list.append(loss.item())
            
            if loss > loss_lim:
                idx_list.append(idx) 
            
    print(f'number of samples to remove: {len(idx_list)}')
    print(f'avg loss: {np.mean(np.array(loss_list))}')
    
    #remove idx
    
    if args.save:

        # Save gan df to json
        to_dict = {}
        for index, row in list(df_data.iterrows()):
            to_dict[index] = dict(row)
        with open(f'{json_gan_p_new}', 'w') as json_file:
            json.dump(to_dict, json_file, indent=3)
            
            print("[INFO] json optimized")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images and related JSON data.')
    parser.add_argument('--data_num', type=int, default= 8, help=' JSON path')
    parser.add_argument('--data_kind', type=str, default='transformed', help='transformed, aligned')
    parser.add_argument('--gan_num', type=str, default= 69)
    parser.add_argument('--gan_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--gan_type', type=str, default='diff_cgan', help='cgan, distil_cgan, mask_cgan')
    parser.add_argument('--save', default=False)
    args = parser.parse_args()

    main(args)
    
    



