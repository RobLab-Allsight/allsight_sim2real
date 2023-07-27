###########################
# Import
###########################
import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
import random
import cv2

random.seed(42)

###########################
# Define paths and data
###########################

pc_name = os.getlogin()
leds = 'rrrgggbbb'
gel = 'clear' #clear / markers
indenter = ['sphere3', 'sphere4', 'sphere5']
data_name_1 = 'sim_train_2'
# data_name_2 = 'sim_test_1k'
real_paths = [f"/home/{pc_name}/Documents/repose/Allsight_sim2real/allsight_sim2real/datasets/data_Allsight/all_data/allsight_sim_dataset/{gel}/{leds}/data/{ind}" for ind in indenter]
JSON_FILE_1 = f"/home/{pc_name}/Documents/repose/Allsight_sim2real/allsight_sim2real/datasets/data_Allsight/json_data/{data_name_1}.json"
# JSON_FILE_2 = f"/home/{pc_name}/Documents/repose/Allsight_sim2real/allsight_sim2real/datasets/data_Allsight/json_data/{data_name_2}.json"

n_sam = 2000   
###########################
# Concat
###########################

# buffer_sim_paths = []
# for p in real_paths:
#     buffer_sim_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], '*.json'))]
# # buffer_sim_paths = [p for p in buffer_sim_paths if ('transformed_annotated' in p)]


# for idx, p in enumerate(buffer_sim_paths):
#     if idx == 1:
#         df_data_sim = pd.read_json(p).transpose()
#     # else:
#     #     df_data_sim = pd.concat([df_data_sim, pd.read_json(p).transpose()], axis=0)
json_sim_p = '/home/roblab20/Documents/repose/Allsight_sim2real/allsight_sim2real/datasets/data_Allsight/all_data/allsight_sim_dataset/clear/rrrgggbbb/data/sphere3/id3_data_2023_07_25-04_48_46/data_2023_07_25-04_48_46.json'
df_data_sim = pd.read_json(json_sim_p).transpose()       
###########################
# Filter and sample
###########################        
k=1
# df_data_sim = df_data_sim.sample(n=n_sam)
print(df_data_sim.shape)
old_path = "allsight_sim_dataset/"
new_path = f"/home/{pc_name}/Documents/repose/Allsight_sim2real/allsight_sim2real/datasets/data_Allsight/all_data/allsight_sim_dataset/"

df_data_sim['frame'] = df_data_sim['frame'].str.replace(old_path, new_path)
df_data_sim['frame'] = df_data_sim['frame'].str.replace(":", "_")
df_train_real = df_data_sim#.iloc[:int(n_sam/2),:]
print(df_train_real.shape)
# df_test_real =  df_data_sim.iloc[int(n_sam/2):,:]
# print(df_test_real.shape)

###########################
# Save real df to json
########################### 
import json

to_dict = {}
for index, row in list(df_train_real.iterrows()):
    to_dict[index] = dict(row)
with open(r'{}_transformed.json'.format(JSON_FILE_1[:-5]), 'w') as json_file:
    json.dump(to_dict, json_file, indent=3)
    
# to_dict2 = {}
# for index, row in list(df_test_real.iterrows()):
#     to_dict2[index] = dict(row)
# with open(r'{}_transformed.json'.format(JSON_FILE_2[:-5]), 'w') as json_file:
#     json.dump(to_dict2, json_file, indent=3)
    
