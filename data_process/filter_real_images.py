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
gel = 'clear'
indenter = ['20', '30', '40']
data_name_1 = 'real_train_1k'
data_name_2 = 'real_test_1k'
real_paths = [f"/home/{pc_name}/Documents/repose/Allsight_sim2real/allsight_sim2real/datasets/data_Allsight/all_data/allsight_dataset/{gel}/{leds}/data/{ind}" for ind in indenter]
JSON_FILE_1 = f"/home/{pc_name}/Documents/repose/Allsight_sim2real/allsight_sim2real/datasets/data_Allsight/json_data/{data_name_1}.json"
JSON_FILE_2 = f"/home/{pc_name}/Documents/repose/Allsight_sim2real/allsight_sim2real/datasets/data_Allsight/json_data/{data_name_2}.json"

n_sam = 2000   
###########################
# Concat
###########################

buffer_real_paths = []
for p in real_paths:
    buffer_real_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], '*.json'))]
buffer_real_paths = [p for p in buffer_real_paths if ('transformed_annotated' in p)]


for idx, p in enumerate(buffer_real_paths):
    if idx == 0:
        df_data_real = pd.read_json(p).transpose()
    else:
        df_data_real = pd.concat([df_data_real, pd.read_json(p).transpose()], axis=0)
        
###########################
# Filter and sample
###########################        
   
df_data_real = df_data_real[df_data_real.time > 3.0]  # only over touching samples!
df_data_real = df_data_real.sample(n=n_sam)

old_path = "/home/osher/catkin_ws/src/allsight/dataset/"
new_path = f"/home/{pc_name}/Documents/repose/Allsight_sim2real/allsight_sim2real/datasets/data_Allsight/all_data/allsight_dataset/"

df_data_real['frame'] = df_data_real['frame'].str.replace(old_path, new_path)

df_train_real = df_data_real.iloc[:int(n_sam/2),:]
print(df_train_real.shape)
df_test_real =  df_data_real.iloc[int(n_sam/2):,:]
print(df_test_real.shape)

###########################
# Save real df to json
########################### 
# import json

# to_dict = {}
# for index, row in list(df_train_real.iterrows()):
#     to_dict[index] = dict(row)
# with open(r'{}_transformed.json'.format(JSON_FILE_1[:-5]), 'w') as json_file:
#     json.dump(to_dict, json_file, indent=3)
    
# to_dict2 = {}
# for index, row in list(df_test_real.iterrows()):
#     to_dict2[index] = dict(row)
# with open(r'{}_transformed.json'.format(JSON_FILE_2[:-5]), 'w') as json_file:
#     json.dump(to_dict2, json_file, indent=3)
    

real_image = (cv2.imread(df_data_real['frame'][20])).astype(np.uint8)
cv2.imshow('real', real_image)
cv2.waitKey(0)