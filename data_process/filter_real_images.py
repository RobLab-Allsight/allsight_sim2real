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
real_paths = [f"/home/{pc_name}/Documents/repose/Allsight_sim2real/allsight_sim2real/datasets/data_Allsight/all_data/allsight_dataset/{gel}/{leds}/data/{ind}" for ind in indenter]

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
   
df_data_real = df_data_real[df_data_real.time > 2.0]  # only over touching samples!
df_data_real = df_data_real.sample(n=n_sam)

old_path = "/home/osher/catkin_ws/src/allsight/dataset/"
new_path = f"/home/{pc_name}/Documents/repose/Allsight_sim2real/allsight_sim2real/datasets/data_Allsight/all_data/allsight_dataset/"

df_data_real['frame'] = df_data_real['frame'].str.replace(old_path, new_path)
    
###########################
# Save real image df
########################### 

# for idx in range(int(n_sam/2)):
#     real_image = (cv2.imread(df_data_real['frame'][idx])).astype(np.uint8)
#     save_path = f'/home/roblab20/Documents/repose/Allsight_sim2real/allsight_sim2real/datasets/data_Allsight/filltered_real_data/train/{idx}.jpg'  # Specify the path where you want to save the image
#     cv2.imwrite(save_path, real_image)

# for idx in range(int(n_sam/2)):
#     real_image = (cv2.imread(df_data_real['frame'][idx+(int(n_sam/2))])).astype(np.uint8)
#     save_path = f'/home/roblab20/Documents/repose/Allsight_sim2real/allsight_sim2real/datasets/data_Allsight/filltered_real_data/test/{idx}.jpg'  # Specify the path where you want to save the image
#     cv2.imwrite(save_path, real_image)
    
###########################
# Show 1 example image
########################### 

real_image = (cv2.imread(df_data_real['frame'][20])).astype(np.uint8)
cv2.imshow('real', real_image)
cv2.waitKey(0)