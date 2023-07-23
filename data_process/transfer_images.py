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

# update paths:
json_p = '/home/roblab20/Documents/repose/Allsight_sim2real/allsight_sim2real/datasets/data_Allsight/json_data/real_train_8k_transformed.json' ##
trans_folder_path = '/home/roblab20/Documents/repose/Allsight_sim2real/allsight_sim2real/datasets/data_Allsight/'
data_type = 'trainA/' # trainA, trainB, testA, testB.


df_data = pd.read_json(json_p).transpose()

###########################
# Save real image df
########################### 
df_data = df_data.sample(n=1391)
for idx in range(len(df_data)):
    real_image = (cv2.imread(df_data['frame'][idx])).astype(np.uint8)
    save_path = trans_folder_path + data_type+ f'{idx}.jpg'  # Specify the path where you want to save the image
    cv2.imwrite(save_path, real_image)
    
###########################
# Show 1 example image
########################### 


# real_image = (cv2.imread(df_data['frame'][20])).astype(np.uint8)
# cv2.imshow('real', real_image)
# cv2.waitKey(0)