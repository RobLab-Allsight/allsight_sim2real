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
from PIL import Image
import re

# update paths:
epoch_model = 150 # latest / epoch num
json_p = '/home/roblab20/Documents/repose/Allsight_sim2real/allsight_sim2real/datasets/data_Allsight/json_data/sim_train_1_transformed.json' ##
images_folder_path = f'/home/roblab20/Documents/repose/Allsight_sim2real/allsight_sim2real/results/allsight_1/test_{epoch_model}/images/'
copy_to_path = '/home/roblab20/Documents/repose/Allsight_sim2real/allsight_sim2real/datasets/data_Allsight/cgan_data/test2/'
name = 'cgan'
data_name_1 = 'cgan_test_2'
JSON_FILE = f"/home/roblab20/Documents/repose/Allsight_sim2real/allsight_sim2real/datasets/data_Allsight/json_data/{data_name_1}.json"
df_data = pd.read_json(json_p).transpose()
df_data['old_frame'] = df_data['frame']

###########################
# copy to new folder
########################### 

def is_image_file(filename):
    # Check if the file has an image extension
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")
    return filename.lower().endswith(image_extensions)

def get_image_paths_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if is_image_file(f)]
    image_paths = [os.path.join(folder_path, image_file) for image_file in image_files]
    return image_paths


def get_image_number(image_path):
    filename = os.path.basename(image_path)
    image_name = os.path.splitext(filename)[0]
    match = re.search(r'\d+', image_name)
    return int(match.group()) if match else float('inf')


images_p = get_image_paths_from_folder(images_folder_path)
new_images_paths = []
print(len(images_p))
for i in range(len(images_p)):
    filename = os.path.basename(images_p[i])
    image_name_without_number = re.sub(r'^\d+_', '', os.path.splitext(filename)[0])
    if image_name_without_number == 'fake_A':
        new_images_paths.append(images_p[i])
        
sorted_image_paths = sorted(new_images_paths, key=get_image_number) 

###########################
# Save real image df
########################### 

for idx in range(len(sorted_image_paths)):
    real_image = (cv2.imread(sorted_image_paths[idx])).astype(np.uint8)
    save_path = copy_to_path + name + f'{idx}.jpg'  # Specify the path where you want to save the image
    cv2.imwrite(save_path, real_image)
    df_data['frame'][idx] = save_path
   
###########################
# Save real df to json
########################### 
import json

to_dict = {}
for index, row in list(df_data.iterrows()):
    to_dict[index] = dict(row)
with open(r'{}_transformed.json'.format(JSON_FILE[:-5]), 'w') as json_file:
    json.dump(to_dict, json_file, indent=3)
    