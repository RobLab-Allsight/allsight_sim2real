###########################
# Import
###########################
import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
import random
import argparse

random.seed(42)

def get_third_element(lst):
    return lst[2]



def main(args):
    
    ###########################
    # Define paths and data
    ###########################
    pc_name = os.getlogin()
    leds = 'rrrgggbbb'
    gel = 'clear' #clear / markers
    indenter = ['sphere3'] # id 3 only 20 (3mm radius)
    data_name_1 = f'real_train_{args.real_data_num}'
    data_name_2 = f'real_test_{args.real_data_num}'
    real_paths = [f"./datasets/data_Allsight/all_data/allsight_dataset/{gel}/{leds}/data/{ind}" for ind in indenter]
    JSON_FILE_1 = f"./datasets/data_Allsight/json_data/{data_name_1}.json"
    JSON_FILE_2 = f"./datasets/data_Allsight/json_data/{data_name_2}.json"

    n_sam = 6000   
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

    df_data_real['ft_z'] = df_data_real['ft_ee_transformed'].apply(get_third_element)
        
    ###########################
    # Filter and sample
    ###########################        

    # df_data_real = df_data_real[df_data_real.time > 3.0]  # only over touching samples!
    # df_data_real = df_data_real[df_data_real.depth > 0.002]
    # df_data_real = df_data_real[df_data_real.depth < 0.01247]
    df_data_real = df_data_real[df_data_real.ft_z < -5.9]
    df_data_real = df_data_real.sample(n=n_sam, random_state=42)

    old_path = "/home/osher/catkin_ws/src/allsight/dataset/"
    new_path = f"./datasets/data_Allsight/all_data/allsight_dataset/"

    df_data_real['frame'] = df_data_real['frame'].str.replace(old_path, new_path)

    df_train_real = df_data_real.iloc[:int(n_sam*0.8),:]
    print(df_train_real.shape)
    df_test_real =  df_data_real.iloc[int(n_sam*0.8):,:]
    print(df_test_real.shape)
    
    # df_train_real = df_train_real.sample(n=1000, random_state=42)
    ###########################
    # Save real df to json
    ########################### 
    if args.save:
        import json

        to_dict = {}
        for index, row in list(df_train_real.iterrows()):
            to_dict[index] = dict(row)
        with open(r'{}_transformed.json'.format(JSON_FILE_1[:-5]), 'w') as json_file:
            json.dump(to_dict, json_file, indent=3)

        to_dict2 = {}
        for index, row in list(df_test_real.iterrows()):
            to_dict2[index] = dict(row)
        with open(r'{}_transformed.json'.format(JSON_FILE_2[:-5]), 'w') as json_file:
            json.dump(to_dict2, json_file, indent=3)
    
    
    
    
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images and related JSON data.')
    parser.add_argument('--real_data_num', type=int, default= 5, help='real JSON path')
    parser.add_argument('--save', default=False)
    args = parser.parse_args()

    main(args)