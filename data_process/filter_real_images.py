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
import json

random.seed(42)

def get_third_element(lst):
    return lst[2]

def get_buffer_paths(leds, gel, indenter, train_sensor_id=[1,2,4], test_sensor_id=[3,]):
    # sensor id = train sensors id
    trained_sensor_id_final = []
    test_sensor_id_final = []

    buffer_paths_to_train = []
    buffer_paths_to_test = []

    if leds == 'combined':
        leds_list = ['rrrgggbbb', 'rgbrgbrgb', 'white']
    else:
        leds_list = [leds]

    for l in leds_list:
        paths = [f"./datasets/data_Allsight/all_data/allsight_dataset/{gel}/{l}/data/{ind}" for ind in indenter]
        buffer_paths = []
        summ_paths = []

        for p in paths:
            buffer_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], '*_transformed_annotated.json'))]
            summ_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], 'summary.json'))]

        for bp, s in zip(buffer_paths, summ_paths):
            with open(s, 'rb') as handle:
                summ = json.load(handle)
                summ['sensor_id'] = summ['sensor_id'] if isinstance(summ['sensor_id'], list) else [summ['sensor_id']]
            
            for sm in summ['sensor_id']:
                if sm in train_sensor_id:
                    buffer_paths_to_train.append(bp)
                    trained_sensor_id_final.append(sm)
                elif sm in test_sensor_id:
                    buffer_paths_to_test.append(bp)
                    test_sensor_id_final.append(sm)
                else: print(f'Sensor {sm} is not in the train set and not in the test set')

    return buffer_paths_to_train, buffer_paths_to_test, list(set(trained_sensor_id_final)), list(set(test_sensor_id_final))

def main(args):
    
    ###########################
    # Define paths and data
    ###########################
    pc_name = os.getlogin()
    leds = args.leds
    gel = 'clear' #clear / markers
    indenter = ['sphere3', 'ellipse', 'square', 'hexagon'] # id 3 only 20 (3mm radius)
    data_name_1 = f'real_train_{args.real_data_num}'
    data_name_2 = f'real_test_{args.real_data_num}'
    real_paths = [f"./datasets/data_Allsight/all_data/allsight_dataset/{gel}/{leds}/data/{ind}" for ind in indenter]
    JSON_FILE_1 = f"./datasets/data_Allsight/json_data/{data_name_1}.json"
    JSON_FILE_2 = f"./datasets/data_Allsight/json_data/{data_name_2}.json"

    n_sam_train = 6000  
    n_sam_test = 6000   
    ###########################
    # Concat
    ###########################

    # buffer_real_paths = []
    # for p in real_paths:
    #     buffer_real_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], '*.json'))]
    # buffer_real_paths = [p for p in buffer_real_paths if ('transformed_annotated' in p)]

    buffer_train_paths, buffer_test_paths, sensors_1, sensors_2 = get_buffer_paths(leds, gel, indenter,  train_sensor_id=[1,2,4], test_sensor_id=[3])

    for idx, p in enumerate(buffer_train_paths):
        if idx == 0:
            df_data_real_train = pd.read_json(p).transpose()
        else:
            df_data_real_train = pd.concat([df_data_real_train, pd.read_json(p).transpose()], axis=0)

    df_data_real_train['ft_z'] = df_data_real_train['ft_ee_transformed'].apply(get_third_element)
    
    for idx, p in enumerate(buffer_test_paths):
        if idx == 0:
            df_data_real_test = pd.read_json(p).transpose()
        else:
            df_data_real_test = pd.concat([df_data_real_test, pd.read_json(p).transpose()], axis=0)

    df_data_real_test['ft_z'] = df_data_real_test['ft_ee_transformed'].apply(get_third_element)    
   
    
        
    ###########################
    # Filter and sample
    ###########################        

    # df_data_real = df_data_real[df_data_real.time > 3.0]  # only over touching samples!
    # df_data_real = df_data_real[df_data_real.depth > 0.002]
    # df_data_real = df_data_real[df_data_real.depth < 0.01247]
    df_data_real_train = df_data_real_train[df_data_real_train.ft_z < -5.9]
    df_data_real_train = df_data_real_train.sample(n=n_sam_train, random_state=42)
    df_data_real_test = df_data_real_test[df_data_real_test.ft_z < -5.9]
    df_data_real_test = df_data_real_test.sample(n=n_sam_test, random_state=42)

    old_path = "/home/osher/catkin_ws/src/allsight/dataset/"
    new_path = f"./datasets/data_Allsight/all_data/allsight_dataset/"

    df_data_real_test['frame'] = df_data_real_test['frame'].str.replace(old_path, new_path)
    df_data_real_test['ref_frame'] = df_data_real_test['ref_frame'].str.replace(old_path, new_path)
    df_data_real_train['frame'] = df_data_real_train['frame'].str.replace(old_path, new_path)
    df_data_real_train['ref_frame'] = df_data_real_train['ref_frame'].str.replace(old_path, new_path)

   
    # df_train_real = df_train_real.sample(n=1000, random_state=42)
    ###########################
    # Save real df to json
    ########################### 
    if args.save:
        
        to_dict = {}
        for index, row in list(df_data_real_train.iterrows()):
            to_dict[index] = dict(row)
        with open(r'{}_transformed.json'.format(JSON_FILE_1[:-5]), 'w') as json_file:
            json.dump(to_dict, json_file, indent=3)

        to_dict2 = {}
        for index, row in list(df_data_real_test.iterrows()):
            to_dict2[index] = dict(row)
        with open(r'{}_transformed.json'.format(JSON_FILE_2[:-5]), 'w') as json_file:
            json.dump(to_dict2, json_file, indent=3)
    
    
    
    
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images and related JSON data.')
    parser.add_argument('--real_data_num', type=int, default= 6, help='real JSON path')
    parser.add_argument('--save', default=False)
    parser.add_argument('--leds', type=str, default='white', help='rrrgggbbb | white')
    args = parser.parse_args()

    main(args)