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

def main(args):
    ###########################
    # Define paths and data
    ###########################

    leds = args.leds
    gel = 'clear' #clear / markers
    indenter = ['sphere3', 'cube']
    data_name_1 = f'sim_train_{args.sim_data_num}'
    real_paths = [f"./datasets/data_Allsight/all_data/allsight_sim_dataset/{gel}/{leds}/data/{ind}" for ind in indenter]
    JSON_FILE_1 = f"./datasets/data_Allsight/json_data/{data_name_1}.json"

    buffer_sim_paths = []
    for p in real_paths:
        buffer_sim_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], '*.json'))]

    for idx, p in enumerate(buffer_sim_paths):
        if idx == 0:
            df_data_sim = pd.read_json(p).transpose()
        else:
            df_data_sim = pd.concat([df_data_sim, pd.read_json(p).transpose()], axis=0)
    
    df_data_sim = df_data_sim.sample(n=args.samples, random_state=42)
    ###########################
    # Filter and sample
    ###########################        
    old_path = "allsight_sim_dataset/"
    new_path = f"./datasets/data_Allsight/all_data/allsight_sim_dataset/"

    df_data_sim['frame'] = df_data_sim['frame'].str.replace(old_path, new_path)
    df_data_sim['frame'] = df_data_sim['frame'].str.replace(":", "_")
    
    df_data_sim['ref_frame'] = df_data_sim['ref_frame'].str.replace(old_path, new_path)
    df_data_sim['ref_frame'] = df_data_sim['ref_frame'].str.replace(":", "_")

    print(df_data_sim.shape)

    ###########################
    # Save real df to json
    ########################### 
    if args.save:
        import json

        to_dict = {}
        for index, row in list(df_data_sim.iterrows()):
            to_dict[index] = dict(row)
        with open(r'{}_transformed.json'.format(JSON_FILE_1[:-5]), 'w') as json_file:
            json.dump(to_dict, json_file, indent=3)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images and related JSON data.')
    parser.add_argument('--sim_data_num', type=int, default= 6, help='sim JSON path')
    parser.add_argument('--samples', type=int, default= 5000, help='sim JSON path')
    parser.add_argument('--leds', type=str, default='white', help='rrrgggbbb | white')
    parser.add_argument('--save', default=False)
    args = parser.parse_args()

    main(args)