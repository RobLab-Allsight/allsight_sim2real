import numpy as np
import pandas as pd
import os
import random
import cv2
import re
import json
import argparse


def is_image_file(filename):
    # Check if the file has an image extension
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")
    return filename.lower().endswith(image_extensions)

def get_image_paths_from_folder(folder_path):
    # Get all image paths from a given folder
    image_files = [f for f in os.listdir(folder_path) if is_image_file(f)]
    image_paths = [os.path.join(folder_path, image_file) for image_file in image_files]
    return image_paths

def get_image_number(image_path):
    # Extract image number from image file name
    filename = os.path.basename(image_path)
    image_name = os.path.splitext(filename)[0]
    match = re.search(r'\d+', image_name)
    return int(match.group()) if match else float('inf')

def main(args):
    random.seed(42)
    
    json_sim_p = f'./datasets/data_Allsight/json_data/sim_train_{args.sim_data_num}_{args.data_kind}.json'
    json_gan_name = f'{args.name}_test_{args.cgan_num}_{args.sim_data_num}_{args.cgan_epoch}'
    images_folder_path = f'./results/allsight_{args.cgan_num}/test_{args.cgan_epoch}/images/'
    copy_to_path = f'./datasets/data_Allsight/{args.name}_data/test{args.cgan_num}_{args.cgan_epoch}/'
    JSON_FILE = f"./datasets/data_Allsight/json_data/{json_gan_name}"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(copy_to_path):
        os.makedirs(copy_to_path)
        
    df_data = pd.read_json(json_sim_p).transpose()
    df_data['old_frame'] = df_data['frame']

    images_p = get_image_paths_from_folder(images_folder_path)
    new_images_paths = [img for img in images_p if 'fake_A' in os.path.basename(img)]

    sorted_image_paths = sorted(new_images_paths, key=get_image_number)

    if args.save:
        # Save real image df
        for idx, img_path in enumerate(sorted_image_paths):
            real_image = (cv2.imread(img_path)).astype(np.uint8)
            save_path = os.path.join(copy_to_path, f'{args.name}{idx}.jpg')
            cv2.imwrite(save_path, real_image)
            df_data['frame'][idx] = save_path

        # Save real df to json
        to_dict = {}
        for index, row in list(df_data.iterrows()):
            to_dict[index] = dict(row)
        with open(f'{JSON_FILE}_transformed.json', 'w') as json_file:
            json.dump(to_dict, json_file, indent=3)
            
            print("[INFO] json saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images and related JSON data.')
    parser.add_argument('--sim_data_num', type=int, default= 2, help='sim JSON path')
    parser.add_argument('--data_kind', type=str, default='transformed', help='transformed, aligned')
    parser.add_argument('--cgan_num', type=str, default= 2)
    parser.add_argument('--cgan_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--name', type=str, default='cgan', help='cgan, distil_cgan')
    parser.add_argument('--save', default=False)
    args = parser.parse_args()

    main(args)