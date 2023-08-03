import numpy as np
import pandas as pd
import random
import cv2
import argparse

  
def main(args):
    random.seed(42)
    
    from_json_p = f'./datasets/data_Allsight/json_data/{args.data_type}_train_{args.data_num}_transformed.json'
    trans_folder_path = './datasets/data_Allsight/'

    df_data = pd.read_json(from_json_p).transpose()
    
    if args.samples != 0:
        df_data = df_data.iloc[:args.samples,:]
        # df_data = df_data.sample(n=args.samples)
    
    for idx in range(len(df_data)):
        real_image = (cv2.imread(df_data['frame'][idx])).astype(np.uint8)
        save_path1 = trans_folder_path + 'train' +args.folder_type + f'/{idx}.jpg'  # Specify the path where you want to save the image
        save_path2 = trans_folder_path + 'test' +args.folder_type + f'/{idx}.jpg'
        cv2.imwrite(save_path1, real_image)
        cv2.imwrite(save_path2, real_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images and related JSON data.')
    parser.add_argument('--data_type', type=str, default='real', help='real, sim')
    parser.add_argument('--data_num', type=int, default=4, help='from JSON path')
    parser.add_argument('--folder_type', type=str, default='A', help='A, B')
    parser.add_argument('--samples', type=int, default=4800, help='Number of samples, if 0 -> not sample take all')
    args = parser.parse_args()

    main(args)