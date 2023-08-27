import numpy as np
import pandas as pd
import random
import cv2
import argparse
import os
import glob

def is_folder_empty(folder_path):
    """
    Check if a folder is empty.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        bool: True if the folder is empty, False if it contains files.
    """
    return not any(os.listdir(folder_path))

def delete_files_in_folder(folder_path):
    """
    Delete all files in a folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        None
    """
    files = glob.glob(os.path.join(folder_path, "*"))
    for file in files:
        try:
            if os.path.isfile(file):
                os.remove(file)
                # print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
    print(f"[INFO] files in {folder_path} deleted")

def main(args):
    print("----------------------")
    print(f"[INFO] start transfer_images for {args.data_type}")

    random.seed(42)
    
    from_json_p = f'./datasets/data_Allsight/json_data/{args.data_type}_train_{args.data_num}_{args.data_kind}.json'
    trans_folder_path = './datasets/data_Allsight/'

    df_data = pd.read_json(from_json_p).transpose()
    
    if args.samples != 0:
        df_data = df_data.iloc[:args.samples,:]
        # df_data = df_data.sample(n=args.samples)
    
    for set in ['train','test']:
            folder_path = f"{trans_folder_path}{set}{args.folder_type}"
            if not is_folder_empty(folder_path):
                print(f"[INFO] folder {folder_path} not empty!")
                delete_files_in_folder(folder_path)
            
    print("[INFO] ready to transfer")
        
    for idx in range(len(df_data)):
        real_image = (cv2.imread(df_data['frame'][idx])).astype(np.uint8)
        save_path1 = trans_folder_path + 'train' +args.folder_type + f'/{idx}.jpg'  # Specify the path where you want to save the image
        save_path2 = trans_folder_path + 'test' +args.folder_type + f'/{idx}.jpg'
        cv2.imwrite(save_path1, real_image)
        cv2.imwrite(save_path2, real_image)
    
    if args.data_type == 'sim' :
        ref_frames = df_data['ref_frame'].unique()
        print("[INFO] saving refrences")
        for ref_path in ref_frames:
            for ref_i in range(1,args.ref_num+1):
                ref_img = cv2.imread(ref_path).astype(np.uint8)
                save_path1 = trans_folder_path + 'trainA' + f'/{idx+ref_i}.jpg'  # Specify the path where you want to save the image
                cv2.imwrite(save_path1, ref_img)
                
                save_path2 = trans_folder_path + 'testA' + f'/{idx+ref_i}.jpg'
                cv2.imwrite(save_path2, ref_img)
                
                save_path3 = trans_folder_path + 'trainB' + f'/{idx+ref_i}.jpg'  # Specify the path where you want to save the image
                cv2.imwrite(save_path3, ref_img)
                
                save_path4 = trans_folder_path + 'testB' + f'/{idx+ref_i}.jpg'
                cv2.imwrite(save_path4, ref_img)
                
            idx+=ref_i
        

    print(f"[INFO] finsih transfer_images for {args.data_type}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images and related JSON data.')
    parser.add_argument('--data_type', type=str, default='sim', help='real, sim')
    parser.add_argument('--data_kind', type=str, default='transformed', help='transformed, aligned')
    parser.add_argument('--data_num', type=int, default=7, help='from JSON path')
    parser.add_argument('--ref_num', type=int, default=1, help='number of each refrence frame in the final dataset')
    parser.add_argument('--folder_type', type=str, default='A', help='A, B')
    parser.add_argument('--samples', type=int, default=4980, help='Number of samples, if 0 -> not sample take all')
    args = parser.parse_args()

    main(args)