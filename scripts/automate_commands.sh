#!/bin/bash

if [ "$1" == "filter_real" ]; then
    # Filter Real Data
    python3 data_process/filter_real_images.py
fi

if [ "$1" == "merge_sim" ]; then
    # Merge Sim Data
    python3 data_process/merge_json_sim.py
fi

# Transfer Images to GAN Folders - Real Data
python3 data_process/transfer_images.py --data_type real --data_num 7 --folder_type A --samples 0 --data_kind transformed

# Transfer Images to GAN Folders - Sim Data
python3 data_process/transfer_images.py --data_type sim --data_num 7 --folder_type B --samples 0 --data_kind transformed

# Train CycleGAN
python3 train.py --dataroot ./datasets/data_Allsight/ --name exp_distil --model distil_cycle_gan --epoch_distil 1 --distil_policy linear --distil_slope 0.03 --lambda_C 0.5

# Test CycleGAN (Replace "#" with the desired epoch)
python3 test.py --dataroot ./datasets/data_Allsight/ --name exp_distil --model distil_cycle_gan --epoch latest

# Create JSON for GAN Images
python3 data_process/sim2gan_json.py --sim_data_num 7 --cgan_num 0 --name distil_cgan --data_kind tranformed --cgan_epoch latest --save True

# Train and Test - Regressor
python3 train_regressor.py --train_type gan --sim_data_num 7 --real_data_num 7 --cgan_num 0 --gan_name distil_cgan --cgan_epoch latest --input_type with_ref_6c --leds white --model_name efficientnet_b0 --aug True
