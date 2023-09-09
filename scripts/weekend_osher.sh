#!/bin/bash

python3 train.py --name allsight_72 --model diff_cycle_gan --n_epochs 45 --n_epochs_decay 25 --epoch_distil 55 --lambda_C 0.1 --lambda_A 20 --lambda_B 25 --lambda_D 30 --lr 0.00002

python3 data_process/sim2gan_json.py --sim_data_num 8 --gan_num 72 --gan_type distil_cgan --gan_epoch latest --save True --diff True

python3 data_process/update_compose_frame_gan.py --sim_data_num 8 --gan_num 72 --gan_type diff_cgan --gan_epoch latest --save True 

python3 train_regressor.py --train_type gan --sim_data_num 8 --real_data_num 8 --gan_num 72 --gan_name diff_cgan --gan_epoch latest --input_type with_ref_6c --leds white --aug True

python3 train_regressor.py --train_type gan --sim_data_num 8 --real_data_num 8 --gan_num 72 --gan_name diff_cgan --gan_epoch latest --input_type with_ref_6c --leds white --aug True --frame_type diff_frame

python3 train.py --name allsight_73 --model diff_cycle_gan --n_epochs 45 --n_epochs_decay 25 --epoch_distil 30 --lambda_C 0.2 --lambda_A 20 --lambda_B 25 --lambda_D 30 --lr 0.00002 --lambda_identity 0.0 

python3 data_process/sim2gan_json.py --sim_data_num 8 --gan_num 73 --gan_type diff_cgan --gan_epoch latest --save True --diff True

python3 data_process/update_compose_frame_gan.py --sim_data_num 8 --gan_num 73 --gan_type diff_cgan --gan_epoch latest --save True 

python3 train_regressor.py --train_type gan --sim_data_num 8 --real_data_num 8 --gan_num 73 --gan_name diff_cgan --gan_epoch latest --input_type with_ref_6c --leds white --aug True

python3 train_regressor.py --train_type gan --sim_data_num 8 --real_data_num 8 --gan_num 73 --gan_name diff_cgan --gan_epoch latest --input_type with_ref_6c --leds white --aug True --frame_type diff_frame
