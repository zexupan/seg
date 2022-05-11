#!/bin/bash 

direc=/home/panzexu/datasets/pose_ted_long/

data_direc=${direc}orig/

train_samples=5000000 # no. of train mixture samples simulated
val_samples=5000 # no. of validation mixture samples simulated
test_samples=3000 # no. of test mixture samples simulated
C=2 # no. of speakers in the mixture
mixture_data_list=mixture_data_list_${C}mix_large.csv #mixture datalist

audio_data_direc=${direc}audio_clean/ # Target audio saved directory

echo 'stage 2: create mixture list'
python 2_create_mixture_list.py \
--C $C \
--train_samples $train_samples \
--val_samples $val_samples \
--test_samples $test_samples \
--audio_data_direc $audio_data_direc \
--mixture_data_list $mixture_data_list \



