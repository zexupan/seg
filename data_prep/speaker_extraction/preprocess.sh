#!/bin/bash 

direc=/home/panzexu/datasets/pose_ted_long/

data_direc=${direc}orig/

train_samples=200000 # no. of train mixture samples simulated
val_samples=5000 # no. of validation mixture samples simulated
test_samples=3000 # no. of test mixture samples simulated
C=2 # no. of speakers in the mixture
mix_db=10 # random db ratio from -10 to 10db
mixture_data_list=mixture_data_list_${C}mix.csv #mixture datalist

audio_data_direc=${direc}audio_clean/ # Target audio saved directory
mixture_audio_direc=${direc}audio_mixture/${C}_mix_min/ # Audio mixture saved directory
pose_data_direc=${direc}visual_embedding/pose/ # The gesture saved directory

# stage 1: save the clean audio and pose from the orig folder
echo 'stage 1: save the clean audio and pose from the orig folder'
python 1_readdata.py \
--data_direc $data_direc \
--audio_data_direc $audio_data_direc \
--pose_data_direc $pose_data_direc  \


# stage 2: Remove repeated datas in pretrain and train set, extract audio from mp4, create mixture list
echo 'stage 2: create mixture list'
python 2_create_mixture_list.py \
--C $C \
--mix_db $mix_db \
--train_samples $train_samples \
--val_samples $val_samples \
--test_samples $test_samples \
--audio_data_direc $audio_data_direc \
--mixture_data_list $mixture_data_list \


# stage 3: create audio mixture from list
echo 'stage 3: create mixture audios'
python 3_create_mixtures.py \
--C $C \
--audio_data_direc $audio_data_direc \
--mixture_audio_direc $mixture_audio_direc \
--mixture_data_list $mixture_data_list 
cp $mixture_data_list $mixture_audio_direc


