#!/bin/sh

gpu_id=0,1,2,3

continue_from='Dprnn_19-01-2022(11:24:00)'

if [ -z ${continue_from} ]; then
	log_name='Dprnn_'$(date '+%d-%m-%Y(%H:%M:%S)')
	mkdir logs/$log_name
else
	log_name=${continue_from}
fi

CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=4 \
--master_port=1553 \
main.py \
\
--batch_size 15 \
--audio_direc '/home/panzexu/datasets/pose_ted_long/audio_clean/' \
--mix_lst_path '/home/panzexu/datasets/pose_ted_long/audio_mixture/3_mix_min/mixture_data_list_3mix.csv' \
--mixture_direc '/home/panzexu/datasets/pose_ted_long/audio_mixture/3_mix_min/' \
--C 3 \
\
--log_name $log_name \
--epochs 100 \
\
--continue_from ${continue_from} \
--use_tensorboard 1 \
>logs/$log_name/console1.txt 2>&1

# --continue_from ${continue_from} \
