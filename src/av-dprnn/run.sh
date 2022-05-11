#!/bin/sh

gpu_id=2,3

continue_from=

if [ -z ${continue_from} ]; then
	log_name='avDprnn_'$(date '+%d-%m-%Y(%H:%M:%S)')
	mkdir logs/$log_name
else
	log_name=${continue_from}
fi

CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=2172 \
main.py \
--log_name $log_name \
\
--audio_direc '/home/panzexu/datasets/pose_ted_long/audio_clean/' \
--visual_direc '/home/panzexu/datasets/pose_ted_long/visual_embedding/pose/' \
--mix_lst_path '/home/panzexu/datasets/pose_ted_long/audio_mixture/2_mix_min/mixture_data_list_2mix.csv' \
--mixture_direc '/home/panzexu/datasets/pose_ted_long/audio_mixture/2_mix_min/' \
--C 2 \
\
--batch_size 16 \
--lr 0.0005 \
--num_workers 4 \
--max_length 10 \
\
--epochs 200 \
--use_tensorboard 1 \
>logs/$log_name/console.txt 2>&1

# --continue_from ${continue_from} \