#!/bin/sh

gpu_id=11,7

continue_from=

if [ -z ${continue_from} ]; then
	log_name='avConv_'$(date '+%Y-%m-%d(%H:%M:%S)')
	mkdir logs/$log_name
else
	log_name=${continue_from}
fi

CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=3201 \
main.py \
\
--log_name $log_name \
\
--audio_direc '/home/panzexu/datasets/pose_ted_long/audio_clean/' \
--visual_direc '/home/panzexu/datasets/pose_ted_long/visual_embedding/pose/' \
--mix_lst_path '/home/panzexu/workspace/avss_pose/data/sync/mixture_data_list_2mix_large.csv' \
--epochs 100 \
--num_workers 4 \
\
--batch_size 16 \
\
--use_tensorboard 1 \
>logs/$log_name/console.txt 2>&1

# --continue_from ${continue_from} \
