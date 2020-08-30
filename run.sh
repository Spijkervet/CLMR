#!/bin/sh
. ~/miniconda3/etc/profile.d/conda.sh
conda activate clmr

python main.py --id 58 --model_name clmr --supervised 1 --backprop_encoder 0 --dataset msd --epochs 500 --batch_size 96 --audio_length 59049 --sample_rate 22050 --projector_layers 2 --projection_dim 128 --temperature 0 --transforms_polarity 0 --transforms_noise 0 --transforms_gain 0 --transforms_filters 0 --transforms_delay 0 --transforms_pitch 0 --transforms_reverb 0 --learning_rate 0.0001 --logistic_lr 0.0001 --checkpoint_epochs 1 --perc_train_data 1.0 --dataparallel 1 --workers 10
exit

# CUDA_VISIBLE_DEVICES=0 python main.py --id 52 --model_name clmr --supervised 1 --backprop_encoder 0 --dataset msd --epochs 500 --batch_size 96 --audio_length 59049 --sample_rate 22050 --projector_layers 2 --projection_dim 128 --temperature 0 --transforms_polarity 0 --transforms_noise 0 --transforms_gain 0 --transforms_filters 0 --transforms_delay 0 --transforms_pitch 0 --transforms_reverb 0 --learning_rate 0.0001 --logistic_lr 0.0001 --checkpoint_epochs 1 --perc_train_data 0.01
# CUDA_VISIBLE_DEVICES=0 python main.py --id 54 --model_name clmr --supervised 1 --backprop_encoder 0 --dataset msd --epochs 500 --batch_size 96 --audio_length 59049 --sample_rate 22050 --projector_layers 2 --projection_dim 128 --temperature 0 --transforms_polarity 0 --transforms_noise 0 --transforms_gain 0 --transforms_filters 0 --transforms_delay 0 --transforms_pitch 0 --transforms_reverb 0 --learning_rate 0.0001 --logistic_lr 0.0001 --checkpoint_epochs 1 --perc_train_data 0.05
# CUDA_VISIBLE_DEVICES=0 python main.py --id 56 --model_name clmr --supervised 1 --backprop_encoder 0 --dataset msd --epochs 500 --batch_size 96 --audio_length 59049 --sample_rate 22050 --projector_layers 2 --projection_dim 128 --temperature 0 --transforms_polarity 0 --transforms_noise 0 --transforms_gain 0 --transforms_filters 0 --transforms_delay 0 --transforms_pitch 0 --transforms_reverb 0 --learning_rate 0.0001 --logistic_lr 0.0001 --checkpoint_epochs 1 --perc_train_data 0.2

exit

# python linear_evaluation.py --dataset magnatagatune --model_path /storage/jspijkervet/logs/42 --epoch_num 500 --logistic_epochs 500 --logistic_lr 0.0001
# python linear_evaluation.py --dataset magnatagatune --model_path /storage/jspijkervet/logs/43 --epoch_num 500 --logistic_epochs 500 --logistic_lr 0.0001

# python main.py --id 34 --model_name clmr --supervised 1 --backprop_encoder 0 --dataset magnatagatune --epochs 500 --batch_size 48 --audio_length 59049 --sample_rate 22050 --projector_layers 2 --projection_dim 128 --temperature 0 --transforms_polarity 0 --transforms_noise 0 --transforms_gain 0 --transforms_filters 0 --transforms_delay 0 --transforms_pitch 0 --learning_rate 0.0001 --logistic_lr 0.0001 --perc_train_data 0.01 --load_ram 1 --workers 6 --dataparallel 1
# python main.py --id 35 --model_name clmr --supervised 1 --backprop_encoder 0 --dataset magnatagatune --epochs 500 --batch_size 48 --audio_length 59049 --sample_rate 22050 --projector_layers 2 --projection_dim 128 --temperature 0 --transforms_polarity 0 --transforms_noise 0 --transforms_gain 0 --transforms_filters 0 --transforms_delay 0 --transforms_pitch 0 --learning_rate 0.0001 --logistic_lr 0.0001 --perc_train_data 0.02 --load_ram 1 --workers 6 --dataparallel 1
# python main.py --id 36 --model_name clmr --supervised 1 --backprop_encoder 0 --dataset magnatagatune --epochs 500 --batch_size 48 --audio_length 59049 --sample_rate 22050 --projector_layers 2 --projection_dim 128 --temperature 0 --transforms_polarity 0 --transforms_noise 0 --transforms_gain 0 --transforms_filters 0 --transforms_delay 0 --transforms_pitch 0 --learning_rate 0.0001 --logistic_lr 0.0001 --perc_train_data 0.05 --load_ram 1 --workers 6 --dataparallel 1
# python main.py --id 37 --model_name clmr --supervised 1 --backprop_encoder 0 --dataset magnatagatune --epochs 500 --batch_size 48 --audio_length 59049 --sample_rate 22050 --projector_layers 2 --projection_dim 128 --temperature 0 --transforms_polarity 0 --transforms_noise 0 --transforms_gain 0 --transforms_filters 0 --transforms_delay 0 --transforms_pitch 0 --learning_rate 0.0001 --logistic_lr 0.0001 --perc_train_data 0.1 --load_ram 1 --workers 6 --dataparallel 1
# python main.py --id 38 --model_name clmr --supervised 1 --backprop_encoder 0 --dataset magnatagatune --epochs 500 --batch_size 48 --audio_length 59049 --sample_rate 22050 --projector_layers 2 --projection_dim 128 --temperature 0 --transforms_polarity 0 --transforms_noise 0 --transforms_gain 0 --transforms_filters 0 --transforms_delay 0 --transforms_pitch 0 --learning_rate 0.0001 --logistic_lr 0.0001 --perc_train_data 0.2 --load_ram 1 --workers 6 --dataparallel 1
# python main.py --id 39 --model_name clmr --supervised 1 --backprop_encoder 0 --dataset magnatagatune --epochs 500 --batch_size 48 --audio_length 59049 --sample_rate 22050 --projector_layers 2 --projection_dim 128 --temperature 0 --transforms_polarity 0 --transforms_noise 0 --transforms_gain 0 --transforms_filters 0 --transforms_delay 0 --transforms_pitch 0 --learning_rate 0.0001 --logistic_lr 0.0001 --perc_train_data 0.5 --load_ram 1 --workers 6 --dataparallel 1
# python main.py --id 40 --model_name clmr --supervised 1 --backprop_encoder 0 --dataset magnatagatune --epochs 500 --batch_size 48 --audio_length 59049 --sample_rate 22050 --projector_layers 2 --projection_dim 128 --temperature 0 --transforms_polarity 0 --transforms_noise 0 --transforms_gain 0 --transforms_filters 0 --transforms_delay 0 --transforms_pitch 0 --learning_rate 0.0001 --logistic_lr 0.0001 --perc_train_data 1.0 --load_ram 1 --workers 6 --dataparallel 1
# python main.py --id 42 --model_name clmr --supervised 0 --backprop_encoder 0 --dataset magnatagatune --epochs 500 --batch_size 48 --audio_length 59049 --sample_rate 22050 --projector_layers 2 --projection_dim 128 --temperature 0.5 --transforms_polarity 0 --transforms_noise 0 --transforms_gain 0 --transforms_filters 0 --transforms_delay 0 --transforms_pitch 0 --transforms_reverb 0.4 --learning_rate 0.0003 --logistic_lr 0.0001 --perc_train_data 1 --load_ram 1 --workers 6 --dataparallel 1 --ablation 1
# python main.py --id 43 --model_name clmr --supervised 0 --backprop_encoder 0 --dataset magnatagatune --epochs 500 --batch_size 48 --audio_length 59049 --sample_rate 22050 --projector_layers 2 --projection_dim 128 --temperature 0.5 --transforms_polarity 0 --transforms_noise 0 --transforms_gain 0 --transforms_filters 0 --transforms_delay 0 --transforms_pitch 0 --transforms_reverb 0.8 --learning_rate 0.0003 --logistic_lr 0.0001 --perc_train_data 1 --load_ram 1 --workers 6 --dataparallel 1 --ablation 1



exit


export NUM_NODES=1
export NUM_GPUS_PER_NODE=2
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))
export OMP_NUM_THREADS=12

python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS_PER_NODE \
        --nnodes=$NUM_NODES \
        --node_rank $NODE_RANK \
        main.py \
        --id 32 \
        --workers 6 \
        --supervised 0 \
        --epochs 100 \
        --perc_train_data 0.5

python -c "import torch; print(torch.__version__)"

DATASET_DIR=./datasets

num_workers=$(nproc --all)
echo "Num threads: $num_workers"

export NUM_NODES=1
export NUM_GPUS_PER_NODE=2
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))
export OMP_NUM_THREADS=$num_workers
num_workers=$(($num_workers/$NUM_GPUS_PER_NODE))


echo "Workers per process: $num_workers"

python -m torch.distributed.launch \
	--nproc_per_node=$NUM_GPUS_PER_NODE \
	--nnodes=$NUM_NODES \
	--node_rank $NODE_RANK \
	main.py --transforms_delay 0.4 --projector_layers 2 --dataset magnatagatune --transforms_noise 0 --perc_train_data 1 --backprop_encoder 0 --id 29 --projection_dim 128 --temperature 0.5 --transforms_polarity 0.8 --transforms_gain 0.4 --supervised 0 --learning_rate 0.0003 --batch_size 24 --audio_length 59049 --epochs 3000 --transforms_pitch 0.6 --logistic_lr 0.0001 --transforms_filters 0.8 --sample_rate 22050 --model_name clmr \
	--data_input_dir $DATASET_DIR \
	--workers $num_workers
