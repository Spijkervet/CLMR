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
