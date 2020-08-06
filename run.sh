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
        --workers 6 \
        --supervised 1
