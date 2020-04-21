python -m detection.pytorch-retinanet.train with \
    model_path=./logs/10 \
    epoch_num=90 \
    dataset=csv \
    csv_train=./detection/data/normalised/deepscores/training.csv \
    csv_classes=./detection/class_mapping.csv \
    csv_val=./detection/data/normalised/deepscores/test.csv \
    depth=50 \
    image_channels=1