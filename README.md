# Contrastive Learning of Musical Representations
PyTorch implementation of Contrastive Learning of Musical Representations.

### Quickstart



### Pre-trained models


### Results

## Usage
Run the following command to setup a conda environment:
```
sh setup.sh
conda activate clmr
```

Or alternatively with pip:
```
pip install -r requirements.txt
```

Then, simply run:
```
python main.py
```

### Testing
To test a trained model, make sure to set the `model_path` variable in the `config/config.yaml` to the log ID of the training (e.g. `logs/0`).
Set the `epoch_num` to the epoch number you want to load the checkpoints from (e.g. `40`).

```
python -m testing.logistic_regression
```

or in place:
```
python -m testing.logistic_regression with model_path=./logs/0 epoch_num=40
```

## Visualise TSNE manifold
```
python validate_latent_space.py with model_path=./logs/audio/magnatagatune/clmr/1 epoch_num=1490 sample_rate=22050  audio_length=59049  -u
```


## Configuration
The configuration of training can be found in: `config/config.yaml`. I personally prefer to use files instead of long strings of arguments when configuring a run. An example `config.yaml` file:
```
# train options
batch_size: 256
workers: 16
start_epoch: 0
epochs: 40

# model options
resnet: "resnet18"
normalize: True
projection_dim: 64

# loss options
temperature: 0.5

# reload options
model_path: "logs/0" # set to the directory containing `checkpoint_##.tar` 
epoch_num: 40 # set to checkpoint number

# logistic regression options
logistic_batch_size: 256
logistic_epochs: 100
```

## Logging and TensorBoard
The `sacred` package is used to log all experiments into the `logs` directory. To view results in TensorBoard, run:
```
tensorboard --logdir logs
```

## Optimizers and learning rate schedule
This implementation features the Adam optimizer and the LARS optimizer, with the option to decay the learning rate using a cosine decay schedule. The optimizer and weight decay can be configured in the `config/config.yaml` file.
<p align="center">
  <img src="https://github.com/Spijkervet/SimCLR/blob/master/media/lr_cosine_decay_schedule.png?raw=true" width="400"/>
</p>

## Inference
```
python inference.py \
    with \
    audio_url="https://www.youtube.com/watch?v=ftjEcrrf7r0" \
    model_path=/storage/jspijkervet/logs_backup_ws7/clmr/2/ \
    epoch_num=1490 \
    finetune_model_path=/storage/jspijkervet/logs_backup_ws7/clmr/4/ \
    finetune_epoch_num=50
```

#### Dependencies
```
torch
torchvision
tensorboard
sacred
pyyaml
```
