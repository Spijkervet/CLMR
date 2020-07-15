# Contrastive Learning of Musical Representations
PyTorch implementation of Contrastive Learning of Musical Representations by J. Spijkervet and J.A. Burgoyne (2020). We adapt SimCLR to the raw audio domain and contribute a pipeline of audio augmentations and encoder suited for pre-training on unlabeled, raw (musical) audio data. We evaluate the performance of the self-supervised learned representations on the task of music classification. 

Despite unsupervised, contrastive pre-training and fine-tuning on the music classification task using *linear* classifier, we achieve competitive results relative to fully supervised training. 

<div align="center">
  <img width="50%" alt="CLMR model" src="https://github.com/Spijkervet/CLMR/blob/master/media/clmr_model.png?raw=true">
</div>
<div align="center">
  An illustration of CLMR.
</div>



## Quickstart
This downloads a pre-trained CLMR model (trained on unlabeled, raw audio data from MagnaTagATune) and fine-tunes a linear classifier on the MagnaTagATune music tagging task, which should receive an ROC-AUC of `±87.7\%` and a PR-AUC of `±34.3%` on the test set.
```
git clone https://github.com/spijkervet/clmr.git && cd clmr
curl -L https://github.com/Spijkervet/CLMR/releases/download/1.0/clmr_checkpoint_1550.pt -O
curl -L https://github.com/Spijkervet/CLMR/releases/download/1.0/features.p -O

# for conda:
sh setup.sh
conda activate clmr

# otherwise:
python3 -m pip install -r requirements.txt

# download MagnaTagATune and train a linear classifier for 20 epochs:
python linear_evaluation.py --dataset magnatagatune --download 1 --model_path . --epoch_num 1550 --logistic_epochs 10 --logistic_lr 0.001

```

The following command downloads MagnaTagATune, pre-processes it and starts self-supervised pre-training:
```
python main.py --dataset magnatagatune --download 1
```


## Results

### MagnaTagATune
| Encoder / Model | Batch-size / epochs | Fine-tune head |  ROC-AUC |  PR-AUC |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| SampleCNN / CLMR | 48 / 1550 | Linear Classifier | 87.71 | 34.27 |
SampleCNN / CLMR | 48 / 1550 | MLP (1 extra hidden layer) |  88.47 | **34.96** |
| [SampleCNN (fully supervised, baseline)](https://www.mdpi.com/2076-3417/8/1/150) | - | - | 88.56 | 34.38 |
| [Pons et al. (fully supervised, reported SOTA)](https://arxiv.org/pdf/1711.02520.pdf) | - | - | **89.05** | 34.92 |

### Million Song Dataset
*Million Song Dataset experiments will follow soon*


## Pre-trained models
*Links go to download*
| Encoder (batch-size, epochs) | Fine-tune head | Pre-train dataset | ROC-AUC | PR-AUC
| ------------- | ------------- | ------------- | ------------- | -------------
[SampleCNN (48, 1550)](https://github.com/Spijkervet/CLMR/releases/download/1.0/clmr_checkpoint_1550.pt) | [Linear Classifier](https://github.com/Spijkervet/CLMR/releases/download/1.0-l/finetuner_checkpoint_20.pt) | MagnaTagATune | 87.71 (88.47) | 34.27 (34.96)


## Usage

### Pre-training
The following commands are used to set-up and pre-train on raw, unlabeled audio data:

Run the following command to setup a conda environment:
```
sh setup.sh
conda activate clmr
```

Or alternatively with pip:
```
pip install -r requirements.txt
```

Then, simply run the following command to pre-train the CLMR model on the MagnaTagATune dataset (and download / pre-process it for the first time, flag can be removed after completion):
```
python main.py --dataset magnatagatune --download 1
```

### Linear evaluation
To test a trained model, make sure to set the `model_path` variable in the `config/config.yaml` to the folder containing the saved pre-trained model (e.g. `./save`). Set the `epoch_num` to the epoch number you want to load the checkpoints from (e.g. `40`).
```
python -m testing.logistic_regression
```

E.g., you can first download the pre-traind model from the table above, move it to the folder containing this repository, and use:
or in place:
```
python -m testing.logistic_regression --model_path=./ --epoch_num=1550
```

## Configuration
The configuration of training can be found in: `config/config.yaml`. I personally prefer to use files instead of long strings of arguments when configuring a run. Every entry in the config file can be overrided with the corresponding flag (e.g. `--epochs 500` if you would like to train with 500 epochs).

## Logging and TensorBoard
To view results in TensorBoard, run:
```
tensorboard --logdir ./runs
```

## Inference
This command performs inference using a pre-trained encoder using CLMR (task-agnostic) and a fine-tuned linear classifier on the task of music classification. It will predict the tags corresponding to the song "Bohemian Rhapsody" by Queen, and yield ROC-AUC/ROC-AP scores and a taggram for the entire song:
```
python inference.py \
    --audio_url="https://www.youtube.com/watch?v=fJ9rUzIMcZQ" \
    --model_path=. \
    --epoch_num=1550 \
    --finetune_model_path=. \
    --finetune_epoch_num=20
```

#### Dependencies
\>= Python 3.7
```
torch
torchvision
tensorboard
essentia
seaborn
PyYAML
opencv-python
youtube-dl
pydot
python-dotenv
matplotlib
imageio
```
