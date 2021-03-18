# Contrastive Learning of Musical Representations
PyTorch implementation of Contrastive Learning of Musical Representations by J. Spijkervet and J.A. Burgoyne (2021). We adapt SimCLR to the raw audio domain and contribute a pipeline of audio augmentations and encoder suited for pre-training on unlabeled, raw (musical) audio data. We evaluate the performance of the self-supervised learned representations on the task of music classification.

Despite unsupervised, contrastive pre-training and fine-tuning on the music classification task using *linear* classifier, we achieve state-of-the-art results on the MTAT dataset relative to fully supervised training.

*This is the CLMR v2 implementation, for the original implementation go to the [`v1`](https://github.com/Spijkervet/CLMR/tree/v1) branch*

<div align="center">
  <img width="50%" alt="CLMR model" src="https://github.com/Spijkervet/CLMR/blob/master/media/clmr_model.png?raw=true">
</div>
<div align="center">
  An illustration of CLMR.
</div>



## Quickstart
```
git clone https://github.com/spijkervet/clmr.git && cd clmr

pip3 install -r requirements.txt
# or
python3 setup.py install
```

The following command downloads MagnaTagATune, preprocesses it and starts self-supervised pre-training:
```
python3 preprocess.py --dataset magnatagatune
python3 main.py --dataset magnatagatune
```


## Results

### MagnaTagATune
| Encoder / Model | Batch-size / pre-training epochs | Fine-tune head |  ROC-AUC |  PR-AUC |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| SampleCNN / CLMR | 48 / 10000 | Linear Classifier | 88.5 | **35.4** |
SampleCNN / CLMR | 48 / 10000 | MLP (1 extra hidden layer) |  **89.3** | **35.9** |
| [SampleCNN (fully supervised, baseline)](https://www.mdpi.com/2076-3417/8/1/150) | - | - | 88.6 | 34.4 |
| [Pons et al. (fully supervised, reported SOTA)](https://arxiv.org/pdf/1711.02520.pdf) | - | - | **89.1** | 34.92 |

### Million Song Dataset
| Encoder / Model | Batch-size / pre-training epochs | Fine-tune head |  ROC-AUC |  PR-AUC |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| SampleCNN / CLMR | 48 / 1000 | Linear Classifier | 85.7 | 25.0 |
| [SampleCNN (fully supervised, baseline)](https://www.mdpi.com/2076-3417/8/1/150) | - | - | **88.4** | - |
| [Pons et al. (fully supervised, reported SOTA)](https://arxiv.org/pdf/1711.02520.pdf) | - | - | 87.4 | **28.5** |


## Pre-trained models
*Links go to download*
| Encoder (batch-size, epochs) | Fine-tune head | Pre-train dataset | ROC-AUC | PR-AUC
| ------------- | ------------- | ------------- | ------------- | -------------
[SampleCNN (96, 10000)](https://github.com/Spijkervet/CLMR/releases/download/2.0/clmr_checkpoint_10000.zip) | [Linear Classifier](https://github.com/Spijkervet/CLMR/releases/download/2.0/finetuner_checkpoint_200.zip) | MagnaTagATune |  88.5 (89.3) | 35.4 (35.9)
[SampleCNN (48, 1550)](https://github.com/Spijkervet/CLMR/releases/download/1.0/clmr_checkpoint_1550.pt) | [Linear Classifier](https://github.com/Spijkervet/CLMR/releases/download/1.0-l/finetuner_checkpoint_20.pt) | MagnaTagATune | 87.71 (88.47) | 34.27 (34.96)

## Training
### 1. Pre-training
Simply run the following command to pre-train the CLMR model on the MagnaTagATune dataset.
```
python main.py --dataset magnatagatune
```

### 2. Linear evaluation
To test a trained model, make sure to set the `checkpoint_path` variable in the `config/config.yaml`, or specify it as an argument:
```
python linear_evaluation.py --checkpoint_path ./clmr_checkpoint_10000.pt
```

## Configuration
The configuration of training can be found in: `config/config.yaml`. I personally prefer to use files instead of long strings of arguments when configuring a run. Every entry in the config file can be overrided with the corresponding flag (e.g. `--max_epochs 500` if you would like to train with 500 epochs).

## Logging and TensorBoard
To view results in TensorBoard, run:
```
tensorboard --logdir ./runs
```
