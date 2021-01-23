# Contrastive Learning of Musical Representations
PyTorch implementation of Contrastive Learning of Musical Representations by J. Spijkervet and J.A. Burgoyne (2020). We adapt SimCLR to the raw audio domain and contribute a pipeline of audio augmentations and encoder suited for pre-training on unlabeled, raw (musical) audio data. We evaluate the performance of the self-supervised learned representations on the task of music classification. 

Despite unsupervised, contrastive pre-training and fine-tuning on the music classification task using *linear* classifier, we achieve state-of-the-art results on the MTAT dataset relative to fully supervised training. 

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
| SampleCNN / CLMR | 48 / 1550 | Linear Classifier | **88.49** | **35.37** |
SampleCNN / CLMR | 48 / 1550 | MLP (1 extra hidden layer) |  **89.25** | **35.89** |
| [SampleCNN (fully supervised, baseline)](https://www.mdpi.com/2076-3417/8/1/150) | - | - | 88.56 | 34.38 |
| [Pons et al. (fully supervised, reported SOTA)](https://arxiv.org/pdf/1711.02520.pdf) | - | - | 89.05 | 34.92 |

### Million Song Dataset
*Million Song Dataset experiments will follow soon*


## Pre-trained models
*Links go to download*
| Encoder (batch-size, epochs) | Fine-tune head | Pre-train dataset | ROC-AUC | PR-AUC
| ------------- | ------------- | ------------- | ------------- | -------------
[SampleCNN (48, 1550)](https://github.com/Spijkervet/CLMR/releases/download/1.0/clmr_checkpoint_1550.pt) | [Linear Classifier](https://github.com/Spijkervet/CLMR/releases/download/1.0-l/finetuner_checkpoint_20.pt) | MagnaTagATune | 87.71 (88.47) | 34.27 (34.96)

## Web interfaces

### Tagger Interface
Assuming the pre-trained models are downloaded in the project's root folder:
```
python -m web.tagger.app \                                                                                                      ~/git/clmr
    --model_path=. \
    --epoch_num=1550 \
    --finetune_model_path=. \
    --finetune_epoch_num=20
```

### Latent listening
```
python3 -m web.latent_listening.get_predictions
python3 -m http.server
Navigate to: localhost:8000/web/latent_listening
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

## Usage

### Downloading & Pre-processing datasets
In the `scripts` folder, various bash scripts can be found that download the dataset and, at the end, provide a command that pre-processes all audio files automatically. These are invoked when the `--download 1` flag is submitted for pre-training or fine-tuning. `ffmpeg` is required to resample audio tracks. All datasets are on the filesystem in below mentioned format. So if the MagnaTagATune's or Million Song Dataset's files are already present on the file system, they can simply be moved to the corresponding dataset's `raw` folder:

- `DATA_INPUT_DIR`/`DATASET_NAME`/`raw` containing the unprocessed version of the dataset
- `DATA_INPUT_DIR`/`DATASET_NAME`/`processed` containing processed audio (sample rate) for training / testing
- `DATA_INPUT_DIR`/`DATASET_NAME`/`processed_annotations` containing annotations of the dataset (if present)

E.g. for MagnaTagATune:
- `./datasets/magnatagatune/raw` - containing the 0-9 and a-f folders with the .mp3 files.

Or the Million Song Dataset:
- `./datasets/million_song_dataset/raw` - containing the folders (and sub-folders) with .mp3 files.

Subseqently, the following Python script can be invoked to pre-process the raw files:
```
python3 -m scripts.datasets.preprocess_dataset --data_input_dir ./datasets --dataset magnatagatune --sample_rate 22050 --audio_length 59049
```

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
python linear_evaluation.py
```

E.g., you can first download the pre-trained model from the table above, move it to the folder containing this repository, and use:
or in place:
```
python linear_evaluation.py --model_path=./ --epoch_num=1550
```

## Configuration
The configuration of training can be found in: `config/config.yaml`. I personally prefer to use files instead of long strings of arguments when configuring a run. Every entry in the config file can be overrided with the corresponding flag (e.g. `--epochs 500` if you would like to train with 500 epochs).

## Logging and TensorBoard
To view results in TensorBoard, run:
```
tensorboard --logdir ./runs
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
