import torchaudio
import torch.nn as nn
from datasets import AUDIO
import matplotlib.pyplot as plt

def test_audioset():
    audio_dataset = AUDIO("tests/data/audioset")
    audio, label = audio_dataset[0]

    sample_rate = 22050
    n_fft = 1024
    f_min = 0.0
    f_max = 11000
    n_mels = 256
    stype = "magnitude"  # magnitude
    top_db = None  # f_max

    transform = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        ),
        torchaudio.transforms.AmplitudeToDB(stype=stype, top_db=top_db),
    )
