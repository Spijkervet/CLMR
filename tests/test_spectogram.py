import torchaudio
import torch.nn as nn

from clmr.datasets import AUDIO

def test_audioset():
    audio_dataset = AUDIO("tests/data/audioset")
    audio, label = audio_dataset[0]

    sample_rate = 22050
    n_fft = 1024
    n_mels = 128
    stype = "magnitude"  # magnitude
    top_db = None  # f_max

    transform = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
        ),
        torchaudio.transforms.AmplitudeToDB(stype=stype, top_db=top_db),
    )
