import unittest
import torchaudio
import torch.nn as nn
from torchaudio_augmentations import *

from clmr.datasets import AUDIO


class TestAudioSet(unittest.TestCase):
    sample_rate = 16000

    def get_audio_transforms(self, num_samples):
        transform = Compose(
            [
                RandomResizedCrop(n_samples=num_samples),
                RandomApply([PolarityInversion()], p=0.8),
                RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3),
                RandomApply([Gain()], p=0.2),
                RandomApply([Delay(sample_rate=self.sample_rate)], p=0.5),
                RandomApply(
                    [PitchShift(n_samples=num_samples, sample_rate=self.sample_rate)],
                    p=0.4,
                ),
                RandomApply([Reverb(sample_rate=self.sample_rate)], p=0.3),
            ]
        )
        return transform

    def test_audioset(self):
        audio_dataset = AUDIO("tests/data/audioset")
        audio, label = audio_dataset[0]

        sample_rate = 22050
        n_fft = 1024
        n_mels = 128
        stype = "magnitude"  # magnitude
        top_db = None  # f_max

        transform = self.get_audio_transforms(num_samples=sample_rate)

        spec_transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
            ),
            torchaudio.transforms.AmplitudeToDB(stype=stype, top_db=top_db),
        )

        audio = transform(audio)
        audio = spec_transform(audio)
        assert audio.shape[1] == 128
        assert audio.shape[2] == 44
