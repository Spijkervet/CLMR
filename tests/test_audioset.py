import unittest
import torchaudio
from torchaudio_augmentations import (
    Compose,
    RandomApply,
    RandomResizedCrop,
    PolarityInversion,
    Noise,
    Gain,
    Delay,
    PitchShift,
    Reverb,
)
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
        audio_dataset = AUDIO("./tests/data/audioset")
        audio, label = audio_dataset[0]
        assert audio.shape[0] == 1
        assert audio.shape[1] == 93680

        num_samples = (
            self.sample_rate * 5
        )  # the test item is approximately 5.8 seconds.
        transform = self.get_audio_transforms(num_samples=num_samples)
        audio = transform(audio)
        torchaudio.save("augmented_sample.wav", audio, sample_rate=self.sample_rate)
