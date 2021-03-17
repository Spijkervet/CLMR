import torchaudio
from torchaudio_augmentations import *
from clmr.datasets import AUDIO

sr = 22050


def test_audioset():
    audio_dataset = AUDIO("tests/data/audioset")
    audio, label = audio_dataset[0]
    assert audio.shape[0] == 1
    assert audio.shape[1] == 93680

    num_samples = sr
    transform = Compose(
        [
            RandomResizedCrop(n_samples=num_samples),
            RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3),
            RandomApply([Gain()], p=0.2),
            RandomApply([Delay(sample_rate=sr)], p=0.5),
            RandomApply([PitchShift(n_samples=num_samples, sample_rate=sr)], p=0.4),
            RandomApply([Reverb(sample_rate=sr)], p=0.3),
        ]
    )

    audio = transform(audio)
    torchaudio.save("augmented_sample.wav", audio, sample_rate=sr)
