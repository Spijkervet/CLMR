from datasets import AUDIO
import torchaudio
from audio_augmentations import *

sr = 22050


def test_audioset():
    audio_dataset = AUDIO("tests/data/audioset")
    audio, label = audio_dataset[0]
    assert audio.shape[0] == 1
    assert audio.shape[1] == 661794

    transform = Compose(
        [
            RandomResizedCrop(sr * 5),
            HighLowPass(sr=sr, p=1.0),
            Reverse(p=1.0),
            PitchShift(audio_length=sr * 5, sr=sr, p=1.0),
            Reverb(sr=sr, p=1.0),
            Reverse(p=1.0),
        ]
    )

    audio = transform(audio)
    torchaudio.save("reverb.wav", audio, sample_rate=sr)
