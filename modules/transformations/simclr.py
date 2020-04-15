import torch
import torchaudio
import numpy as np
import random
import essentia
import essentia.standard
import librosa


class RandomResizedCrop:
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __call__(self, audio):
        max_samples = audio.size(1)

        assert (
            max_samples - self.n_samples
        ) > 0, "max samples exceeds number of samples in crop"

        start_idx = np.random.randint(0, max_samples - self.n_samples)

        audio = audio[:, start_idx : start_idx + self.n_samples]
        return audio


class InvertSignal:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            audio = audio * -1
        return audio

class Noise:
    def __init__(self, p=0.8):
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            noise = torch.LongTensor(*audio.size()).random_(-1, 1) * 0.01 # 0.1 gain
            audio = noise + audio
        return audio

class BandPass:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            bp = essentia.standard.BandPass(sampleRate=16000)
            audio = audio.squeeze()  # reshape, since essentia takes (samples, channels)
            audio = bp(audio.numpy())
            audio = torch.from_numpy(audio).reshape(
                1, -1
            )  # reshape back to (channels, samples)

        return audio

class LowPass:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            lp = essentia.standard.LowPass(sampleRate=16000)
            audio = audio.squeeze()  # reshape, since essentia takes (samples, channels)
            audio = lp(audio.numpy())
            audio = torch.from_numpy(audio).reshape(
                1, -1
            )  # reshape back to (channels, samples)

        return audio


class HighPass:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            hp = essentia.standard.HighPass(sampleRate=16000)
            audio = audio.squeeze()  # reshape, since essentia takes (samples, channels)
            audio = hp(audio.numpy())
            audio = torch.from_numpy(audio).reshape(
                1, -1
            )  # reshape back to (channels, samples)

        return audio


class PitchShift:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            audio = audio.squeeze()

            pitches = [-2, -1, 1, 2]
            n_steps = np.random.choice(pitches)
            # time_stretch = [1.5, 1.25, 0.75, 0.5]
            # stretch = time_stretch[pitches.index(n_steps)]

            audio = audio.numpy()
            # audio = librosa.effects.time_stretch(audio, rate=stretch)
            audio = librosa.effects.pitch_shift(
                audio, sr=16000, n_steps=n_steps
            )
            audio = torch.from_numpy(audio).reshape(
                1, -1
            )  # reshape back to (channels, samples)
        return audio


class Reverse:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            audio = torch.flip(audio, dims=[0, 1])

        return audio


class AudioTransforms:
    """
    A stochastic data augmentation module that transforms any given data example randomly 
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, args):

        self.lin_eval = args.lin_eval

        self.transformations = [
            RandomResizedCrop(n_samples=args.audio_length),
            InvertSignal(p=0.5), # "horizontal flip"
            Noise(p=0.8),
            BandPass(p=0.5),
            # LowPass(p=0.25),
            # HighPass(p=0.25),
            # PitchShift(p=0.25)
            # Reverse(p=0.5),
        ]

    def transform(self, x):
        if self.lin_eval:
            x = self.transformations[0](x)  # only crop in eval
        else:
            for t in self.transformations:
                x = t(x)
        return x

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class TransformsSimCLR:
    pass
