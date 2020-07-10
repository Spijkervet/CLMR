import torch
import torchaudio
import random
import essentia
import essentia.standard
import librosa
import numpy as np

# from torchaudio.transforms import Vol


class RandomResizedCrop:
    def __init__(self, sr, n_samples):
        self.sr = sr
        self.n_samples = n_samples

    def __call__(self, audio):
        max_samples = audio.size(1)

        assert (
            max_samples - self.n_samples
        ) >= 0, "max samples exceeds number of samples in crop"

        start_idx = random.randint(0, max_samples - self.n_samples)  # * 2))
        audio = audio[:, start_idx : start_idx + self.n_samples]
        return audio, start_idx


class InvertSignal:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            audio = audio.squeeze()
            audio = audio * -1
            audio = audio.reshape(1, -1)
        return audio, None


class Noise:
    def __init__(self, sr, p=0.8):
        self.sr = sr
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            audio = audio.squeeze()
            audio = audio + (torch.FloatTensor(*audio.shape).normal_(0, 1) * 0.001)
            audio = audio.reshape(1, -1)
        return audio, None


class HighLowBandPass:
    def __init__(self, sr, highpass_freq, lowpass_freq, p=0.5):
        self.sr = sr
        self.p = p
        self.highpass_freq = highpass_freq
        self.lowpass_freq = lowpass_freq

    def __call__(self, audio):
        highlowband = random.randint(0, 1)
        if random.random() < self.p:
            if highlowband == 0:
                filt = essentia.standard.HighPass(
                    cutoffFrequency=self.highpass_freq, sampleRate=self.sr
                )
            elif highlowband == 1:
                filt = essentia.standard.LowPass(
                    cutoffFrequency=self.lowpass_freq, sampleRate=self.sr
                )

            audio = audio.squeeze()  # reshape, since essentia takes (samples, channels)
            audio = filt(audio.numpy())
            audio = torch.from_numpy(audio).reshape(
                1, -1
            )  # reshape back to (channels, samples)

        return audio, None


class Gain:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p

    def __call__(self, audio):
        gain = random.randint(-6, 0)  # input was normalized to max(x)
        if random.random() < self.p:
            pass
            # vol = Vol(gain, gain_type="db")
            # audio = vol(audio)
        return audio, None


class PitchShift:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            audio = audio.squeeze()

            pitches = [-2, -1, 1, 2]
            n_steps = random.choice(pitches)
            # time_stretch = [1.5, 1.25, 0.75, 0.5]
            # stretch = time_stretch[pitches.index(n_steps)]

            audio = audio.numpy()
            # audio = librosa.effects.time_stretch(audio, rate=stretch)
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
            audio = torch.from_numpy(audio).reshape(
                1, -1
            )  # reshape back to (channels, samples)
        return audio, None


class Reverse:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            audio = torch.flip(audio, dims=[0, 1])

        return audio, None


class AudioTransforms:
    """
    A stochastic data augmentation module that transforms any given data example randomly 
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, args):
        self.args = args
        sr = args.sample_rate

        self.train_transform = [
            RandomResizedCrop(n_samples=args.audio_length, sr=sr),
            InvertSignal(p=args.transforms_phase, sr=sr),
            Noise(p=args.transforms_noise, sr=sr),
            # Gain(p=args.transforms_gain, sr=sr),
            HighLowBandPass(
                p=args.transforms_filters,
                highpass_freq=args.transforms_highpass_freq,
                lowpass_freq=args.transforms_lowpass_freq,
                sr=sr
            ),
            # PitchShift(p=0.1, sr=sr)
            # Reverse(p=0.5, sr=sr),
        ]

        self.test_transform = []

    def __call__(self, x, mean, std):
        x0, transformations = self.transform(x)
        x1, _ = self.transform(x, prev_transforms=transformations)

        # clamp the values again between [-1, 1], in case any
        # unwanted transformations went to [-inf, inf]
        x0 = torch.clamp(x0, min=-1, max=1)
        x1 = torch.clamp(x1, min=-1, max=1)

        # pseudo-standardise
        if mean is not None and std is not None:
            x0 = self.normalise(x0, mean, std)
            x1 = self.normalise(x1, mean, std)
        return x0, x1

    def normalise(self, audio, mean, std):
        return (audio - mean) / std

    def transform(self, x, prev_transforms=None):
        transformations = {}
        for t in self.train_transform:
            x, transformation = t(x)
            transformations[t.__class__.__name__] = transformation
        return x, transformations
