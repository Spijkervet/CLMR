import torch
import torchaudio
import random
import essentia
import essentia.standard
import librosa
import numpy as np

from torchaudio.transforms import Vol


class RandomResizedCrop:
    def __init__(self, sr, n_samples):
        self.sr = sr
        self.n_samples = n_samples

    def __call__(self, audio, prev_transform=None):
        # do not end at end of audio (silence)
        max_samples = audio.size(1)  #  - (self.n_samples * 4)

        assert (
            max_samples - self.n_samples
        ) > 0, "max samples exceeds number of samples in crop"

        # keep a frame of 1 x n_samples so we have a margin
        start_sample = (
            self.n_samples * 4
        )  # do not start at the start of the audio (silence)

        # TODO start sample
        start_idx = random.randint(0, max_samples - self.n_samples)  # * 2))

        # if x0 is cropped, crop x1 within a frame of 5 seconds (do not get "too" global) # TODO variable
        # if prev_transform and abs(start_idx - prev_transform) > (3 * self.sr):
        #     start_idx = random.randint(self.n_samples, prev_transform)

        audio = audio[:, start_idx : start_idx + self.n_samples]
        return audio, start_idx


class InvertSignal:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p

    def __call__(self, audio, prev_transform=None):
        if random.random() < self.p:
            audio = audio.squeeze()
            audio = audio * -1
            audio = audio.reshape(1, -1)
        return audio, None


class Noise:
    def __init__(self, sr, p=0.8):
        self.sr = sr
        self.p = p

    def __call__(self, audio, prev_transform=None):
        if random.random() < self.p:
            audio = audio.squeeze()
            audio = audio + (torch.FloatTensor(*audio.shape).normal_(0, 1) * 0.001)
            audio = audio.reshape(1, -1)
            # target_snr_db = random.randint(2, 10)  # signal-to-noise ratio
            # x = audio[0]
            # sig_avg_watts = abs(x.mean())
            # sig_avg_db = 10 * np.log10(sig_avg_watts)
            # noise_avg_db = sig_avg_db - target_snr_db
            # noise_avg_watts = 10 ** (noise_avg_db / 10)
            # noise_volts = torch.empty(len(x)).normal_(
            #     mean=0, std=np.sqrt(noise_avg_watts)
            # )
            # audio = x + noise_volts
            # audio = audio.reshape(1, -1)

        return audio, None


class HighLowBandPass:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p

    def __call__(self, audio, prev_transform=None):
        highlowband = random.randint(0, 1)
        if random.random() < self.p:
            if highlowband == 0:
                filt = essentia.standard.HighPass(
                    cutoffFrequency=1000, sampleRate=self.sr
                )
            elif highlowband == 1:
                filt = essentia.standard.LowPass(
                    cutoffFrequency=4000, sampleRate=self.sr
                )
            # else:
            #     filt = essentia.standard.BandPass(bandwidth=1000, cutoffFrequency=1500, sampleRate=self.sr)

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

    def __call__(self, audio, prev_transform=None):
        gain = random.randint(-6, 0)  # input was normalized to max(x)
        if random.random() < self.p:
            vol = Vol(gain, gain_type="db")
            audio = vol(audio)
        return audio, None


class PitchShift:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p

    def __call__(self, audio, prev_transform=None):
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

    def __call__(self, audio, prev_transform=None):
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
        self.lin_eval = args.lin_eval
        sr = args.sample_rate

        self.train_transform = [
            RandomResizedCrop(n_samples=args.audio_length, sr=sr),
            InvertSignal(p=args.transforms_phase, sr=sr),
            # Noise(p=1.0, sr=sr),
            Gain(p=args.transforms_gain, sr=sr),
            HighLowBandPass(p=args.transforms_filters, sr=sr),
            # PitchShift(p=0.1, sr=sr)
            # Reverse(p=0.5, sr=sr),
        ]

        self.test_transform = []

    def transform(self, x, prev_transforms=None):
        transformations = {}
        if self.lin_eval:
            x, transformation = self.train_transform[0](x)  # only crop in eval
        else:
            for t in self.train_transform:
                prev_transform = None
                # if prev_transforms:
                #     prev_transform = prev_transforms[t.__class__.__name__]
                x, transformation = t(x, prev_transform=prev_transform)
                transformations[t.__class__.__name__] = transformation
        return x, transformations

    def __call__(self, x):
        x0, transformations = self.transform(x)
        x1, transformations = self.transform(x, prev_transforms=transformations)

        # clamp the values again between [-1, 1], in case any
        # unwanted transformations went to [-inf, inf]
        x0 = torch.clamp(x0, min=-1, max=1)
        x1 = torch.clamp(x1, min=-1, max=1)

        # randomly get segment
        max_samples = x.size(1)
        start_idx = random.randint(0, max_samples - self.args.audio_length)
        x_test = x[:, start_idx : start_idx + self.args.audio_length]
        return x0, x1, x_test
