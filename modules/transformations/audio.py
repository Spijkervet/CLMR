import torch
import torchaudio
import random
import essentia.standard
import librosa
import numpy as np
import audioop
from torchaudio.transforms import Vol


class RandomResizedCrop:
    def __init__(self, sr, n_samples):
        self.sr = sr
        self.n_samples = n_samples

    def __call__(self, audio):
        max_samples = audio.shape[0]
        start_idx = random.randint(0, max_samples - self.n_samples)  # * 2))
        audio = audio[start_idx : start_idx + self.n_samples]
        return audio


class InvertSignal:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            audio = np.negative(audio) # considerably faster
        return audio


class Noise:
    def __init__(self, sr, p=0.8):
        self.sr = sr
        self.p = p
        self.snr = 80

    def __call__(self, audio):
        if random.random() < self.p:
            RMS_s = np.sqrt(np.mean(audio**2))
            RMS_n = np.sqrt(RMS_s**2 / (pow(10, self.snr / 20)))
            noise = np.random.normal(0, RMS_n, audio.shape[0]).astype('float32')
            audio = audio + noise
            audio = np.clip(audio, -1, 1)
        return audio


class HighLowBandPass:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p

    def __call__(self, audio):
        highlowband = random.randint(0, 1)
        if random.random() < self.p:
            if highlowband == 0:
                highpass_freq = random.randint(200, 1200)
                filt = essentia.standard.HighPass(
                    cutoffFrequency=highpass_freq, sampleRate=self.sr
                )
            elif highlowband == 1:
                lowpass_freq = random.randint(2200, 4000)
                filt = essentia.standard.LowPass(
                    cutoffFrequency=lowpass_freq, sampleRate=self.sr
                )
            # else:
            #     filt = essentia.standard.BandPass(bandwidth=1000, cutoffFrequency=1500, sampleRate=self.sr)
            audio = filt(audio)

        return audio


class Gain:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            gain = random.randint(-20, -1)  # input was normalized to max(x)
            audio = torch.from_numpy(audio)
            audio = Vol(gain, gain_type="db")(audio)  # takes Tensor
            audio = audio.numpy()
        return audio


class PitchShift:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            pitches = [-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7]
            n_steps = random.choice(pitches)
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
        return audio


class Reverse:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            audio = torch.flip(audio, dims=[0, 1])

        return audio


class Delay:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p
        self.factor = 0.5  # volume factor of delay signal

    def calc_offset(self, ms):
        return int(ms * (self.sr / 1000))

    def __call__(self, audio):
        if random.random() < self.p:
            # delay between 200 - 500ms with 50ms intervals
            mss = np.arange(200, 500, 50)
            ms = random.choice(mss)

            # calculate delay
            offset = self.calc_offset(ms)
            beginning = [0.0] * offset
            end = audio[:-offset]
            delayed_signal = np.hstack((beginning, end))
            delayed_signal = delayed_signal * self.factor
            audio = (audio + delayed_signal) / 2
            audio = audio.astype(np.float32)

        return audio


class AudioTransforms:
    """
    A stochastic data augmentation module that transforms any given data example randomly 
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, args):
        self.args = args
        self.ablation = args.ablation
        sr = args.sample_rate

        self.train_transform = [
            RandomResizedCrop(n_samples=args.audio_length, sr=sr),
            InvertSignal(p=args.transforms_polarity, sr=sr),
            Noise(p=args.transforms_noise, sr=sr),
            Gain(p=args.transforms_gain, sr=sr),
            HighLowBandPass(p=args.transforms_filters, sr=sr),
            Delay(p=args.transforms_delay, sr=sr),
            PitchShift(p=args.transforms_pitch, sr=sr)
            # Reverse(p=0.5, sr=sr),
        ]
        self.test_transform = []

    def __call__(self, x, mean, std):
        x0 = self.transform(x, 0)
        x1 = self.transform(x, 1)

        # to PyTorch format (channels, samples)
        x0 = x0.reshape(1, -1)
        x1 = x1.reshape(1, -1)

        # clamp the values again between [-1, 1], in case any
        # unwanted transformations went to [-inf, inf]
        # x0 = torch.clamp(x0, min=-1, max=1)
        # x1 = torch.clamp(x1, min=-1, max=1)
        return x0, x1

    def transform(self, x, num):
        # assymetric ablation
        if self.ablation and num == 1:
            x = self.train_transform[0](x)
        else:
            for t in self.train_transform:
                x = t(x)
        return x
