import torch
import torchaudio
import random
import essentia
import essentia.standard
import librosa


class RandomResizedCrop:
    def __init__(self, sr, n_samples):
        self.sr = sr
        self.n_samples = n_samples

    def __call__(self, audio, prev_transform=None):
        max_samples = audio.size(1)

        assert (
            max_samples - self.n_samples
        ) > 0, "max samples exceeds number of samples in crop"

        # keep a frame of 1 x n_samples so we have a margin
        start_idx = random.randint(self.n_samples, max_samples - (self.n_samples * 2))

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
            audio = audio * -1
        return audio, None

class Noise:
    def __init__(self, sr, p=0.8):
        self.sr = sr
        self.p = p

    def __call__(self, audio, prev_transform=None):
        if random.random() < self.p:
            noise = torch.LongTensor(*audio.size()).random_(-1, 1) * 0.01 # 0.1 gain
            audio = noise + audio
        return audio, None

class BandPass:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p

    def __call__(self, audio, prev_transform=None):
        if random.random() < self.p:
            bp = essentia.standard.BandPass(sampleRate=self.sr)
            audio = audio.squeeze()  # reshape, since essentia takes (samples, channels)
            audio = bp(audio.numpy())
            audio = torch.from_numpy(audio).reshape(
                1, -1
            )  # reshape back to (channels, samples)

        return audio, None

class LowPass:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p

    def __call__(self, audio, prev_transform=None):
        if random.random() < self.p:
            lp = essentia.standard.LowPass(sampleRate=self.sr)
            audio = audio.squeeze()  # reshape, since essentia takes (samples, channels)
            audio = lp(audio.numpy())
            audio = torch.from_numpy(audio).reshape(
                1, -1
            )  # reshape back to (channels, samples)

        return audio, None


class HighPass:
    def __init__(self, sr, p=0.5):
        self.sr = sr
        self.p = p

    def __call__(self, audio, prev_transform=None):
        if random.random() < self.p:
            hp = essentia.standard.HighPass(sampleRate=self.sr)
            audio = audio.squeeze()  # reshape, since essentia takes (samples, channels)
            audio = hp(audio.numpy())
            audio = torch.from_numpy(audio).reshape(
                1, -1
            )  # reshape back to (channels, samples)

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
            audio = librosa.effects.pitch_shift(
                audio, sr=16000, n_steps=n_steps
            )
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

        self.lin_eval = args.lin_eval
        sr = args.sample_rate

        self.transformations = [
            RandomResizedCrop(n_samples=args.audio_length, sr=sr),
            InvertSignal(p=0.5, sr=sr), # "horizontal flip"
            Noise(p=0.8, sr=sr),
            BandPass(p=0.2, sr=sr),
            LowPass(p=0.2, sr=sr),
            HighPass(p=0.2, sr=sr),
            # PitchShift(p=0.25, sr=sr)
            # Reverse(p=0.5, sr=sr),
        ]

    def transform(self, x, prev_transforms=None):
        transformations = {}
        if self.lin_eval:
            x, transformation = self.transformations[0](x)  # only crop in eval
        else:
            for t in self.transformations:
                prev_transform = None
                if prev_transforms:
                    prev_transform = prev_transforms[t.__class__.__name__]
                x, transformation = t(x, prev_transform=prev_transform)
                transformations[t.__class__.__name__] = transformation
        return x, transformations

    def __call__(self, x):
        x0, transformations = self.transform(x)
        # print("x0", transformations)
        x1, transformations = self.transform(x, prev_transforms=transformations)
        # print("x1", transformations)
        return x0, x1 


class TransformsSimCLR:
    pass
