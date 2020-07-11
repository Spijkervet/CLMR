import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset as TorchDataset

# inherits Dataset from PyTorch
class Dataset(TorchDataset):

    def __init__(self, sample_rate, audio_length, tracks_list, tracks_list_all, mean, std):
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.tracks_list = tracks_list
        self.tracks_list_all = tracks_list_all
        self.mean = mean
        self.std = std

    def loader(self, path):
        audio, sr = torchaudio.load(path, normalization=True)
        return audio, sr

    def get_audio(self, fp):
        audio, sr = self.loader(fp)
        max_samples = audio.shape[1]

        if sr != self.sample_rate:
            raise Exception("Sample rate is not consistent throughout the dataset")

        if max_samples - self.audio_length <= 0:
            raise Exception("Max samples exceeds number of samples in crop")

        if torch.isnan(audio).any():
            raise Exception("Audio contains NaN values")

        return audio

    def __len__(self):
        return len(self.tracks_list)

    def get_full_size_audio(self, fp):
        audio = self.get_audio(fp)

        # normalise audio
        # audio = self.normalise_audio(audio)

        # split into equally sized tensors of self.audio_length
        batch = torch.split(audio, self.audio_length, dim=1)

        # remove last, since it is not a full self.audio_length, and stack
        batch = torch.cat(batch[:-1])

        # reshape to B x 1 x N
        batch = batch.reshape(batch.shape[0], 1, -1)
        return batch

    def normalise_audio(self, audio):
        return (audio - self.mean) / self.std

    def denormalise_audio(self, norm_audio):
        return (norm_audio * self.std) + self.mean

    def sample_audio_by_track_id(self, track_id, batch_size=20):
        """
        Get audio samples based on the track_id (batch_size = num_samples)
        used for plotting the latent representations of different tracks
        """
        batch = torch.zeros(batch_size, 1, self.audio_length)
        for idx in range(batch_size):
            index = self.tracks_dict[track_id][0]
            _, fp, label, _ = self.tracks_list_all[index]  # from non-dup!!

            audio = self.get_audio(fp)

            start_idx = idx * self.audio_length

            # too large
            if (start_idx + self.audio_length) > audio.size(1):
                return None

            batch[idx, 0, :] = audio[:, start_idx : start_idx + self.audio_length]
        return batch
