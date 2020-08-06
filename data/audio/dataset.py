import os
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset as TorchDataset
import numpy as np

from utils.audio import process_wav

# inherits Dataset from PyTorch
class Dataset(TorchDataset):

    def __init__(self, name, split, sample_rate, audio_length, tracks_list, num_segments, audio_proc_dir, mean, std):
        self.name = name
        self.split = split
        self.audio_proc_dir
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.tracks_list = tracks_list
        self.num_segments = num_segments 
        self.mean = mean
        self.std = std

        print(f"[{self.name} {self.split}]: Loaded {len(self.tracks_list)} audio segments")


    def indexer(self, ids, id2audio_path, id2gt, dataset):
        index = []
        tmp = []
        track_index = defaultdict(list)
        track_idx = 0
        clip_idx = 0
        for clip_id in ids:
            fp = id2audio_path[clip_id]
            label = id2gt[clip_id]

            if dataset == "magnatagatune":
                track_id = "".join(Path(fp).stem.split("-")[:-2])
                if track_id not in tmp:
                    tmp.append(track_id)
                    track_idx += 1
                fp = f"{track_idx}-{clip_id}-{self.sample_rate}.wav"
                fp = os.path.join(self.audio_proc_dir, self.split, fp)
            else:
                track_idx = clip_id
                clip_id = clip_idx
                fp = os.path.join(self.audio_proc_dir, fp)
                clip_idx += 1

            for s in range(self.num_segments): 
                index.append([track_idx, clip_id, s, fp, label])
                track_index[track_idx].append([clip_id, s, fp, label])
        return index, track_index

    def loader(self, path):
        audio, sr = process_wav(self.sample_rate, path, False)
        return audio, sr

    def get_audio(self, fp):
        audio, sr = self.loader(fp)
        max_samples = audio.shape[0]
        if sr != self.sample_rate:
            raise Exception("Sample rate is not consistent throughout the dataset")

        if max_samples - self.audio_length <= 0:
            raise Exception("Max samples exceeds number of samples in crop")

        if np.isnan(audio).any():
            raise Exception("Audio contains NaN values")

        return audio

    def __len__(self):
        return len(self.tracks_list)

    def get_full_size_audio(self, fp):
        audio = self.get_audio(fp)

        # normalise audio
        # audio = self.normalise_audio(audio)

        # split into equally sized tensors of self.audio_length
        audio = torch.from_numpy(audio).reshape(1, -1)
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
            _, _, fp, _ = self.track_index[track_id][0]

            audio = self.get_audio(fp)

            start_idx = idx * self.audio_length

            # too large
            if (start_idx + self.audio_length) > audio.shape[0]:
                return None

            batch[idx, 0, :] = torch.from_numpy(audio[start_idx : start_idx + self.audio_length])
        return batch
