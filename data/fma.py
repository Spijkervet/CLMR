import sys

sys.path.append("../")

import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from collections import defaultdict
from pathlib import Path


from fma.utils import get_audio_path


def default_loader(path):
    return torchaudio.load(path, normalization=False)


def default_indexer(path, tracks, labels, sample_rate):
    items = []
    track_dict = defaultdict(list)
    for idx, t in enumerate(tracks.index):
        fp = get_audio_path(path, t)
        resampled_fp = f"{os.path.splitext(fp)[0]}_{sample_rate}.mp3"
        if os.path.exists(resampled_fp):
            dir_id = str(Path(fp).parent.name)
            track_id = str(Path(fp).stem)
            label = labels[idx]
            items.append((track_id, dir_id, label))
            track_dict[track_id].append(idx)

    return items, track_dict


class FmaDataset(Dataset):
    def __init__(
        self,
        args,
        root_dir,
        tracks_df,
        labels,
        audio_length,
        indexer=default_indexer,
        loader=default_loader,
        transform=None,
    ):
        self.root_dir = root_dir
        self.tracks = tracks_df
        self.sample_rate = args.sample_rate
        
        self.audio_length = audio_length
        self.labels = labels

        self.indexer = indexer
        self.loader = loader
        self.transform = transform

        self.tracks_list, self.tracks_dict = self.indexer(root_dir, self.tracks, self.labels, self.sample_rate)

    def __getitem__(self, index):
        track_id, dir_id, label = self.tracks_list[index]
        fp = os.path.join(self.root_dir, dir_id, track_id + f"_{self.sample_rate}.mp3")

        audio, sr = self.loader(fp)

        audio = audio.mean(axis=0).reshape(1, -1)  # to mono

        assert (
            sr == self.sample_rate
        ), "Sample rate is not consistent throughout the dataset"

        if self.transform:
            audio = self.transform(audio)

        return audio, label

    def __len__(self):
        return len(self.tracks_list)

    def get_audio_by_track_id(self, track_id, batch_size=20):
        """
        Get audio samples based on the track_id
        used for plotting the latent representations of different tracks
        """
        # batch_size = min(len(self.tracks_dict[track_id]), batch_size)
        batch = torch.zeros(batch_size, 1, self.audio_length)
        for idx in range(batch_size):
            batch[idx, 0, :], _, _, _ = self.__getitem__(
                self.tracks_dict[track_id][idx]
            )

        return batch
