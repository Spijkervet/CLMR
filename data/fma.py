import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torchaudio
from collections import defaultdict
from repositories.fma.utils import get_audio_path
from pathlib import Path

def default_loader(path):
    return torchaudio.load(path, normalization=False)


def default_indexer(path, tracks):
    items = []
    track_dict = defaultdict(list)
    for idx, t in enumerate(tracks.index):
        fp = get_audio_path(path, t)
        if os.path.exists(fp):
            dir_id = str(Path(fp).parent.name)
            track_id = str(Path(fp).stem)
            items.append((track_id, dir_id))
            track_dict[track_id].append(idx)

    return items, track_dict


class FmaDataset(Dataset):
    def __init__(
        self, options, root_dir, tracks_df, audio_length=20480, indexer=default_indexer, loader=default_loader
    ):
        self.root_dir = root_dir
        self.tracks = tracks_df
        self.sample_rate = 16000 # TODO: make this a variable

        self.audio_length = audio_length
        self.indexer = indexer
        self.loader = loader

        self.tracks_list, self.tracks_dict = self.indexer(root_dir, self.tracks)

    def __getitem__(self, index):
        track_id, dir_id = self.tracks_list[index]
        fp = os.path.join(self.root_dir, dir_id, track_id + '.mp3')

        audio, sr = self.loader(fp)

        audio = audio.mean(axis=0).reshape(1, -1)

        assert (
            sr == self.sample_rate
        ), "Sample rate is not consistent throughout the dataset"

        # discard last part that is not a full 10ms
        ms = self.sample_rate / 100

        max_length = audio.size(1) // ms * ms
        # print(ms, max_length, self.audio_length)

        audio_range = np.arange(ms, max_length - self.audio_length - 0, ms)

        start_idx = int(
            np.random.choice(audio_range)
        )

        audio = audio[:, start_idx : start_idx + self.audio_length]  # assume mono

        # Normalize audio with mean / std?
        audio = (audio - self.mean) / self.std
        return audio, fp, index, start_idx

    def __len__(self):
        return len(self.tracks_list)

    def get_audio_by_track_id(self, track_id, batch_size=20):
        """
        Get audio samples based on the track_id
        used for plotting the latent representations of different tracks
        """
        # batch_size = min(len(self.speaker_dict[track_id]), batch_size)
        batch = torch.zeros(batch_size, 1, self.audio_length)
        for idx in range(batch_size):
            batch[idx, 0, :], _, _, _ = self.__getitem__(
                self.speaker_dict[track_id][idx]
            )

        return batch
