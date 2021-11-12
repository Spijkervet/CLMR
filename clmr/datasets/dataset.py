import math
import os
import random
import subprocess
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from typing import Tuple
from abc import abstractmethod


def preprocess_audio(source, target, sample_rate):
    p = subprocess.Popen(
        ["ffmpeg", "-i", source, "-ar", str(sample_rate), target, "-loglevel", "quiet"]
    )
    p.wait()


class Dataset(TorchDataset):

    _ext_audio = ".wav"

    def __init__(self, root: str):
        pass

    @abstractmethod
    def file_path(self, n: int):
        pass

    def target_file_path(self, n: int) -> str:
        fp = self.file_path(n)
        file_basename, _ = os.path.splitext(fp)
        return file_basename + self._ext_audio

    def preprocess(self, n: int, sample_rate: int):
        fp = self.file_path(n)
        target_fp = self.target_file_path(n)

        if not os.path.exists(target_fp):
            preprocess_audio(fp, target_fp, sample_rate)

    def load(self, n):
        target_fp = self.target_file_path(n)
        try:
            audio, sample_rate = torchaudio.load(target_fp)
        except OSError as e:
            print("File not found, try running `python preprocess.py` first.\n\n", e)
            return
        return audio, sample_rate


class SplitMusicDataset(Dataset):
    def __init__(self, dataset: Dataset, max_audio_length: int):
        self.dataset = dataset
        self.max_audio_length = max_audio_length
        self.n_classes = dataset.n_classes

        number_of_tracks = len(self.dataset)
        number_of_samples = self.dataset[0][0].shape[1]
        number_of_full_chunks = math.floor(number_of_samples / self.max_audio_length)

        self.index = []
        for track_idx in range(number_of_tracks):
            for sample_idx in range(number_of_full_chunks):
                self.index.append([track_idx, sample_idx])

        random.shuffle(self.index)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        track_idx, sample_idx = self.index[idx]
        x, y = self.dataset[track_idx]
        x = x[
            :,
            sample_idx
            * self.max_audio_length : (sample_idx + 1)
            * self.max_audio_length,
        ]
        return x, y
