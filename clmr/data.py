"""Wrapper for Torch Dataset class to enable contrastive training
"""
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio_augmentations import Compose
from typing import Tuple, List


class ContrastiveDataset(Dataset):
    def __init__(self, dataset: Dataset, input_shape: List[int], transform: Compose):
        self.dataset = dataset
        self.transform = transform
        self.input_shape = input_shape
        self.ignore_idx = []

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        if idx in self.ignore_idx:
            return self[idx + 1]

        audio, label = self.dataset[idx]

        if audio.shape[1] < self.input_shape[1]:
            self.ignore_idx.append(idx)
            return self[idx + 1]

        if self.transform:
            audio = self.transform(audio)
        return audio, label

    def __len__(self) -> int:
        return len(self.dataset)

    def concat_clip(self, n: int, audio_length: float) -> Tensor:
        audio, _ = self.dataset[n]
        batch = torch.split(audio, audio_length, dim=1)
        batch = torch.cat(batch[:-1])
        batch = batch.unsqueeze(dim=1)

        if self.transform:
            batch = self.transform(batch)

        return batch
