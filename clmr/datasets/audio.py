import os
from glob import glob
from collections import defaultdict
from typing import Any, Tuple, Optional

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


def load_audio(audio_path):
    audio, sr = torchaudio.load(audio_path)
    return audio


class AUDIO(Dataset):
    """Create a Dataset for any folder of audio files.
    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        folder_in_archive (str, optional): The top-level directory of the dataset.
        subset (str, optional): Which subset of the dataset to use.
            One of ``"training"``, ``"validation"``, ``"testing"`` or ``None``.
            If ``None``, the entire dataset is used. (default: ``None``).
    """

    _ext_audio = ".wav"

    def __init__(
        self,
        root: str,
    ) -> None:

        self.root = root
        self._path = root

        self.fl = glob(os.path.join(self.root, "*.wav"))
        self.fl.extend(glob(os.path.join(self.root, "*.mp3")))
        self.fl.extend(glob(os.path.join(self.root, "**", "*.wav")))
        self.fl.extend(glob(os.path.join(self.root, "**", "*.mp3")))

        if len(self.fl) == 0:
            raise RuntimeError(
                "Dataset not found. Please place the audio files in the {} folder.".format(
                    self._path
                )
            )

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, label)``
        """
        audio_path = self.fl[n]
        audio = load_audio(audio_path)
        label = None
        return audio, label

    def __len__(self) -> int:
        return len(self.fl)
