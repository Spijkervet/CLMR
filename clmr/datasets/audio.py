import os
import subprocess
from glob import glob
from collections import defaultdict
from typing import Any, Tuple, Optional

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


def preprocess_audio(source, target, sample_rate):
    p = subprocess.Popen(
        ["ffmpeg", "-i", source, "-ar", str(sample_rate), target, "-loglevel", "quiet"]
    )
    p.wait()


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

    def __init__(self, root: str,) -> None:

        self._path = root
        self.n_classes = 1

        self.fl = glob(os.path.join(self._path, "*{}".format(self._ext_audio)))
        self.fl.extend(
            glob(os.path.join(self._path, "**", "*{}".format(self._ext_audio)))
        )

        if len(self.fl) == 0:
            raise RuntimeError(
                "Dataset not found. Please place the audio files in the {} folder.".format(
                    self._path
                )
            )

    def file_path(self, n: int) -> str:
        fp = self.fl[n]
        return fp

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

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, label)``
        """
        audio, _ = self.load(n)
        label = []
        return audio, label

    def __len__(self) -> int:
        return len(self.fl)
