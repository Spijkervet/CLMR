import os
import subprocess
import torchaudio
from torch.utils.data import Dataset as TorchDataset
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
