import os
import warnings
import subprocess
import torch
import numpy as np
import zipfile
from collections import defaultdict
from typing import Any, Tuple, Optional
from tqdm import tqdm

import soundfile as sf
import torchaudio
torchaudio.set_audio_backend("soundfile")
from torch import Tensor, FloatTensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)


FOLDER_IN_ARCHIVE = "magnatagatune"
_CHECKSUMS = {
    "http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.001": "",
    "http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.002": "",
    "http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.003": "",
    "http://mi.soi.city.ac.uk/datasets/magnatagatune/annotations_final.csv": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/binary.npy": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/tags.npy": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/test.npy": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/train.npy": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/valid.npy": "",
}


def preprocess_audio(source, target):
    p = subprocess.Popen(["ffmpeg", "-i", source, target, "-loglevel", "quiet"])
    p.wait()


class MAGNATAGATUNE(Dataset):
    """Create a Dataset for MagnaTagATune.
    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        folder_in_archive (str, optional): The top-level directory of the dataset.
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        subset (str, optional): Which subset of the dataset to use.
            One of ``"training"``, ``"validation"``, ``"testing"`` or ``None``.
            If ``None``, the entire dataset is used. (default: ``None``).
    """

    _ext_audio = ".wav"

    def __init__(
        self,
        root: str,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
        subset: Optional[str] = None,
    ) -> None:

        # super(GTZAN, self).__init__()
        self.root = root
        self.folder_in_archive = folder_in_archive
        self.download = download
        self.subset = subset

        assert subset is None or subset in ["train", "valid", "test"], (
            "When `subset` not None, it must take a value from "
            + "{'train', 'valid', 'test'}."
        )

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                os.makedirs(self._path)

            zip_files = []
            for url, checksum in _CHECKSUMS.items():
                target_fn = os.path.basename(url)
                target_fp = os.path.join(self._path, target_fn)
                if ".zip" in target_fp:
                    zip_files.append(target_fp)
                
                if not os.path.exists(target_fp):
                    download_url(
                        url,
                        self._path,
                        filename=target_fn,
                        hash_value=checksum,
                        hash_type="md5",
                    )

            if not os.path.exists(os.path.join(self._path, "f", "american_bach_soloists-j_s__bach_solo_cantatas-01-bwv54__i_aria-30-59.mp3")):
                merged_zip = os.path.join(self._path, "mp3.zip")
                print("Merging zip files...")
                with open(merged_zip, 'wb') as f:           
                    for filename in zip_files:                                  
                        with open(filename, 'rb') as g:
                            f.write(g.read())
                                                                                
                extract_archive(merged_zip)

        self.binary = np.load(os.path.join(self._path, "binary.npy"))
        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        if self.subset == "train":
            self.fl = np.load(os.path.join(self._path, "train.npy"))
        elif self.subset == "valid":
            self.fl = np.load(os.path.join(self._path, "valid.npy"))
        elif self.subset == "test":
            self.fl = np.load(os.path.join(self._path, "test.npy"))

        self.n_classes = self.binary.shape[1]
        # self.audio = {}
        # for f in tqdm(self.fl):
        #     clip_id, fp = f.split("\t")
        #     if clip_id not in self.audio.keys():
        #         audio, _ = load_magnatagatune_item(fp, self._path, self._ext_audio)
        #         self.audio[clip_id] = audio

    def file_path(self, n: int) -> str:
        _, fp = self.fl[n].split("\t")
        return os.path.join(self._path, fp) 

    def target_file_path(self, n: int) -> str:
        fp = self.file_path(n)
        file_basename, _ = os.path.splitext(fp)
        return file_basename + self._ext_audio

    def preprocess(self, n: int):
        fp = self.file_path(n)
        target_fp = self.target_file_path(n)

        if not os.path.exists(target_fp):
            preprocess_audio(fp, target_fp)

    def load(self, n):
        target_fp = self.target_file_path(n)
        audio, sample_rate = torchaudio.load(target_fp)
        return audio, sample_rate


    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, label)``
        """
        clip_id, fp = self.fl[n].split("\t")
        label = self.binary[int(clip_id)]

        # audio = self.audio[clip_id]
        audio, _ = self.load(n)
        label = FloatTensor(label)
        return audio, label

    def __len__(self) -> int:
        return len(self.fl)
