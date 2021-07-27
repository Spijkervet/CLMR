import os
import pickle
import torch
import torchaudio
from collections import defaultdict
from pathlib import Path
from torch import Tensor, FloatTensor
from tqdm import tqdm
from typing import Any, Tuple, Optional

from clmr.datasets import Dataset


def load_id2gt(gt_file, msd_7d):
    ids = []
    with open(gt_file) as f:
        id2gt = dict()
        for line in f.readlines():
            msd_id, gt = line.strip().split("\t")  # id is string
            id_7d = msd_7d[msd_id]
            id2gt[msd_id] = eval(gt)  # gt is array
            ids.append(msd_id)
    return ids, id2gt


def load_id2path(index_file, msd_7d):
    paths = []
    with open(index_file) as f:
        id2path = dict()
        for line in f.readlines():
            msd_id, msd_path = line.strip().split("\t")
            id_7d = msd_7d[msd_id]
            path = os.path.join(id_7d[0], id_7d[1], f"{id_7d}.clip.mp3")
            id2path[msd_id] = path
            paths.append(path)
        return paths, id2path


def default_indexer(ids, id2audio_path, id2gt):
    index = []
    track_index = defaultdict(list)
    track_idx = 0
    clip_idx = 0
    for clip_id in ids:
        fp = id2audio_path[clip_id]
        label = id2gt[clip_id]
        track_idx = clip_id
        clip_id = clip_idx
        clip_idx += 1
        index.append([track_idx, clip_id, fp, label])
        track_index[track_idx].append([clip_id, fp, label])
    return index, track_index


def default_loader(path):
    audio, sr = torchaudio.load(path)
    audio = audio.mean(dim=0, keepdim=True)
    return audio, sr


class MillionSongDataset(Dataset):

    _base_dir = "million_song_dataset"
    _ext_audio = ".wav"

    def __init__(
        self,
        root: str,
        base_dir: str = _base_dir,
        download: bool = False,
        subset: Optional[str] = None,
    ):
        if download:
            raise Exception("The Million Song Dataset is not publicly available")

        self.root = root
        self.base_dir = base_dir
        self.subset = subset

        assert subset is None or subset in ["train", "valid", "test"], (
            "When `subset` not None, it must take a value from "
            + "{'train', 'valid', 'test'}."
        )

        self._path = os.path.join(self.root, self.base_dir)

        if not os.path.exists(self._path):
            raise RuntimeError(
                "Dataset not found. Please place the MSD files in the {} folder.".format(
                    self._path
                )
            )

        msd_processed_annot = Path(self._path, "processed_annotations")

        if self.subset == "train":
            self.annotations_file = Path(msd_processed_annot) / "train_gt_msd.tsv"
        elif self.subset == "valid":
            self.annotations_file = Path(msd_processed_annot) / "val_gt_msd.tsv"
        else:
            self.annotations_file = Path(msd_processed_annot) / "test_gt_msd.tsv"

        with open(Path(msd_processed_annot) / "MSD_id_to_7D_id.pkl", "rb") as f:
            self.msd_to_7d = pickle.load(f)

        # int to label
        with open(Path(msd_processed_annot) / "output_labels_msd.txt", "r") as f:
            lines = f.readlines()
            self.tags = eval(lines[1][lines[1].find("[") :])
            self.n_classes = len(self.tags)

        [audio_repr_paths, id2audio_path] = load_id2path(
            Path(msd_processed_annot) / "index_msd.tsv", self.msd_to_7d
        )
        [ids, id2gt] = load_id2gt(self.annotations_file, self.msd_to_7d)

        self.index, self.track_index = default_indexer(ids, id2audio_path, id2gt)

    def file_path(self, n: int) -> str:
        _, _, fp, _ = self.index[n]
        return os.path.join(self._path, "preprocessed", fp)

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        track_id, clip_id, fp, label = self.index[n]
        label = torch.FloatTensor(label)

        try:
            audio, _ = self.load(n)
        except Exception as e:
            print(f"Skipped {track_id, fp}, could not load audio: {e}")
            return self.__getitem__(n + 1)
        return audio, label

    def __len__(self) -> int:
        return len(self.index)
