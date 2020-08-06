import torch
import torchaudio
import os
import random
import numpy as np
import subprocess
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import time

from .dataset import Dataset
from scripts.datasets.utils import write_statistics
from utils import random_undersample_balanced
from utils.audio import load_tracks, concat_tracks


def load_id2gt(gt_file):
    ids = []
    fgt = open(gt_file)
    id2gt = dict()
    for line in fgt.readlines():
        id, gt = line.strip().split("\t")  # id is string
        id2gt[id] = eval(gt)  # gt is array
        ids.append(id)
    return ids, id2gt


def load_id2path(index_file):
    paths = []
    fspec = open(index_file)
    id2path = dict()
    for line in fspec.readlines():
        id, path = line.strip().split("\t")
        id2path[id] = path
        paths.append(path)
    return paths, id2path


def get_dataset_stats(loader, tracks_list):
    means = []
    stds = []
    for track_id, fp, label, _ in tqdm(tracks_list):
        audio, sr = loader(fp)
        means.append(audio.mean())
        stds.append(audio.std())
    return np.array(means).mean(), np.array(stds).mean()


class MTTDataset(Dataset):

    base_dir = "magnatagatune"
    splits = ("train", "valid", "test")

    def __init__(
        self, args, split, pretrain, download=False, transform=None,
    ):
        self.name = "magnatagatune"
        if download:
            subprocess.call(
                [
                    "sh",
                    "./scripts/download_magnatagatune.sh",
                    args.data_input_dir,
                    str(args.sample_rate),
                ]
            )

        self.split = split
        self.pretrain = pretrain
        self.transform = transform
        self.sample_rate = args.sample_rate
        self.audio_length = args.audio_length
        self.load_ram = args.load_ram
        self.supervised = args.supervised

        # MagnaTagATune has clips of 30s, of which the last is not a full |audio_length|
        self.num_segments = (30 * self.sample_rate) // self.audio_length - 1

        self.audio_dir = os.path.join(args.data_input_dir, self.base_dir, "raw")
        self.audio_proc_dir = os.path.join(
            args.data_input_dir, self.base_dir, "processed"
        )

        mtt_processed_annot = Path(
            args.data_input_dir, self.base_dir, "processed_annotations"
        )

        if split == "train":
            self.annotations_file = Path(mtt_processed_annot) / "train_gt_mtt.tsv"
        elif split == "valid":
            self.annotations_file = Path(mtt_processed_annot) / "val_gt_mtt.tsv"
        else:
            self.annotations_file = Path(mtt_processed_annot) / "test_gt_mtt.tsv"

        # int to label
        with open(Path(mtt_processed_annot) / f"output_labels_mtt.txt", "r") as f:
            lines = f.readlines()
            self.tags = eval(lines[1][lines[1].find("[") :])
            self.num_tags = len(self.tags)

        [audio_repr_paths, id2audio_path] = load_id2path(
            Path(mtt_processed_annot) / "index_mtt.tsv"
        )
        [ids, id2gt] = load_id2gt(self.annotations_file)
        self.id2audio_path = id2audio_path

        self.index, self.track_index = self.indexer(
            ids, id2audio_path, id2gt, "magnatagatune"
        )

        if not self.supervised and self.pretrain:
            # we already load a full fragment of audio (with 10 segments)
            self.index = [
                [track_id, clip_id, segment, fp, label]
                for track_id, clip_id, segment, fp, label in self.index
                if segment == 0
            ]

        # reduce dataset to n%
        if pretrain and args.perc_train_data < 1.0 and split == "train":  # only on train set
            print("Train dataset size:", len(self.index))
            train_X_indices = np.array(
                [idx for idx in range(len(self.index))]
            ).reshape(-1, 1)
            train_y = np.array([label for _, _, _, _, label in self.index])
            train_X_indices, _ = random_undersample_balanced(
                train_X_indices, train_y, args.perc_train_data
            )

            new_index = []
            for idx, (track_id, clip_id, segment, fp, label) in enumerate(self.index):
                if idx in train_X_indices:
                    new_index.append([track_id, clip_id, segment, fp, label])

            self.index = new_index
            print("Undersampled train dataset size:", len(self.index))

        # print(f"Num tracks: {len(self.tracks_list)}")

        ## get dataset statistics
        self.mean = None
        self.std = None

        if not self.supervised and self.pretrain and self.split == "train":
            # track index contains track_ids matched with clip_ids, so filtering is easier
            print(
                "Concatenating tracks for pre-training (to avoid positive samples in the negative samples batch)"
            )

            # new track_id filtered index (unique track_ids)
            self.index = concat_tracks(
                args.sample_rate, self.audio_proc_dir, self.split, self.track_index
            )

        if self.load_ram and self.pretrain and self.split == "train":
            print("Loading train data into memory for faster training")
            self.audios = load_tracks(args.sample_rate, self.index)

        super(MTTDataset, self).__init__(
            self.name,
            self.split,
            self.sample_rate,
            self.audio_length,
            self.index,
            self.num_segments,
            self.audio_proc_dir,
            self.mean,
            self.std,
        )

        # memory leak bug in DDP
        if args.world_size > 1:
            self.index = np.array(self.index)
            self.track_index = {k: np.array(v) for k, v in self.track_index.items()}

    # get one segment (==59049 samples) and its 50-d label
    def __getitem__(self, idx):
        track_id, clip_id, segment, fp, label = self.index[idx]
        label = torch.FloatTensor(label)
        try:
            if self.load_ram and self.pretrain and self.split == "train":
                audio = self.audios["{}-{}".format(track_id, clip_id)]
            else:
                audio = self.get_audio(fp)

        except Exception as e:
            print(f"Skipped {track_id, fp}, could not load audio: {e}")
            return self.__getitem__(idx + 1)

        # only transform if unsupervised training
        if not self.supervised and self.pretrain and self.transform:
            audio = self.transform(audio, self.mean, self.std)
        # elif self.model_name == "cpc":
        #     max_samples = audio.size(1)
        #     start_idx = random.randint(0, max_samples - self.audio_length)
        #     audio = audio[:, start_idx : start_idx + self.audio_length]
        #     audio = self.normalise_audio(audio)
        #     audio = (audio, audio)
        else:
            start_idx = int(segment) * self.audio_length
            audio = audio[start_idx : start_idx + self.audio_length]
            audio = audio.reshape(1, -1)  # [channels, samples]
            audio = (audio, audio)

        return audio, label

