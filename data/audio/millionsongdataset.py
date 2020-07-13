import torch
import torchaudio
import os
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from scripts.datasets.utils import write_statistics
from utils import random_undersample_balanced
from tqdm import tqdm

import random

from .dataset import Dataset


"""
From Pons et al.
"""


def load_id2gt(gt_file, msd_7d):
    ids = []
    fgt = open(gt_file)
    id2gt = dict()
    for line in fgt.readlines():
        msd_id, gt = line.strip().split("\t")  # id is string
        id_7d = msd_7d[msd_id]
        id2gt[msd_id] = eval(gt)  # gt is array
        ids.append(msd_id)
    return ids, id2gt


def load_id2path(index_file, msd_7d):
    paths = []
    fspec = open(index_file)
    id2path = dict()
    for line in fspec.readlines():
        msd_id, msd_path = line.strip().split("\t")
        id_7d = msd_7d[msd_id]
        path = os.path.join(id_7d[0], id_7d[1], f"{id_7d}.clip.wav")
        id2path[msd_id] = path
        paths.append(path)
    return paths, id2path


def default_indexer(args, path, id2audio, id2gt):
    items = []
    tracks_dict = defaultdict(list)
    index = {}
    index_num = 0
    for idx, (clip_id, label) in tqdm(enumerate(id2gt.items())):
        if idx > 100:
            break

        fp = os.path.join(path, id2audio[clip_id])
        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            index_name = Path(id2audio[clip_id].split(".")[0]).stem  # 7 digital ID
            if index_name not in index.keys():
                index[index_name] = index_num
                index_num += 1

            label = torch.FloatTensor(label)
            items.append((index[index_name], fp, label))
            tracks_dict[index[index_name]].append(idx)
        else:
            print("File not found: {}".format(fp))
    return items, tracks_dict


def default_loader(path):
    audio, sr = torchaudio.load(path, normalization=True)
    return audio, sr


def get_dataset_stats(loader, tracks_list):
    means = []
    stds = []
    for track_id, fp, label in tqdm(tracks_list):
        audio, sr = loader(fp)
        means.append(audio.mean())
        stds.append(audio.std())
    return np.array(means).mean(), np.array(stds).mean()


class MSDDataset(Dataset):

    base_dir = "million_song_dataset"

    def __init__(self, args, split, pretrain, download=False, transform=None):
        if download:
            raise Exception("The Million Song Dataset is not publicly available")

        self.split = split
        self.pretrain = pretrain
        self.transform = transform
        self.sample_rate = args.sample_rate
        self.audio_length = args.audio_length

        # Million Song Dataset has clips of 30s, of which the last is not a full |audio_length|
        self.num_segments = (30 * self.sample_rate) // self.audio_length - 1

        self.mean = None  # -1.1771917e-05
        self.std = None  # 0.14238065

        dir_name = f"processed_{args.sample_rate}"
        self.audio_dir = os.path.join(
            args.data_input_dir, "million_song_dataset", dir_name
        )
        self.audio_proc_dir = os.path.join(
            args.data_input_dir, self.base_dir, "processed"
        )

        self.audio_raw_dir = os.path.join(
            args.data_input_dir, self.base_dir, "raw"
        )

        msd_processed_annot = Path(
            args.data_input_dir, "million_song_dataset", "processed_annotations"
        )

        if split == "train":
            self.annotations_file = Path(msd_processed_annot) / "train_gt_msd.tsv"
            split = "Train"
        elif split == "validation":
            self.annotations_file = Path(msd_processed_annot) / "val_gt_msd.tsv"
            split = "Validation"
        else:
            self.annotations_file = Path(msd_processed_annot) / "test_gt_msd.tsv"
            split = "Test"

        self.msd_to_7d = pickle.load(
            open(Path(msd_processed_annot) / "MSD_id_to_7D_id.pkl", "rb")
        )

        # int to label
        with open(Path(msd_processed_annot) / f"output_labels_msd.txt", "r") as f:
            lines = f.readlines()
            self.tags = eval(lines[1][lines[1].find("[") :])
            self.num_tags = len(self.tags)

        [audio_repr_paths, id2audio_path] = load_id2path(
            Path(msd_processed_annot) / "index_msd.tsv", self.msd_to_7d
        )
        [ids, id2gt] = load_id2gt(self.annotations_file, self.msd_to_7d)

        self.index, self.track_index = self.indexer(ids, id2audio_path, id2gt, dataset="million_song_dataset")
        
        # we already load a full fragment of audio, and then select a segment from all N segments)
        if self.pretrain:
            self.index = [
                [track_id, clip_id, segment, fp, label]
                for track_id, clip_id, segment, fp, label in self.index
                if segment == 0
            ]

        # from new_data import preprocess_tracks
        # preprocess_tracks(self.sample_rate, self.audio_raw_dir, self.audio_proc_dir, self.split, self.track_index, id2audio_path)

        # reduce dataset to n%
        # if args.perc_train_data < 1.0 and (train or validation): # only on train set
        #     print("Train dataset size:", len(self.tracks_list))
        #     train_X_indices = np.array([idx for idx in range(len(self.tracks_list))]).reshape(-1, 1)
        #     train_y = np.array([label.numpy() for _, _, label, _ in self.tracks_list])
        #     train_X_indices, _ = random_undersample_balanced(train_X_indices, train_y, args.perc_train_data)

        #     new_tracks_list = []
        #     for idx, (track_id, fp, label, segment) in enumerate(self.tracks_list):
        #         if idx in train_X_indices:
        #             new_tracks_list.append([track_id, fp, label, segment])

        #     self.tracks_list = new_tracks_list
        #     print("Undersampled train dataset size:", len(self.tracks_list))

        # print(f"Num tracks: {len(self.tracks_list)}")

        print(f"[{split} dataset ({args.dataset}_{self.sample_rate})]")

        super(MSDDataset, self).__init__(
            self.split,
            self.sample_rate,
            self.audio_length,
            self.index,
            self.num_segments,
            self.audio_proc_dir,
            self.mean,
            self.std,
        )

    # get one segment (==59049 samples) and its 50-d label
    def __getitem__(self, index):
        track_id, clip_id, segment, fp, label = self.tracks_list[index]

        try:
            audio = self.get_audio(fp)
        except Exception as e:
            print(f"Skipped {track_id, fp}, could not load audio: {e}")
            return self.__getitem__(index + 1)

        # only transform if unsupervised training
        if self.pretrain and self.transform:
            audio = self.transform(audio, self.mean, self.std)
        # elif self.model_name == "cpc":
        #     max_samples = audio.size(1)
        #     start_idx = random.randint(0, max_samples - self.audio_length)
        #     audio = audio[:, start_idx : start_idx + self.audio_length]
        #     audio = (audio, audio)
        else:
            start_idx = segment * self.audio_length
            audio = audio[start_idx : start_idx + self.audio_length]
            audio = audio.reshape(1, -1) # [channels, samples]
            audio = (audio, audio)

        return audio, label
