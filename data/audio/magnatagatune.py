import torch
import torchaudio
import os
import random
import numpy as np
import subprocess
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from .dataset import Dataset
from scripts.datasets.utils import write_statistics
from utils import random_undersample_balanced


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


def default_indexer(args, path, id2audio, id2gt):
    items = []
    tracks_dict = defaultdict(list)
    index = {}
    index_num = 0
    for idx, (clip_id, label) in enumerate(id2gt.items()):
        fn = Path(id2audio[clip_id].split(".")[0])
        d = fn.parent.stem
        fn = fn.stem
        index_name = "-".join(fn.split("-")[:-2])
        fp = os.path.join(path, d, index_name + "-full.wav")

        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            if index_name not in index.keys():
                index[index_name] = index_num
                index_num += 1

            label = torch.FloatTensor(label)

            # if supervised/eval
            items.append(
                (index[index_name], fp, label, 0)
            )  # only one segment, since full track
            tracks_dict[index[index_name]].append(idx)
        else:
            print("File not found: {}".format(fp))
    return items, tracks_dict


def get_dataset_stats(loader, tracks_list):
    means = []
    stds = []
    for track_id, fp, label, _ in tqdm(tracks_list):
        audio, sr = loader(fp)
        means.append(audio.mean())
        stds.append(audio.std())
    return np.array(means).mean(), np.array(stds).mean()


class MTTDataset(Dataset):

    base_folder = "magnatagatune"
    splits = ("train", "validation", "test")

    def __init__(
        self, args, split, pretrain, download=False, transform=None, indexer=default_indexer, 
    ):

        if download:
            subprocess.call(
                [
                    "sh",
                    "./scripts/download_magnatagatune.sh",
                    args.data_input_dir,
                    str(args.sample_rate),
                ]
            )

        self.supervised = args.supervised
        self.indexer = indexer
        self.transform = transform
        self.sample_rate = args.sample_rate
        self.audio_length = args.audio_length
        self.model_name = args.model_name

        if pretrain:
            dir_name = f"processed_full_{self.sample_rate}_wav"
        else:
            dir_name = f"processed_{self.sample_rate}_wav"

        self.audio_dir = os.path.join(args.data_input_dir, "magnatagatune", dir_name)

        mtt_processed_annot = Path(
            args.data_input_dir, "magnatagatune", "processed_annotations"
        )

        if split == "train":
            self.annotations_file = Path(mtt_processed_annot) / "train_gt_mtt.tsv"
        elif split == "validation":
            self.annotations_file = Path(mtt_processed_annot) / "val_gt_mtt.tsv"
        else:
            self.annotations_file = Path(mtt_processed_annot) / "test_gt_mtt.tsv"

        # int to label
        with open(Path(mtt_processed_annot) / f"output_labels_mtt.txt", "r") as f:
            lines = f.readlines()
            self.tags = eval(lines[1][lines[1].find("[") :])
            self.num_tags = len(self.tags)

        [audio_repr_paths, id2audio_repr_path] = load_id2path(
            Path(mtt_processed_annot) / "index_mtt.tsv"
        )
        [ids, id2gt] = load_id2gt(self.annotations_file)
        self.tracks_list_all, self.tracks_dict = self.indexer(
            args, self.audio_dir, id2audio_repr_path, id2gt
        )

        if pretrain:
            self.nodups = []
            self.indexes = []
            for track_id, fp, label, segment in self.tracks_list_all:
                if track_id not in self.indexes:
                    self.nodups.append([track_id, fp, label, segment])
                    self.indexes.append(track_id)

            print(
                "Removed duplicates from:",
                len(self.tracks_list_all),
                "to:",
                len(self.nodups),
            )

            self.tracks_list = self.nodups

        # reduce dataset to n%
        if args.perc_train_data < 1.0 and train and not validation:  # only on train set
            print("Train dataset size:", len(self.tracks_list))
            train_X_indices = np.array(
                [idx for idx in range(len(self.tracks_list))]
            ).reshape(-1, 1)
            train_y = np.array([label.numpy() for _, _, label, _ in self.tracks_list])
            train_X_indices, _ = random_undersample_balanced(
                train_X_indices, train_y, args.perc_train_data
            )

            new_tracks_list = []
            for idx, (track_id, fp, label, segment) in enumerate(self.tracks_list):
                if idx in train_X_indices:
                    new_tracks_list.append([track_id, fp, label, segment])

            self.tracks_list = new_tracks_list
            print("Undersampled train dataset size:", len(self.tracks_list))

        print(f"Num tracks: {len(self.tracks_list)}")

        ## get dataset statistics
        self.mean = None
        self.std = None

        # stats_path = os.path.join(args.data_input_dir, "magnatagatune", f"statistics_{self.sample_rate}.csv")
        # print(stats_path)
        # if not os.path.exists(stats_path):
        #     print(f"[{name} dataset]: Fetching dataset statistics (mean/std) for {version}_{self.sample_rate} version")
        #     if train:
        #         self.mean, self.std = get_dataset_stats(
        #             self.loader, self.tracks_list
        #         )
        #         write_statistics(self.mean, self.std, len(self.tracks_list), stats_path)
        #     else:
        #         raise FileNotFoundError(
        #             f"{stats_path} does not exist, no mean/std from train set"
        #         )
        # else:
        #     with open(stats_path, "r") as f:
        #         l = f.readlines()
        #         stats = l[1].split(";")
        #         self.mean = float(stats[0])
        #         self.std = float(stats[1])
        print(
            f"[{split} dataset ({args.dataset}_{self.sample_rate})]: Loaded mean/std: {self.mean}, {self.std}"
        )

        super(MTTDataset, self).__init__(self.sample_rate, self.audio_length, self.tracks_list, self.tracks_list_all, self.mean, self.std)

    # get one segment (==59049 samples) and its 50-d label
    def __getitem__(self, index):
        track_id, fp, label, segment = self.tracks_list[index]

        try:
            audio = self.get_audio(fp)
        except Exception as e:
            print(f"Skipped {track_id, fp}, could not load audio: {e}")
            return self.__getitem__(index + 1)

        # only transform if unsupervised training
        if self.model_name == "clmr" and self.transform:
            audio = self.transform(audio, self.mean, self.std)
        elif self.model_name == "cpc":
            max_samples = audio.size(1)
            start_idx = random.randint(0, max_samples - self.audio_length)
            audio = audio[:, start_idx : start_idx + self.audio_length]
            audio = self.normalise_audio(audio)
            audio = (audio, audio)
        else:
            raise Exception("Transformation unknown")

        return audio, label, track_id

