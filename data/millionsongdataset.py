import torch
import torchaudio
from torch.utils.data import Dataset
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

# much faster loading
import soundfile as sf
from scipy.io import wavfile


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
        id2gt[id_7d] = eval(gt)  # gt is array
        ids.append(id_7d)
    return ids, id2gt


def load_id2path(index_file, msd_7d):
    paths = []
    fspec = open(index_file)
    id2path = dict()
    for line in fspec.readlines():
        msd_id, msd_path = line.strip().split("\t")
        id_7d = msd_7d[msd_id]
        path = os.path.join(id_7d[0], id_7d[1], f"{id_7d}.clip.wav")
        id2path[id_7d] = path
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
            index_name = Path(id2audio[clip_id].split(".")[0]).stem # 7 digital ID
            if index_name not in index.keys():
                index[index_name] = index_num
                index_num += 1

            label = torch.FloatTensor(label)
            items.append(
                (index[index_name], fp, label)
            )
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
    def __init__(
        self, args, train, validation=False, loader=default_loader, transform=None, indexer=default_indexer
    ):
        self.supervised = args.supervised
        self.indexer = indexer
        self.loader = loader
        self.transform = transform
        self.sample_rate = args.sample_rate
        self.audio_length = args.audio_length
        self.model_name = args.model_name
        self.mean = None # -1.1771917e-05
        self.std = None # 0.14238065


        dir_name = f"processed_{args.sample_rate}"
        self.audio_dir = os.path.join(args.data_input_dir, "million_song_dataset", dir_name)

        msd_processed_annot = Path(
            args.data_input_dir, "million_song_dataset", "processed_annotations"
        )

        if train:
            self.annotations_file = Path(msd_processed_annot) / "train_gt_msd.tsv"
            split = "Train"
        elif validation:
            self.annotations_file = Path(msd_processed_annot) / "val_gt_msd.tsv"
            split = "Validation"
        else:
            self.annotations_file = Path(msd_processed_annot) / "test_gt_msd.tsv"
            split = "Test"

        self.msd_to_7d = pickle.load(open(Path(msd_processed_annot) / "MSD_id_to_7D_id.pkl", "rb"))

        # int to label
        with open(Path(msd_processed_annot) / f"output_labels_msd.txt", "r") as f:
            lines = f.readlines()
            self.tags = eval(lines[1][lines[1].find("["):])
            self.num_tags = len(self.tags)
            
        [audio_repr_paths, id2audio_repr_path] = load_id2path(
            Path(msd_processed_annot) / "index_msd.tsv", self.msd_to_7d
        )
        [ids, id2gt] = load_id2gt(self.annotations_file, self.msd_to_7d)
        self.tracks_list, self.tracks_dict = self.indexer(
            args, self.audio_dir, id2audio_repr_path, id2gt
        )

        # reduce dataset to n%
        if args.perc_train_data < 1.0 and (train or validation): # only on train set
            print("Train dataset size:", len(self.tracks_list))
            train_X_indices = np.array([idx for idx in range(len(self.tracks_list))]).reshape(-1, 1)
            train_y = np.array([label.numpy() for _, _, label, _ in self.tracks_list])
            train_X_indices, _ = random_undersample_balanced(train_X_indices, train_y, args.perc_train_data)

            new_tracks_list = []
            for idx, (track_id, fp, label, segment) in enumerate(self.tracks_list):
                if idx in train_X_indices:
                    new_tracks_list.append([track_id, fp, label, segment])
                
            self.tracks_list = new_tracks_list
            print("Undersampled train dataset size:", len(self.tracks_list))

        print(f"Num tracks: {len(self.tracks_list)}")
        
        print(f"[{split} dataset ({args.dataset}_{self.sample_rate})]")

    # get one segment (==59049 samples) and its 50-d label
    def __getitem__(self, index):
        track_id, fp, label = self.tracks_list[index]

        try:
            audio = self.get_audio(fp)
        except Exception as e:
            print(f"Skipped {track_id, fp}, could not load audio: {e}")
            return self.__getitem__(index+1)

        # only transform if unsupervised training
        if self.model_name == "clmr" and self.transform:
            audio = self.transform(audio, self.mean, self.std)
        elif self.model_name == "cpc":
            max_samples = audio.size(1)
            start_idx = random.randint(0, max_samples - self.audio_length)
            audio = audio[:, start_idx : start_idx + self.audio_length]
            audio = (audio, audio)
        else:
            raise Exception("Transformation unknown")

        return audio, label, track_id

    def __len__(self):
        return len(self.tracks_list)

    def get_audio(self, fp):
        audio, sr = self.loader(fp)
        max_samples = audio.shape[1]

        if sr != self.sample_rate:
            raise Exception("Sample rate is not consistent throughout the dataset")

        if max_samples - self.audio_length <= 0:
            raise Exception("Max samples exceeds number of samples in crop")

        if torch.isnan(audio).any():
            raise Exception("Audio contains NaN values")

        return audio

    def normalise_audio(self, audio):
        return (audio - self.mean) / self.std

    def denormalise_audio(self, norm_audio):
        return (norm_audio * self.std) + self.mean

    def get_full_size_audio(self, fp):
        audio = self.get_audio(fp)
        
        # normalise audio
        if self.mean:
            audio = self.normalise_audio(audio)

        # split into equally sized tensors of self.audio_length
        batch = torch.split(audio, self.audio_length, dim=1)

        # remove last, since it is not a full self.audio_length, and stack
        batch = torch.cat(batch[:-1])

        # reshape to B x 1 x N
        batch = batch.reshape(batch.shape[0], 1, -1)
        return batch

    def sample_audio_by_track_id(self, track_id, batch_size=20):
        """
        Get audio samples based on the track_id (batch_size = num_samples)
        used for plotting the latent representations of different tracks
        """
        batch = torch.zeros(batch_size, 1, self.audio_length)
        for idx in range(batch_size):
            index = self.tracks_dict[track_id][0]
            _, fp, label = self.tracks_list[index]  # from non-dup!!

            audio = self.get_audio(fp)

            start_idx = idx * self.audio_length

            # too large
            if (start_idx + self.audio_length) > audio.size(1):
                return None

            batch[idx, 0, :] = audio[:, start_idx : start_idx + self.audio_length]
        return batch
