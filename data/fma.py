
import os
import sys
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from collections import defaultdict
from pathlib import Path

import sklearn.preprocessing

sys.path.append("../")
from fma.utils import get_audio_path
from fma.utils import load as load_fma


def default_loader(path):
    return torchaudio.load(path, normalization=True)


def default_indexer(path, tracks, labels, sample_rate):
    items = []
    track_dict = defaultdict(list)
    for idx, t in enumerate(tracks.index):
        fp = get_audio_path(path, t)
        resampled_fp = f"{os.path.splitext(fp)[0]}.mp3"
        if os.path.exists(resampled_fp):
            dir_id = str(Path(fp).parent.name)
            track_id = str(Path(fp).stem)
            label = labels[idx] # TODO
            items.append((track_id, resampled_fp, label))
            track_dict[track_id].append(idx)

    return items, track_dict


class FmaDataset(Dataset):
    def __init__(
        self,
        args,
        train,
        audio_length,
        indexer=default_indexer,
        loader=default_loader,
        transform=None,
    ):
        self.root_dir = os.path.join(args.data_input_dir, "fma", f"{args.fma_version}_{args.sample_rate}")
        self.sample_rate = args.sample_rate
        self.audio_length = audio_length
        self.indexer = indexer
        self.loader = loader
        self.transform = transform


        tracks_file = os.path.join(args.data_input_dir, "fma", "fma_metadata", "tracks.csv")

        tracks = load_fma(tracks_file)

        train_df = tracks["set", "split"] == "training"
        val_df = tracks["set", "split"] == "validation"
        test_df = tracks["set", "split"] == "test"

        # subset
        subset = tracks["set", "subset"] <= "medium"

        train_df = tracks.loc[subset & train_df | val_df]
        # val_df = tracks.loc[subset & val_df]
        test_df = tracks.loc[subset & test_df]


        ## labels
        y_train = train_df["track"]["genre_top"]
        y_test = test_df["track"]["genre_top"]

        print(y_train)
        exit(0) # TODO

        # fill missing genres with unknown token
        y_train = y_train.cat.add_categories("<UNK>")
        y_test = y_test.cat.add_categories("<UNK>")
        y_train = y_train.fillna("<UNK>")
        y_test = y_test.fillna("<UNK>")

        enc = sklearn.preprocessing.LabelEncoder()
        y_train = enc.fit_transform(y_train)
        y_test = enc.transform(y_test)

        all_y = np.concatenate((y_train, y_test), axis=0)
        args.n_classes = len(np.unique(all_y))

        if train:
            self.tracks = train_df
            self.labels = y_train
        else:
            self.tracks = test_df
            self.labels = y_test

        self.tracks_list, self.tracks_dict = self.indexer(self.root_dir, self.tracks, self.labels, self.sample_rate)

        train_test = "Train" if train else "Test"
        print(f"[{train_test}]: Loaded {len(self.tracks_list)} tracks")

    def get_audio(self, fp):
        audio, sr = self.loader(fp)
        audio = audio.mean(axis=0).reshape(1, -1)  # to mono
        max_samples = audio.size(1)

        if sr != self.sample_rate:
            raise Exception("Sample rate is not consistent throughout the dataset")

        if max_samples - self.audio_length <= 0:
            raise Exception("Max samples exceeds number of samples in crop")

        return audio

    def __getitem__(self, index):
        track_id, fp, label = self.tracks_list[index]

        try:
            audio = self.get_audio(fp)
        except:
            pass
            print(f"Skipped {track_id, fp}, could not load audio")
            return self.__getitem__(index+1)

        if self.transform:
            audio = self.transform(audio)

        return audio, label

    def __len__(self):
        return len(self.tracks_list)

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
