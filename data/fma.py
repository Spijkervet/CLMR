
import os
import sys
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import sklearn.preprocessing

sys.path.append("../")
from fma.utils import get_audio_path
from fma.utils import load as load_fma

from datasets.utils.utils import write_statistics

def default_loader(path):
    # with audio normalisation
    audio, sr = torchaudio.load(path, normalization=lambda x: torch.abs(x).max())
    return audio, sr


def get_dataset_stats(loader, tracks_list):
    means = []
    stds = []
    for track_id, fp, label, _ in tqdm(tracks_list):
        audio, sr = loader(fp)
        if np.isnan(audio.mean()) or np.isnan(audio.std()):
            continue
        means.append(audio.mean())
        stds.append(audio.std())
    return np.array(means).mean(), np.array(stds).mean()


def default_indexer(path, tracks, labels, sample_rate):
    items = []
    track_dict = defaultdict(list)
    for idx, t in enumerate(tracks.index):
        fp = get_audio_path(path, t)
        fp = f"{os.path.splitext(fp)[0]}.wav"
        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            dir_id = str(Path(fp).parent.name)
            track_id = str(Path(fp).stem)
            label = labels[idx] # TODO
            segment = 0
            items.append((track_id, fp, label, segment))
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
        self.root_dir = os.path.join(args.data_input_dir, "fma", f"processed_segments_{args.fma_version}_{args.sample_rate}_wav")
        self.sample_rate = args.sample_rate
        self.audio_length = audio_length
        self.indexer = indexer
        self.loader = loader
        self.transform = transform

        at_least_one_pos = ""
        if args.at_least_one_pos:
            at_least_one_pos = "_onepos"

        tracks_file = os.path.join(args.data_input_dir, "fma", "fma_metadata", "tracks.csv")

        tracks = load_fma(tracks_file)

        train_df = tracks["set", "split"] == "training"
        val_df = tracks["set", "split"] == "validation"
        test_df = tracks["set", "split"] == "test"

        # subset
        subset = tracks["set", "subset"] <= "medium"

        train_df = tracks.loc[subset & train_df | val_df] # we do not use the val. set.
        # train_df = tracks.loc[subset & train_df | val_df]
        # val_df = tracks.loc[subset & val_df]
        test_df = tracks.loc[subset & test_df]


        ## labels
        y_train = train_df["track"]["genre_top"]
        y_test = test_df["track"]["genre_top"]

        # TODO: labels

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


        name = "Train" if train else "Test"
        version = args.fma_version
        stats_path = os.path.join(args.data_input_dir, args.dataset, f"statistics_{version}_{self.sample_rate}{at_least_one_pos}.csv")
        if not os.path.exists(stats_path):
            print(f"[{name} dataset]: Fetching dataset statistics (mean/std) for {version}_{self.sample_rate}{at_least_one_pos} version")
            if train:
                self.mean, self.std = get_dataset_stats(
                    self.loader, self.tracks_list
                )
                write_statistics(self.mean, self.std, len(self.tracks_list), stats_path)
            else:
                raise FileNotFoundError(
                    f"{stats_path} does not exist, no mean/std from train set"
                )
        else:
            with open(stats_path, "r") as f:
                l = f.readlines()
                stats = l[1].split(";")
                self.mean = float(stats[0])
                self.std = float(stats[1])
        
        print(f"[{name} dataset ({version}_{self.sample_rate})]: Loaded mean/std: {self.mean}, {self.std}")
        print(f"[{name}]: Loaded {len(self.tracks_list)} tracks")

    def get_audio(self, fp):
        audio, sr = self.loader(fp)
        audio = audio.mean(axis=0).reshape(1, -1)  # to mono

        max_samples = audio.size(1)

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

    def __getitem__(self, index):
        track_id, fp, label, _ = self.tracks_list[index]
        
        try:
            audio = self.get_audio(fp)
        except Exception as e:
            print(f"Skipped {track_id, fp}, could not load audio: {e}")
            return self.__getitem__(index+1)

        if self.transform:
            audio = self.transform(audio, self.mean, self.std)

        return audio, label, track_id

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
            _, fp, label, _ = self.tracks_list[index]  # from non-dup!!

            audio = self.get_audio(fp)
            audio = self.normalise_audio(audio)

            start_idx = idx * self.audio_length

            # too large
            if (start_idx + self.audio_length) > audio.size(1):
                return None

            batch[idx, 0, :] = audio[:, start_idx : start_idx + self.audio_length]
        return batch