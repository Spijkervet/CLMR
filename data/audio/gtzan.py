import torch
import torchaudio
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from scripts.datasets.utils import write_statistics
import random

# much faster loading
from scipy.io import wavfile


def path2gt(path):
    """ Given the audio path, it returns the ground truth label.
    """
    if 'blues' in path:
        return 0
    elif 'classical' in path:
        return 1
    elif 'country' in path:
        return 2
    elif 'disco' in path:
        return 3
    elif 'hiphop' in path:
        return 4
    elif 'jazz' in path:
        return 5
    elif 'metal' in path:
        return 6
    elif 'pop' in path:
        return 7
    elif 'reggae' in path:
        return 8
    elif 'rock' in path:
        return 9
    else:
        raise Exception('Did not find the corresponding ground truth (' + str(path) + ')!')


def default_indexer(args, path, annotations_file):
    items = []
    tracks_dict = defaultdict(list)
    index = {}
    index_num = 0
    with open(annotations_file, "r") as f:
        lines = f.read().splitlines()
        for idx, fp in enumerate(lines):
            label = path2gt(fp)

            fp = os.path.join(path, fp)
            if os.path.exists(fp) and os.path.getsize(fp) > 0:
                audio_example, _ = default_loader(fp)
                num_segments = audio_example.size(1) // args.audio_length

                index_name = Path(fp).stem
                if index_name not in index.keys():
                    index[index_name] = index_num
                    index_num += 1

                # if supervised/eval
                if args.model_name == "supervised" or args.lin_eval:
                    # n segments so it sees all data
                    for n in range(num_segments):
                        items.append((index[index_name], fp, label, n))
                else:
                    items.append(
                        (index[index_name], fp, label, 0)
                    )  # only one segment, since full track
                tracks_dict[index[index_name]].append(idx)
            else:
                print("File not found: {}".format(fp))
    return items, tracks_dict


def default_loader(path):
    # with audio normalisation
    audio, sr = torchaudio.load(path, normalization=lambda x: torch.abs(x).max())
    return audio, sr


def get_dataset_stats(loader, tracks_list, stats_path):
    means = []
    stds = []
    for track_id, fp, label, _ in tqdm(tracks_list):
        audio, sr = loader(fp)
        means.append(audio.mean())
        stds.append(audio.std())
    return np.array(means).mean(), np.array(stds).mean()



class GTZANDataset(Dataset):
    def __init__(
        self, args, train, unlabeled, loader=default_loader, indexer=default_indexer, transform=None,
    ):
        self.lin_eval = args.lin_eval
        self.indexer = indexer
        self.loader = loader
        self.transform = transform
        self.sample_rate = args.sample_rate
        self.audio_length = args.audio_length
        self.model_name = args.model_name
        self.num_tags = 10

        if unlabeled:
            version = "unlabeled" # TODO
            split_name = "unlabeled"
        elif train:
            version = "segments"
            split_name = "train"
        else:
            version = "segments"
            split_name = "test"

        self.audio_dir = os.path.join(args.data_input_dir, "gtzan", f"processed_segments_{self.sample_rate}_wav")

        annotations_dir = Path(
            args.data_input_dir, "gtzan", "annotations"
        )
        self.annotations_file = Path(annotations_dir) / f"{split_name}_filtered.txt"

        self.tracks_list_all, self.tracks_dict = self.indexer(
            args, self.audio_dir, self.annotations_file
        )

        self.nodups = []
        self.indexes = []
        for track_id, fp, label, segment in self.tracks_list_all:
            if track_id not in self.indexes:
                self.nodups.append([track_id, fp, label, segment])
                self.indexes.append(track_id)

        if args.lin_eval or args.model_name == "supervised":
            print("### Linear / Supervised training, using segmented dataset ###")
            self.tracks_list = self.tracks_list_all
        else:
            print("### Pre-training, using whole dataset ###")
            print(
                "Removed duplicates from:",
                len(self.tracks_list_all),
                "to:",
                len(self.nodups),
            )
            self.tracks_list = self.nodups

        self.tracks_list_test = []
        for track_id, fp, label, segment in self.tracks_list:
            if (
                segment == 0
            ):  # remove segment, since we will evaluate with get_full_size_audio()
                self.tracks_list_test.append([track_id, fp, label, -1])

        print(f"Num segments: {len(self.tracks_list)}")
        print(f"Num tracks: {len(self.tracks_list_test)}")

        ## get dataset statistics
        name = "Train" if train else "Test"
        stats_path = os.path.join(args.data_input_dir, args.dataset, f"statistics_{version}.csv")
        if not os.path.exists(stats_path):
            print(f"[{name} dataset]: Fetching dataset statistics (mean/std)")
            if train:
                self.mean, self.std = get_dataset_stats(
                    self.loader, self.tracks_list, stats_path
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

        print(f"[{name} dataset]: Loaded mean/std: {self.mean}, {self.std}")

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

    def get_full_size_audio(self, track_id, fp):
        # segments = self.tracks_dict[track_id]
        # batch_size = len(segments)
        audio = self.get_audio(fp)
        
        # normalise audio
        audio = self.normalise_audio(audio)

        # split into equally sized tensors of self.audio_length
        batch = torch.split(audio, self.audio_length, dim=1)

        # remove last, since it is not a full self.audio_length, and stack
        batch = torch.cat(batch[:-1])

        # reshape to B x 1 x N
        batch = batch.reshape(batch.shape[0], 1, -1)
        return batch

    def normalise_audio(self, audio):
        return (audio - self.mean) / self.std

    def denormalise_audio(self, norm_audio):
        return (norm_audio * self.std) + self.mean

    # get one segment (==59049 samples) and its 50-d label
    def __getitem__(self, index):
        track_id, fp, label, segment = self.tracks_list[index]

        try:
            audio = self.get_audio(fp)
        except:
            pass
            print(f"Skipped {track_id, fp}, could not load audio")
            return self.__getitem__(index+1)

        # only transform if unsupervised training
        if self.lin_eval or self.model_name == "supervised":
            start_idx = segment * self.audio_length
            audio = audio[:, start_idx : start_idx + self.audio_length]
            audio = self.normalise_audio(audio)
            audio = (audio, audio)
        elif self.model_name == "clmr" and self.transform:
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
            _, fp, label, _ = self.tracks_list_all[index]  # from non-dup!!

            audio = self.get_audio(fp)

            start_idx = idx * self.audio_length

            # too large
            if (start_idx + self.audio_length) > audio.size(1):
                return None

            batch[idx, 0, :] = audio[:, start_idx : start_idx + self.audio_length]
        return batch
