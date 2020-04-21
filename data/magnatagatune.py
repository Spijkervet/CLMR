import torch
import torchaudio
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import librosa
import warnings
warnings.filterwarnings('ignore') # to supress librosa

def default_loader(path):
    # audio, sr = librosa.core.load(path, sr=22050)
    # return audio
    return torchaudio.load(path) #, normalization=False)


def default_indexer(args, path, tracks, labels, sample_rate, num_tags, tag_list):
    items = []
    tracks_dict = defaultdict(list)
    index = 0
    prev_index = None
    for idx, t in enumerate(tracks.index):
        fn = Path(tracks.iloc[idx]["mp3_path"].split(".")[0])
        if args.lin_eval:
            fp = os.path.join(path, str(fn) + ".mp3") 
            index_name = "-".join(fn.stem.split("-")[:-2]) # get all tracks together
        else:
            d = fn.parent.stem
            fn = fn.stem
            index_name = "-".join(fn.split("-")[:-2])
            fp = os.path.join(path, d, index_name + "-0-full.mp3")
        if os.path.exists(fp):
            if index_name != prev_index:
                prev_index = index_name
                index += 1

            # build label in the order of 50_tags.txt
            label = np.zeros(num_tags)
            for i, tag in enumerate(tag_list):
                if tag == "":
                    continue

                if tracks[tag].iloc[idx] == 1:
                    label[i] = 1
            label = torch.FloatTensor(label)

            items.append((index, fp, label))
            tracks_dict[index].append(idx)
    return items, tracks_dict


class MTTDataset(Dataset):
    def __init__(self, args, annotations_file, indexer=default_indexer, transform=None):
        """
        Args : 
            csvfile : train/val/test csvfiles
            audio_dir : directory that contains folders 0 - f
        """

        self.indexer = indexer
        self.tag_list = open(args.list_of_tags, "r").read().split("\n")
        self.audio_dir = args.mtt_processed_audio
        self.num_tags = args.num_tags
        self.annotations_file = annotations_file

        self.annotations_frame = pd.read_csv(self.annotations_file, delimiter="\t")

        self.labels = self.annotations_frame.drop(["clip_id", "mp3_path"], axis=1)

        self.transform = transform

        self.sample_rate = args.sample_rate
        self.audio_length = args.audio_length

        self.tracks_list_all, self.tracks_dict = self.indexer(
            args,
            self.audio_dir,
            self.annotations_frame,
            self.labels,
            self.sample_rate,
            self.num_tags,
            self.tag_list,
        )

        self.nodups = []
        self.indexes = []
        for track_id, fp, label in self.tracks_list_all:
            if track_id not in self.indexes:
                self.nodups.append([track_id, fp, label])
                self.indexes.append(track_id)
        
        print(len(self.nodups), len(self.tracks_list_all))
        if args.lin_eval:
            print("### Linear evaluation, using segmented dataset ###")
            self.tracks_list = self.tracks_list_all
        else:
            print("### Pre-training, using whole dataset ###")
            self.tracks_list = self.nodups
        # print(len(self.tracks_list_all), len(self.tracks_list))

    def get_audio(self, index, fp):
        audio, sr = default_loader(fp)
        # audio = torch.from_numpy(audio)
        # audio = audio.reshape(1, -1)

        max_samples = audio.size(1)
        assert (
            max_samples - self.audio_length
        ) > 0, "max samples exceeds number of samples in crop"
        return audio

    # get one segment (==59049 samples) and its 50-d label
    def __getitem__(self, index):
        track_id, fp, label = self.tracks_list[index]

        audio = self.get_audio(index, fp)

        # this is done in the RandomResizeCrop transformation
        # max_samples = audio.size(1)
        # start_idx = np.random.randint(0, max_samples - self.audio_length)
        # audio = audio[:, start_idx : start_idx + self.audio_length]

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
            _, fp, label = self.tracks_list[index]
            audio = self.get_audio(index, fp)

            start_idx = idx * self.audio_length
            batch[idx, 0, :] = audio[:, start_idx : start_idx + self.audio_length]
        return batch
