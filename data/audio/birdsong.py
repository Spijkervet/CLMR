import torch
import torchaudio
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import time

from .dataset import Dataset
from scripts.datasets.utils import write_statistics
from utils import random_undersample_balanced
from utils.audio import load_tracks, concat_tracks

def load_csv(fp):
    df = pd.read_csv(fp)
    return df



class BirdsongDataset(Dataset):

    base_dir = "birdsong"
    splits = ("train", "valid")

    def __init__(
        self, args, split, pretrain, download=False, transform=None,
    ):
        self.name = "birdsong"
        self.split = split
        self.pretrain = pretrain
        self.transform = transform
        self.sample_rate = args.sample_rate
        self.audio_length = args.audio_length
        self.load_ram = args.load_ram
        self.supervised = args.supervised
        self.num_segments = None
        self.skip = []

        # MagnaTagATune has clips of 30s, of which the last is not a full |audio_length|

        self.audio_dir = os.path.join(args.data_input_dir, self.base_dir, "raw")
        self.audio_proc_dir = os.path.join(
            args.data_input_dir, self.base_dir, "processed"
        )

        processed_annot = Path(
            args.data_input_dir, self.base_dir, "processed_annotations"
        )

        if split == "train":
            self.annotations_file = Path(processed_annot) / "train.csv"
        elif split == "valid":
            self.annotations_file = Path(processed_annot) / "train.csv"
        else:
            self.annotations_file = Path(processed_annot) / "test.csv"

        df = load_csv(self.annotations_file)

        if split == "test":
            self.tags = []
            self.index = []
            self.track_index = []
        else:
            self.tags = sorted(df.ebird_code.unique().tolist(), reverse=True)
            self.num_tags = len(self.tags)
            self.index = []
            self.track_index = defaultdict(list)
            segment = 0
            for clip_id, row in df.iterrows():
                # fp = os.path.join(row["ebird_code"], row["filename"])
                track_id = row["ebird_code"]
                label = np.zeros(self.num_tags)
                label[self.tags.index(track_id)] = 1
                fp = f"{track_id}-{clip_id}-{self.sample_rate}.wav"
                fp = os.path.join(self.audio_proc_dir, self.split, fp)
                self.index.append([track_id, clip_id, segment, fp, label])
                self.track_index[track_id].append([clip_id, segment, fp, label])
            
        print(len(self.index))

        ## get dataset statistics
        self.mean = None
        self.std = None

        if self.load_ram and self.pretrain and self.split == "train":
            print("Loading train data into memory for faster training")
            self.audios = load_tracks(args.sample_rate, self.index)

        if self.split == "valid":
            self.index = self.index[:5]

        super(BirdsongDataset, self).__init__(
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
        if fp in self.skip:
            return self.__getitem__(idx + 1)

        ## for faster i/o
        # clip_id, segment, fp, label = random.choice(self.track_index[track_id])
        ## fp = os.path.splitext(fp)[0] + "-" + str(segment) + ".wav"

        label = torch.FloatTensor(label)
        try:
            if self.load_ram and self.pretrain and self.split == "train":
                audio = self.audios["{}-{}".format(track_id, clip_id)]
            else:
                audio = self.get_audio(fp)

        except Exception as e:
            print(f"Skipped {track_id, fp}, could not load audio: {e}")
            self.skip.append(fp)
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
            audio = torch.from_numpy(audio)
            audio = (audio, audio)

        return audio, label

