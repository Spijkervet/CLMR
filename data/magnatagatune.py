import torch
import torchaudio
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from datasets.utils.utils import write_statistics
import random

# much faster loading
import soundfile as sf
from scipy.io import wavfile


"""
From Pons et al.
"""


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


def pons_indexer(args, path, id2audio, id2gt):
    items = []
    tracks_dict = defaultdict(list)
    # index = 0
    prev_index = None
    index = {}
    index_num = 0
    for idx, (clip_id, label) in enumerate(id2gt.items()):
        fn = Path(id2audio[clip_id].split(".")[0])
        if args.lin_eval or args.supervised:
            fp = os.path.join(path, str(fn) + ".wav")
            index_name = "-".join(fn.stem.split("-")[:-2])  # get all tracks together
        else:
            d = fn.parent.stem
            fn = fn.stem
            index_name = "-".join(fn.split("-")[:-2])
            fp = os.path.join(path, d, index_name + "-0-full.wav")

        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            if index_name not in index.keys():
                index[index_name] = index_num
                index_num += 1

            label = torch.FloatTensor(label)

            # if supervised/eval
            if args.supervised or args.lin_eval:
                # n segments so it sees all data
                num_segments = 10
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


"""
"""


def default_loader(path):
    # with audio normalisation
    audio, sr = torchaudio.load(path, normalization=lambda x: torch.abs(x).max())
    # audio, sr = torchaudio.load(path, normalization=True)

    # is a bit slower with multiprocessing loading into the dataloader (num_workers > 1)
    # rate, sig = wavfile.read(path)
    # sig = sig.astype('float32') / 32767 # normalise 16 bit PCM between -1 and 1
    # audio = torch.FloatTensor(sig).reshape(1, -1)

    # soundfile, also fast
    # audio, sr = sf.read(path)
    # audio = audio / (1 << 16) # normalise
    # audio = torch.from_numpy(audio).float().reshape(1, -1)
    return audio, sr


def get_dataset_stats(loader, tracks_list):
    means = []
    stds = []
    for track_id, fp, label, _ in tqdm(tracks_list):
        audio, sr = loader(fp)
        means.append(audio.mean())
        stds.append(audio.std())
    return np.array(means).mean(), np.array(stds).mean()


class MTTDataset(Dataset):
    def __init__(
        self, args, train, validation=False, loader=default_loader, transform=None,
    ):
        self.supervised = args.supervised
        self.num_tags = args.num_tags
        self.lin_eval = args.lin_eval
        self.indexer = pons_indexer
        self.loader = loader
        self.transform = transform
        self.sample_rate = args.sample_rate
        self.audio_length = args.audio_length
        self.model_name = args.model_name

        if args.lin_eval or args.supervised:
            version = "segments"
        else:
            version = "concat"
            
        dir_name = f"processed_{version}_{self.sample_rate}_wav"

        self.audio_dir = os.path.join(args.data_input_dir, "magnatagatune", dir_name)

        mtt_processed_annot = Path(
            args.data_input_dir, "magnatagatune", "processed_annotations"
        )

        at_least_one_pos = ""
        if args.at_least_one_pos:
            at_least_one_pos = "_onepos"

        if train:
            self.annotations_file = Path(mtt_processed_annot) / f"train_gt_mtt{at_least_one_pos}.tsv"
        elif validation:
            self.annotations_file = Path(mtt_processed_annot) / f"val_gt_mtt{at_least_one_pos}.tsv"
        else:
            self.annotations_file = Path(mtt_processed_annot) / f"test_gt_mtt{at_least_one_pos}.tsv"

        [audio_repr_paths, id2audio_repr_path] = load_id2path(
            Path(mtt_processed_annot) / "index_mtt.tsv"
        )
        [ids, id2gt] = load_id2gt(self.annotations_file)
        self.tracks_list_all, self.tracks_dict = self.indexer(
            args, self.audio_dir, id2audio_repr_path, id2gt
        )

        self.nodups = []
        self.indexes = []
        for track_id, fp, label, segment in self.tracks_list_all:
            if track_id not in self.indexes:
                self.nodups.append([track_id, fp, label, segment])
                self.indexes.append(track_id)

        if version == "segments":
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
        if validation:
            name = "Validation"
        if args.pretrain_dataset == "billboard":
            version = "unlabeled"
        if args.pretrain_dataset == "fma":
            version = "medium"
            
        stats_path = os.path.join(args.data_input_dir, args.pretrain_dataset, f"statistics_{version}_{self.sample_rate}{at_least_one_pos}.csv")
        print(stats_path)
        if not os.path.exists(stats_path):
            print(f"[{name} dataset]: Fetching dataset statistics (mean/std) from pre-trained {args.pretrain_dataset} for {version}_{self.sample_rate}{at_least_one_pos} version")
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
        
        print(f"[{name} dataset ({args.pretrain_dataset}_{version}_{self.sample_rate})]: Loaded mean/std: {self.mean}, {self.std}")

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
        if self.lin_eval or self.supervised:
            start_idx = random.randint(0, segment * self.audio_length) # audio.size(1) - self.audio_length) # 
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
