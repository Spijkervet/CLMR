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
import librosa


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
        if args.lin_eval:
            fp = os.path.join(path, str(fn) + ".mp3")
            index_name = "-".join(fn.stem.split("-")[:-2])  # get all tracks together
        else:
            d = fn.parent.stem
            fn = fn.stem
            index_name = "-".join(fn.split("-")[:-2])
            fp = os.path.join(path, d, index_name + "-0-full.mp3")

        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            if index_name not in index.keys():
                index[index_name] = index_num
                index_num += 1

            label = torch.FloatTensor(label)
            items.append((index[index_name], fp, label))
            tracks_dict[index[index_name]].append(idx)
        else:
            print("File not found: {}".format(fp))
    return items, tracks_dict


"""
"""


def default_loader(path):
    # audio, sr = librosa.core.load(path, sr=22050)
    # audio = torch.from_numpy(audio)
    # audio = audio.reshape(1, -1)
    # return audio ,sr
    # return audio
    return torchaudio.load(path, normalization=True)
    # return torchaudio.load(path, normalization=lambda x: torch.abs(x).max())


def get_dataset_stats(loader, tracks_list, stats_path):
    means = []
    stds = []
    for track_id, fp, label in tqdm(tracks_list):
        audio, sr = loader(fp)
        means.append(audio.mean())
        stds.append(audio.std())

    return np.array(means).mean(), np.array(stds).mean()


def default_indexer(args, path, tracks, labels, sample_rate, num_tags, tag_list):
    items = []
    tracks_dict = defaultdict(list)
    index = 0
    prev_index = None
    for idx, track in tracks.iterrows():
        fn = Path(track["mp3_path"].split(".")[0])
        if args.lin_eval:
            fp = os.path.join(path, str(fn) + ".mp3")
            index_name = "-".join(fn.stem.split("-")[:-2])  # get all tracks together
        else:
            d = fn.parent.stem
            fn = fn.stem
            index_name = "-".join(fn.split("-")[:-2])
            fp = os.path.join(path, d, index_name + "-0-full.mp3")

        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            if index_name != prev_index:
                prev_index = index_name
                index += 1

            # build label in the order of 50_tags.txt
            label = np.zeros(num_tags)
            for i, tag in enumerate(tag_list):
                if tag == "":
                    continue

                if track[tag] == 1:
                    label[i] = 1
            label = torch.FloatTensor(label)

            labels_vec = torch.FloatTensor(list(labels.iloc[idx]))
            if (label == labels_vec).all() == False:
                raise Exception("Labels do not match")

            items.append((index, fp, label))
            tracks_dict[index].append(idx)
        else:
            print(f"File not found or empty: {fp}")
    return items, tracks_dict


class MTTDataset(Dataset):
    def __init__(
        self,
        args,
        annotations_file,
        train,
        indexer=default_indexer,
        loader=default_loader,
        transform=None,
    ):
        """
        Args : 
            csvfile : train/val/test csvfiles
            audio_dir : directory that contains folders 0 - f
        """

        self.indexer = indexer
        self.loader = loader
        self.tag_list = open(args.list_of_tags, "r").read().split("\n")
        self.audio_dir = args.mtt_processed_audio
        self.num_tags = args.num_tags
        self.annotations_file = annotations_file
        self.transform = transform
        self.sample_rate = args.sample_rate
        self.audio_length = args.audio_length
        
        if args.mtt_pons_eval:
            [audio_repr_paths, id2audio_repr_path] = load_id2path(
                Path(args.mtt_processed_annot) / "index_mtt.tsv"
            )
            [ids, id2gt] = load_id2gt(self.annotations_file)
            self.tracks_list_all, self.tracks_dict = pons_indexer(
                args, self.audio_dir, id2audio_repr_path, id2gt
            )

            # oversample to balance classes for logistic regression
            # if args.lin_eval:
            #     print(self.annotations_file)
            #     exit(0)


        else:
            self.annotations_frame = pd.read_csv(self.annotations_file, delimiter="\t")
            self.labels = self.annotations_frame.drop(["clip_id", "mp3_path"], axis=1)

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

        if args.lin_eval:
            print("### Linear evaluation, using segmented dataset ###")
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
        # print(len(self.tracks_list_all), len(self.tracks_list))

        # normalise entire dataset
        # name = "Train" if train else "Test"
        # stats_path = os.path.join(args.data_input_dir, args.dataset, "statistics.csv")
        # if not os.path.exists(stats_path):
        #     print(f"[{name} dataset]: Fetching dataset statistics (mean/std)")
        #     if train:
        #         self.mean, self.std = get_dataset_stats(self.loader, self.tracks_list, stats_path)
        #         write_statistics(self.mean, self.std, len(self.tracks_list), stats_path)
        #     else:
        #         raise FileNotFoundError(f"{stats_path} does not exist, no mean/std from train set")
        # else:
        #     with open(stats_path, "r") as f:
        #         l = f.readlines()
        #         stats = l[1].split(";")
        #         self.mean = float(stats[0])
        #         self.std = float(stats[1])

        # print(f"[{name} dataset]: Loaded mean/std: {self.mean}, {self.std}")

    def get_audio(self, fp):
        audio, sr = self.loader(fp)

        max_samples = audio.size(1)
        assert (
            max_samples - self.audio_length
        ) > 0, "max samples exceeds number of samples in crop"
        return audio

    def get_full_size_audio(self, track_id, fp):
        # segments = self.tracks_dict[track_id]
        # batch_size = len(segments)
        audio = self.get_audio(fp)

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
        track_id, fp, label = self.tracks_list[index]

        audio = self.get_audio(fp)

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
            _, fp, label = self.tracks_list_all[index]  # from non-dup!!

            audio = self.get_audio(fp)

            start_idx = idx * self.audio_length

            # too large
            if (start_idx + self.audio_length) > audio.size(1):
                return None

            batch[idx, 0, :] = audio[:, start_idx : start_idx + self.audio_length]
        return batch
