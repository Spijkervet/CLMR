import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torchaudio
from collections import defaultdict
import mirdata
import mirdata.billboard
import scipy.signal as signal
from matplotlib import pyplot as plt
import librosa


def default_loader(path):
    return torchaudio.load(path, normalization=True)


def default_indexer(path, fp, tracks, sr):
    items = []
    tracks_dict = defaultdict(list)
    index = 0
    with open(fp, "r") as f:
        for l in f.read().splitlines():
            line = l.split("\t")
            track_id = int(line[0])
            sample_id = int(line[1])
            start_idx = int(line[2])
            end_idx = int(line[3])
            labels = eval(line[4])

            fp = os.path.join(
                path,
                str(track_id),
                "{}-{}-{}.wav".format(sample_id, start_idx, end_idx),
            )
            if os.path.exists(fp):
                labels = torch.FloatTensor(labels)
                items.append((track_id, fp, labels))
                tracks_dict[track_id].append(index)
                index += 1

    return items, tracks_dict


def track_index(args):
    if args.dataset == "billboard":
        mirdata.billboard.download()
        # mirdata.billboard.validate()
        tracks = mirdata.billboard.load()
    elif args.dataset == "beatles":
        mirdata.beatles.download()
        # mirdata.billboard.validate()
        tracks = mirdata.beatles.load()
    else:
        raise Exception("Invalid dataset")
    return tracks


def get_audio_path(path, sr):
    return os.path.join(os.path.dirname(path), str(sr) + os.path.splitext(path)[1])


class MIRDataset(Dataset):
    def __init__(
        self,
        args,
        train,
        unlabeled,
        audio_length,
        indexer=default_indexer,
        loader=default_loader,
        track_index=track_index,
        diff_train_dataset=None,
        transform=None,
    ):
        # self.tracks = tracks

        self.root_dir = os.path.join(args.data_input_dir, args.dataset, "samples")

        if unlabeled:
            split_name = "unlabeled"
        elif train:
            split_name = "train"
        else:
            split_name = "test"

        self.labels_file = os.path.join(
            args.data_input_dir, args.dataset, "labels", f"{split_name}_split.txt"
        )

        self.sample_rate = args.sample_rate

        self.audio_length = audio_length
        self.indexer = indexer
        self.loader = loader
        self.track_index = track_index
        self.transform = transform

        self.tracks_index = self.track_index(args)
        self.tracks_list, self.tracks_dict = self.indexer(
            self.root_dir, self.labels_file, self.tracks_index, sr=self.sample_rate
        )

        if not diff_train_dataset:
            with open(
                os.path.join(args.data_input_dir, f"{args.dataset}_statistics.csv"), "r"
            ) as f:
                l = f.readlines()
                stats = l[1].split(";")
                mean = float(stats[0])
                std = float(stats[1])
                chroma_mean = float(stats[2])
                chroma_std = float(stats[3])
        else:
            print("### USING DIFFERENT TRAIN DATASET MEAN/STD ###")
            mean = diff_train_dataset.mean
            std = diff_train_dataset.std
            chroma_mean = diff_train_dataset.chroma_mean
            std_mean = diff_train_dataset.chroma_std
            print(mean, std, chroma_mean, chroma_std)
            print("###")

        self.mean = mean
        self.std = std

        self.chroma_mean = chroma_mean
        self.chroma_std = chroma_std

        print(f"Train Dataset mean: {mean}, std: {std}")
        print(f"[{split_name}]: {len(self.tracks_list)} segments")

    def get_audio_fp(self, track_id, sample_id, start_idx, end_idx, label):

        return fp

    def get_audio(self, fp):
        audio, sr = self.loader(fp)

        max_samples = audio.size(1)
        if sr != self.sample_rate:
            raise Exception("Sample rate is not consistent throughout the dataset")

        if max_samples - self.audio_length < 0:
            raise Exception("Max samples exceeds number of samples in crop")

        return audio

    def __getitem__(self, index):
        track_id, fp, labels = self.tracks_list[index]

        try:
            audio = self.get_audio(fp)
        except:
            pass
            print(f"Skipped {track_id, fp}, could not load audio")
            return self.__getitem__(index+1)


        # Normalize audio
        # audio = (audio - self.mean) / self.std

        if self.transform:
            audio = self.transform(audio)

        return audio, labels

        ## chroma
        # fp = os.path.join(
        #     self.root_dir,
        #     track_id,
        #     "{}-{}-{}-{}.chroma".format(sample_id, start_idx, end_idx, label),
        # )
        # chroma = torch.load(fp)
        # chroma = chroma[r_start_idx : r_start_idx + self.audio_length]
        # chroma = (chroma - self.chroma_mean) / self.chroma_std
        # chroma = chroma.reshape(1, -1)

        # specgram = torchaudio.transforms.MelSpectrogram()(audio)
        # specgram = specgram.log2()[0,:,:]

        # S = librosa.feature.melspectrogram(audio[0].cpu().numpy(), sr=self.sample_rate, n_mels=128)
        # specgram2 = librosa.power_to_db(S, ref=np.max)/40+1

        # print(specgram.shape)
        # return ((audio, audio), label) #, label, track_id, r_start_idx

    def __len__(self):
        return len(self.tracks_list)

    def sample_audio_by_track_id(self, track_id, batch_size=20):
        """
        Get audio samples based on the track_id
        used for plotting the latent representations of different tracks
        """
        batch = torch.zeros(batch_size, 1, self.audio_length)
        for idx in range(batch_size):
            index = self.tracks_dict[track_id][0]
            track_id, fp, label = self.tracks_list[index]
            audio = self.get_audio(fp)

            start_idx = idx * self.audio_length
            # too large
            if (start_idx + self.audio_length) > audio.size(1):
                return None

            batch[idx, 0, :] = audio[:, start_idx : start_idx + self.audio_length]

        return batch

    def get_full_size_test_item(self, index):
        """
        get audio samples that cover the full length of the input files
        used for testing the phone classification performance
        """
        track_id, fp, _ = self.tracks_list[index]

        audio, sr = self.loader(fp)

        assert (
            sr == self.sample_rate
        ), "Sample rate is not consistent throughout the dataset"

        # discard last part that is not a full 10ms
        ms = self.sample_rate / 100

        max_length = int(audio.size(1) // ms * ms)
        audio = audio[:max_length]

        norm_audio = (audio - self.mean) / self.std

        return norm_audio, audio, track_id, sample_id

    def unnorm(self, norm_audio):
        audio = (norm_audio * self.std) + self.mean
        return audio
