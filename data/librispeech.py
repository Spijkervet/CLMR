import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict


def default_loader(path):
    return torchaudio.load(path, normalization=False)


def default_flist_reader(flist):
    item_list = []
    speaker_dict = defaultdict(list)
    index = 0
    with open(flist, "r") as rf:
        for line in rf.readlines():
            speaker_id, dir_id, sample_id = line.replace("\n", "").split("-")
            item_list.append((speaker_id, dir_id, sample_id))
            speaker_dict[speaker_id].append(index)
            index += 1

    return item_list, speaker_dict


class LibriDataset(Dataset):
    def __init__(
        self,
        opt,
        root,
        flist,
        audio_length=20480,
        flist_reader=default_flist_reader,
        loader=default_loader,
    ):
        self.root = root
        self.opt = opt

        self.file_list, self.speaker_dict = flist_reader(flist)

        self.loader = loader
        self.audio_length = audio_length

        self.mean = -1456218.7500
        self.std = 135303504.0
    
        self.speaker_id_dict = {}
        for idx, key in enumerate(self.speaker_dict):
            self.speaker_id_dict[key] = idx

    def __getitem__(self, index):
        speaker_id, dir_id, sample_id = self.file_list[index]
        filename = "{}-{}-{}".format(speaker_id, dir_id, sample_id)
        audio, samplerate = self.loader(
            os.path.join(self.root, speaker_id, dir_id, "{}.flac".format(filename))
        )

        assert (
            samplerate == 16000
        ), "Watch out, samplerate is not consistent throughout the dataset!"

        # discard last part that is not a full 10ms
        max_length = audio.size(1) // 160 * 160

        start_idx = np.random.choice(
            np.arange(160, max_length - self.audio_length - 0, 160)
        )

        audio = audio[:, start_idx : start_idx + self.audio_length]

        # normalize the audio samples
        audio = (audio - self.mean) / self.std

        return audio, self.speaker_id_dict[speaker_id], filename, start_idx

    def __len__(self):
        return len(self.file_list)


    def get_audio_by_speaker(self, speaker_id, batch_size):
        batch_size = min(len(self.speaker_dict[speaker_id]), batch_size)
        batch = torch.zeros(batch_size, 1, self.audio_length)
        for idx in range(batch_size):
            batch[idx, 0, :], _, _, _ = self.__getitem__(
                self.speaker_dict[speaker_id][idx]
            )

        return batch
