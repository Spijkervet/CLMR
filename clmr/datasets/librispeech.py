import os
import torchaudio
from torch.utils.data import Dataset


class LIBRISPEECH(Dataset):

    subset_map = {"train": "train-clean-100", "test": "test-clean"}

    def __init__(self, root, download, subset):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root, download=download, url=self.subset_map[subset]
        )

        self.speaker2idx = {}

        if not os.path.exists(self.dataset._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        self.speaker_ids = list(map(int, os.listdir(self.dataset._path)))
        for idx, speaker_id in enumerate(sorted(self.speaker_ids)):
            self.speaker2idx[speaker_id] = idx

        self.n_classes = len(self.speaker2idx.keys())

    def __getitem__(self, idx):
        (
            audio,
            sample_rate,
            utterance,
            speaker_id,
            chapter_id,
            utterance_id,
        ) = self.dataset[idx]
        label = self.speaker2idx[speaker_id]
        return audio, label

    def __len__(self):
        return len(self.dataset)
