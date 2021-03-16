import torchaudio
from torchaudio.datasets.gtzan import gtzan_genres
from torch.utils.data import Dataset


class GTZAN(Dataset):

    subset_map = {"train": "training", "valid": "validation", "test": "testing"}

    def __init__(self, root, download, subset):
        self.dataset = torchaudio.datasets.GTZAN(
            root=root, download=download, subset=self.subset_map[subset]
        )
        self.labels = gtzan_genres

        self.label2idx = {}
        for idx, label in enumerate(self.labels):
            self.label2idx[label] = idx

        self.n_classes = len(self.label2idx.keys())

    def __getitem__(self, idx):
        audio, sr, label = self.dataset[idx]
        label = self.label2idx[label]
        return audio, label

    def __len__(self):
        return len(self.dataset)
