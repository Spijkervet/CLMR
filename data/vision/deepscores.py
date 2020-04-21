import torch
import torch.utils.data as data
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
import torchvision.transforms

from .deepscores_processor import DeepScoresProcessor

class DeepScoresDataset(data.Dataset):
    def __init__(self, args, train=True, transform=None, target_transform=None):
        self.args = args
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.data_dir = os.path.join(args.data_input_dir, "deepscores")
        
        # if download:
        #     self.download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

        self.class_names = {}
        with open(Path(self.data_dir) / "class_names.csv", "r") as f:
            for l in f.readlines():
                class_id, name = l.split(",")
                class_id = int(class_id)
                name = name.rstrip()
                self.class_names[class_id] = name

        train_data_path = Path(self.data_dir) / "train.npy"
        if not os.path.exists(train_data_path):
            print("### Processing DeepScores dataset ###")
            processor = DeepScoresProcessor(path=self.data_dir, seed=args.seed)
            processor.read_images()

        if self.train:
            self.data = np.load(train_data_path)
            self.targets = np.load(
                Path(self.data_dir) / "train_annotations.npy"
            )
        else:
            self.data = np.load(Path(self.data_dir) / "test.npy")
            self.targets = np.load(
                Path(self.data_dir) / "test_annotations.npy"
            )

        self.targets_dict = defaultdict(list)
        for x, y in zip(self.data, self.targets):
            class_id = y.argmax()  # one-hot
            self.targets_dict[class_id].append(x)

        self.sample_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor()
            ]
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = target.astype(np.float32)
        return img, target

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

    def get_class_name(self, class_id):
        return self.class_names[class_id]

    def sample_from_class_id(self, class_id, batch_size=40):
        batch_size = min(batch_size, len(self.targets_dict[class_id]))
        batch = torch.zeros(
            batch_size,
            self.args.image_channels,
            self.args.image_height,
            self.args.image_width,
        )
        for idx in range(batch_size):
            img = Image.fromarray(self.targets_dict[class_id][idx])
            img = self.sample_transform(img)
            batch[idx, :] = img
        return batch
